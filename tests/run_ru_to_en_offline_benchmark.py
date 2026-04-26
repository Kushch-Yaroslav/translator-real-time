from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.benchmark.paths import (  # noqa: E402
    BENCHMARK_AUDIO_PATH,
    BENCHMARK_RUNS_DIR,
    BENCHMARK_SOURCE_DIR,
    ensure_benchmark_dirs,
)
from core.audio.audio_engine import AudioEngine  # noqa: E402
from core.audio.chunk_processor import ChunkProcessor, ChunkProcessorConfig  # noqa: E402
from core.config.app_config import (  # noqa: E402
    AppConfig,
    TranslationBranchConfig,
    load_app_config,
)
from core.pipeline.branch_runtime import (  # noqa: E402
    SPEAK_BRANCH_PROFILE,
    build_branch_runtime_config,
    resolve_runtime_branch_config,
)


@dataclass
class BenchmarkLogEvent:
    at_sec: float
    message: str


@dataclass
class OfflineBenchmarkReport:
    input_path: str
    duration_sec: float
    first_speech_at_sec: float | None
    first_queue_at_sec: float | None
    first_translate_at_sec: float | None
    first_tts_ready_at_sec: float | None
    queue_start_delay_sec: float | None
    translate_start_delay_sec: float | None
    tts_ready_start_delay_sec: float | None
    queued_segments: list[str]
    translated_segments: list[str]
    final_segments: list[str]
    duplicate_queued_segments: list[str]
    duplicate_translated_segments: list[str]
    long_tts_segments: list[str]
    long_translation_gaps: list[str]
    log_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless offline benchmark for the RU=>EN realtime AudioEngine pipeline."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Optional path to the benchmark audio file. If omitted, recordings/benchmarks/ru_to_en/source/current_benchmark.wav is used.",
    )
    parser.add_argument("--pace", dest="pace", action="store_true", help="Feed audio in real time.")
    parser.add_argument("--no-pace", dest="pace", action="store_false", help="Feed audio as fast as possible.")
    parser.add_argument(
        "--pace-factor",
        type=float,
        default=1.0,
        help="Realtime pacing factor when --pace is enabled. 1.0 = real time, 2.0 = twice as fast.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=40,
        help="Chunk size sent into realtime STT. 40ms is closer to live streaming than large offline chunks.",
    )
    parser.add_argument(
        "--tts-long-sec",
        type=float,
        default=2.8,
        help="Report TTS chunks longer than this threshold.",
    )
    parser.add_argument(
        "--gap-long-sec",
        type=float,
        default=2.2,
        help="Report translation gaps longer than this threshold.",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        help="Prewarm runtime models before the benchmark run.",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip runtime prewarm before the benchmark run.",
    )
    parser.add_argument(
        "--use-cuda-tts",
        choices=("auto", "on", "off"),
        default="auto",
        help="Override Piper CUDA usage. STT/translation still use runtime defaults.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BENCHMARK_RUNS_DIR),
        help="Directory where run artifacts will be written.",
    )
    parser.set_defaults(pace=True, warmup=True)
    return parser.parse_args()


def pick_input_path(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    if not BENCHMARK_AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark audio file does not exist yet: {BENCHMARK_AUDIO_PATH}"
        )
    return BENCHMARK_AUDIO_PATH


def decode_audio_to_float32_mono(path: Path, target_samplerate: int) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_samplerate),
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False), target_samplerate


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_compare_text(text: str) -> str:
    text = normalize_text(text).lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split()).strip()


def find_first_speech_at_sec(audio: np.ndarray, samplerate: int, threshold: float) -> float | None:
    if audio.size == 0:
        return None

    window_samples = max(1, int(0.04 * samplerate))
    for start in range(0, len(audio), window_samples):
        chunk = audio[start: start + window_samples]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-10))
        if rms >= threshold:
            return start / float(samplerate)
    return None


def detect_duplicates(items: list[str]) -> list[str]:
    duplicates: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_compare_text(item)
        if not normalized:
            continue
        if normalized in seen:
            duplicates.append(item)
            continue
        seen.add(normalized)
    return duplicates


def extract_payload(message: str, prefix: str) -> str:
    if not message.startswith(prefix):
        return ""
    return normalize_text(message[len(prefix):])


def extract_tts_duration(message: str) -> float | None:
    if not message.startswith("TTS audio ready: "):
        return None
    value = message.removeprefix("TTS audio ready: ").removesuffix(" sec").strip()
    try:
        return float(value)
    except ValueError:
        return None


def build_runtime_config(use_cuda_tts: str) -> tuple[AppConfig, TranslationBranchConfig]:
    base_config = load_app_config()
    runtime_config = build_branch_runtime_config(base_config, SPEAK_BRANCH_PROFILE)
    active_branch = resolve_runtime_branch_config(base_config, SPEAK_BRANCH_PROFILE)

    if use_cuda_tts == "on":
        runtime_config.tts.use_cuda = True
    elif use_cuda_tts == "off":
        runtime_config.tts.use_cuda = False

    return runtime_config, active_branch


def start_headless_engine(
    config: AppConfig,
    active_branch: TranslationBranchConfig,
    *,
    samplerate: int,
    channels: int,
    blocksize: int,
    log_events: list[BenchmarkLogEvent],
) -> tuple[AudioEngine, float]:
    engine = AudioEngine(config, active_branch_config=active_branch)
    started_at = time.perf_counter()
    log_lock = threading.Lock()

    def on_log(message: str) -> None:
        with log_lock:
            log_events.append(
                BenchmarkLogEvent(
                    at_sec=time.perf_counter() - started_at,
                    message=normalize_text(message),
                )
            )

    engine.on_log = on_log
    engine.on_error = on_log
    engine.current_samplerate = samplerate
    engine.current_channels = channels
    engine.current_blocksize = blocksize
    engine._stt_speech_hangover_chunks = 0
    engine._stt_speech_hangover_max_chunks = max(
        1,
        int(
            round(
                engine.app_config.stt.noise_gate_hangover_sec
                / (blocksize / float(samplerate))
            )
        ),
    )

    engine.last_final_text = ""
    engine.last_translated_text = ""
    engine.last_enqueued_final_text = ""
    engine.last_emitted_source_text = ""
    engine.last_translated_at = 0.0
    engine._last_stt_activity_at = 0.0
    engine._clear_final_text_queue()
    engine._clear_low_latency_queue()
    engine._clear_tts_text_queue()
    engine._clear_pending_final()
    engine._clear_partial_state()

    engine.processor = ChunkProcessor(
        ChunkProcessorConfig(
            samplerate=samplerate,
            channels=channels,
        )
    )

    branch_config = engine._get_active_branch_config()
    if not engine._stt_backend_outputs_translated_text():
        engine.translation_service, _translation_cached = engine._get_or_create_translation_service(branch_config)
    else:
        engine.translation_service = None

    engine.tts_service, _tts_cached = engine._get_or_create_tts_service(branch_config)
    engine.realtime_stt, _stt_label, _stt_cached = engine._get_or_create_realtime_stt_service(branch_config)
    engine.realtime_stt.start(
        partial_callback=engine._on_realtime_partial,
        final_callback=engine._on_realtime_final,
    )

    engine.running = True
    engine._tts_stop_event.clear()
    engine.tts_worker_thread = threading.Thread(
        target=engine._tts_worker_loop,
        daemon=True,
        name="offline-tts-worker",
    )
    engine.tts_worker_thread.start()

    engine.phrase_worker_thread = threading.Thread(
        target=engine._phrase_processing_loop,
        daemon=True,
        name="offline-phrase-worker",
    )
    engine.phrase_worker_thread.start()

    engine.final_worker_thread = threading.Thread(
        target=engine._final_debounce_loop,
        daemon=True,
        name="offline-final-worker",
    )
    engine.final_worker_thread.start()

    return engine, started_at


def stop_headless_engine(engine: AudioEngine) -> None:
    engine.running = False
    engine._tts_stop_event.set()

    if engine.realtime_stt is not None:
        try:
            engine.realtime_stt.stop()
        except Exception as error:
            engine._handle_error(f"Realtime STT stop error: {error}")

    for thread in (
        engine.phrase_worker_thread,
        engine.final_worker_thread,
        engine.tts_worker_thread,
    ):
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    engine.realtime_stt = None
    engine.translation_service = None
    engine.tts_service = None
    engine.phrase_worker_thread = None
    engine.final_worker_thread = None
    engine.tts_worker_thread = None
    engine._clear_final_text_queue()
    engine._clear_low_latency_queue()
    engine._clear_tts_text_queue()
    engine._clear_pending_final()
    engine._clear_partial_state()


def wait_for_pipeline_to_flush(engine: AudioEngine, max_wait_sec: float = 20.0) -> None:
    deadline = time.monotonic() + max_wait_sec
    stable_empty_since: float | None = None

    while time.monotonic() < deadline:
        queues_empty = (
            engine.low_latency_text_queue.empty()
            and engine.final_text_queue.empty()
            and engine.tts_text_queue.empty()
        )
        if queues_empty:
            if stable_empty_since is None:
                stable_empty_since = time.monotonic()
            elif (time.monotonic() - stable_empty_since) >= 0.6:
                return
        else:
            stable_empty_since = None
        time.sleep(0.05)


def run_benchmark(args: argparse.Namespace) -> OfflineBenchmarkReport:
    ensure_benchmark_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("benchmark: build runtime config", flush=True)
    runtime_config, active_branch = build_runtime_config(args.use_cuda_tts)
    samplerate = runtime_config.audio.samplerate
    print("benchmark: pick input", flush=True)
    input_path = pick_input_path(args.input_path)
    print(f"benchmark: decode audio {input_path}", flush=True)
    audio, samplerate = decode_audio_to_float32_mono(input_path, samplerate)

    first_speech_at = find_first_speech_at_sec(
        audio,
        samplerate,
        runtime_config.stt.noise_gate_threshold,
    )

    log_events: list[BenchmarkLogEvent] = []
    print("benchmark: start headless engine", flush=True)
    engine, log_started_at = start_headless_engine(
        runtime_config,
        active_branch,
        samplerate=samplerate,
        channels=runtime_config.audio.channels,
        blocksize=runtime_config.audio.blocksize,
        log_events=log_events,
    )

    if args.warmup:
        print("benchmark: warmup runtime", flush=True)
        engine.prewarm_runtime()

    chunk_samples = max(1, int((args.chunk_ms / 1000.0) * samplerate))
    feed_started_offset_sec: float | None = None

    try:
        if engine.realtime_stt is None:
            raise RuntimeError("Realtime STT was not initialized")

        print("benchmark: feed audio", flush=True)
        feed_started_offset_sec = time.perf_counter() - log_started_at
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset: offset + chunk_samples]
            if chunk.size == 0:
                continue

            chunk_2d = chunk.reshape(-1, 1).astype(np.float32, copy=False)
            processed_chunk = engine.process_chunk(chunk_2d)
            stt_chunk = engine._prepare_chunk_for_stt(processed_chunk)
            engine._append_utterance_audio_chunk(stt_chunk)
            engine.realtime_stt.send_audio_chunk(stt_chunk, samplerate)

            if args.pace:
                pace_factor = max(args.pace_factor, 0.01)
                time.sleep((args.chunk_ms / 1000.0) / pace_factor)

        print("benchmark: commit + done", flush=True)
        engine.realtime_stt.commit()
        engine.realtime_stt.send_done()
        print("benchmark: wait flush", flush=True)
        wait_for_pipeline_to_flush(engine)
        time.sleep(0.35)
    finally:
        print("benchmark: stop headless engine", flush=True)
        stop_headless_engine(engine)

    print("benchmark: summarize", flush=True)
    queued_segments: list[str] = []
    translated_segments: list[str] = []
    final_segments: list[str] = []
    long_tts_segments: list[str] = []
    long_translation_gaps: list[str] = []

    first_queue_at: float | None = None
    first_translate_at: float | None = None
    first_tts_ready_at: float | None = None
    last_translated_at: float | None = None

    for event in log_events:
        adjusted_at_sec = event.at_sec
        if feed_started_offset_sec is not None:
            adjusted_at_sec -= feed_started_offset_sec
        message = event.message

        queued = extract_payload(message, "LOWLAT sentence queued: ")
        if not queued:
            queued = extract_payload(message, "LOWLAT final tail queued: ")
        if queued:
            queued_segments.append(queued)
            if first_queue_at is None:
                first_queue_at = adjusted_at_sec
            continue

        translated = extract_payload(message, "TRANSLATED: ")
        if translated:
            translated_segments.append(translated)
            if first_translate_at is None:
                first_translate_at = adjusted_at_sec
            if last_translated_at is not None:
                gap = adjusted_at_sec - last_translated_at
                if gap > args.gap_long_sec:
                    long_translation_gaps.append(
                        f"{gap:.2f}s gap before '{translated}'"
                    )
            last_translated_at = adjusted_at_sec
            continue

        final_text = extract_payload(message, "FINAL: ")
        if final_text:
            final_segments.append(final_text)
            continue

        tts_duration = extract_tts_duration(message)
        if tts_duration is not None:
            if first_tts_ready_at is None:
                first_tts_ready_at = adjusted_at_sec
            if tts_duration > args.tts_long_sec:
                long_tts_segments.append(f"{tts_duration:.2f}s")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = output_dir / f"offline_benchmark_{timestamp}.log"
    report_path = output_dir / f"offline_benchmark_{timestamp}.json"

    log_path.write_text(
        "\n".join(
            f"[{event.at_sec:7.3f}s] {event.message}"
            for event in log_events
        ) + "\n",
        encoding="utf-8",
    )

    def delta_or_none(value: Optional[float], origin: Optional[float]) -> float | None:
        if value is None or origin is None:
            return None
        return round(value - origin, 3)

    report = OfflineBenchmarkReport(
        input_path=str(input_path),
        duration_sec=round(len(audio) / float(samplerate), 3),
        first_speech_at_sec=None if first_speech_at is None else round(first_speech_at, 3),
        first_queue_at_sec=None if first_queue_at is None else round(first_queue_at, 3),
        first_translate_at_sec=None if first_translate_at is None else round(first_translate_at, 3),
        first_tts_ready_at_sec=None if first_tts_ready_at is None else round(first_tts_ready_at, 3),
        queue_start_delay_sec=delta_or_none(first_queue_at, first_speech_at),
        translate_start_delay_sec=delta_or_none(first_translate_at, first_speech_at),
        tts_ready_start_delay_sec=delta_or_none(first_tts_ready_at, first_speech_at),
        queued_segments=queued_segments,
        translated_segments=translated_segments,
        final_segments=final_segments,
        duplicate_queued_segments=detect_duplicates(queued_segments),
        duplicate_translated_segments=detect_duplicates(translated_segments),
        long_tts_segments=long_tts_segments,
        long_translation_gaps=long_translation_gaps,
        log_path=str(log_path),
    )

    report_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"input: {input_path}")
    print(f"duration_sec: {report.duration_sec:.2f}")
    print(f"first_speech_at_sec: {report.first_speech_at_sec}")
    print(f"queue_start_delay_sec: {report.queue_start_delay_sec}")
    print(f"translate_start_delay_sec: {report.translate_start_delay_sec}")
    print(f"tts_ready_start_delay_sec: {report.tts_ready_start_delay_sec}")
    print(f"queued_segments: {len(report.queued_segments)}")
    print(f"translated_segments: {len(report.translated_segments)}")
    print(f"duplicate_queued_segments: {len(report.duplicate_queued_segments)}")
    print(f"duplicate_translated_segments: {len(report.duplicate_translated_segments)}")
    print(f"log: {log_path}")
    print(f"report: {report_path}")

    return report


def main() -> int:
    args = parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
