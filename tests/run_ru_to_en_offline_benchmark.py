from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

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
class BenchmarkMetrics:
    queued_segments_count: int = 0
    translated_segments_count: int = 0
    final_segments_count: int = 0

    duplicate_queued_count: int = 0
    duplicate_translated_count: int = 0
    semantic_overlap_count: int = 0

    long_tts_segments_count: int = 0
    long_translation_gaps_count: int = 0

    avg_translation_gap_sec: float | None = None
    max_translation_gap_sec: float | None = None

    avg_tts_duration_sec: float | None = None
    max_tts_duration_sec: float | None = None

    total_pipeline_time_sec: float | None = None
    realtime_factor: float | None = None

    expected_word_count: int | None = None
    final_word_count: int | None = None
    word_count_diff: int | None = None
    coverage_ratio: float | None = None


@dataclass
class SemanticOverlapDiagnostic:
    stream: str
    index_a: int
    segment_a: str
    index_b: int
    segment_b: str
    overlap_score: float
    shared_keywords: list[str] = field(default_factory=list)


@dataclass
class FragmentCoverageDiagnostic:
    fragment_index: int
    expected_fragment: str
    best_queued_index: int | None = None
    best_queued_segment: str = ""
    queued_overlap_score: float = 0.0
    queued_shared_keywords: list[str] = field(default_factory=list)
    best_final_index: int | None = None
    best_final_segment: str = ""
    final_overlap_score: float = 0.0
    final_shared_keywords: list[str] = field(default_factory=list)
    status: str = "missing"


@dataclass
class QueuedFinalCoverageDiagnostic:
    queued_index: int
    queued_segment: str
    best_final_index: int | None = None
    best_final_segment: str = ""
    overlap_score: float = 0.0
    shared_keywords: list[str] = field(default_factory=list)
    status: str = "missing_in_final"


@dataclass
class OfflineBenchmarkReport:
    test_id: str
    input_path: str
    expected_path: str | None = None
    expected_text: str | None = None
    duration_sec: float = 0.0

    first_speech_at_sec: float | None = None
    first_queue_at_sec: float | None = None
    first_translate_at_sec: float | None = None
    first_tts_ready_at_sec: float | None = None

    queue_start_delay_sec: float | None = None
    translate_start_delay_sec: float | None = None
    tts_ready_start_delay_sec: float | None = None

    queued_segments: list[str] = field(default_factory=list)
    translated_segments: list[str] = field(default_factory=list)
    final_segments: list[str] = field(default_factory=list)

    duplicate_queued_segments: list[str] = field(default_factory=list)
    duplicate_translated_segments: list[str] = field(default_factory=list)
    suspicious_semantic_overlaps: list[SemanticOverlapDiagnostic] = field(default_factory=list)
    suspicious_returns: list[SemanticOverlapDiagnostic] = field(default_factory=list)
    final_coverage_by_fragment: list[FragmentCoverageDiagnostic] = field(default_factory=list)
    lost_expected_fragments: list[FragmentCoverageDiagnostic] = field(default_factory=list)
    queued_not_covered_by_final: list[QueuedFinalCoverageDiagnostic] = field(default_factory=list)

    long_tts_segments: list[str] = field(default_factory=list)
    long_translation_gaps: list[str] = field(default_factory=list)

    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    status: str = "success"
    error: str | None = None
    log_path: str = ""


@dataclass
class BenchmarkSummaryTestItem:
    test_id: str
    status: str
    result_path: str


@dataclass
class BenchmarkSummaryAverage:
    duration_sec: float = 0.0
    queue_start_delay_sec: float = 0.0
    translate_start_delay_sec: float = 0.0
    tts_ready_start_delay_sec: float = 0.0
    total_pipeline_time_sec: float = 0.0
    realtime_factor: float = 0.0
    duplicate_queued_count: float = 0.0
    duplicate_translated_count: float = 0.0
    semantic_overlap_count: float = 0.0
    long_translation_gaps_count: float = 0.0
    long_tts_segments_count: float = 0.0
    coverage_ratio: float = 0.0


@dataclass
class BenchmarkSummary:
    run_id: str
    started_at: str
    finished_at: str
    mode: str
    concurrency: int
    total_tests: int
    success_tests: int
    failed_tests: int
    average: BenchmarkSummaryAverage
    tests: list[BenchmarkSummaryTestItem]
    comparison_path: str | None = None


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
        "--category",
        help="Run all benchmarks in the specified category.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks in all categories.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max parallel benchmark runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where run artifacts will be written. If None, a new run_TIMESTAMP dir is created in BENCHMARK_RUNS_DIR.",
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


def normalize_translated_compare_text(text: str) -> str:
    text = normalize_text(text).lower()
    return "".join(ch for ch in text if ch.isalnum())


_SEMANTIC_STOP_WORDS = {
    "я",
    "ты",
    "он",
    "она",
    "оно",
    "мы",
    "вы",
    "они",
    "это",
    "как",
    "что",
    "чем",
    "чтобы",
    "в",
    "во",
    "на",
    "по",
    "из",
    "у",
    "к",
    "ко",
    "с",
    "со",
    "за",
    "от",
    "до",
    "под",
    "над",
    "при",
    "и",
    "но",
    "то",
    "так",
    "потому",
    "если",
    "когда",
    "мой",
    "моя",
    "мои",
    "моё",
    "его",
    "ее",
    "её",
    "их",
    "наш",
    "наша",
    "наши",
    "ваш",
    "ваша",
    "ваши",
    "для",
    "через",
    "эт",
    "этот",
    "эта",
    "эти",
    "также",
    "еще",
    "ещё",
}

_RUSSIAN_STEM_SUFFIXES = (
    "аются", "яются", "аются", "яются", "ились", "ались", "ились",
    "ывать", "ивать", "овать", "ировать",
    "аться", "яться", "иться", "еться", "нуть",
    "аешь", "яешь", "аете", "яете", "ают", "яют",
    "уешь", "уете", "уют", "ишь", "ите", "им", "ит", "ят", "ют",
    "ался", "ялся", "илась", "илась", "илось", "ались", "ялись",
    "ал", "ала", "ало", "али", "ял", "яла", "яло", "яли",
    "аю", "яю", "ую", "юю", "ешь", "ете", "ем", "ут", "ют",
    "ить", "ать", "ять", "еть", "оть", "уть", "ти",
    "иями", "ями", "ами", "его", "ого", "ему", "ому", "ыми", "ими",
    "иях", "ах", "ях", "ия", "ья", "ие", "ье", "ий", "ый", "ой",
    "ая", "яя", "ое", "ее", "ую", "юю", "ом", "ем", "ам", "ям",
    "ов", "ев", "ей", "иям", "ием",
    "а", "я", "ы", "и", "е", "о", "у", "ю", "ь",
)


def normalize_semantic_word(word: str) -> str:
    word = normalize_compare_text(word)
    if not word:
        return ""

    if word.endswith(("ся", "сь")) and len(word) > 4:
        word = word[:-2]

    if len(word) <= 4:
        return word

    for suffix in _RUSSIAN_STEM_SUFFIXES:
        if len(word) - len(suffix) < 4:
            continue
        if word.endswith(suffix):
            word = word[:-len(suffix)]
            break

    if len(word) > 5 and word.endswith(("ова", "ева")):
        word = word[:-3]

    if len(word) > 5 and word.endswith(("ир", "ыва", "ива")):
        word = word[:-1] if word.endswith("ир") else word[:-3]

    return word


def extract_semantic_keywords(text: str) -> list[str]:
    normalized = normalize_compare_text(text)
    if not normalized:
        return []

    keywords: list[str] = []
    seen: set[str] = set()
    for raw_word in normalized.split():
        if raw_word in _SEMANTIC_STOP_WORDS:
            continue
        keyword = normalize_semantic_word(raw_word)
        if not keyword or keyword in _SEMANTIC_STOP_WORDS or keyword in seen:
            continue
        seen.add(keyword)
        keywords.append(keyword)
    return keywords


def analyze_semantic_overlaps(
    items: list[str],
    *,
    stream: str,
    window_size: int = 5,
) -> list[SemanticOverlapDiagnostic]:
    diagnostics: list[SemanticOverlapDiagnostic] = []
    if len(items) < 2:
        return diagnostics

    for index_b, segment_b in enumerate(items):
        compare_b = normalize_compare_text(segment_b)
        if not compare_b:
            continue
        keywords_b = extract_semantic_keywords(segment_b)
        if not keywords_b:
            continue
        set_b = set(keywords_b)

        start_index = max(0, index_b - window_size)
        for index_a in range(start_index, index_b):
            segment_a = items[index_a]
            compare_a = normalize_compare_text(segment_a)
            if not compare_a or compare_a == compare_b:
                continue

            keywords_a = extract_semantic_keywords(segment_a)
            if not keywords_a:
                continue
            set_a = set(keywords_a)

            shared = sorted(set_a & set_b)
            if not shared:
                continue

            overlap_score = len(shared) / float(min(len(set_a), len(set_b)))
            if len(shared) >= 2:
                is_suspicious = overlap_score >= 0.5
            else:
                is_suspicious = overlap_score >= 0.8 and min(len(set_a), len(set_b)) <= 2

            if not is_suspicious:
                continue

            diagnostics.append(
                SemanticOverlapDiagnostic(
                    stream=stream,
                    index_a=index_a,
                    segment_a=segment_a,
                    index_b=index_b,
                    segment_b=segment_b,
                    overlap_score=round(overlap_score, 2),
                    shared_keywords=shared,
                )
            )

    return diagnostics


def split_text_into_semantic_fragments(text: str) -> list[str]:
    text = normalize_text((text or "").replace("\n", " "))
    if not text:
        return []

    fragments: list[str] = []
    buffer = ""
    for character in text:
        buffer += character
        if character in ".!?":
            fragment = normalize_text(buffer)
            if fragment:
                fragments.append(fragment)
            buffer = ""
            continue

        if character in ",;:":
            candidate = normalize_text(buffer)
            if len(candidate.split()) >= 4:
                fragments.append(candidate)
                buffer = ""

    trailing = normalize_text(buffer)
    if trailing:
        fragments.append(trailing)

    return fragments


def _semantic_match_score(
    source_keywords: list[str],
    target_keywords: list[str],
) -> tuple[float, list[str]]:
    if not source_keywords or not target_keywords:
        return 0.0, []

    shared = sorted(set(source_keywords) & set(target_keywords))
    if not shared:
        return 0.0, []

    score = len(shared) / float(len(set(source_keywords)))
    return round(score, 2), shared


def _is_meaningful_fragment_match(
    score: float,
    shared_keywords: list[str],
    source_keywords: list[str],
    target_keywords: list[str],
) -> bool:
    if not shared_keywords:
        return False
    if len(shared_keywords) >= 3 and score >= 0.5:
        return True
    if len(shared_keywords) >= 2 and score >= 0.6:
        return True
    if score >= 1.0 and min(len(set(source_keywords)), len(set(target_keywords))) <= 2:
        return True
    return False


def _find_best_fragment_match(
    fragment: str,
    candidates: list[str],
) -> tuple[int | None, str, float, list[str]]:
    source_keywords = extract_semantic_keywords(fragment)
    if not source_keywords:
        return None, "", 0.0, []

    best_index: int | None = None
    best_segment = ""
    best_score = 0.0
    best_shared: list[str] = []
    best_candidate_keywords: list[str] = []

    for index, candidate in enumerate(candidates):
        candidate_keywords = extract_semantic_keywords(candidate)
        score, shared = _semantic_match_score(source_keywords, candidate_keywords)
        if score > best_score or (score == best_score and len(shared) > len(best_shared)):
            best_index = index
            best_segment = candidate
            best_score = score
            best_shared = shared
            best_candidate_keywords = candidate_keywords

    if best_index is None:
        return None, "", 0.0, []

    if not _is_meaningful_fragment_match(best_score, best_shared, source_keywords, best_candidate_keywords):
        return None, "", 0.0, []

    return best_index, best_segment, best_score, best_shared


def build_fragment_coverage_report(
    expected_text: str,
    queued_segments: list[str],
    final_segments: list[str],
) -> tuple[list[FragmentCoverageDiagnostic], list[FragmentCoverageDiagnostic], list[QueuedFinalCoverageDiagnostic]]:
    expected_fragments = split_text_into_semantic_fragments(expected_text)
    coverage_report: list[FragmentCoverageDiagnostic] = []
    lost_fragments: list[FragmentCoverageDiagnostic] = []

    for fragment_index, expected_fragment in enumerate(expected_fragments):
        queued_index, queued_segment, queued_score, queued_shared = _find_best_fragment_match(
            expected_fragment,
            queued_segments,
        )
        final_index, final_segment, final_score, final_shared = _find_best_fragment_match(
            expected_fragment,
            final_segments,
        )

        if final_index is not None:
            status = "covered_by_final"
        elif queued_index is not None:
            status = "covered_by_queue_only"
        else:
            status = "missing_before_queue"

        diagnostic = FragmentCoverageDiagnostic(
            fragment_index=fragment_index,
            expected_fragment=expected_fragment,
            best_queued_index=queued_index,
            best_queued_segment=queued_segment,
            queued_overlap_score=queued_score,
            queued_shared_keywords=queued_shared,
            best_final_index=final_index,
            best_final_segment=final_segment,
            final_overlap_score=final_score,
            final_shared_keywords=final_shared,
            status=status,
        )
        coverage_report.append(diagnostic)
        if status != "covered_by_final":
            lost_fragments.append(diagnostic)

    queued_not_in_final: list[QueuedFinalCoverageDiagnostic] = []
    for queued_index, queued_segment in enumerate(queued_segments):
        final_index, final_segment, score, shared = _find_best_fragment_match(
            queued_segment,
            final_segments,
        )
        status = "covered_by_final" if final_index is not None else "missing_in_final"
        if status == "missing_in_final":
            queued_not_in_final.append(
                QueuedFinalCoverageDiagnostic(
                    queued_index=queued_index,
                    queued_segment=queued_segment,
                    best_final_index=final_index,
                    best_final_segment=final_segment,
                    overlap_score=score,
                    shared_keywords=shared,
                    status=status,
                )
            )

    return coverage_report, lost_fragments, queued_not_in_final


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


def detect_duplicates(items: list[str], *, normalizer=normalize_compare_text) -> list[str]:
    duplicates: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalizer(item)
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

    engine.stop()


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


def run_benchmark(
    args: argparse.Namespace,
    input_path: Path,
    output_dir: Path,
    test_id: str = "current",
) -> OfflineBenchmarkReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_path = input_path.with_suffix(".expected.txt")
    expected_text: str | None = None
    if expected_path.exists():
        expected_text = expected_path.read_text(encoding="utf-8").strip()

    runtime_config, active_branch = build_runtime_config(args.use_cuda_tts)
    samplerate = runtime_config.audio.samplerate
    audio, samplerate = decode_audio_to_float32_mono(input_path, samplerate)

    first_speech_at = find_first_speech_at_sec(
        audio,
        samplerate,
        runtime_config.stt.noise_gate_threshold,
    )

    log_events: list[BenchmarkLogEvent] = []
    engine, log_started_at = start_headless_engine(
        runtime_config,
        active_branch,
        samplerate=samplerate,
        channels=runtime_config.audio.channels,
        blocksize=runtime_config.audio.blocksize,
        log_events=log_events,
    )

    if args.warmup:
        engine.prewarm_runtime()

    chunk_samples = max(1, int((args.chunk_ms / 1000.0) * samplerate))
    feed_started_offset_sec: float | None = None
    start_pipeline_time = time.perf_counter()

    try:
        if engine.realtime_stt is None:
            raise RuntimeError("Realtime STT was not initialized")

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

        engine.realtime_stt.commit()
        engine.realtime_stt.send_done()
        wait_for_pipeline_to_flush(engine)
        time.sleep(0.35)
    except Exception as e:
        return OfflineBenchmarkReport(
            test_id=test_id,
            input_path=str(input_path),
            status="failed",
            error=str(e),
        )
    finally:
        stop_headless_engine(engine)

    total_pipeline_time = time.perf_counter() - start_pipeline_time

    queued_segments: list[str] = []
    translated_segments: list[str] = []
    final_segments: list[str] = []
    long_tts_segments: list[str] = []
    long_translation_gaps: list[str] = []
    tts_durations: list[float] = []
    translation_gaps: list[float] = []

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
                translation_gaps.append(gap)
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
            tts_durations.append(tts_duration)
            if first_tts_ready_at is None:
                first_tts_ready_at = adjusted_at_sec
            if tts_duration > args.tts_long_sec:
                long_tts_segments.append(f"{tts_duration:.2f}s")

    test_name = Path(input_path).stem
    log_path = output_dir / f"{test_name}.log"
    report_path = output_dir / f"{test_name}.json"

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

    duration_sec = round(len(audio) / float(samplerate), 3)
    suspicious_semantic_overlaps = [
        *analyze_semantic_overlaps(queued_segments, stream="queued"),
        *analyze_semantic_overlaps(final_segments, stream="final"),
    ]
    final_coverage_by_fragment: list[FragmentCoverageDiagnostic] = []
    lost_expected_fragments: list[FragmentCoverageDiagnostic] = []
    queued_not_covered_by_final: list[QueuedFinalCoverageDiagnostic] = []
    suspicious_returns = [
        overlap for overlap in suspicious_semantic_overlaps
        if overlap.stream == "queued"
    ]

    metrics = BenchmarkMetrics(
        queued_segments_count=len(queued_segments),
        translated_segments_count=len(translated_segments),
        final_segments_count=len(final_segments),
        duplicate_queued_count=len(detect_duplicates(queued_segments)),
        duplicate_translated_count=len(
            detect_duplicates(
                translated_segments,
                normalizer=normalize_translated_compare_text,
            )
        ),
        semantic_overlap_count=len(suspicious_semantic_overlaps),
        long_tts_segments_count=len(long_tts_segments),
        long_translation_gaps_count=len(long_translation_gaps),
        avg_translation_gap_sec=round(sum(translation_gaps) / len(translation_gaps), 3) if translation_gaps else None,
        max_translation_gap_sec=round(max(translation_gaps), 3) if translation_gaps else None,
        avg_tts_duration_sec=round(sum(tts_durations) / len(tts_durations), 3) if tts_durations else None,
        max_tts_duration_sec=round(max(tts_durations), 3) if tts_durations else None,
        total_pipeline_time_sec=round(total_pipeline_time, 3),
        realtime_factor=round(total_pipeline_time / duration_sec, 3) if duration_sec > 0 else None,
    )

    if expected_text:
        metrics.expected_word_count = len(expected_text.split())
        final_full_text = " ".join(final_segments)
        metrics.final_word_count = len(final_full_text.split())
        metrics.word_count_diff = metrics.final_word_count - metrics.expected_word_count
        if metrics.expected_word_count > 0:
            metrics.coverage_ratio = round(
                metrics.final_word_count / float(metrics.expected_word_count),
                3,
            )
        (
            final_coverage_by_fragment,
            lost_expected_fragments,
            queued_not_covered_by_final,
        ) = build_fragment_coverage_report(
            expected_text,
            queued_segments,
            final_segments,
        )

    report = OfflineBenchmarkReport(
        test_id=test_id,
        input_path=str(input_path),
        expected_path=str(expected_path) if expected_path.exists() else None,
        expected_text=expected_text,
        duration_sec=duration_sec,
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
        duplicate_translated_segments=detect_duplicates(
            translated_segments,
            normalizer=normalize_translated_compare_text,
        ),
        suspicious_semantic_overlaps=suspicious_semantic_overlaps,
        suspicious_returns=suspicious_returns,
        final_coverage_by_fragment=final_coverage_by_fragment,
        lost_expected_fragments=lost_expected_fragments,
        queued_not_covered_by_final=queued_not_covered_by_final,
        long_tts_segments=long_tts_segments,
        long_translation_gaps=long_translation_gaps,
        metrics=metrics,
        log_path=str(log_path),
        status="success",
    )

    report_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def compare_runs(current_summary: Dict[str, Any], previous_summary: Dict[str, Any]) -> Dict[str, Any]:
    def get_status(diff: float, field_name: str) -> str:
        if abs(diff) < 1e-6:
            return "same"
        # For these metrics, lower is better
        better_if_lower = [
            "realtime_factor",
            "duplicate_queued_count",
            "duplicate_translated_count",
            "semantic_overlap_count",
            "long_translation_gaps_count",
            "long_tts_segments_count",
            "queue_start_delay_sec",
            "translate_start_delay_sec",
            "tts_ready_start_delay_sec",
            "total_pipeline_time_sec",
            "word_count_diff",
        ]
        if field_name == "coverage_ratio":
            return "improved" if diff > 0 else "worse"
        if field_name in better_if_lower:
            return "improved" if diff < 0 else "worse"
        return "unknown"

    comparison = {
        "current_run_id": current_summary["run_id"],
        "previous_run_id": previous_summary["run_id"],
        "average_diff": {},
        "tests_diff": [],
    }

    fields_to_compare = [
        "queue_start_delay_sec",
        "translate_start_delay_sec",
        "tts_ready_start_delay_sec",
        "total_pipeline_time_sec",
        "realtime_factor",
        "duplicate_queued_count",
        "duplicate_translated_count",
        "semantic_overlap_count",
        "long_translation_gaps_count",
        "long_tts_segments_count",
        "word_count_diff",
        "coverage_ratio",
    ]

    curr_avg = current_summary.get("average", {})
    prev_avg = previous_summary.get("average", {})

    for field in fields_to_compare:
        curr_val = curr_avg.get(field)
        prev_val = prev_avg.get(field)

        if curr_val is not None and prev_val is not None:
            diff = curr_val - prev_val
            comparison["average_diff"][field] = {
                "previous": prev_val,
                "current": curr_val,
                "diff": round(diff, 4),
                "status": get_status(diff, field),
            }
        else:
            comparison["average_diff"][field] = {
                "previous": prev_val,
                "current": curr_val,
                "diff": None,
                "status": "unknown",
            }

    # Compare individual tests
    curr_tests = {t["test_id"]: t for t in current_summary.get("tests", [])}
    prev_tests = {t["test_id"]: t for t in previous_summary.get("tests", [])}

    for test_id, curr_test in curr_tests.items():
        if test_id in prev_tests:
            prev_test = prev_tests[test_id]
            # Try to load detailed reports if possible to compare more metrics
            try:
                curr_rep = json.loads(Path(curr_test["result_path"]).read_text(encoding="utf-8"))
                prev_rep = json.loads(Path(prev_test["result_path"]).read_text(encoding="utf-8"))

                test_metrics_diff = {}
                # Compare main metrics for the test
                # We can reuse fields_to_compare but need to map them to report structure
                # In report, some are top level, some are in .metrics
                
                def get_val(rep, field):
                    if field in rep: return rep[field]
                    if "metrics" in rep and field in rep["metrics"]: return rep["metrics"][field]
                    return None

                for field in fields_to_compare:
                    cv = get_val(curr_rep, field)
                    pv = get_val(prev_rep, field)
                    if cv is not None and pv is not None:
                        d = cv - pv
                        test_metrics_diff[field] = {
                            "previous": pv,
                            "current": cv,
                            "diff": round(d, 4),
                            "status": get_status(d, field)
                        }

                comparison["tests_diff"].append({
                    "test_id": test_id,
                    "status": "compared",
                    "metrics": test_metrics_diff
                })
            except Exception:
                comparison["tests_diff"].append({
                    "test_id": test_id,
                    "status": "unknown",
                    "metrics": {}
                })

    return comparison


def main() -> int:
    args = parse_args()
    ensure_benchmark_dirs()

    input_files: list[tuple[str, Path]] = []
    mode = "single"

    if args.all:
        mode = "all"
        for category_dir in BENCHMARK_SOURCE_DIR.iterdir():
            if category_dir.is_dir():
                for wav_file in category_dir.glob("*.wav"):
                    test_id = f"{category_dir.name}/{wav_file.stem}"
                    input_files.append((test_id, wav_file))
    elif args.category:
        mode = f"category:{args.category}"
        category_dir = BENCHMARK_SOURCE_DIR / args.category
        if category_dir.exists() and category_dir.is_dir():
            for wav_file in category_dir.glob("*.wav"):
                test_id = f"{args.category}/{wav_file.stem}"
                input_files.append((test_id, wav_file))
    else:
        input_path = pick_input_path(args.input_path)
        test_id = "current"
        if args.input_path:
            p = Path(args.input_path)
            if p.parent.parent == BENCHMARK_SOURCE_DIR:
                test_id = f"{p.parent.name}/{p.stem}"
        input_files.append((test_id, input_path))

    if not input_files:
        print("No benchmark files found.")
        return 1

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"run_{run_timestamp}"
    
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = BENCHMARK_RUNS_DIR / run_id

    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    started_at = datetime.now()
    reports: list[OfflineBenchmarkReport] = []

    print(f"Starting benchmark run: {run_id} (mode={mode}, concurrency={args.concurrency})")

    def run_one(item: tuple[str, Path]) -> OfflineBenchmarkReport:
        test_id, input_path = item
        # If multi-file, create subdirs for categories
        if "/" in test_id:
            category, _ = test_id.split("/", 1)
            test_output_dir = base_output_dir / category
        else:
            test_output_dir = base_output_dir
            
        print(f"  [RUN] {test_id} ...")
        rep = run_benchmark(args, input_path, test_output_dir, test_id=test_id)
        print(f"  [DONE] {test_id} (status={rep.status})")
        return rep

    if len(input_files) == 1:
        reports.append(run_one(input_files[0]))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            reports = list(executor.map(run_one, input_files))

    finished_at = datetime.now()
    
    # Calculate Summary
    success_reports = [r for r in reports if r.status == "success"]
    
    avg = BenchmarkSummaryAverage()
    if success_reports:
        count = len(success_reports)
        avg.duration_sec = round(sum(r.duration_sec for r in success_reports) / count, 2)
        avg.queue_start_delay_sec = round(sum((r.queue_start_delay_sec or 0) for r in success_reports) / count, 2)
        avg.translate_start_delay_sec = round(sum((r.translate_start_delay_sec or 0) for r in success_reports) / count, 2)
        avg.tts_ready_start_delay_sec = round(sum((r.tts_ready_start_delay_sec or 0) for r in success_reports) / count, 2)
        avg.total_pipeline_time_sec = round(sum((r.metrics.total_pipeline_time_sec or 0) for r in success_reports) / count, 2)
        avg.realtime_factor = round(sum((r.metrics.realtime_factor or 0) for r in success_reports) / count, 2)
        avg.duplicate_queued_count = round(sum(r.metrics.duplicate_queued_count for r in success_reports) / count, 2)
        avg.duplicate_translated_count = round(sum(r.metrics.duplicate_translated_count for r in success_reports) / count, 2)
        avg.semantic_overlap_count = round(sum(r.metrics.semantic_overlap_count for r in success_reports) / count, 2)
        avg.long_translation_gaps_count = round(sum(r.metrics.long_translation_gaps_count for r in success_reports) / count, 2)
        avg.long_tts_segments_count = round(sum(r.metrics.long_tts_segments_count for r in success_reports) / count, 2)
        coverage_values = [r.metrics.coverage_ratio for r in success_reports if r.metrics.coverage_ratio is not None]
        if coverage_values:
            avg.coverage_ratio = round(sum(coverage_values) / len(coverage_values), 3)

    summary = BenchmarkSummary(
        run_id=run_id,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        mode=mode,
        concurrency=args.concurrency,
        total_tests=len(input_files),
        success_tests=len(success_reports),
        failed_tests=len(input_files) - len(success_reports),
        average=avg,
        tests=[
            BenchmarkSummaryTestItem(
                test_id=r.test_id,
                status=r.status,
                result_path=str(base_output_dir / f"{r.test_id}.json" if "/" in r.test_id else base_output_dir / f"{Path(r.input_path).stem}.json")
            ) for r in reports
        ]
    )

    summary_path = base_output_dir / "summary.json"
    
    # Run Comparison
    comparison_path = None
    all_runs = sorted(BENCHMARK_RUNS_DIR.glob("run_*"))
    # The current run is likely the last one in all_runs if it's already in the directory
    # or it will be added after we create summary.json.
    # To be safe, let's find the previous run before this one.
    previous_run_dir = None
    for rdir in reversed(all_runs):
        if rdir.name != run_id:
            if (rdir / "summary.json").exists():
                previous_run_dir = rdir
                break
    
    if previous_run_dir:
        try:
            prev_summary = json.loads((previous_run_dir / "summary.json").read_text(encoding="utf-8"))
            comparison = compare_runs(asdict(summary), prev_summary)
            comp_file = base_output_dir / "comparison.json"
            comp_file.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            comparison_path = str(comp_file)
            summary.comparison_path = comparison_path
        except Exception as e:
            print(f"Failed to create comparison: {e}")

    summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Limit to 3 runs
    all_runs_updated = sorted(BENCHMARK_RUNS_DIR.glob("run_*"))
    if len(all_runs_updated) > 3:
        runs_to_delete = all_runs_updated[:-3]
        for rdir in runs_to_delete:
            print(f"Deleting old run: {rdir}")
            try:
                import shutil
                shutil.rmtree(rdir)
            except Exception as e:
                print(f"Failed to delete {rdir}: {e}")

    print(f"\nRun finished. Total: {summary.total_tests}, Success: {summary.success_tests}, Failed: {summary.failed_tests}")
    if comparison_path:
        print(f"Comparison created: {comparison_path}")
    print(f"Summary: {summary_path}")
    
    return 0 if summary.failed_tests == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
