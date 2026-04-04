from __future__ import annotations

import argparse
import json
import re
import time
import wave
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
import threading

import numpy as np

from core.sst.nim_realtime_stt_service import (
    NIMRealtimeSTTConfig,
    NIMRealtimeSTTService,
)
from core.translation.translation_service import (
    TranslationConfig,
    TranslationDirection,
    TranslationService,
)
from core.tts.tts_service import TTSConfig, TTSService


@dataclass
class PhraseEvaluation:
    index: int
    expected_text: str
    recognized_text: str
    translated_text: str
    wav_path: str
    raw_audio_seconds: float
    trimmed_audio_seconds: float
    stt_seconds: float
    translation_seconds: float
    tts_seconds: float
    tts_audio_seconds: float
    stt_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценить записанные русские тестовые фразы через локальный RU->EN pipeline."
    )
    parser.add_argument(
        "--dataset-dir",
        default="recordings/test_phrases_ru_to_en",
        help="Каталог с metadata.json и wav-файлами.",
    )
    parser.add_argument(
        "--output-json",
        default="recordings/test_phrases_ru_to_en/evaluation.json",
        help="Куда сохранить отчет.",
    )
    parser.add_argument(
        "--stt-url",
        default="http://localhost:9000",
        help="Base URL локального realtime STT.",
    )
    parser.add_argument(
        "--stt-ws-url",
        default="ws://localhost:9000/v1/realtime?intent=transcription",
        help="WebSocket URL локального realtime STT.",
    )
    parser.add_argument(
        "--tts-data-dir",
        default="/media/yaroslav/DATA/ai_models/piper",
        help="Каталог моделей Piper.",
    )
    parser.add_argument(
        "--tts-voice-name",
        default="en_US-lessac-medium",
        help="Английский голос Piper для оценки RU->EN.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.006,
        help="Порог RMS для обрезки тишины.",
    )
    parser.add_argument(
        "--padding-ms",
        type=float,
        default=120.0,
        help="Сколько аудио оставлять вокруг найденной речи.",
    )
    return parser.parse_args()


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        samplerate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported sample width: {sample_width * 8} bits")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32, copy=False), samplerate


def trim_silence(
    audio: np.ndarray,
    samplerate: int,
    threshold: float,
    padding_ms: float,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    frame_size = max(1, int(0.03 * samplerate))
    rms_values: list[float] = []

    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]
        if frame.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame)) + 1e-10))
        rms_values.append(rms)

    active_indices = [idx for idx, rms in enumerate(rms_values) if rms >= threshold]
    if not active_indices:
        return audio

    first_frame = active_indices[0]
    last_frame = active_indices[-1]
    padding_samples = int((padding_ms / 1000.0) * samplerate)

    start_sample = max(0, first_frame * frame_size - padding_samples)
    end_sample = min(len(audio), (last_frame + 1) * frame_size + padding_samples)
    return audio[start_sample:end_sample].astype(np.float32, copy=False)


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split())


def collapse_repeated_sentences(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    sentence_pattern = re.compile(r"[^.!?]+[.!?]?")
    raw_sentences = [segment.strip() for segment in sentence_pattern.findall(text) if segment.strip()]
    if len(raw_sentences) < 2:
        return text

    collapsed_sentences: list[str] = []
    last_norm = ""

    for sentence in raw_sentences:
        sentence_norm = normalize_text(sentence)
        if not sentence_norm:
            continue
        if sentence_norm == last_norm:
            continue

        collapsed_sentences.append(sentence)
        last_norm = sentence_norm

    collapsed_text = " ".join(collapsed_sentences).strip()
    return collapsed_text or text


def similarity_score(expected: str, actual: str) -> float:
    return SequenceMatcher(
        None,
        normalize_text(expected),
        normalize_text(actual),
    ).ratio()


def transcribe_with_realtime_stt(
    audio: np.ndarray,
    samplerate: int,
    base_url: str,
    ws_url: str,
) -> str:
    final_segments: list[str] = []
    done_event = threading.Event()

    def on_final(text: str) -> None:
        normalized = collapse_repeated_sentences((text or "").strip())
        if normalized:
            final_segments.append(normalized)
        done_event.set()

    service = NIMRealtimeSTTService(
        NIMRealtimeSTTConfig(
            base_url=base_url,
            ws_url=ws_url,
            language="ru-RU",
            sample_rate_hz=16000,
            num_channels=1,
            timeout=10.0,
            commit_interval_sec=0.35,
            enable_automatic_punctuation=True,
            on_log=None,
        )
    )

    try:
        service.start(partial_callback=None, final_callback=on_final)

        chunk_samples = max(1, int(0.2 * samplerate))
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if chunk.size == 0:
                continue
            service.send_audio_chunk(chunk, samplerate)

        service.commit()
        service.send_done()
        done_event.wait(timeout=8.0)
    finally:
        service.stop()

    if final_segments:
        return collapse_repeated_sentences(" ".join(final_segments).strip())

    return collapse_repeated_sentences(service.get_last_final_text().strip())


def mean_value(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    print("Loading translation model RU->EN...")
    translation_service = TranslationService(
        TranslationConfig(
            direction=TranslationDirection.RU_TO_EN,
            enabled=True,
        )
    )

    print("Loading English TTS voice...")
    tts_service = TTSService(
        TTSConfig(
            voice_name=args.tts_voice_name,
            data_dir=args.tts_data_dir,
            use_cuda=None,
        )
    )

    evaluations: list[PhraseEvaluation] = []

    for item in metadata:
        wav_path = Path(item["wav_path"])
        expected_text = item["text"]

        audio, samplerate = load_wav(wav_path)
        trimmed_audio = trim_silence(
            audio,
            samplerate=samplerate,
            threshold=args.silence_threshold,
            padding_ms=args.padding_ms,
        )

        stt_started = time.perf_counter()
        recognized_text = transcribe_with_realtime_stt(
            trimmed_audio.reshape(-1, 1),
            samplerate=samplerate,
            base_url=args.stt_url,
            ws_url=args.stt_ws_url,
        )
        stt_seconds = time.perf_counter() - stt_started

        translation_started = time.perf_counter()
        translated_text = translation_service.translate(recognized_text)
        translation_seconds = time.perf_counter() - translation_started

        tts_started = time.perf_counter()
        tts_audio = tts_service.synthesize(translated_text, target_samplerate=48000)
        tts_seconds = time.perf_counter() - tts_started

        evaluation = PhraseEvaluation(
            index=int(item["index"]),
            expected_text=expected_text,
            recognized_text=recognized_text,
            translated_text=translated_text,
            wav_path=str(wav_path),
            raw_audio_seconds=len(audio) / float(samplerate),
            trimmed_audio_seconds=len(trimmed_audio) / float(samplerate),
            stt_seconds=stt_seconds,
            translation_seconds=translation_seconds,
            tts_seconds=tts_seconds,
            tts_audio_seconds=tts_audio.shape[0] / 48000.0 if tts_audio.size > 0 else 0.0,
            stt_similarity=similarity_score(expected_text, recognized_text),
        )
        evaluations.append(evaluation)

        print(
            f"[{evaluation.index:02d}] "
            f"stt={evaluation.stt_seconds:.3f}s "
            f"mt={evaluation.translation_seconds:.3f}s "
            f"tts={evaluation.tts_seconds:.3f}s "
            f"sim={evaluation.stt_similarity:.3f}"
        )
        print(f"  expected:   {evaluation.expected_text}")
        print(f"  recognized: {evaluation.recognized_text}")
        print(f"  translated: {evaluation.translated_text}")

    report = {
        "summary": {
            "count": len(evaluations),
            "avg_stt_seconds": mean_value(item.stt_seconds for item in evaluations),
            "avg_translation_seconds": mean_value(item.translation_seconds for item in evaluations),
            "avg_tts_seconds": mean_value(item.tts_seconds for item in evaluations),
            "avg_stt_similarity": mean_value(item.stt_similarity for item in evaluations),
            "tts_backend": tts_service.get_runtime_backend_label(),
            "tts_voice_name": args.tts_voice_name,
        },
        "items": [asdict(item) for item in evaluations],
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(f"Saved evaluation report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
