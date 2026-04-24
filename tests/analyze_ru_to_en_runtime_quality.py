from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.sst.faster_whisper_realtime_stt_service import (  # noqa: E402
    FasterWhisperRealtimeSTTConfig,
    FasterWhisperRealtimeSTTService,
)
from core.translation.translation_service import (  # noqa: E402
    TranslationConfig,
    TranslationDirection,
    TranslationService,
)


LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| [A-Z]+ \| \[SPEAK\] (?P<message>.*)$"
)
OGG_FILENAME_RE = re.compile(
    r"audio_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.ogg$"
)


@dataclass
class LogEvent:
    at: datetime
    message: str


@dataclass
class SegmentIssue:
    kind: str
    details: str


@dataclass
class AudioQualityReport:
    audio_path: str
    audio_started_at: str
    duration_sec: float
    source_transcript_ogg: str
    source_transcript_log: str
    expected_translation: str
    spoken_similarity_pct: float
    voiced_similarity_pct: float
    queued_segments: list[str]
    voiced_segments: list[str]
    duplicate_queued_segments: list[str]
    duplicate_voiced_segments: list[str]
    long_tts_segments: list[str]
    long_translation_gaps: list[str]
    issues: list[SegmentIssue]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнить .ogg запись с тем, что услышал и озвучил RU=>EN pipeline."
    )
    parser.add_argument("audio_paths", nargs="+")
    parser.add_argument("--log-path", default="logs/speak_ru_to_en.log")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--window-before-sec", type=float, default=3.0)
    parser.add_argument("--window-after-sec", type=float, default=25.0)
    parser.add_argument("--tts-long-sec", type=float, default=2.8)
    parser.add_argument("--gap-long-sec", type=float, default=2.2)
    parser.add_argument("--skip-translation", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


def decode_audio_to_float32_mono(path: Path) -> tuple[np.ndarray, int]:
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
        "48000",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False), 48000


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def normalize_compare_text(text: str) -> str:
    text = normalize_text(text).lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(text.split()).strip()


def similarity_pct(left: str, right: str) -> float:
    return round(
        SequenceMatcher(None, normalize_compare_text(left), normalize_compare_text(right)).ratio() * 100.0,
        1,
    )


def parse_audio_start(path: Path) -> datetime:
    match = OGG_FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse timestamp from audio filename: {path.name}")
    date_part = match.group("date")
    time_part = match.group("time").replace("-", ":")
    return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")


def load_log_events(path: Path) -> list[LogEvent]:
    events: list[LogEvent] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_LINE_RE.match(raw_line)
        if not match:
            continue
        events.append(
            LogEvent(
                at=datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S"),
                message=match.group("message"),
            )
        )
    return events


def build_stt_service(device: str) -> FasterWhisperRealtimeSTTService:
    return FasterWhisperRealtimeSTTService(
        FasterWhisperRealtimeSTTConfig(
            language="ru",
            sample_rate_hz=16000,
            device=device,
            partial_interval_sec=0.12,
            min_window_sec=0.45,
            max_window_sec=12.0,
            min_silence_duration_ms=90,
            speech_pad_ms=40,
            speech_threshold=0.55,
            whisper_model_size="large-v3-turbo",
            compute_type="float16" if device == "cuda" else "int8",
            beam_size=1,
            best_of=1,
            patience=1.0,
            on_log=None,
        )
    )


def transcribe_audio(path: Path, device: str) -> tuple[str, float]:
    audio, samplerate = decode_audio_to_float32_mono(path)
    service = build_stt_service(device)
    final_segments: list[str] = []
    done_event = threading.Event()

    def on_final(text: str) -> None:
        normalized = normalize_text(text)
        if normalized:
            final_segments.append(normalized)
        done_event.set()

    started_at = time.perf_counter()
    service.start(partial_callback=None, final_callback=on_final)
    try:
        chunk_samples = max(1, int(0.2 * samplerate))
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset: offset + chunk_samples]
            if chunk.size == 0:
                continue
            service.send_audio_chunk(chunk, samplerate)
        service.commit()
        service.send_done()
        done_event.wait(timeout=20.0)
        if device == "cuda":
            time.sleep(0.20)
    finally:
        last_final_text = normalize_text(service.get_last_final_text())
        service.stop()

    transcript = normalize_text(" ".join(final_segments)) or last_final_text
    return transcript, len(audio) / float(samplerate)


def filter_events_for_audio(
    events: list[LogEvent],
    started_at: datetime,
    duration_sec: float,
    window_before_sec: float,
    window_after_sec: float,
) -> list[LogEvent]:
    window_start = started_at - timedelta(seconds=window_before_sec)
    window_end = started_at + timedelta(seconds=duration_sec + window_after_sec)
    return [event for event in events if window_start <= event.at <= window_end]


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


def detect_duplicate_segments(segments: Iterable[str]) -> list[str]:
    duplicates: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        normalized = normalize_compare_text(segment)
        if not normalized:
            continue
        if normalized in seen:
            duplicates.append(segment)
            continue
        seen.add(normalized)
    return duplicates


def detect_long_translation_gaps(events: list[LogEvent], gap_long_sec: float) -> list[str]:
    translated_events = [
        event for event in events
        if event.message.startswith("TRANSLATED: ")
    ]
    findings: list[str] = []
    for previous, current in zip(translated_events, translated_events[1:]):
        delta = (current.at - previous.at).total_seconds()
        if delta > gap_long_sec:
            findings.append(
                f"{previous.at.time()} -> {current.at.time()} gap={delta:.2f}s"
            )
    return findings


def build_report(
    audio_path: Path,
    events: list[LogEvent],
    source_transcript_ogg: str,
    duration_sec: float,
    skip_translation: bool,
    tts_long_sec: float,
    gap_long_sec: float,
) -> AudioQualityReport:
    source_log_segments = [
        extract_payload(event.message, "FINAL: ")
        for event in events
        if event.message.startswith("FINAL: ")
    ]
    queued_segments = [
        extract_payload(event.message, "LOWLAT sentence queued: ")
        for event in events
        if event.message.startswith("LOWLAT sentence queued: ")
    ]
    queued_segments.extend(
        extract_payload(event.message, "LOWLAT final tail queued: ")
        for event in events
        if event.message.startswith("LOWLAT final tail queued: ")
    )
    queued_segments = [segment for segment in queued_segments if segment]

    voiced_segments = [
        extract_payload(event.message, "TRANSLATED: ")
        for event in events
        if event.message.startswith("TRANSLATED: ")
    ]
    voiced_segments = [segment for segment in voiced_segments if segment]

    long_tts_segments = []
    for event, voiced_segment in zip(
        [event for event in events if event.message.startswith("TTS audio ready: ")],
        voiced_segments,
    ):
        duration = extract_tts_duration(event.message)
        if duration is not None and duration > tts_long_sec:
            long_tts_segments.append(f"{duration:.2f}s | {voiced_segment}")

    source_transcript_log = normalize_text(" ".join(source_log_segments))
    expected_translation = ""
    if not skip_translation and source_transcript_ogg:
        try:
            translation_service = TranslationService(
                TranslationConfig(direction=TranslationDirection.RU_TO_EN, enabled=True)
            )
            expected_translation = normalize_text(
                translation_service.translate(source_transcript_ogg)
            )
        except Exception as error:
            issues = [
                SegmentIssue(
                    "translation_reference_unavailable",
                    str(error),
                )
            ]
        else:
            issues = []
    else:
        issues = []

    duplicate_queued_segments = detect_duplicate_segments(queued_segments)
    duplicate_voiced_segments = detect_duplicate_segments(voiced_segments)
    long_translation_gaps = detect_long_translation_gaps(events, gap_long_sec)

    for segment in duplicate_queued_segments:
        issues.append(SegmentIssue("duplicate_queue", segment))
    for segment in duplicate_voiced_segments:
        issues.append(SegmentIssue("duplicate_voice", segment))
    for item in long_tts_segments:
        issues.append(SegmentIssue("long_tts_segment", item))
    for item in long_translation_gaps:
        issues.append(SegmentIssue("long_translation_gap", item))

    if source_transcript_log and normalize_compare_text(source_transcript_log) != normalize_compare_text(source_transcript_ogg):
        issues.append(
            SegmentIssue(
                "stt_mismatch",
                f"ogg='{source_transcript_ogg}' | log='{source_transcript_log}'",
            )
        )

    voiced_joined = normalize_text(" ".join(voiced_segments))
    return AudioQualityReport(
        audio_path=str(audio_path),
        audio_started_at=parse_audio_start(audio_path).isoformat(sep=" "),
        duration_sec=round(duration_sec, 2),
        source_transcript_ogg=source_transcript_ogg,
        source_transcript_log=source_transcript_log,
        expected_translation=expected_translation,
        spoken_similarity_pct=similarity_pct(source_transcript_ogg, source_transcript_log),
        voiced_similarity_pct=similarity_pct(expected_translation, voiced_joined) if expected_translation else 0.0,
        queued_segments=queued_segments,
        voiced_segments=voiced_segments,
        duplicate_queued_segments=duplicate_queued_segments,
        duplicate_voiced_segments=duplicate_voiced_segments,
        long_tts_segments=long_tts_segments,
        long_translation_gaps=long_translation_gaps,
        issues=issues,
    )


def print_report(report: AudioQualityReport) -> None:
    print(f"audio: {report.audio_path}")
    print(f"started_at: {report.audio_started_at}")
    print(f"duration_sec: {report.duration_sec:.2f}")
    print(f"spoken_similarity_pct: {report.spoken_similarity_pct:.1f}")
    if report.expected_translation:
        print(f"voiced_similarity_pct: {report.voiced_similarity_pct:.1f}")
    print()
    print(f"OGG transcript: {report.source_transcript_ogg}")
    print(f"Log transcript: {report.source_transcript_log}")
    if report.expected_translation:
        print(f"Expected EN:    {report.expected_translation}")
    print(f"Voiced EN:      {' '.join(report.voiced_segments)}")
    print()

    if report.queued_segments:
        print("Queued segments:")
        for segment in report.queued_segments:
            print(f"  - {segment}")

    if report.duplicate_queued_segments:
        print("Duplicate queued segments:")
        for segment in report.duplicate_queued_segments:
            print(f"  - {segment}")

    if report.duplicate_voiced_segments:
        print("Duplicate voiced segments:")
        for segment in report.duplicate_voiced_segments:
            print(f"  - {segment}")

    if report.long_tts_segments:
        print("Long TTS segments:")
        for item in report.long_tts_segments:
            print(f"  - {item}")

    if report.long_translation_gaps:
        print("Long translation gaps:")
        for item in report.long_translation_gaps:
            print(f"  - {item}")

    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"  - [{issue.kind}] {issue.details}")

    print()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    log_events = load_log_events(log_path)
    reports: list[AudioQualityReport] = []

    for raw_audio_path in args.audio_paths:
        audio_path = Path(raw_audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        source_transcript_ogg, duration_sec = transcribe_audio(audio_path, device=args.device)
        window_events = filter_events_for_audio(
            log_events,
            started_at=parse_audio_start(audio_path),
            duration_sec=duration_sec,
            window_before_sec=args.window_before_sec,
            window_after_sec=args.window_after_sec,
        )
        report = build_report(
            audio_path=audio_path,
            events=window_events,
            source_transcript_ogg=source_transcript_ogg,
            duration_sec=duration_sec,
            skip_translation=args.skip_translation,
            tts_long_sec=args.tts_long_sec,
            gap_long_sec=args.gap_long_sec,
        )
        reports.append(report)
        print_report(report)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(report) for report in reports]
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved_json: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
