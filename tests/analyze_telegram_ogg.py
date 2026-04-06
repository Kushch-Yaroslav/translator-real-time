from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.sst.nim_realtime_stt_service import (  # noqa: E402
    NIMRealtimeSTTConfig,
    NIMRealtimeSTTService,
)
from core.sst.riva_realtime_stt_service import (  # noqa: E402
    RivaRealtimeSTTConfig,
    RivaRealtimeSTTService,
)
from core.translation.translation_service import (  # noqa: E402
    TranslationConfig,
    TranslationDirection,
    TranslationService,
)


@dataclass
class TimelineEvent:
    at_sec: float
    kind: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ Telegram .ogg через realtime STT с таймлайном partial/final."
    )
    parser.add_argument("input_path")
    parser.add_argument("--stt-backend", choices=("nim", "riva"), default="riva")
    parser.add_argument("--language", default="ru-RU")
    parser.add_argument("--direction", choices=("ru_to_en", "en_to_ru"), default="ru_to_en")
    parser.add_argument("--stt-url", default="http://localhost:9000")
    parser.add_argument("--stt-ws-url", default="ws://localhost:9000/v1/realtime?intent=transcription")
    parser.add_argument("--riva-uri", default="localhost:50051")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--pace", action="store_true")
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


def build_stt_service(args: argparse.Namespace):
    if args.stt_backend == "riva":
        return RivaRealtimeSTTService(
            RivaRealtimeSTTConfig(
                uri=args.riva_uri,
                language=args.language,
                sample_rate_hz=16000,
                num_channels=1,
                timeout=10.0,
                enable_automatic_punctuation=True,
                on_log=None,
            )
        )

    return NIMRealtimeSTTService(
        NIMRealtimeSTTConfig(
            base_url=args.stt_url,
            ws_url=args.stt_ws_url,
            language=args.language,
            sample_rate_hz=16000,
            num_channels=1,
            timeout=10.0,
            commit_interval_sec=0.35,
            enable_automatic_punctuation=True,
            on_log=None,
        )
    )


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    audio, samplerate = decode_audio_to_float32_mono(input_path)
    service = build_stt_service(args)

    translation_direction = (
        TranslationDirection.RU_TO_EN
        if args.direction == "ru_to_en"
        else TranslationDirection.EN_TO_RU
    )
    translation_service = TranslationService(
        TranslationConfig(direction=translation_direction, enabled=True)
    )

    events: list[TimelineEvent] = []
    done_event = threading.Event()
    started_at = 0.0

    def add_event(kind: str, text: str) -> None:
        normalized = normalize_text(text)
        if not normalized:
            return
        events.append(
            TimelineEvent(
                at_sec=time.perf_counter() - started_at,
                kind=kind,
                text=normalized,
            )
        )

    def on_partial(text: str) -> None:
        add_event("partial", text)

    def on_final(text: str) -> None:
        add_event("final", text)
        done_event.set()

    service.start(partial_callback=on_partial, final_callback=on_final)
    started_at = time.perf_counter()

    chunk_samples = max(1, int((args.chunk_ms / 1000.0) * samplerate))
    try:
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset: offset + chunk_samples]
            if chunk.size == 0:
                continue

            service.send_audio_chunk(chunk, samplerate)
            if args.pace:
                time.sleep(args.chunk_ms / 1000.0)

        service.commit()
        service.send_done()
        done_event.wait(timeout=12.0)
    finally:
        final_text = normalize_text(service.get_last_final_text())
        service.stop()

    translated_text = ""
    if final_text:
        translated_text = normalize_text(translation_service.translate(final_text))

    print(f"file: {input_path}")
    print(f"duration_sec: {len(audio) / float(samplerate):.2f}")
    print(f"backend: {args.stt_backend}")
    print(f"language: {args.language}")
    print()
    for event in events:
        print(f"[{event.at_sec:6.2f}s] {event.kind.upper():7s} {event.text}")

    print()
    print(f"FINAL: {final_text}")
    print(f"TRANSLATED: {translated_text}")

    if args.output_json:
        payload = {
            "input_path": str(input_path),
            "duration_sec": len(audio) / float(samplerate),
            "backend": args.stt_backend,
            "language": args.language,
            "final_text": final_text,
            "translated_text": translated_text,
            "events": [asdict(event) for event in events],
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved_json: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
