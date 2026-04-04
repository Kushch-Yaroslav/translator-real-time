from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import wave
from dataclasses import asdict, dataclass

import numpy as np
import sounddevice as sd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PHRASES = [
    "Hello everyone.",
    "My name is Yaroslav.",
    "I am from Ukraine and I am twenty six years old.",
    "I work as a software engineer.",
    "This is a test of the English to Russian translation pipeline.",
    "Please translate my speech clearly and without long delays.",
    "Short phrase.",
    "This sentence is a little longer so we can test natural pacing and segmentation.",
]


@dataclass
class PhraseRecord:
    index: int
    text: str
    wav_path: str
    duration_seconds: float
    lead_in_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a fixed set of English test phrases into separate wav files."
    )
    parser.add_argument(
        "--output-dir",
        default="recordings/test_phrases",
        help="Directory where wav files and metadata will be saved.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=48000,
        help="Recording sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index.",
    )
    parser.add_argument(
        "--lead-in",
        type=float,
        default=1.5,
        help="Seconds to wait after showing each phrase before recording starts.",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=1.0,
        help="Extra seconds to keep recording after the estimated speech duration.",
    )
    return parser.parse_args()


def estimate_duration_seconds(text: str, lead_in: float, tail: float) -> float:
    words = len(text.split())
    speech_seconds = max(2.0, words * 0.55)
    return lead_in + speech_seconds + tail


def save_wav(path: Path, audio: np.ndarray, samplerate: int) -> None:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1 if pcm16.ndim == 1 else pcm16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Available input devices:")
    print(sd.query_devices())
    print()

    records: list[PhraseRecord] = []

    for index, phrase in enumerate(DEFAULT_PHRASES, start=1):
        duration_seconds = estimate_duration_seconds(
            phrase,
            lead_in=args.lead_in,
            tail=args.tail,
        )

        print("=" * 80)
        print(f"Phrase {index}/{len(DEFAULT_PHRASES)}")
        print(phrase)
        print(f"Recording will start in {args.lead_in:.1f} sec")
        time.sleep(args.lead_in)

        frames = int(duration_seconds * args.samplerate)
        print(f"Recording {duration_seconds:.1f} sec...")
        audio = sd.rec(
            frames,
            samplerate=args.samplerate,
            channels=args.channels,
            dtype="float32",
            device=args.device,
        )
        sd.wait()

        wav_path = output_dir / f"{index:02d}.wav"
        save_wav(wav_path, audio, args.samplerate)
        print(f"Saved {wav_path}")

        records.append(
            PhraseRecord(
                index=index,
                text=phrase,
                wav_path=str(wav_path),
                duration_seconds=duration_seconds,
                lead_in_seconds=args.lead_in,
            )
        )

        print("Pause for 2.0 sec before next phrase")
        time.sleep(2.0)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    phrases_path = output_dir / "phrases.txt"
    phrases_path.write_text(
        "\n".join(f"{record.index:02d}. {record.text}" for record in records) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Done. Metadata saved to {metadata_path}")
    print(f"Phrase list saved to {phrases_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
