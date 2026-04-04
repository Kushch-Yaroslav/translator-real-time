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
    "Привет, меня зовут Ярослав.",
    "Я из Украины, и мне двадцать шесть лет.",
    "Я работаю инженером-программистом.",
    "Это тест русского в английский переводчика.",
    "Пожалуйста, переведи мою речь четко и без большой задержки.",
    "Короткая фраза.",
    "Это предложение длиннее, чтобы проверить естественный темп и сегментацию.",
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
        description="Записать фиксированный набор русских тестовых фраз в отдельные wav-файлы."
    )
    parser.add_argument(
        "--output-dir",
        default="recordings/test_phrases_ru_to_en",
        help="Каталог для wav-файлов и метаданных.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=48000,
        help="Частота дискретизации записи.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Количество каналов.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Необязательный индекс входного устройства sounddevice.",
    )
    parser.add_argument(
        "--lead-in",
        type=float,
        default=1.5,
        help="Сколько ждать после показа фразы до начала записи.",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=1.0,
        help="Сколько секунд оставить после ожидаемого окончания фразы.",
    )
    return parser.parse_args()


def estimate_duration_seconds(text: str, lead_in: float, tail: float) -> float:
    words = len(text.split())
    speech_seconds = max(2.2, words * 0.60)
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

    print("Доступные входные устройства:")
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
        print(f"Фраза {index}/{len(DEFAULT_PHRASES)}")
        print(phrase)
        print(f"Запись начнется через {args.lead_in:.1f} сек")
        time.sleep(args.lead_in)

        frames = int(duration_seconds * args.samplerate)
        print(f"Запись {duration_seconds:.1f} сек...")
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
        print(f"Сохранено: {wav_path}")

        records.append(
            PhraseRecord(
                index=index,
                text=phrase,
                wav_path=str(wav_path),
                duration_seconds=duration_seconds,
                lead_in_seconds=args.lead_in,
            )
        )

        print("Пауза 2.0 сек перед следующей фразой")
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
    print(f"Готово. Метаданные: {metadata_path}")
    print(f"Список фраз: {phrases_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
