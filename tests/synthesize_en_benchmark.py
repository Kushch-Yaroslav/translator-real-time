"""Synthesize English benchmark speech for the EN=>RU pipeline.

Produces:
  recordings/benchmarks/en_to_ru/source/normal/<id>.wav (+ .expected.txt)
  recordings/benchmarks/en_to_ru/source/fast/<id>.wav     (+ .expected.txt)
  recordings/benchmarks/en_to_ru/source/continuous_fast/continuous_fast.wav (+ .expected.txt)

normal  = length_scale 1.0 (regular pace)
fast    = length_scale 0.7 (fast speech, the user's main complaint scenario)
continuous_fast = one long fast utterance with minimal pauses

Uses the Piper en_US-ryan voice. CPU only (no CUDA needed for synthesis here;
Piper falls back to CPU automatically if CUDA provider is unavailable/disabled).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru"
VOICE_PATH = "/media/yaroslav/DATA/ai_models/piper/en/en_US/ryan/medium/en_US-ryan-medium.onnx"

NORMAL_SENTENCES = [
    ("greeting", "Hi, how are you doing today?"),
    ("project", "I'm working on a new project for our client this week."),
    ("deadline", "Can you send me the report before Friday?"),
    ("university", "The University of California was founded in 1868 and it is very famous."),
    ("meeting", "I think we should meet next week to discuss the budget."),
    ("airport", "Yesterday I was at the airport and I missed my flight because of the traffic."),
    ("weather", "The weather is really nice, so let's go for a walk in the park."),
    ("software", "I have a question about the new software update and the release date."),
    ("payment", "Could you please explain how the payment process works in your country?"),
    ("conference", "We are planning a small conference in October, and I need your help with the program."),
]

CONTINUOUS_FAST_TEXT = (
    "So we talked about the project last time and you said the design is almost ready. "
    "That is great news. I need you to check the budget with the finance team before Monday, "
    "because the client wants a final proposal by the end of the week. "
    "Also, can you invite the engineering lead to the meeting? She has a lot of questions about the API."
)


def _save_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def synthesize(voice: PiperVoice, text: str, length_scale: float) -> tuple[np.ndarray, int]:
    config = SynthesisConfig(length_scale=length_scale)
    chunks = []
    rate = None
    for chunk in voice.synthesize(text, syn_config=config):
        if rate is None:
            rate = chunk.sample_rate
        chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
    return np.concatenate(chunks), int(rate)


def main() -> None:
    voice = PiperVoice.load(VOICE_PATH)

    for name, sentences in (
        ("normal", NORMAL_SENTENCES),
        ("fast", NORMAL_SENTENCES),
    ):
        length_scale = 1.0 if name == "normal" else 0.7
        out_dir = BENCH_ROOT / "source" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for test_id, text in sentences:
            audio, rate = synthesize(voice, text, length_scale)
            _save_wav(audio, rate, out_dir / f"{test_id}.wav")
            (out_dir / f"{test_id}.expected.txt").write_text(text + "\n", encoding="utf-8")
            duration = len(audio) / rate
            print(f"{name}/{test_id}.wav {duration:.1f}s len_scale={length_scale}")

    cont_dir = BENCH_ROOT / "source" / "continuous_fast"
    cont_dir.mkdir(parents=True, exist_ok=True)
    audio, rate = synthesize(voice, CONTINUOUS_FAST_TEXT, 0.65)
    _save_wav(audio, rate, cont_dir / "continuous_fast.wav")
    (cont_dir / "continuous_fast.expected.txt").write_text(CONTINUOUS_FAST_TEXT + "\n", encoding="utf-8")
    print(f"continuous_fast/continuous_fast.wav {len(audio) / rate:.1f}s len_scale=0.65")

    print(f"Done. Audio dir: {BENCH_ROOT / 'source'}")


if __name__ == "__main__":
    main()
