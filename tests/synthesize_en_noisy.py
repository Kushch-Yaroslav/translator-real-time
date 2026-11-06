"""Create noisy/crosstalk variants of the EN benchmark (realistic conversation conditions).

For each target sentence:
  - background speech: a different Piper voice (en_US-lessac) speaking unrelated English,
    mixed at -12 dB (crosstalk from the other side / speakers)
  - plus low-level broadband noise at -24 dB

Output: recordings/benchmarks/en_to_ru/source/noisy_crosstalk/<id>.wav (+ .expected.txt)
Expected text remains the TARGET sentence (what the STT should hear).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru" / "source" / "noisy_crosstalk"

TARGET_VOICE = "/media/yaroslav/DATA/ai_models/piper/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
BG_VOICE = "/media/yaroslav/DATA/ai_models/piper/en/en_US/lessac/medium/en_US-lessac-medium.onnx"

TARGETS = {
    "greeting": "Hi, how are you doing today?",
    "project": "I'm working on a new project for our client this week.",
    "deadline": "Can you send me the report before Friday?",
    "university": "The University of California was founded in 1868 and it is very famous.",
    "meeting": "I think we should meet next week to discuss the budget.",
    "airport": "Yesterday I was at the airport and I missed my flight because of the traffic.",
    "weather": "The weather is really nice, so let's go for a walk in the park.",
    "software": "I have a question about the new software update and the release date.",
    "payment": "Could you please explain how the payment process works in your country?",
    "conference": "We are planning a small conference in October, and I need your help with the program.",
}

BG_LINES = [
    "Anyway, I was thinking about the weekend plans and the restaurant reservations.",
    "The server is down again, can you check the logs when you get a chance?",
    "I ordered pizza for tonight, and we should probably buy more drinks.",
    "The flight was delayed, so the hotel is holding our room until midnight.",
    "Do you remember that conference in Berlin last spring? It was quite interesting.",
    "I need to call the bank about the credit card statement they sent me.",
    "The new laptop is much faster, and the battery lasts almost all day.",
    "We should water the plants before going on vacation next month.",
    "The meeting was cancelled because the client had a family emergency.",
    "I found a great coffee shop near the station, we should try it tomorrow.",
]


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2
        rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data, rate


def synth_to_wav(voice: PiperVoice, text: str, path: Path, length_scale: float = 1.0) -> int:
    chunks = []
    rate = None
    for chunk in voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale)):
        if rate is None:
            rate = chunk.sample_rate
        chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
    audio = np.concatenate(chunks)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())
    return rate


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_voice = PiperVoice.load(TARGET_VOICE)
    bg_voice = PiperVoice.load(BG_VOICE)

    tmp = OUT_DIR / "_tmp"
    tmp.mkdir(exist_ok=True)

    for i, (test_id, text) in enumerate(TARGETS.items()):
        target_path = tmp / f"t_{test_id}.wav"
        bg_path = tmp / f"b_{test_id}.wav"
        synth_to_wav(target_voice, text, target_path, length_scale=0.85)
        # background: two different lines, slightly different pace, start before target ends
        synth_to_wav(bg_voice, BG_LINES[i % len(BG_LINES)] + " " + BG_LINES[(i + 3) % len(BG_LINES)], bg_path, length_scale=1.1)

        target, rate = load_wav(target_path)
        bg, _ = load_wav(bg_path)
        if len(bg) < len(target):
            bg = np.tile(bg, 1 + len(target) // max(len(bg), 1))
        bg = bg[: len(target)]

        rng = np.random.default_rng(42 + i)
        noise = rng.standard_normal(len(target)) * 10 ** (-24 / 20)
        mix = target + bg * 10 ** (-12 / 20) + noise
        mix = np.clip(mix, -1.0, 1.0)
        pcm16 = (mix * 32767).astype(np.int16)
        out_path = OUT_DIR / f"{test_id}.wav"
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm16.tobytes())
        (OUT_DIR / f"{test_id}.expected.txt").write_text(text + "\n", encoding="utf-8")
        print(f"noisy_crosstalk/{test_id}.wav {len(target) / rate:.1f}s")

    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print(f"Done: {OUT_DIR}")


if __name__ == "__main__":
    main()
