"""A/B benchmark: whisper.cpp models on CPU for the EN=>RU branch.

Runs whisper-server (CPU mode, -ng) per model, transcribes all
recordings/benchmarks/en_to_ru/source wavs and reports WER + latency.

Usage:
  CUDA_VISIBLE_DEVICES="" python tests/run_en_stt_ab.py

CPU only on purpose: the user's GPU is busy. Latency numbers are CPU
relative values (RTF); on GPU both models will be faster.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru" / "source"
WHISPER_SERVER = PROJECT_ROOT / "tmp" / "whisper.cpp" / "build" / "bin" / "whisper-server"
MODEL_DIR = Path("/media/yaroslav/DATA/ai_models/whisper.cpp/models")
PORT = 8199
BASE = f"http://127.0.0.1:{PORT}"

MODELS = [
    "ggml-medium.en.bin",
    "ggml-large-v3-turbo.bin",
]


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").split()
    hyp = hypothesis.lower().replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").split()
    if not ref:
        return 0.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[0, :] = np.arange(len(hyp) + 1)
    d[:, 0] = np.arange(len(ref) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)]) / len(ref)


def wait_ready(timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            requests.get(f"{BASE}/health", timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def transcribe(wav_path: Path) -> tuple[str, float]:
    with open(wav_path, "rb") as f:
        started = time.perf_counter()
        resp = requests.post(
            f"{BASE}/inference",
            files={
                "file": (wav_path.name, f, "audio/wav"),
                "language": (None, "en"),
                "response_format": (None, "json"),
                "temperature": (None, "0.0"),
                "temperature_inc": (None, "0.0"),
                "no_timestamps": (None, "true"),
                "suppress_nst": (None, "true"),
            },
            timeout=300,
        )
        elapsed = time.perf_counter() - started
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip(), elapsed


def run_model(model_name: str) -> dict:
    model_path = MODEL_DIR / model_name
    proc = subprocess.Popen(
        [
            str(WHISPER_SERVER),
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "-m", str(model_path),
            "-l", "en",
            "-nt",
            "-sns",
            "-ng",          # CPU only
            "-t", "16",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready()
        results = []
        for test_dir in sorted(BENCH_DIR.iterdir()):
            if not test_dir.is_dir():
                continue
            for wav in sorted(test_dir.glob("*.wav")):
                expected = (wav.with_suffix(".expected.txt")).read_text(encoding="utf-8").strip()
                text, elapsed = transcribe(wav)
                results.append(
                    {
                        "test": f"{test_dir.name}/{wav.stem}",
                        "expected": expected,
                        "text": text,
                        "wer": round(wer(expected, text), 4),
                        "latency_sec": round(elapsed, 3),
                    }
                )
                print(f"  [{model_name}] {test_dir.name}/{wav.stem}: WER={results[-1]['wer']} t={elapsed:.2f}s")
        wers = [r["wer"] for r in results]
        lats = [r["latency_sec"] for r in results]
        return {
            "model": model_name,
            "mean_wer": round(float(np.mean(wers)), 4),
            "max_wer": round(float(np.max(wers)), 4),
            "mean_latency_sec": round(float(np.mean(lats)), 3),
            "max_latency_sec": round(float(np.max(lats)), 3),
            "results": results,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    all_results = [run_model(m) for m in MODELS]

    print("\n===== SUMMARY =====")
    for r in all_results:
        print(
            f"{r['model']}: mean_WER={r['mean_wer']} max_WER={r['max_wer']} "
            f"mean_latency={r['mean_latency_sec']}s max_latency={r['max_latency_sec']}s"
        )

    out_path = PROJECT_ROOT / "recordings" / "benchmarks" / "en_to_ru" / "stt_ab_cpu.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
