from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "recordings" / "benchmarks" / "ru_to_en"
BENCHMARK_SOURCE_DIR = BENCHMARK_ROOT / "source"
BENCHMARK_RUNS_DIR = BENCHMARK_ROOT / "runs"
BENCHMARK_AUDIO_PATH = BENCHMARK_SOURCE_DIR / "current_benchmark.wav"


def ensure_benchmark_dirs() -> None:
    BENCHMARK_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
