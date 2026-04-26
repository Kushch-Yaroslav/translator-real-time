from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "recordings" / "benchmarks" / "ru_to_en"
BENCHMARK_SOURCE_DIR = BENCHMARK_ROOT / "source"
BENCHMARK_RUNS_DIR = BENCHMARK_ROOT / "runs"
BENCHMARK_AUDIO_PATH = BENCHMARK_SOURCE_DIR / "current_benchmark.wav"

CATEGORY_MAPPING = {
    "simple_text": "Простой текст",
    "natural_speech": "Естественная речь",
    "pauses_and_hesitation": "Паузы и запинки",
    "difficult_phrases": "Сложные фразы",
    "noisy_or_unclear": "Шумная / нечеткая речь",
}

# Backward compatibility mapping
LEGACY_CATEGORY_MAP = {
    "simple": "simple_text",
    "medium": "natural_speech",
    "pauses": "pauses_and_hesitation",
    "difficult": "difficult_phrases",
}


def ensure_benchmark_dirs() -> None:
    BENCHMARK_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure all new category directories exist
    for cat_dir_name in CATEGORY_MAPPING.keys():
        (BENCHMARK_SOURCE_DIR / cat_dir_name).mkdir(parents=True, exist_ok=True)


def get_category_source_dir(category: str) -> Path:
    # Handle legacy categories
    mapped_category = LEGACY_CATEGORY_MAP.get(category, category)
    return BENCHMARK_SOURCE_DIR / mapped_category


def get_test_paths(category: str, test_id: str) -> tuple[Path, Path]:
    """Returns (audio_path, expected_txt_path)"""
    category_dir = get_category_source_dir(category)
    audio_path = category_dir / f"{test_id}.wav"
    expected_path = category_dir / f"{test_id}.expected.txt"
    return audio_path, expected_path
