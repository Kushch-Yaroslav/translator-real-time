from dataclasses import replace

from core.app_config import DEFAULT_CONFIG
from core.audio_engine import AudioEngine


def _build_engine(direction: str, backend: str = "faster_whisper") -> AudioEngine:
    config = replace(
        DEFAULT_CONFIG,
        stt=replace(DEFAULT_CONFIG.stt, backend=backend),
        branches=replace(
            DEFAULT_CONFIG.branches,
            primary=replace(
                DEFAULT_CONFIG.branches.primary,
                translation_direction=direction,
                stt_language="ru-RU" if direction == "ru_to_en" else "en-US",
            ),
        ),
    )
    return AudioEngine(config)


def test_low_latency_direct_pipeline_enabled_for_faster_whisper_en_to_ru() -> None:
    engine = _build_engine("en_to_ru")

    assert engine._uses_low_latency_direct_pipeline() is True


def test_low_latency_direct_pipeline_enabled_for_faster_whisper_ru_to_en() -> None:
    engine = _build_engine("ru_to_en")

    assert engine._uses_low_latency_direct_pipeline() is True


def test_low_latency_direct_pipeline_enabled_for_whispercpp_en_to_ru() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert engine._uses_low_latency_direct_pipeline() is True
    assert engine._supports_low_latency_followup_partial() is True


def test_low_latency_direct_pipeline_disabled_for_non_faster_whisper() -> None:
    engine = _build_engine("en_to_ru", backend="riva")

    assert engine._uses_low_latency_direct_pipeline() is False



def test_en_to_ru_partial_candidate_can_emit_first_completed_sentence() -> None:
    engine = _build_engine("en_to_ru")

    assert (
        engine._select_low_latency_partial_candidate("Hello, my name is Yeroslav.")
        == "Hello, my name is Yeroslav."
    )
    assert (
        engine._select_low_latency_partial_candidate(
            "Hello, my name is Yeroslav. I am from Ukraine."
        )
        == "Hello, my name is Yeroslav."
    )


def test_low_latency_final_tail_skips_single_word_you() -> None:
    engine = _build_engine("en_to_ru")
    engine._low_latency_emitted_text = "Hello everyone, my name is Yeroslav."

    assert engine._should_skip_low_latency_final_tail("you") is True
    assert engine._should_skip_low_latency_final_tail("Thanks") is True
    assert engine._should_skip_low_latency_final_tail("me 26 years old") is False


def test_whispercpp_known_startup_hallucination_is_skipped_before_first_emit() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert (
        engine._should_skip_known_low_latency_source_hallucination(
            "Welcome to the American League of Legends."
        )
        is True
    )


def test_whispercpp_followup_partial_accepts_four_word_continuation() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert (
        engine._select_low_latency_followup_partial_candidate_from_anchor(
            "Hello, my name is Yaroslav.",
            "Hello, my name is Yaroslav. I am from Ukraine.",
        )
        == "I am from Ukraine."
    )


def test_whispercpp_sanitizes_repeated_intro_from_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")
    engine._low_latency_emitted_text = "Hello, my name is Yaroslav."

    assert (
        engine._sanitize_low_latency_partial_candidate(
            "Hello, my name is Yaroslav, I'm from Ukraine and me 26 years old."
        )
        == "I'm from Ukraine and me 26 years old."
    )


def test_whispercpp_skips_weak_meet_the_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert engine._sanitize_low_latency_partial_candidate("I'm from Ukraine and meet the") == ""


def test_extract_incremental_text_handles_intro_overlap_variation() -> None:
    previous = "Hello, my name is Yaroslav."
    current = "Hello everyone, my name is Yaroslav. I am from Ukraine and me 26 years old."

    assert (
        AudioEngine._extract_incremental_text(previous, current)
        == "I am from Ukraine and me 26 years old."
    )


def test_extract_incremental_text_drops_repeated_name_when_intro_name_changes() -> None:
    previous = "Hello, my name is Ericka."
    current = "Hello, my name is Yeroslav. I am from Ukraine and me 26 years old."

    assert (
        AudioEngine._extract_incremental_text(previous, current)
        == "I am from Ukraine and me 26 years old"
    )
