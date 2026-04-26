from dataclasses import replace

from core.config.app_config import DEFAULT_CONFIG, LISTEN_BRANCH_ID, get_branch_config, replace_branch_config
from core.audio.audio_engine import AudioEngine


def _build_engine(direction: str, backend: str = "faster_whisper") -> AudioEngine:
    branch_config = replace(
        get_branch_config(DEFAULT_CONFIG, LISTEN_BRANCH_ID),
        translation_direction=direction,
        stt_language="ru-RU" if direction == "ru_to_en" else "en-US",
    )
    config = replace_branch_config(
        replace(
            DEFAULT_CONFIG,
            stt=replace(DEFAULT_CONFIG.stt, backend=backend),
        ),
        branch_config,
    )
    return AudioEngine(config)


def test_low_latency_direct_pipeline_enabled_for_faster_whisper_en_to_ru() -> None:
    engine = _build_engine("en_to_ru")

    assert engine._uses_low_latency_direct_pipeline() is True


def test_low_latency_direct_pipeline_enabled_for_faster_whisper_ru_to_en() -> None:
    engine = _build_engine("ru_to_en")

    assert engine._uses_low_latency_direct_pipeline() is True


def test_low_latency_direct_pipeline_disabled_when_partials_are_disabled() -> None:
    engine = _build_engine("ru_to_en")
    engine.app_config = replace(
        engine.app_config,
        stt=replace(engine.app_config.stt, partial_emit_enabled=False),
    )

    assert engine._uses_low_latency_direct_pipeline() is False


def test_low_latency_direct_pipeline_enabled_for_whispercpp_en_to_ru() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert engine._uses_low_latency_direct_pipeline() is True
    assert engine._supports_low_latency_followup_partial() is True


def test_low_latency_direct_pipeline_disabled_for_non_faster_whisper() -> None:
    engine = _build_engine("en_to_ru", backend="riva")

    assert engine._uses_low_latency_direct_pipeline() is False


def test_ru_to_en_final_reconstruction_uses_fragment_buffer() -> None:
    engine = _build_engine("ru_to_en")
    engine._remember_ru_to_en_fragment("всем привет")
    engine._remember_ru_to_en_fragment("меня зовут Ярослав")
    engine._remember_ru_to_en_fragment("мне 20 6 лет")
    engine._remember_ru_to_en_fragment("мне 26 лет")
    engine._remember_ru_to_en_fragment("я из Украины")
    engine._remember_ru_to_en_fragment("я работал в компании I'm days")
    engine._remember_ru_to_en_fragment("я работал в компании AMDays")

    assert (
        engine._reconstruct_ru_to_en_final_text(
            "я работал в компании AEMDays и делал лендинги"
        )
        == "Я работал в компании AEMDays и делал лендинги."
    )


def test_ru_to_en_fragment_buffer_skips_consecutive_duplicates() -> None:
    engine = _build_engine("ru_to_en")

    engine._remember_ru_to_en_fragment("мне 26 лет")
    engine._remember_ru_to_en_fragment("мне 26 лет")
    engine._remember_ru_to_en_fragment("я из Украины")

    assert engine._ru_to_en_current_fragments == [
        "мне 26 лет",
        "я из Украины",
    ]


def test_ru_to_en_final_reconstruction_uses_local_fragment_window() -> None:
    engine = _build_engine("ru_to_en")
    engine._remember_ru_to_en_fragment("Так как у лейдингов обычно нет сложной логики,")
    engine._remember_ru_to_en_fragment("то основное влияние на производительность вызывают картинки.")
    engine._remember_ru_to_en_fragment("Сначала я пользовался бесплатными сервисами.")
    engine._remember_ru_to_en_fragment("но опирался то в лимит,")

    reconstructed = engine._reconstruct_ru_to_en_final_text(
        "Сначала я пользовался бесплатными сервисами, но опирался то в лимит, то в отсутствие гибкости."
    )

    assert "Так как у лейдингов" not in reconstructed
    assert reconstructed.startswith("Сначала я пользовался бесплатными сервисами")


def test_ru_to_en_can_replace_last_emitted_fragment_with_refined_company_phrase() -> None:
    engine = _build_engine("ru_to_en")
    engine._remember_ru_to_en_emitted_phrase("я делал лендинги.")
    engine._remember_ru_to_en_fragment("я делал лендинги.")

    assert engine._replace_last_ru_to_en_emitted_fragment("AMDays я делал лендинги.") is True
    assert engine._ru_to_en_emitted_phrases[-1] == "AMDays я делал лендинги."


def test_ru_to_en_can_replace_last_emitted_fragment_with_refined_wording() -> None:
    engine = _build_engine("ru_to_en")
    engine._remember_ru_to_en_emitted_phrase("но упирался то в лимит,")
    engine._remember_ru_to_en_fragment("но упирался то в лимит,")

    assert engine._replace_last_ru_to_en_emitted_fragment("опирался то в лимит,") is True
    assert engine._ru_to_en_emitted_phrases[-1] == "опирался то в лимит,"


def test_ru_to_en_defers_short_conjunction_partial_phrase() -> None:
    engine = _build_engine("ru_to_en")

    assert engine._should_defer_ru_to_en_partial_phrase("но упирался то в лимит,") is True
    assert engine._should_defer_ru_to_en_partial_phrase("я из Украины.") is False


def test_ru_to_en_defers_short_landing_phrase_after_company_context() -> None:
    engine = _build_engine("ru_to_en")
    engine._remember_ru_to_en_fragment("Когда я работал в компании AMD,")

    assert engine._should_defer_ru_to_en_partial_phrase("я делал лендинги.") is True


def test_ru_to_en_defers_short_landing_phrase_from_partial_context() -> None:
    engine = _build_engine("ru_to_en")
    engine._ru_to_en_last_partial_context = "Когда я работал в компании AMD, я делал лендинги."

    assert engine._should_defer_ru_to_en_partial_phrase("я делал лендинги.") is True


def test_ru_to_en_defers_short_company_prefix_partial_phrase() -> None:
    engine = _build_engine("ru_to_en")

    assert engine._should_defer_ru_to_en_partial_phrase("Когда я работал в компании AMD,") is True
    assert engine._should_defer_ru_to_en_partial_phrase("Когда я работал в компании AMDays я делал лендинги.") is False


def test_ru_to_en_defers_weaker_comfort_phrase() -> None:
    engine = _build_engine("ru_to_en")

    assert engine._should_defer_ru_to_en_partial_phrase(
        "так как мой английский слабее уровня комфорта."
    ) is True
    assert engine._should_defer_ru_to_en_partial_phrase(
        "так как мой английский слабее уровня комфортного общения."
    ) is False


def test_ru_to_en_defers_weaker_free_systems_phrase() -> None:
    engine = _build_engine("ru_to_en")

    assert engine._should_defer_ru_to_en_partial_phrase(
        "Сначала я пользовался бесплатными системами."
    ) is True
    assert engine._should_defer_ru_to_en_partial_phrase(
        "Сначала я пользовался бесплатными сервисами."
    ) is False


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


def test_whispercpp_skips_weak_me_too_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert engine._sanitize_low_latency_partial_candidate("I'm from Ukraine and me too.") == ""


def test_whispercpp_skips_russian_country_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert engine._sanitize_low_latency_partial_candidate("P26 is a Russian country.") == ""


def test_whispercpp_skips_mid_century_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert (
        engine._sanitize_low_latency_partial_candidate("I'm from Ukraine and mid-20th century.")
        == ""
    )


def test_whispercpp_skips_mi26_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert engine._sanitize_low_latency_partial_candidate("I am from Ukraine and Mi-26.") == ""


def test_whispercpp_skips_repeated_name_in_followup_partial() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")

    assert (
        engine._sanitize_low_latency_partial_candidate("I'm from Ukraine and my name is Jaroslav.")
        == ""
    )


def test_whispercpp_defers_age_only_partial_after_from_ukraine_anchor() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")
    engine._low_latency_last_queued_text = "I am from Ukraine."

    assert engine._should_defer_whispercpp_age_partial("and me 26 years old") is True


def test_whispercpp_strips_repeated_from_ukraine_context_from_followup() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")
    engine._low_latency_last_queued_text = "I am from Ukraine."

    assert (
        engine._sanitize_low_latency_partial_candidate("I'm from Ukraine and I'm 26 years old.")
        == "I'm 26 years old."
    )


def test_whispercpp_skips_pure_repeated_from_ukraine_followup() -> None:
    engine = _build_engine("en_to_ru", backend="whisper_cpp")
    engine._low_latency_last_queued_text = "I am from Ukraine."

    assert (
        engine._sanitize_low_latency_partial_candidate("I am from Ukraine and I am from Ukraine.")
        == ""
    )


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
