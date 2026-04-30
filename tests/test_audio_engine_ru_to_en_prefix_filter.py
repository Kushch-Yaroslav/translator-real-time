from dataclasses import replace

from core.audio.audio_engine import AudioEngine
from core.config.app_config import (
    DEFAULT_CONFIG,
    LISTEN_BRANCH_ID,
    get_branch_config,
    replace_branch_config,
)


def _build_engine() -> AudioEngine:
    branch_config = replace(
        get_branch_config(DEFAULT_CONFIG, LISTEN_BRANCH_ID),
        translation_direction="ru_to_en",
        stt_language="ru-RU",
    )
    config = replace_branch_config(DEFAULT_CONFIG, branch_config)
    return AudioEngine(config)


def test_prepare_phrase_skips_backfilled_thought() -> None:
    engine = _build_engine()
    engine._remember_ru_to_en_emitted_phrase("я делал лендинги.")

    assert (
        engine._prepare_ru_to_en_phrase_for_queue(
            "Когда я работал в компании AMDays я делал лендинги."
        )
        == ""
    )


def test_prepare_phrase_skips_unsafe_short_refinement_tail() -> None:
    engine = _build_engine()
    engine._remember_ru_to_en_emitted_phrase(
        "так как мой английский слабее уровня комфорта."
    )

    assert (
        engine._prepare_ru_to_en_phrase_for_queue(
            "так как мой английский слабее уровня комфортного общения."
        )
        == ""
    )


def test_prepare_phrase_keeps_safe_continuation_tail() -> None:
    engine = _build_engine()
    engine._remember_ru_to_en_emitted_phrase("Я решил создать свой инструмент,")

    assert (
        engine._prepare_ru_to_en_phrase_for_queue(
            "Я решил создать свой инструмент, которым активно пользуюсь до сих пор."
        )
        == "которым активно пользуюсь до сих пор."
    )


def test_strict_short_translated_fragment_skip_requires_all_signals() -> None:
    engine = _build_engine()
    engine._recent_translated_texts = ["I solved it because of the language barrier."]

    assert (
        engine._should_skip_strict_short_translated_fragment(
            "потому что",
            "because of the",
        )
        is True
    )


def test_strict_short_translated_fragment_keeps_new_thought() -> None:
    engine = _build_engine()
    engine._recent_translated_texts = ["I solved it because of the language barrier."]

    assert (
        engine._should_skip_strict_short_translated_fragment(
            "и я решил",
            "And I decided",
        )
        is False
    )


def test_strict_short_translated_fragment_keeps_contentful_short_phrase() -> None:
    engine = _build_engine()
    engine._recent_translated_texts = ["My English became weaker over time."]

    assert (
        engine._should_skip_strict_short_translated_fragment(
            "потому что мой английский слабее",
            "Because my English is weaker",
        )
        is False
    )


def test_strict_short_translated_fragment_keeps_named_entity() -> None:
    engine = _build_engine()
    engine._recent_translated_texts = ["When I worked at AMD, I built landing pages."]

    assert (
        engine._should_skip_strict_short_translated_fragment(
            "когда я работал в amd",
            "when I worked at AMD",
        )
        is False
    )


def test_strict_short_translated_fragment_skips_five_word_retry_tail() -> None:
    engine = _build_engine()
    engine._recent_translated_source_texts = ["но опирался то в лимит,"]

    assert (
        engine._should_skip_strict_short_translated_fragment(
            "упирался то в лимит,",
            "and clung to the limit,",
        )
        is True
    )


def test_strict_short_translated_fragment_requires_source_retry_coverage() -> None:
    engine = _build_engine()
    engine._recent_translated_source_texts = ["сначала я пользовался бесплатными сервисами."]

    assert (
        engine._should_skip_strict_short_translated_fragment(
            "упирался то в лимит,",
            "and clung to the limit,",
        )
        is False
    )
