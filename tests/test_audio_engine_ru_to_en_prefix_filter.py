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
