from __future__ import annotations

from core.audio.audio_engine import AudioEngine
from core.audio.ru_to_en_stream_controller import RuToEnStreamController


def _build_controller() -> RuToEnStreamController:
    return RuToEnStreamController(
        normalize_text=AudioEngine._normalize_text,
        normalize_compare_text=AudioEngine._normalize_compare_text,
        extract_incremental_text=AudioEngine._extract_incremental_text,
        merge_final_texts=AudioEngine._merge_final_texts,
    )


def test_one_preview_per_utterance_then_final_tail() -> None:
    controller = _build_controller()

    preview = controller.push_partial(
        "Сейчас мы общаемся через мой переводчик.",
        now=0.0,
    )
    assert [chunk.text for chunk in preview] == ["Сейчас мы общаемся через мой переводчик."]
    assert [chunk.is_final for chunk in preview] == [False]

    assert (
        controller.push_partial(
            "Сейчас мы общаемся через мой переводчик в реальном времени.",
            now=0.40,
        )
        == []
    )

    final_chunks = controller.push_final(
        "Сейчас мы общаемся через мой переводчик в реальном времени.",
        now=0.80,
    )
    assert [chunk.text for chunk in final_chunks] == ["в реальном времени."]
    assert [chunk.is_final for chunk in final_chunks] == [True]


def test_preview_emits_earlier_for_meaningful_comma_clause() -> None:
    controller = _build_controller()

    preview = controller.push_partial(
        "Всем привет, меня зовут Ярослав,",
        now=0.0,
    )
    assert [chunk.text for chunk in preview] == ["Всем привет, меня зовут Ярослав,"]
    assert [chunk.is_preview for chunk in preview] == [True]


def test_final_does_not_backfill_old_prefix_after_newer_preview() -> None:
    controller = _build_controller()

    controller.push_partial("Сейчас мы общаемся через мой переводчик.", now=0.0)
    controller.push_partial("Сейчас мы общаемся через мой переводчик.", now=0.25)

    chunks = controller.push_final(
        "мне 26 лет, я из Украины. Сейчас мы общаемся через мой переводчик в реальном времени.",
        now=1.0,
    )

    assert [chunk.text for chunk in chunks] == ["в реальном времени."]


def test_final_clause_streaming_avoids_micro_fragments_for_landing_sentence() -> None:
    controller = _build_controller()

    chunks = controller.push_final(
        "Так как у лендингов обычно нет сложной логики, то основное влияние на производительность вызывают картинки.",
        now=1.0,
    )

    assert [chunk.text for chunk in chunks] == [
        "Так как у лендингов обычно нет сложной логики,",
        "то основное влияние на производительность вызывают картинки.",
    ]
    assert [chunk.is_final for chunk in chunks] == [False, True]


def test_controller_keeps_three_word_final_tail() -> None:
    controller = _build_controller()

    chunks = controller.push_final("Очень много лендингов.", now=1.0)

    assert [chunk.text for chunk in chunks] == ["Очень много лендингов."]


def test_controller_drops_weak_noise_final_tail() -> None:
    controller = _build_controller()

    chunks = controller.push_final("называют картинки.", now=1.0)

    assert chunks == []
