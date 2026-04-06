from core.app_config import DEFAULT_CONFIG
from core.audio_engine import AudioEngine


def test_strip_known_stt_hallucination_tail_drops_standalone_phrase() -> None:
    engine = AudioEngine(DEFAULT_CONFIG)

    assert engine._strip_known_stt_hallucination_tail("Продолжение следует...") == ""


def test_strip_known_stt_hallucination_tail_removes_trailing_phrase() -> None:
    engine = AudioEngine(DEFAULT_CONFIG)

    text = "Сегодня я проснулся поздно и продолжаю разрабатывать приложение. Продолжение следует..."

    assert (
        engine._strip_known_stt_hallucination_tail(text)
        == "Сегодня я проснулся поздно и продолжаю разрабатывать приложение."
    )


def test_strip_known_stt_hallucination_tail_keeps_regular_text() -> None:
    engine = AudioEngine(DEFAULT_CONFIG)

    text = "Сегодня я проснулся поздно и продолжаю разрабатывать приложение."

    assert engine._strip_known_stt_hallucination_tail(text) == text
