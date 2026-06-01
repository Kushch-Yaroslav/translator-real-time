from core.ui.main_window import MainWindow


def test_user_log_shows_clean_final_playback_started_once() -> None:
    window = MainWindow.__new__(MainWindow)
    window._ui_spoken_item_ids = set()

    message = (
        "PLAYBACK started: Привет, я Оли, основатель и CEO MicroOne. "
        "id=tts-1 sourceType=final status=playing textLen=42 queueSize=3 "
        "itemAge=0.120s alreadyPlaying=False generation=0"
    )

    assert (
        window._format_user_facing_spoken_log(message, target_language="RU")
        == "RU: Привет, я Оли, основатель и CEO MicroOne."
    )
    assert window._format_user_facing_spoken_log(message, target_language="RU") == ""


def test_user_log_hides_debug_and_partial_playback_events() -> None:
    window = MainWindow.__new__(MainWindow)
    window._ui_spoken_item_ids = set()

    assert (
        window._format_user_facing_spoken_log(
            "TTS item queued: id=tts-1 sourceType=final status=queued text=Привет",
            target_language="RU",
        )
        == ""
    )
    assert (
        window._format_user_facing_spoken_log(
            "PLAYBACK started: черновик id=tts-2 sourceType=partial status=playing",
            target_language="RU",
        )
        == ""
    )
