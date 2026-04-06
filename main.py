import faulthandler
import sys
import threading
from pathlib import Path
from PySide6.QtWidgets import QApplication
from core.main_window import MainWindow
from core.app_config import load_app_config
from core.stt_runtime import ensure_stt_runtime_for_app_config


def _startup_log(message: str) -> None:
    text = f"[startup] {message}"
    print(text, flush=True)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    startup_log_path = log_dir / "startup.log"
    with startup_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")


def _warm_stt_runtime(app_config) -> None:
    try:
        _startup_log("warming STT runtime in background")
        ensure_stt_runtime_for_app_config(app_config)
        _startup_log("STT runtime ready")
    except Exception as error:
        _startup_log(f"Background STT warmup skipped: {error}")


def main():
    faulthandler.enable()
    _startup_log("creating QApplication")
    app = QApplication(sys.argv)
    _startup_log("QApplication created")

    try:
        _startup_log("loading app config")
        app_config = load_app_config()
        _startup_log(
            "app config loaded: "
            f"backend={app_config.stt.backend}, "
            f"direction={app_config.branches.primary.translation_direction}, "
            f"nim_container={app_config.branches.primary.nim_container_id}"
        )
    except Exception as error:
        _startup_log(f"Config startup error: {error}")
        print(f"Config startup error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)

    _startup_log("creating MainWindow")
    window = MainWindow()
    _startup_log("MainWindow created")
    window.show()
    _startup_log("MainWindow shown")
    threading.Thread(
        target=_warm_stt_runtime,
        args=(app_config,),
        daemon=True,
        name="startup-stt-warmup",
    ).start()
    _startup_log("entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
