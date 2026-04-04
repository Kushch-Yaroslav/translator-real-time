import faulthandler
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from core.main_window import MainWindow
from core.app_config import load_app_config
from core.nim_runtime import ensure_nim_runtime_for_app_config


def _startup_log(message: str) -> None:
    text = f"[startup] {message}"
    print(text, flush=True)

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    startup_log_path = log_dir / "startup.log"
    with startup_log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")


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
            f"direction={app_config.branches.primary.translation_direction}, "
            f"nim_container={app_config.branches.primary.nim_container_id}"
        )
        _startup_log("ensuring NIM runtime")
        ensure_nim_runtime_for_app_config(app_config)
        _startup_log("NIM runtime ready")
    except Exception as error:
        _startup_log(f"NIM startup error: {error}")
        print(f"NIM startup error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)

    _startup_log("creating MainWindow")
    window = MainWindow()
    _startup_log("MainWindow created")
    window.show()
    _startup_log("MainWindow shown, entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
