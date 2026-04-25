import faulthandler
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from core.runtime.backend_manager import BackendManager
from core.ui.backend_status_window import BackendStatusWindow
from core.ui.benchmark_recorder_window import BenchmarkRecorderWindow
from core.ui.main_window import MainWindow
from core.config.app_config import get_default_branch_config, load_app_config


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
    app.setQuitOnLastWindowClosed(False)
    _startup_log("QApplication created")

    try:
        _startup_log("loading app config")
        app_config = load_app_config()
        default_branch = get_default_branch_config(app_config)
        _startup_log(
            "app config loaded: "
            f"backend={app_config.stt.backend}, "
            f"direction={default_branch.translation_direction}, "
            f"nim_container={default_branch.nim_container_id}"
        )
    except Exception as error:
        _startup_log(f"Config startup error: {error}")
        print(f"Config startup error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)

    _startup_log("creating BackendManager")
    backend_manager = BackendManager(app_config, Path(__file__).resolve().parent)
    backend_manager.ensure_started_async()
    _startup_log("BackendManager created")

    _startup_log("creating BackendStatusWindow")
    status_window = BackendStatusWindow(backend_manager)
    status_window.show()
    _startup_log("BackendStatusWindow shown")

    _startup_log("creating MainWindow")
    window = MainWindow(backend_manager=backend_manager)
    _startup_log("MainWindow created")
    window.show()
    _startup_log("MainWindow shown")

    _startup_log("creating BenchmarkRecorderWindow")
    benchmark_window = BenchmarkRecorderWindow()
    benchmark_window.show()
    _startup_log("BenchmarkRecorderWindow shown")

    _startup_log("entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
