import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from core.main_window import MainWindow
from core.nim_runtime import ensure_nim_runtime


def main():
    app = QApplication(sys.argv)

    try:
        ensure_nim_runtime()
    except Exception as error:
        QMessageBox.critical(
            None,
            "NIM Startup Error",
            str(error),
        )
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
