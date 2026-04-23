from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.runtime.backend_manager import BackendManager
from core.pipeline.branch_definitions import LISTEN_LANE_DEFINITION, SPEAK_LANE_DEFINITION


class BackendStatusWindow(QWidget):
    def __init__(self, backend_manager: BackendManager):
        super().__init__()
        self.backend_manager = backend_manager

        self.setWindowTitle("Статус backend-ов")
        self.resize(560, 240)

        self._build_ui()
        self._bind_events()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(500)
        self.refresh_timer.timeout.connect(self.refresh_statuses)
        self.refresh_timer.start()
        self.refresh_statuses()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.en_group = self._build_backend_group(
            LISTEN_LANE_DEFINITION.backend_title or LISTEN_LANE_DEFINITION.title
        )
        self.ru_group = self._build_backend_group(
            SPEAK_LANE_DEFINITION.backend_title or SPEAK_LANE_DEFINITION.title
        )
        layout.addWidget(self.en_group["box"])
        layout.addWidget(self.ru_group["box"])

    def _build_backend_group(self, title: str) -> dict[str, QWidget]:
        box = QGroupBox(title)
        layout = QGridLayout(box)

        status_label = QLabel("—")
        detail_label = QLabel("—")
        detail_label.setWordWrap(True)
        restart_button = QPushButton("Перезапуск")
        stop_button = QPushButton("Остановить")

        layout.addWidget(QLabel("Статус:"), 0, 0)
        layout.addWidget(status_label, 0, 1)
        layout.addWidget(QLabel("Детали:"), 1, 0)
        layout.addWidget(detail_label, 1, 1, 1, 2)
        layout.addWidget(restart_button, 2, 1)
        layout.addWidget(stop_button, 2, 2)

        return {
            "box": box,
            "status_label": status_label,
            "detail_label": detail_label,
            "restart_button": restart_button,
            "stop_button": stop_button,
        }

    def _bind_events(self) -> None:
        self.en_group["restart_button"].clicked.connect(self.backend_manager.restart_en_to_ru_async)
        self.en_group["stop_button"].clicked.connect(self.backend_manager.stop_en_to_ru)
        self.ru_group["restart_button"].clicked.connect(self.backend_manager.restart_ru_to_en_async)
        self.ru_group["stop_button"].clicked.connect(self.backend_manager.stop_ru_to_en)

    def refresh_statuses(self) -> None:
        statuses = {
            status.backend_id: status
            for status in self.backend_manager.get_status_snapshot()
        }

        self._apply_status(self.en_group, statuses.get(LISTEN_LANE_DEFINITION.backend_id))
        self._apply_status(self.ru_group, statuses.get(SPEAK_LANE_DEFINITION.backend_id))

    def _apply_status(self, group: dict[str, QWidget], status) -> None:
        if status is None:
            return

        color = {
            "ready": "#177245",
            "starting": "#8b6b11",
            "checking": "#5a5a5a",
            "stopped": "#7a7a7a",
            "error": "#7d1f1f",
        }.get(status.state, "#5a5a5a")

        group["status_label"].setText(status.state)
        group["status_label"].setStyleSheet(
            "QLabel { "
            f"background: {color}; color: white; border-radius: 8px; "
            "padding: 6px 10px; font-weight: 600; }"
        )
        group["detail_label"].setText(status.detail)

    def closeEvent(self, event) -> None:
        app = QApplication.instance()
        if app is not None:
            app.quit()
        super().closeEvent(event)
