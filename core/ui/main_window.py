from __future__ import annotations

import re
import threading
import time

from PySide6.QtCore import Qt, Signal, QSize, QRectF, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config.app_config import get_default_config_path, load_app_config
from core.audio.audio_engine import AudioEngine
from core.audio.audio_service import (
    TRANSLATOR_LISTEN_SINK_NAME,
    TRANSLATOR_MIC_SOURCE_NAME,
    TRANSLATOR_SINK_NAME,
    cleanup_translator_loopbacks,
    ensure_translator_listen_sink_exists,
    ensure_translator_mic_source_exists,
    ensure_translator_sink_exists,
    get_default_real_sink_name,
    get_default_real_source_name,
    get_monitor_source_name_for_sink,
    load_source_loopback,
    repair_default_audio_devices,
    set_sink_volume_percent,
    set_loopback_volume_percent,
    unload_pulse_module,
)
from core.runtime.backend_manager import BackendManager
from core.pipeline.branch_controller import BranchController
from core.pipeline.branch_registry import BranchRegistry
from core.pipeline.branch_definitions import (
    LISTEN_LANE_DEFINITION,
    SPEAK_LANE_DEFINITION,
    get_default_lane_definitions,
)
from core.logging.file_logger import AppFileLogger
from core.pipeline.pipeline_orchestrator import PipelineOrchestrator
from core.runtime.stt_runtime import ensure_stt_runtime_for_app_config


class AudioLevelMeter(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._level = 0.0
        self.setMinimumWidth(120)
        self.setMaximumWidth(180)
        self.setMinimumHeight(18)
        self.setMaximumHeight(18)

    def sizeHint(self) -> QSize:
        return QSize(150, 18)

    def set_level(self, level: float) -> None:
        bounded = max(0.0, min(1.0, float(level)))
        if abs(bounded - self._level) < 0.01:
            return
        self._level = bounded
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect().adjusted(0, 2, 0, -2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2f2f2f"))
        painter.drawRoundedRect(rect, 6, 6)

        bars = 18
        gap = 3.0
        active_bars = int(round(self._level * bars))
        bar_width = max(2.0, (rect.width() - gap * (bars + 1)) / float(bars))

        for index in range(bars):
            x = rect.x() + gap + index * (bar_width + gap)
            level_factor = (index + 1) / float(bars)
            bar_height = max(4.0, rect.height() * level_factor)
            y = rect.y() + (rect.height() - bar_height) / 2.0
            bar_rect = QRectF(x, y, bar_width, bar_height)

            if index < active_bars:
                color = QColor("#1fbf65") if index < bars * 0.7 else QColor("#d6a21d")
            else:
                color = QColor("#555555")

            path = QPainterPath()
            path.addRoundedRect(bar_rect, 1.8, 1.8)
            painter.fillPath(path, color)


class MainWindow(QWidget):
    log_signal = Signal(str)
    error_signal = Signal(str)
    pipeline_start_succeeded_signal = Signal()
    pipeline_start_failed_signal = Signal(str)
    listen_level_signal = Signal(float)
    speak_level_signal = Signal(float)

    ORIGINAL_MODE_MUTED = "muted"
    ORIGINAL_MODE_DUCKED = "ducked"
    ORIGINAL_MODE_FULL = "full"
    def __init__(self, backend_manager: BackendManager | None = None):
        super().__init__()

        self.setWindowTitle("Голосовой перевод в реальном времени")
        self.resize(1080, 920)

        self.backend_manager = backend_manager
        self.app_config_path = get_default_config_path()
        self.base_config = load_app_config(self.app_config_path)
        self.file_logger = AppFileLogger()
        self.file_logger.session_started()

        self.branch_registry = BranchRegistry(self.base_config, get_default_lane_definitions())
        self.pipeline_orchestrator = PipelineOrchestrator(
            self.branch_registry,
            engine_factory=self._build_lane_engine,
        )

        self.pipeline_running = False
        self.pipeline_starting = False
        self._ui_spoken_item_ids: set[str] = set()

        self.original_audio_mode = self.ORIGINAL_MODE_DUCKED
        self.original_duck_percent = 50

        self.original_sink_name = ""
        self.listen_input_name = ""
        self.speak_input_name = ""

        self.original_loopback_module_id: str | None = None
        self.speak_passthrough_loopback_module_id: str | None = None
        self._original_audio_fade_token = 0
        self._speak_audio_fade_token = 0
        self._pending_reset_restore_listen_active: bool | None = None
        self._pending_reset_restore_speak_active: bool | None = None

        self._build_ui()
        self._bind_events()

        self.log_signal.connect(self._append_log_to_ui)
        self.error_signal.connect(self._append_error_to_ui)
        self.pipeline_start_succeeded_signal.connect(self._on_pipeline_started)
        self.pipeline_start_failed_signal.connect(self._on_pipeline_start_failed)
        self.listen_level_signal.connect(self._set_listen_level)
        self.speak_level_signal.connect(self._set_speak_level)

        self._bind_branch_engine(self.listen_branch, self.listen_level_signal.emit)
        self._bind_branch_engine(self.speak_branch, self.speak_level_signal.emit)

        self.refresh_routes()
        self._update_controls_state()
        self._start_background_prewarm()

        if self.backend_manager:
            self.refresh_timer = QTimer(self)
            self.refresh_timer.setInterval(500)
            self.refresh_timer.timeout.connect(self.refresh_backend_statuses)
            self.refresh_timer.start()
            self.refresh_backend_statuses()

    @property
    def listen_engine(self) -> AudioEngine:
        return self.listen_branch.engine

    @property
    def speak_engine(self) -> AudioEngine:
        return self.speak_branch.engine

    @property
    def listen_branch(self) -> BranchController:
        return self.pipeline_orchestrator.get_controller(LISTEN_LANE_DEFINITION.lane_key)

    @property
    def speak_branch(self) -> BranchController:
        return self.pipeline_orchestrator.get_controller(SPEAK_LANE_DEFINITION.lane_key)

    @property
    def branch_controllers(self) -> tuple[BranchController, ...]:
        return self.pipeline_orchestrator.list_controllers()

    @property
    def listen_active(self) -> bool:
        return self.listen_branch.active

    @listen_active.setter
    def listen_active(self, value: bool) -> None:
        self.listen_branch.active = bool(value)

    @property
    def speak_active(self) -> bool:
        return self.speak_branch.active

    @speak_active.setter
    def speak_active(self, value: bool) -> None:
        self.speak_branch.active = bool(value)

    def _build_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Локальный двунаправленный переводчик.\n"
            "EN=>RU: входящий английский берётся из TranslatorListen и переводится в наушники.\n"
            "RU=>EN: выбери в приложении микрофон TranslatorMicrophone.\n"
            f"Конфиг: {self.app_config_path}\n"
            f"Лог: {self.file_logger.log_path}"
        )
        self.info_label.setWordWrap(True)
        self.main_layout.addWidget(self.info_label)

        self.main_layout.addWidget(self._build_routes_group())
        self.main_layout.addWidget(self._build_pipeline_group())
        self.main_layout.addWidget(self._build_speak_group())
        self.main_layout.addWidget(self._build_listen_group())
        self.main_layout.addWidget(self._build_logs_group())
        if self.backend_manager:
            self.main_layout.addWidget(self._build_backend_status_group())

    def _build_backend_status_group(self) -> QGroupBox:
        group = QGroupBox("Статус backend-ов")
        layout = QHBoxLayout(group)

        self.en_backend_group = self._build_single_backend_status_ui(
            LISTEN_LANE_DEFINITION.backend_title or LISTEN_LANE_DEFINITION.title
        )
        self.ru_backend_group = self._build_single_backend_status_ui(
            SPEAK_LANE_DEFINITION.backend_title or SPEAK_LANE_DEFINITION.title
        )

        layout.addWidget(self.en_backend_group["box"], stretch=1)
        layout.addWidget(self.ru_backend_group["box"], stretch=1)
        return group

    def _build_single_backend_status_ui(self, title: str) -> dict:
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

    def _build_routes_group(self) -> QGroupBox:
        group = QGroupBox("Маршрут")
        layout = QFormLayout(group)

        self.speak_source_value_label = QLabel("—")
        self.listen_source_value_label = QLabel("—")
        self.headphones_value_label = QLabel("—")
        self.virtual_mic_value_label = QLabel(TRANSLATOR_MIC_SOURCE_NAME)
        self.listen_virtual_value_label = QLabel(TRANSLATOR_LISTEN_SINK_NAME)

        for label in (
            self.speak_source_value_label,
            self.listen_source_value_label,
            self.headphones_value_label,
            self.virtual_mic_value_label,
            self.listen_virtual_value_label,
        ):
            label.setWordWrap(True)

        layout.addRow("Говорить RU=>EN:", self.speak_source_value_label)
        layout.addRow("Слушать EN=>RU:", self.listen_source_value_label)
        layout.addRow("Наушники:", self.headphones_value_label)
        layout.addRow("Микрофон RU=>EN:", self.virtual_mic_value_label)
        layout.addRow("Virtual sink EN=>RU:", self.listen_virtual_value_label)
        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Пайплайн")
        layout = QHBoxLayout(group)

        self.refresh_button = QPushButton("Обновить устройства")
        self.reset_audio_button = QPushButton("Reset")
        self.start_button = QPushButton("Запустить пайплайн")
        self.stop_button = QPushButton("Остановить пайплайн")
        self.stop_button.setEnabled(False)

        layout.addWidget(self.refresh_button)
        layout.addWidget(self.reset_audio_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        return group

    def _build_speak_group(self) -> QGroupBox:
        group = QGroupBox(self.speak_branch.definition.title)
        layout = QHBoxLayout(group)

        self.speak_status_label = QLabel()
        self.speak_level_meter = AudioLevelMeter()
        self.speak_toggle_button = QPushButton()
        self.speak_toggle_button.setEnabled(False)

        layout.addWidget(self.speak_status_label)
        layout.addWidget(self.speak_level_meter)
        layout.addStretch(1)
        layout.addWidget(self.speak_toggle_button)
        return group

    def _build_listen_group(self) -> QGroupBox:
        group = QGroupBox(self.listen_branch.definition.title)
        layout = QVBoxLayout(group)

        top_row = QHBoxLayout()
        self.listen_status_label = QLabel()
        self.listen_level_meter = AudioLevelMeter()
        self.listen_toggle_button = QPushButton()
        self.listen_toggle_button.setEnabled(False)
        top_row.addWidget(self.listen_status_label)
        top_row.addWidget(self.listen_level_meter)
        top_row.addStretch(1)
        top_row.addWidget(self.listen_toggle_button)
        layout.addLayout(top_row)

        mode_row = QHBoxLayout()
        self.listen_mute_button = QPushButton("Заглушить")
        self.listen_duck_button = QPushButton("Приглушить")
        self.listen_full_button = QPushButton("Слушать 100%")
        for button in (self.listen_mute_button, self.listen_duck_button, self.listen_full_button):
            button.setEnabled(False)
            mode_row.addWidget(button)
        layout.addLayout(mode_row)

        slider_row = QHBoxLayout()
        self.original_volume_caption_label = QLabel("Слышимость оригинала:")
        self.original_volume_value_label = QLabel("50%")
        self.original_volume_slider = QSlider(Qt.Horizontal)
        self.original_volume_slider.setRange(0, 100)
        self.original_volume_slider.setSingleStep(5)
        self.original_volume_slider.setPageStep(10)
        self.original_volume_slider.setValue(self.original_duck_percent)
        self.original_volume_slider.setEnabled(False)
        slider_row.addWidget(self.original_volume_caption_label)
        slider_row.addWidget(self.original_volume_slider, stretch=1)
        slider_row.addWidget(self.original_volume_value_label)
        layout.addLayout(slider_row)

        return group

    def _build_logs_group(self) -> QGroupBox:
        group = QGroupBox("Логи")
        layout = QHBoxLayout(group)

        speak_column = QVBoxLayout()
        speak_header = QHBoxLayout()
        speak_label = QLabel("Что сказал я")
        self.clear_speak_logs_button = QPushButton("Очистить")
        self.clear_speak_logs_button.setMaximumWidth(90)
        speak_header.addWidget(speak_label)
        speak_header.addStretch(1)
        speak_header.addWidget(self.clear_speak_logs_button)
        self.speak_log_output = QTextEdit()
        self.speak_log_output.setReadOnly(True)
        speak_column.addLayout(speak_header)
        speak_column.addWidget(self.speak_log_output)

        listen_column = QVBoxLayout()
        listen_header = QHBoxLayout()
        listen_label = QLabel("Что сказали мне")
        self.clear_listen_logs_button = QPushButton("Очистить")
        self.clear_listen_logs_button.setMaximumWidth(90)
        listen_header.addWidget(listen_label)
        listen_header.addStretch(1)
        listen_header.addWidget(self.clear_listen_logs_button)
        self.listen_log_output = QTextEdit()
        self.listen_log_output.setReadOnly(True)
        listen_column.addLayout(listen_header)
        listen_column.addWidget(self.listen_log_output)

        layout.addLayout(speak_column, stretch=1)
        layout.addLayout(listen_column, stretch=1)
        return group

    def _bind_events(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_routes)
        self.reset_audio_button.clicked.connect(self.reset_audio)
        self.clear_speak_logs_button.clicked.connect(self._clear_speak_logs)
        self.clear_listen_logs_button.clicked.connect(self._clear_listen_logs)
        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.speak_toggle_button.clicked.connect(self.toggle_speak)
        self.listen_toggle_button.clicked.connect(self.toggle_listen)
        self.listen_mute_button.clicked.connect(self.set_original_audio_muted)
        self.listen_duck_button.clicked.connect(self.set_original_audio_ducked)
        self.listen_full_button.clicked.connect(self.set_original_audio_full)
        self.original_volume_slider.valueChanged.connect(self._on_duck_volume_changed)

        if self.backend_manager:
            self.en_backend_group["restart_button"].clicked.connect(self.backend_manager.restart_en_to_ru_async)
            self.en_backend_group["stop_button"].clicked.connect(self.backend_manager.stop_en_to_ru)
            self.ru_backend_group["restart_button"].clicked.connect(self.backend_manager.restart_ru_to_en_async)
            self.ru_backend_group["stop_button"].clicked.connect(self.backend_manager.stop_ru_to_en)

    def refresh_routes(self) -> None:
        try:
            ensure_translator_sink_exists()
            ensure_translator_listen_sink_exists()
            ensure_translator_mic_source_exists()
            if not self.pipeline_running and not self.pipeline_starting:
                for change in repair_default_audio_devices():
                    self._emit_log(f"Audio defaults repaired | {change}")
            self.speak_input_name = get_default_real_source_name() or ""
            self.original_sink_name = get_default_real_sink_name() or ""
            self.listen_input_name = get_monitor_source_name_for_sink(TRANSLATOR_LISTEN_SINK_NAME)

            self.speak_source_value_label.setText(self.speak_input_name or "—")
            self.listen_source_value_label.setText(self.listen_input_name or "—")
            self.headphones_value_label.setText(self.original_sink_name or "—")
            self.virtual_mic_value_label.setText(TRANSLATOR_MIC_SOURCE_NAME)
            self.listen_virtual_value_label.setText(TRANSLATOR_LISTEN_SINK_NAME)

            self._emit_log(
                "Auto routes refreshed | "
                f"speak_input='{self.speak_input_name}' "
                f"listen_input='{self.listen_input_name}' "
                f"headphones='{self.original_sink_name}' "
                f"virtual_mic='{TRANSLATOR_SINK_NAME}'"
            )
        except Exception as error:
            QMessageBox.critical(self, "Ошибка маршрута", str(error))

    def start_pipeline(self) -> None:
        if self.pipeline_running or self.pipeline_starting:
            return

        self._ui_spoken_item_ids.clear()
        self.refresh_routes()
        for branch in self.branch_controllers:
            self._emit_log(
                f"{branch.definition.log_prefix} "
                f"Lane backend={branch.engine.app_config.stt.backend} "
                f"branch='{branch.engine.active_branch_config.label}' "
                f"direction='{branch.engine.active_branch_config.translation_direction}'"
            )
        removed_loopbacks = cleanup_translator_loopbacks(self._emit_log)
        if removed_loopbacks:
            self._emit_log(f"Translator loopbacks cleaned before start: {removed_loopbacks}")

        if not self.listen_input_name or not self.original_sink_name:
            QMessageBox.warning(self, "Ошибка маршрута", "Не удалось определить маршрут для EN=>RU.")
            return
        if not self.speak_input_name:
            QMessageBox.warning(self, "Ошибка маршрута", "Не удалось определить микрофон для RU=>EN.")
            return

        if self.backend_manager is not None:
            self.backend_manager.ensure_started_async()

        self.pipeline_starting = True
        self._update_controls_state()
        self._emit_log("Dual pipeline starting...")
        threading.Thread(
            target=self._start_pipeline_worker,
            daemon=True,
            name="start-pipeline-worker",
        ).start()

    def stop_pipeline(self) -> None:
        if self.pipeline_starting:
            return

        self.listen_engine.set_translation_paused(False)
        self.speak_engine.set_translation_paused(False)

        self._safe_stop_engine(self.listen_engine)
        self._safe_stop_engine(self.speak_engine)
        self._stop_original_loopback()
        self._stop_speak_passthrough_loopback()
        cleanup_translator_loopbacks(self._emit_log)

        self.pipeline_running = False
        self.listen_active = False
        self.speak_active = False
        self._ui_spoken_item_ids.clear()
        self._set_listen_level(0.0)
        self._set_speak_level(0.0)
        self._update_controls_state()
        self._emit_log("Dual pipeline stopped")

    def reset_audio(self) -> None:
        if self.pipeline_starting:
            return

        was_running = self.pipeline_running
        restore_listen_active = self.listen_active
        restore_speak_active = self.speak_active

        self._emit_log("Audio reset requested...")

        if was_running:
            self.stop_pipeline()
            time.sleep(0.3)
        else:
            self._stop_original_loopback()
            self._stop_speak_passthrough_loopback()
            removed_loopbacks = cleanup_translator_loopbacks(self._emit_log)
            if removed_loopbacks:
                self._emit_log(f"Translator loopbacks cleaned during reset: {removed_loopbacks}")

        self.refresh_routes()
        set_sink_volume_percent(TRANSLATOR_LISTEN_SINK_NAME, 100)
        set_sink_volume_percent(TRANSLATOR_SINK_NAME, 100)

        if was_running:
            self._pending_reset_restore_listen_active = restore_listen_active
            self._pending_reset_restore_speak_active = restore_speak_active
            self.start_pipeline()
            return

        self._emit_log("Audio reset completed")

    def _start_pipeline_worker(self) -> None:
        try:
            for branch in self.branch_controllers:
                ensure_stt_runtime_for_app_config(
                    branch.engine.app_config,
                    branch.engine.active_branch_config,
                )

            for branch in self.branch_controllers:
                self._start_branch(branch)
            time.sleep(0.08)
            self._start_original_loopback()
        except Exception as error:
            self._safe_stop_engine(self.listen_engine)
            self._safe_stop_engine(self.speak_engine)
            self._stop_original_loopback()
            self._stop_speak_passthrough_loopback()
            self.pipeline_start_failed_signal.emit(str(error))
            return

        self.pipeline_start_succeeded_signal.emit()

    def _on_pipeline_started(self) -> None:
        self.pipeline_starting = False
        self.pipeline_running = True
        self.listen_active = True
        self.speak_active = True

        self.listen_engine.set_translation_paused(False)
        self.speak_engine.set_translation_paused(False)

        self._stop_speak_passthrough_loopback()
        self._apply_current_original_audio_mode(fade_in=True)

        restore_listen_active = self._pending_reset_restore_listen_active
        restore_speak_active = self._pending_reset_restore_speak_active
        self._pending_reset_restore_listen_active = None
        self._pending_reset_restore_speak_active = None

        if restore_speak_active is False:
            self.toggle_speak()
        if restore_listen_active is False:
            self.toggle_listen()
        if restore_listen_active is not None or restore_speak_active is not None:
            self._emit_log("Audio reset completed")

        self._update_controls_state()
        self._emit_log("Dual pipeline started")

    def _on_pipeline_start_failed(self, message: str) -> None:
        self._pending_reset_restore_listen_active = None
        self._pending_reset_restore_speak_active = None
        self.pipeline_starting = False
        self.pipeline_running = False
        self.listen_active = False
        self.speak_active = False
        self._update_controls_state()
        QMessageBox.critical(self, "Ошибка запуска", message)

    def toggle_speak(self) -> None:
        if not self.pipeline_running:
            return

        if self.speak_active:
            self.speak_branch.set_paused(True)
            self._start_speak_passthrough_loopback()
            self._emit_log(self.speak_branch.definition.paused_log_text)
        else:
            self._stop_speak_passthrough_loopback()
            self.speak_branch.set_paused(False)
            self._emit_log(self.speak_branch.definition.resumed_log_text)

        self._update_controls_state()

    def toggle_listen(self) -> None:
        if not self.pipeline_running:
            return

        if self.listen_active:
            self.listen_branch.set_paused(True)
            self._set_original_loopback_volume(100)
            self._emit_log(self.listen_branch.definition.paused_log_text)
        else:
            self.listen_branch.set_paused(False)
            self._apply_current_original_audio_mode(fade_in=True)
            self._emit_log(self.listen_branch.definition.resumed_log_text)

        self._update_controls_state()

    def set_original_audio_muted(self) -> None:
        self.original_volume_slider.setValue(0)

    def set_original_audio_ducked(self) -> None:
        self.original_volume_slider.setValue(50)

    def set_original_audio_full(self) -> None:
        self.original_volume_slider.setValue(100)

    def _on_duck_volume_changed(self, value: int) -> None:
        self.original_duck_percent = int(value)
        self.original_volume_value_label.setText(f"{self.original_duck_percent}%")
        self.original_audio_mode = self._classify_original_audio_mode(self.original_duck_percent)
        self._apply_current_original_audio_mode()
        self._update_controls_state()

    def _apply_current_original_audio_mode(self, *, fade_in: bool = False) -> None:
        if not self.pipeline_running or not self.original_loopback_module_id:
            return

        target_percent = self.original_duck_percent
        if fade_in:
            self._fade_original_loopback_volume(target_percent)
            success = True
        else:
            success = self._set_original_loopback_volume(target_percent)
        self._emit_log(
            f"Original audio mode: {self.original_audio_mode} "
            f"({target_percent}%) -> {'OK' if success else 'FAILED'}"
        )

    def _set_original_loopback_volume(self, percent: int) -> bool:
        if not self.original_loopback_module_id:
            return False
        if percent <= 0:
            effective_percent = 0
        else:
            # UI percent is intentionally non-linear so low values can approach a real mute.
            effective_percent = max(1, int(round((percent * percent) / 100.0)))
        return set_loopback_volume_percent(self.original_loopback_module_id, effective_percent)

    def _start_original_loopback(self) -> None:
        self._stop_original_loopback()

        module_id = load_source_loopback(
            self.listen_input_name,
            self.original_sink_name,
            latency_msec=30,
        )
        if not module_id:
            raise RuntimeError("Не удалось создать loopback оригинального звука.")

        self.original_loopback_module_id = module_id
        time.sleep(0.05)
        self._set_original_loopback_volume(0)
        self._emit_log(
            "Original audio loopback started | "
            f"module_id={module_id} source='{self.listen_input_name}' sink='{self.original_sink_name}'"
        )

    def _stop_original_loopback(self) -> None:
        module_id = self.original_loopback_module_id
        self.original_loopback_module_id = None
        if not module_id:
            return

        success = unload_pulse_module(module_id)
        self._emit_log(
            f"Original audio loopback stopped | module_id={module_id} -> {'OK' if success else 'FAILED'}"
        )

    def _fade_original_loopback_volume(self, target_percent: int) -> None:
        module_id = self.original_loopback_module_id
        if not module_id:
            return

        self._original_audio_fade_token += 1
        fade_token = self._original_audio_fade_token

        def worker() -> None:
            steps = 6
            step_delay_sec = 0.02
            self._set_original_loopback_volume(0)
            for step in range(1, steps + 1):
                if (
                    fade_token != self._original_audio_fade_token
                    or module_id != self.original_loopback_module_id
                    or not self.pipeline_running
                ):
                    return
                level = int(round(target_percent * step / float(steps)))
                self._set_original_loopback_volume(level)
                time.sleep(step_delay_sec)

        threading.Thread(
            target=worker,
            daemon=True,
            name="original-audio-fade-in",
        ).start()

    def _set_speak_passthrough_volume(self, percent: int) -> bool:
        if not self.speak_passthrough_loopback_module_id:
            return False
        return set_loopback_volume_percent(self.speak_passthrough_loopback_module_id, percent)

    def _start_speak_passthrough_loopback(self) -> None:
        self._stop_speak_passthrough_loopback()

        module_id = load_source_loopback(
            self.speak_input_name,
            TRANSLATOR_SINK_NAME,
            latency_msec=20,
        )
        if not module_id:
            raise RuntimeError("Не удалось создать mic loopback для RU=>EN.")

        self.speak_passthrough_loopback_module_id = module_id
        self._set_speak_passthrough_volume(0)
        self._emit_log(
            "Speak passthrough loopback started | "
            f"module_id={module_id} source='{self.speak_input_name}' sink='{TRANSLATOR_SINK_NAME}'"
        )
        self._fade_speak_passthrough_volume(100)

    def _stop_speak_passthrough_loopback(self) -> None:
        module_id = self.speak_passthrough_loopback_module_id
        self.speak_passthrough_loopback_module_id = None
        if not module_id:
            return

        success = unload_pulse_module(module_id)
        self._emit_log(
            f"Speak passthrough loopback stopped | module_id={module_id} -> {'OK' if success else 'FAILED'}"
        )

    def _fade_speak_passthrough_volume(self, target_percent: int) -> None:
        module_id = self.speak_passthrough_loopback_module_id
        if not module_id:
            return

        self._speak_audio_fade_token += 1
        fade_token = self._speak_audio_fade_token

        def worker() -> None:
            steps = 5
            step_delay_sec = 0.02
            self._set_speak_passthrough_volume(0)
            for step in range(1, steps + 1):
                if (
                    fade_token != self._speak_audio_fade_token
                    or module_id != self.speak_passthrough_loopback_module_id
                    or not self.pipeline_running
                ):
                    return
                level = int(round(target_percent * step / float(steps)))
                self._set_speak_passthrough_volume(level)
                time.sleep(step_delay_sec)

        threading.Thread(
            target=worker,
            daemon=True,
            name="speak-audio-fade-in",
        ).start()

    def _start_branch(self, branch: BranchController) -> None:
        branch.start(
            input_name=self._resolve_branch_input_name(branch),
            output_name=self._resolve_branch_output_name(branch),
        )

    def reassign_lane_branch(self, lane_key: str, branch_id: str) -> None:
        was_running = self.pipeline_running
        controller = self.pipeline_orchestrator.get_controller(lane_key)
        was_active = controller.active

        replacement = self.pipeline_orchestrator.reassign_lane_branch(lane_key, branch_id)
        self._bind_branch_engine(
            replacement,
            self.listen_level_signal.emit
            if lane_key == LISTEN_LANE_DEFINITION.lane_key
            else self.speak_level_signal.emit,
        )

        if not was_running:
            if not was_active:
                replacement.set_paused(True)
            self._update_controls_state()
            return

        ensure_stt_runtime_for_app_config(
            replacement.engine.app_config,
            replacement.engine.active_branch_config,
        )
        self._start_branch(replacement)
        if not was_active:
            replacement.set_paused(True)
        self._update_controls_state()

    def _resolve_branch_input_name(self, branch: BranchController) -> str:
        if branch.definition.input_route_role == "real_source":
            return self.speak_input_name
        if branch.definition.input_route_role == "listen_monitor":
            return self.listen_input_name
        raise RuntimeError(f"Unsupported input route role: {branch.definition.input_route_role}")

    def _resolve_branch_output_name(self, branch: BranchController) -> str:
        if branch.definition.output_route_role == "real_sink":
            return self.original_sink_name
        if branch.definition.output_route_role == "translator_sink":
            return TRANSLATOR_SINK_NAME
        raise RuntimeError(f"Unsupported output route role: {branch.definition.output_route_role}")

    def _classify_original_audio_mode(self, percent: int) -> str:
        if percent <= 30:
            return self.ORIGINAL_MODE_MUTED
        if percent >= 75:
            return self.ORIGINAL_MODE_FULL
        return self.ORIGINAL_MODE_DUCKED

    def _start_background_prewarm(self) -> None:
        worker = threading.Thread(
            target=self._background_prewarm,
            daemon=True,
            name="runtime-prewarm",
        )
        worker.start()

    def _background_prewarm(self) -> None:
        for branch in self.branch_controllers:
            if self.backend_manager is not None and branch is self.speak_branch:
                continue
            try:
                branch.prewarm_runtime()
            except Exception as error:
                self._emit_log(f"{branch.definition.title} runtime prewarm skipped: {error}")

    def _bind_branch_engine(self, branch: BranchController, input_level_handler) -> None:
        self.pipeline_orchestrator.bind_lane_callbacks(
            branch.definition.lane_key,
            emit_log=self._emit_log,
            emit_error=self._emit_error,
            emit_input_level=input_level_handler,
        )

    def _build_lane_engine(self, lane_key: str) -> AudioEngine:
        return AudioEngine(
            self.branch_registry.build_runtime_config(lane_key),
            active_branch_config=self.branch_registry.resolve_runtime_branch_config(lane_key),
        )

    def _safe_stop_engine(self, engine: AudioEngine) -> None:
        try:
            engine.stop()
        except Exception as error:
            self._emit_error(str(error))

    def _update_controls_state(self) -> None:
        self.start_button.setEnabled(not self.pipeline_running)
        self.stop_button.setEnabled(self.pipeline_running and not self.pipeline_starting)

        self.refresh_button.setEnabled(not self.pipeline_starting)
        self.reset_audio_button.setEnabled(not self.pipeline_starting)
        self.speak_toggle_button.setEnabled(self.pipeline_running and not self.pipeline_starting)
        self.listen_toggle_button.setEnabled(self.pipeline_running and not self.pipeline_starting)
        for button in (self.listen_mute_button, self.listen_duck_button, self.listen_full_button):
            button.setEnabled(self.pipeline_running and not self.pipeline_starting)
        self.original_volume_slider.setEnabled(self.pipeline_running and not self.pipeline_starting)

        self.reset_audio_button.setStyleSheet(
            "QPushButton { "
            "background: #c99612; color: black; border: none; "
            "padding: 8px 14px; border-radius: 8px; font-weight: 700; }"
        )

        self._set_status_chip(
            self.speak_status_label,
            self.speak_branch.definition.active_status_text
            if self.speak_active and self.pipeline_running
            else self.speak_branch.definition.paused_status_text,
            "#177245" if self.speak_active and self.pipeline_running else "#7a7a7a",
        )
        self._set_button_style(
            self.speak_toggle_button,
            "Пауза" if self.speak_active and self.pipeline_running else "Продолжить",
            "#b74b2a" if self.speak_active and self.pipeline_running else "#177245",
        )

        self._set_status_chip(
            self.listen_status_label,
            self.listen_branch.definition.active_status_text
            if self.listen_active and self.pipeline_running
            else self.listen_branch.definition.paused_status_text,
            "#177245" if self.listen_active and self.pipeline_running else "#7a7a7a",
        )
        self._set_button_style(
            self.listen_toggle_button,
            "Пауза" if self.listen_active and self.pipeline_running else "Продолжить",
            "#b74b2a" if self.listen_active and self.pipeline_running else "#177245",
        )

        self._set_mode_button_style(
            self.listen_mute_button,
            active=self.original_audio_mode == self.ORIGINAL_MODE_MUTED and self.pipeline_running,
            active_color="#7d1f1f",
        )
        self._set_mode_button_style(
            self.listen_duck_button,
            active=self.original_audio_mode == self.ORIGINAL_MODE_DUCKED and self.pipeline_running,
            active_color="#8b6b11",
        )
        self._set_mode_button_style(
            self.listen_full_button,
            active=self.original_audio_mode == self.ORIGINAL_MODE_FULL and self.pipeline_running,
            active_color="#1b5f9e",
        )
        self.original_volume_value_label.setText(f"{self.original_duck_percent}%")

    @staticmethod
    def _set_status_chip(label: QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            "QLabel { "
            f"background: {color}; color: white; border-radius: 8px; "
            "padding: 6px 10px; font-weight: 600; }"
        )

    @staticmethod
    def _set_button_style(button: QPushButton, text: str, color: str) -> None:
        button.setText(text)
        button.setStyleSheet(
            "QPushButton { "
            f"background: {color}; color: white; border: none; "
            "padding: 8px 14px; border-radius: 8px; font-weight: 600; }"
        )

    @staticmethod
    def _set_mode_button_style(button: QPushButton, *, active: bool, active_color: str) -> None:
        color = active_color if active else "#545454"
        button.setStyleSheet(
            "QPushButton { "
            f"background: {color}; color: white; border: none; "
            "padding: 8px 12px; border-radius: 8px; font-weight: 600; }"
        )

    def _append_log_to_ui(self, message: str) -> None:
        stripped_message = message.strip()
        if not stripped_message:
            return

        speak_prefix = f"{self.speak_branch.definition.log_prefix} "
        listen_prefix = f"{self.listen_branch.definition.log_prefix} "

        if stripped_message.startswith(speak_prefix):
            ui_message = self._format_user_facing_spoken_log(
                stripped_message[len(speak_prefix):],
                target_language="EN",
            )
            if ui_message:
                self.speak_log_output.append(ui_message)
            return

        if stripped_message.startswith(listen_prefix):
            ui_message = self._format_user_facing_spoken_log(
                stripped_message[len(listen_prefix):],
                target_language="RU",
            )
            if ui_message:
                self.listen_log_output.append(ui_message)
            return

    def _format_user_facing_spoken_log(self, message: str, *, target_language: str) -> str:
        if not message.startswith("PLAYBACK started: "):
            return ""

        if " sourceType=final " not in message or " status=playing " not in message:
            return ""

        item_match = re.search(r"\bid=(tts-\d+)\b", message)
        if item_match:
            item_id = item_match.group(1)
            if item_id in self._ui_spoken_item_ids:
                return ""
            self._ui_spoken_item_ids.add(item_id)

        text = message[len("PLAYBACK started: "):].strip()
        text = re.sub(r"\s+id=tts-\d+\b.*$", "", text).strip()
        if not text:
            return ""

        return f"{target_language}: {text}"

    def _append_error_to_ui(self, message: str) -> None:
        stripped_message = message.strip()
        speak_prefix = f"{self.speak_branch.definition.log_prefix} "
        listen_prefix = f"{self.listen_branch.definition.log_prefix} "

        if stripped_message.startswith(speak_prefix):
            self.speak_log_output.append(f"ОШИБКА: {stripped_message[len(speak_prefix):]}")
            return

        if stripped_message.startswith(listen_prefix):
            self.listen_log_output.append(f"ОШИБКА: {stripped_message[len(listen_prefix):]}")
            return

        self.speak_log_output.append(f"ОШИБКА: {stripped_message}")
        self.listen_log_output.append(f"ОШИБКА: {stripped_message}")

    def _emit_log(self, message: str) -> None:
        self.log_signal.emit(message)
        self.file_logger.info(message)

    def _emit_error(self, message: str) -> None:
        self.error_signal.emit(message)
        self.file_logger.error(message)

    def _set_listen_level(self, level: float) -> None:
        self.listen_level_meter.set_level(level)

    def _set_speak_level(self, level: float) -> None:
        self.speak_level_meter.set_level(level)

    def _clear_speak_logs(self) -> None:
        self.speak_log_output.clear()

    def _clear_listen_logs(self) -> None:
        self.listen_log_output.clear()

    def refresh_backend_statuses(self) -> None:
        if not self.backend_manager:
            return
        statuses = {
            status.backend_id: status
            for status in self.backend_manager.get_status_snapshot()
        }
        self._apply_backend_status(self.en_backend_group, statuses.get(LISTEN_LANE_DEFINITION.backend_id))
        self._apply_backend_status(self.ru_backend_group, statuses.get(SPEAK_LANE_DEFINITION.backend_id))

    def _apply_backend_status(self, group: dict, status) -> None:
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
        if self.pipeline_running:
            self.stop_pipeline()
        super().closeEvent(event)
