from __future__ import annotations

import threading
import time

import sounddevice as sd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
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

from core.app_config import AppConfig, get_default_config_path, load_app_config
from core.audio_engine import AudioEngine
from core.audio_service import (
    TRANSLATOR_SINK_NAME,
    ensure_translator_sink_exists,
    get_default_real_sink_name,
    load_source_loopback,
    set_loopback_volume_percent,
    unload_pulse_module,
)
from core.audio_utils import find_sounddevice_device_index_by_name
from core.file_logger import AppFileLogger
from core.stt_runtime import ensure_stt_runtime_for_app_config


class MainWindow(QWidget):
    log_signal = Signal(str)
    error_signal = Signal(str)

    ORIGINAL_MODE_MUTED = "muted"
    ORIGINAL_MODE_DUCKED = "ducked"
    ORIGINAL_MODE_FULL = "full"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Мой переводчик")
        self.resize(1080, 860)

        self.app_config_path = get_default_config_path()
        self.base_config = load_app_config(self.app_config_path)
        self.file_logger = AppFileLogger()
        self.file_logger.session_started()

        self.engine = AudioEngine(self._build_runtime_config())

        self.pipeline_running = False
        self.listen_active = True
        self.original_audio_mode = self.ORIGINAL_MODE_DUCKED
        self.original_duck_percent = 50
        self.original_sink_name = ""
        self.listen_input_name = ""
        self.original_loopback_module_id: str | None = None
        self._original_audio_fade_token = 0

        self._build_ui()
        self._bind_events()

        self.log_signal.connect(self._append_log_to_ui)
        self.error_signal.connect(self._append_error_to_ui)

        self.engine.on_log = lambda message: self._emit_log(f"[LISTEN] {message}")
        self.engine.on_error = lambda message: self._emit_error(f"[LISTEN] {message}")

        self.refresh_routes()
        self._update_controls_state()
        self._start_background_prewarm()

    def _build_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Локальный переводчик входящего английского в русский.\n"
            "Оригинал подаётся в приложение через TranslatorMic, перевод звучит в наушниках.\n"
            f"Конфиг: {self.app_config_path}\n"
            f"Лог: {self.file_logger.log_path}"
        )
        self.info_label.setWordWrap(True)
        self.main_layout.addWidget(self.info_label)

        self.main_layout.addWidget(self._build_routes_group())
        self.main_layout.addWidget(self._build_pipeline_group())
        self.main_layout.addWidget(self._build_listen_group())

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.main_layout.addWidget(self.log_output)

    def _build_routes_group(self) -> QGroupBox:
        group = QGroupBox("Маршрут")
        layout = QFormLayout(group)

        self.listen_source_value_label = QLabel("—")
        self.headphones_value_label = QLabel("—")

        for label in (
            self.listen_source_value_label,
            self.headphones_value_label,
        ):
            label.setWordWrap(True)

        layout.addRow("Слушать EN=>RU:", self.listen_source_value_label)
        layout.addRow("Наушники:", self.headphones_value_label)
        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Пайплайн")
        layout = QHBoxLayout(group)

        self.refresh_button = QPushButton("Обновить устройства")
        self.start_button = QPushButton("Запустить пайплайн")
        self.stop_button = QPushButton("Остановить пайплайн")
        self.stop_button.setEnabled(False)

        layout.addWidget(self.refresh_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        return group

    def _build_listen_group(self) -> QGroupBox:
        group = QGroupBox("Слушать EN=>RU")
        layout = QVBoxLayout(group)

        top_row = QHBoxLayout()
        self.listen_status_label = QLabel()
        self.listen_toggle_button = QPushButton()
        self.listen_toggle_button.setEnabled(False)
        top_row.addWidget(self.listen_status_label)
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

    def _bind_events(self) -> None:
        self.refresh_button.clicked.connect(self.refresh_routes)
        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.listen_toggle_button.clicked.connect(self.toggle_listen)
        self.listen_mute_button.clicked.connect(self.set_original_audio_muted)
        self.listen_duck_button.clicked.connect(self.set_original_audio_ducked)
        self.listen_full_button.clicked.connect(self.set_original_audio_full)
        self.original_volume_slider.valueChanged.connect(self._on_duck_volume_changed)

    def refresh_routes(self) -> None:
        try:
            ensure_translator_sink_exists()
            original_sink_name = get_default_real_sink_name() or "—"
            listen_input_name = f"{TRANSLATOR_SINK_NAME}.monitor"

            self.original_sink_name = original_sink_name if original_sink_name != "—" else ""
            self.listen_input_name = listen_input_name if listen_input_name != "—" else ""

            self.listen_source_value_label.setText(listen_input_name)
            self.headphones_value_label.setText(original_sink_name)

            self._emit_log(
                "Auto routes refreshed | "
                f"listen_input='{self.listen_input_name}' "
                f"headphones='{self.original_sink_name}'"
            )
        except Exception as error:
            QMessageBox.critical(self, "Ошибка маршрута", str(error))

    def start_pipeline(self) -> None:
        if self.pipeline_running:
            return

        self.refresh_routes()

        if not self.listen_input_name or not self.original_sink_name:
            QMessageBox.warning(self, "Ошибка маршрута", "Не удалось определить маршрут для EN=>RU.")
            return

        try:
            ensure_stt_runtime_for_app_config(self.engine.app_config)
            self._start_original_loopback()
            self._start_listen_engine()
        except Exception as error:
            self._safe_stop_engine(self.engine)
            self._stop_original_loopback()
            self.pipeline_running = False
            self.listen_active = False
            self._update_controls_state()
            QMessageBox.critical(self, "Ошибка запуска", str(error))
            return

        self.pipeline_running = True
        self.listen_active = True
        self.engine.set_translation_paused(False)
        self._apply_current_original_audio_mode(fade_in=True)
        self._update_controls_state()
        self._emit_log("EN=>RU pipeline started")

    def stop_pipeline(self) -> None:
        self.engine.set_translation_paused(False)
        self._safe_stop_engine(self.engine)
        self._stop_original_loopback()

        self.pipeline_running = False
        self.listen_active = False
        self._update_controls_state()
        self._emit_log("EN=>RU pipeline stopped")

    def toggle_listen(self) -> None:
        if not self.pipeline_running:
            return

        if self.listen_active:
            self.engine.set_translation_paused(True)
            self.listen_active = False
            self._set_original_loopback_volume(100)
            self._emit_log("Listen EN=>RU paused")
        else:
            try:
                self.engine.set_translation_paused(False)
                self.listen_active = True
                self._apply_current_original_audio_mode(fade_in=True)
                self._emit_log("Listen EN=>RU resumed")
            except Exception as error:
                QMessageBox.critical(self, "Ошибка EN=>RU", str(error))
        self._update_controls_state()

    def set_original_audio_muted(self) -> None:
        self.original_volume_slider.setValue(20)

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
        if not self.original_sink_name:
            return

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
        return set_loopback_volume_percent(self.original_loopback_module_id, percent)

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

    def _start_listen_engine(self) -> None:
        input_name = self.listen_input_name
        output_name = self.original_sink_name

        input_index = find_sounddevice_device_index_by_name(input_name, min_input_channels=1)
        output_index = find_sounddevice_device_index_by_name(output_name, min_output_channels=1)

        if input_index is None:
            raise RuntimeError(f"Не удалось сопоставить listen input: {input_name}")
        if output_index is None:
            raise RuntimeError(f"Не удалось сопоставить listen output: {output_name}")

        input_sd_name = sd.query_devices(input_index)["name"]
        output_sd_name = sd.query_devices(output_index)["name"]

        self._emit_log(f"[LISTEN] Selected pactl input: {input_name}")
        self._emit_log(f"[LISTEN] Selected pactl output: {output_name}")
        self._emit_log(f"[LISTEN] Mapped sounddevice input index: {input_index}, name: {input_sd_name}")
        self._emit_log(f"[LISTEN] Mapped sounddevice output index: {output_index}, name: {output_sd_name}")

        self.engine.start(
            input_device_index=input_index,
            output_device_index=output_index,
            selected_pactl_input_name=input_name,
            selected_pactl_output_name=output_name,
            samplerate=self.engine.app_config.audio.samplerate,
            channels=self.engine.app_config.audio.channels,
            blocksize=self.engine.app_config.audio.blocksize,
            stt_window_seconds=1.0,
        )

    def _build_runtime_config(self) -> AppConfig:
        return self.base_config

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
        try:
            self.engine.prewarm_runtime()
        except Exception as error:
            self._emit_log(f"Runtime prewarm skipped: {error}")

    def _safe_stop_engine(self, engine: AudioEngine) -> None:
        try:
            engine.stop()
        except Exception as error:
            self._emit_error(str(error))

    def _update_controls_state(self) -> None:
        self.start_button.setEnabled(not self.pipeline_running)
        self.stop_button.setEnabled(self.pipeline_running)

        self.listen_toggle_button.setEnabled(self.pipeline_running)
        for button in (self.listen_mute_button, self.listen_duck_button, self.listen_full_button):
            button.setEnabled(self.pipeline_running)
        self.original_volume_slider.setEnabled(self.pipeline_running)

        self._set_status_chip(
            self.listen_status_label,
            "EN=>RU активно" if self.listen_active and self.pipeline_running else "EN=>RU на паузе",
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
        self.log_output.append(message)

    def _append_error_to_ui(self, message: str) -> None:
        self.log_output.append(f"ОШИБКА: {message}")

    def _emit_log(self, message: str) -> None:
        self.log_signal.emit(message)
        self.file_logger.info(message)

    def _emit_error(self, message: str) -> None:
        self.error_signal.emit(message)
        self.file_logger.error(message)
