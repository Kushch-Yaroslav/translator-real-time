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
    set_loopback_volume_percent,
    temporary_pulse_stream_properties,
    unload_pulse_module,
)
from core.audio_utils import find_sounddevice_device_index_by_name
from core.backend_manager import BackendManager, build_ru_to_en_runtime_config
from core.file_logger import AppFileLogger
from core.stt_runtime import ensure_stt_runtime_for_app_config


class MainWindow(QWidget):
    log_signal = Signal(str)
    error_signal = Signal(str)
    pipeline_start_succeeded_signal = Signal()
    pipeline_start_failed_signal = Signal(str)

    ORIGINAL_MODE_MUTED = "muted"
    ORIGINAL_MODE_DUCKED = "ducked"
    ORIGINAL_MODE_FULL = "full"
    LISTEN_STREAM_TAG = "TranslatorListenEngine"
    SPEAK_STREAM_TAG = "TranslatorSpeakEngine"

    def __init__(self, backend_manager: BackendManager | None = None):
        super().__init__()

        self.setWindowTitle("Голосовой перевод в реальном времени")
        self.resize(1080, 920)

        self.backend_manager = backend_manager
        self.app_config_path = get_default_config_path()
        self.base_config = load_app_config(self.app_config_path)
        self.file_logger = AppFileLogger()
        self.file_logger.session_started()

        self.listen_engine = AudioEngine(self._build_listen_config())
        self.speak_engine = (
            backend_manager.get_ru_to_en_engine()
            if backend_manager is not None
            else AudioEngine(self._build_speak_config())
        )

        self.pipeline_running = False
        self.pipeline_starting = False
        self.listen_active = True
        self.speak_active = True

        self.original_audio_mode = self.ORIGINAL_MODE_DUCKED
        self.original_duck_percent = 50

        self.original_sink_name = ""
        self.listen_input_name = ""
        self.speak_input_name = ""

        self.original_loopback_module_id: str | None = None
        self.speak_passthrough_loopback_module_id: str | None = None
        self._original_audio_fade_token = 0
        self._speak_audio_fade_token = 0

        self._build_ui()
        self._bind_events()

        self.log_signal.connect(self._append_log_to_ui)
        self.error_signal.connect(self._append_error_to_ui)
        self.pipeline_start_succeeded_signal.connect(self._on_pipeline_started)
        self.pipeline_start_failed_signal.connect(self._on_pipeline_start_failed)

        self.listen_engine.on_log = lambda message: self._emit_log(f"[LISTEN] {message}")
        self.listen_engine.on_error = lambda message: self._emit_error(f"[LISTEN] {message}")
        self.speak_engine.on_log = lambda message: self._emit_log(f"[SPEAK] {message}")
        self.speak_engine.on_error = lambda message: self._emit_error(f"[SPEAK] {message}")

        self.refresh_routes()
        self._update_controls_state()
        self._start_background_prewarm()

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

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.main_layout.addWidget(self.log_output)

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
        self.start_button = QPushButton("Запустить пайплайн")
        self.stop_button = QPushButton("Остановить пайплайн")
        self.stop_button.setEnabled(False)

        layout.addWidget(self.refresh_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        return group

    def _build_speak_group(self) -> QGroupBox:
        group = QGroupBox("Говорить RU=>EN")
        layout = QHBoxLayout(group)

        self.speak_status_label = QLabel()
        self.speak_toggle_button = QPushButton()
        self.speak_toggle_button.setEnabled(False)

        layout.addWidget(self.speak_status_label)
        layout.addStretch(1)
        layout.addWidget(self.speak_toggle_button)
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
        self.speak_toggle_button.clicked.connect(self.toggle_speak)
        self.listen_toggle_button.clicked.connect(self.toggle_listen)
        self.listen_mute_button.clicked.connect(self.set_original_audio_muted)
        self.listen_duck_button.clicked.connect(self.set_original_audio_ducked)
        self.listen_full_button.clicked.connect(self.set_original_audio_full)
        self.original_volume_slider.valueChanged.connect(self._on_duck_volume_changed)

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

        self.refresh_routes()
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
        self._update_controls_state()
        self._emit_log("Dual pipeline stopped")

    def _start_pipeline_worker(self) -> None:
        try:
            ensure_stt_runtime_for_app_config(self.listen_engine.app_config)
            ensure_stt_runtime_for_app_config(self.speak_engine.app_config)

            self._start_listen_engine()
            self._start_speak_engine()
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

        self._update_controls_state()
        self._emit_log("Dual pipeline started")

    def _on_pipeline_start_failed(self, message: str) -> None:
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
            self.speak_engine.set_translation_paused(True)
            self.speak_active = False
            self._start_speak_passthrough_loopback()
            self._emit_log("Speak RU=>EN paused")
        else:
            self._stop_speak_passthrough_loopback()
            self.speak_engine.set_translation_paused(False)
            self.speak_active = True
            self._emit_log("Speak RU=>EN resumed")

        self._update_controls_state()

    def toggle_listen(self) -> None:
        if not self.pipeline_running:
            return

        if self.listen_active:
            self.listen_engine.set_translation_paused(True)
            self.listen_active = False
            self._set_original_loopback_volume(100)
            self._emit_log("Listen EN=>RU paused")
        else:
            self.listen_engine.set_translation_paused(False)
            self.listen_active = True
            self._apply_current_original_audio_mode(fade_in=True)
            self._emit_log("Listen EN=>RU resumed")

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

        with temporary_pulse_stream_properties(
            application_name=self.LISTEN_STREAM_TAG,
            media_name=self.LISTEN_STREAM_TAG,
        ):
            self.listen_engine.start(
                input_device_index=input_index,
                output_device_index=output_index,
                selected_pactl_input_name=input_name,
                selected_pactl_output_name=output_name,
                samplerate=self.listen_engine.app_config.audio.samplerate,
                channels=self.listen_engine.app_config.audio.channels,
                blocksize=self.listen_engine.app_config.audio.blocksize,
                stt_window_seconds=1.0,
                pulse_stream_tag=self.LISTEN_STREAM_TAG,
            )

    def _start_speak_engine(self) -> None:
        input_name = self.speak_input_name
        output_name = TRANSLATOR_SINK_NAME

        input_index = find_sounddevice_device_index_by_name(input_name, min_input_channels=1)
        output_index = find_sounddevice_device_index_by_name(output_name, min_output_channels=1)

        if input_index is None:
            raise RuntimeError(f"Не удалось сопоставить speak input: {input_name}")
        if output_index is None:
            raise RuntimeError(f"Не удалось сопоставить speak output: {output_name}")

        input_sd_name = sd.query_devices(input_index)["name"]
        output_sd_name = sd.query_devices(output_index)["name"]

        self._emit_log(f"[SPEAK] Selected pactl input: {input_name}")
        self._emit_log(f"[SPEAK] Selected pactl output: {output_name}")
        self._emit_log(f"[SPEAK] Mapped sounddevice input index: {input_index}, name: {input_sd_name}")
        self._emit_log(f"[SPEAK] Mapped sounddevice output index: {output_index}, name: {output_sd_name}")

        with temporary_pulse_stream_properties(
            application_name=self.SPEAK_STREAM_TAG,
            media_name=self.SPEAK_STREAM_TAG,
        ):
            self.speak_engine.start(
                input_device_index=input_index,
                output_device_index=output_index,
                selected_pactl_input_name=input_name,
                selected_pactl_output_name=output_name,
                samplerate=self.speak_engine.app_config.audio.samplerate,
                channels=self.speak_engine.app_config.audio.channels,
                blocksize=self.speak_engine.app_config.audio.blocksize,
                stt_window_seconds=1.0,
                pulse_stream_tag=self.SPEAK_STREAM_TAG,
            )

    def _build_listen_config(self) -> AppConfig:
        return self.base_config

    def _build_speak_config(self) -> AppConfig:
        return build_ru_to_en_runtime_config(self.base_config)

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
            self.listen_engine.prewarm_runtime()
        except Exception as error:
            self._emit_log(f"Listen runtime prewarm skipped: {error}")

        if self.backend_manager is None:
            try:
                self.speak_engine.prewarm_runtime()
            except Exception as error:
                self._emit_log(f"Speak runtime prewarm skipped: {error}")

    def _safe_stop_engine(self, engine: AudioEngine) -> None:
        try:
            engine.stop()
        except Exception as error:
            self._emit_error(str(error))

    def _update_controls_state(self) -> None:
        self.start_button.setEnabled(not self.pipeline_running)
        self.stop_button.setEnabled(self.pipeline_running and not self.pipeline_starting)

        self.refresh_button.setEnabled(not self.pipeline_starting)
        self.speak_toggle_button.setEnabled(self.pipeline_running and not self.pipeline_starting)
        self.listen_toggle_button.setEnabled(self.pipeline_running and not self.pipeline_starting)
        for button in (self.listen_mute_button, self.listen_duck_button, self.listen_full_button):
            button.setEnabled(self.pipeline_running and not self.pipeline_starting)
        self.original_volume_slider.setEnabled(self.pipeline_running and not self.pipeline_starting)

        self._set_status_chip(
            self.speak_status_label,
            "RU=>EN активно" if self.speak_active and self.pipeline_running else "RU=>EN на паузе",
            "#177245" if self.speak_active and self.pipeline_running else "#7a7a7a",
        )
        self._set_button_style(
            self.speak_toggle_button,
            "Пауза" if self.speak_active and self.pipeline_running else "Продолжить",
            "#b74b2a" if self.speak_active and self.pipeline_running else "#177245",
        )

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

    def closeEvent(self, event) -> None:
        if self.pipeline_running:
            self.stop_pipeline()
        super().closeEvent(event)
