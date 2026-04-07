from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from typing import List

import sounddevice as sd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.app_config import (
    AppConfig,
    AppRuntimeConfig,
    AudioConfig,
    BranchesConfig,
    STTConfig,
    TTSRuntimeConfig,
    TranslationBranchConfig,
    TranslationRuntimeConfig,
    get_default_config_path,
    get_primary_branch_config,
    get_profiles_dir,
    list_profile_paths,
    load_app_config,
    save_app_config,
)
from core.audio_engine import AudioEngine
from core.audio_service import (
    AudioDevice,
    TRANSLATOR_SINK_NAME,
    enrich_default_flags,
    ensure_translator_sink_exists,
    list_input_devices,
    list_output_devices,
)
from core.audio_utils import find_sounddevice_device_index_by_name
from core.chunk_processor import ProcessingMode
from core.file_logger import AppFileLogger
from core.stt_runtime import ensure_stt_runtime_for_app_config


class MainWindow(QWidget):
    log_signal = Signal(str)
    error_signal = Signal(str)

    PRESET_BALANCED = "balanced"
    PRESET_LOW_LATENCY = "low_latency"
    PRESET_HIGH_QUALITY = "high_quality"

    MODE_ALL = "all"
    MODE_LISTEN_ONLY = "listen_only"
    MODE_SPEAK_ONLY = "speak_only"

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Мой переводчик")
        self.resize(1080, 860)

        self.app_config_path = get_default_config_path()
        self.profiles_dir = get_profiles_dir()
        self.app_config = load_app_config(self.app_config_path)
        self.file_logger = AppFileLogger()
        self.file_logger.session_started()
        self.engine = AudioEngine(self.app_config)

        self.input_devices: List[AudioDevice] = []
        self.output_devices: List[AudioDevice] = []

        self._build_ui()
        self._bind_events()

        self.log_signal.connect(self._append_log_to_ui)
        self.error_signal.connect(self._append_error_to_ui)

        self.engine.on_log = self._handle_engine_log
        self.engine.on_error = self._handle_engine_error

        self._apply_config_to_ui(self.app_config)
        self._reload_profiles()
        self.load_devices()
        self._start_background_prewarm()

    def _build_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Локальный пайплайн: микрофон -> STT -> перевод -> TTS -> TranslatorMic\n"
            "В Telegram выбирай устройство записи: Monitor of TranslatorMic\n"
            f"Конфиг: {self.app_config_path}\n"
            f"Лог: {self.file_logger.log_path}"
        )
        self.main_layout.addWidget(self.info_label)

        self.main_layout.addWidget(self._build_devices_group())
        self.main_layout.addWidget(self._build_profiles_group())
        self.main_layout.addWidget(self._build_runtime_group())
        self.main_layout.addWidget(self._build_stt_group())
        self.main_layout.addWidget(self._build_tts_group())
        self.main_layout.addWidget(self._build_audio_group())
        self.main_layout.addWidget(self._build_controls_group())

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.main_layout.addWidget(self.log_output)

    def _build_devices_group(self) -> QGroupBox:
        group = QGroupBox("Устройства")
        layout = QFormLayout(group)

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()

        layout.addRow("Микрофон:", self.input_combo)
        layout.addRow("Выход приложения:", self.output_combo)
        return group

    def _build_profiles_group(self) -> QGroupBox:
        group = QGroupBox("Профили и пресеты")
        layout = QVBoxLayout(group)

        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Сбалансированный", self.PRESET_BALANCED)
        self.preset_combo.addItem("Низкая задержка", self.PRESET_LOW_LATENCY)
        self.preset_combo.addItem("Стабильное качество", self.PRESET_HIGH_QUALITY)
        self.apply_preset_button = QPushButton("Применить пресет")
        preset_layout.addWidget(QLabel("Макрос:"))
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.apply_preset_button)
        layout.addLayout(preset_layout)

        profile_layout = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.load_profile_button = QPushButton("Загрузить профиль")
        self.save_profile_button = QPushButton("Сохранить как профиль")
        profile_layout.addWidget(QLabel("Профиль:"))
        profile_layout.addWidget(self.profile_combo)
        profile_layout.addWidget(self.load_profile_button)
        profile_layout.addWidget(self.save_profile_button)
        layout.addLayout(profile_layout)

        return group

    def _build_runtime_group(self) -> QGroupBox:
        group = QGroupBox("Режим работы")
        layout = QFormLayout(group)

        self.conversation_mode_combo = QComboBox()
        self.conversation_mode_combo.addItem("Всё сразу", self.MODE_ALL)
        self.conversation_mode_combo.addItem("Только слушать", self.MODE_LISTEN_ONLY)
        self.conversation_mode_combo.addItem("Только говорить", self.MODE_SPEAK_ONLY)

        self.translation_direction_combo = QComboBox()
        self.translation_direction_combo.addItem("EN -> RU", "en_to_ru")
        self.translation_direction_combo.addItem("RU -> EN", "ru_to_en")

        self.translation_enabled_checkbox = QCheckBox("Включить перевод")
        self.translation_enabled_checkbox.setChecked(True)

        self.partial_emit_checkbox = QCheckBox("Разрешить ранний вывод по partial")
        self.partial_emit_checkbox.setChecked(True)

        self.mode_note_label = QLabel(
            "Режимы общения сейчас сохраняются в профиле. Полная логика "
            "`только слушать / только говорить` будет подключена отдельным шагом."
        )
        self.mode_note_label.setWordWrap(True)

        layout.addRow("Сценарий общения:", self.conversation_mode_combo)
        layout.addRow("Направление перевода:", self.translation_direction_combo)
        layout.addRow("", self.translation_enabled_checkbox)
        layout.addRow("", self.partial_emit_checkbox)
        layout.addRow("", self.mode_note_label)
        return group

    def _build_stt_group(self) -> QGroupBox:
        group = QGroupBox("Распознавание и сегментация")
        layout = QFormLayout(group)

        self.stt_backend_combo = QComboBox()
        self.stt_backend_combo.addItem("NVIDIA NIM", "nim")
        self.stt_backend_combo.addItem("NVIDIA Riva", "riva")
        self.stt_backend_combo.addItem("Silero VAD + faster-whisper", "faster_whisper")
        self.stt_backend_combo.addItem("whisper.cpp HTTP + Silero VAD", "whisper_cpp")
        self.stt_backend_combo.addItem("NVIDIA Canary AST", "canary_ast")
        self.commit_interval_spin = self._build_double_spin(0.1, 2.0, 0.05, 2)
        self.final_debounce_spin = self._build_double_spin(0.1, 2.0, 0.05, 2)
        self.partial_stability_spin = self._build_double_spin(0.1, 2.0, 0.05, 2)
        self.partial_min_words_spin = self._build_int_spin(1, 12, 1)
        self.noise_gate_threshold_spin = self._build_double_spin(0.001, 0.05, 0.001, 3)
        self.noise_gate_hangover_spin = self._build_double_spin(0.05, 1.5, 0.05, 2)
        self.stt_window_combo = QComboBox()
        self.stt_window_combo.addItem("0.5 сек", 0.5)
        self.stt_window_combo.addItem("1.0 сек", 1.0)
        self.stt_window_combo.addItem("2.0 сек", 2.0)

        layout.addRow("Backend STT:", self.stt_backend_combo)
        layout.addRow("Окно STT в UI:", self.stt_window_combo)
        layout.addRow("Интервал commit:", self.commit_interval_spin)
        layout.addRow("Ожидание final:", self.final_debounce_spin)
        layout.addRow("Стабильность partial:", self.partial_stability_spin)
        layout.addRow("Мин. слов в partial:", self.partial_min_words_spin)
        layout.addRow("Порог шумоподавления:", self.noise_gate_threshold_spin)
        layout.addRow("Хвост шумоподавления:", self.noise_gate_hangover_spin)
        return group

    def _build_tts_group(self) -> QGroupBox:
        group = QGroupBox("Синтез речи")
        layout = QFormLayout(group)

        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Русский Dmitri Medium", "ru_RU-dmitri-medium")
        self.voice_combo.addItem("English Lessac Medium", "en_US-lessac-medium")
        self.voice_combo.addItem("English Ryan Medium", "en_US-ryan-medium")

        self.max_queue_latency_spin = self._build_double_spin(0.1, 3.0, 0.05, 2)

        layout.addRow("Голос:", self.voice_combo)
        layout.addRow("Макс. задержка очереди TTS:", self.max_queue_latency_spin)
        return group

    def _build_audio_group(self) -> QGroupBox:
        group = QGroupBox("Аудио")
        layout = QFormLayout(group)

        self.samplerate_spin = self._build_int_spin(8000, 96000, 1000)
        self.blocksize_spin = self._build_int_spin(128, 4096, 64)
        self.channels_combo = QComboBox()
        self.channels_combo.addItem("1 (моно)", 1)
        self.channels_combo.addItem("2 (стерео)", 2)

        layout.addRow("Частота дискретизации:", self.samplerate_spin)
        layout.addRow("Размер блока:", self.blocksize_spin)
        layout.addRow("Каналы:", self.channels_combo)
        return group

    def _build_controls_group(self) -> QGroupBox:
        group = QGroupBox("Управление")
        layout = QVBoxLayout(group)

        mode_layout = QHBoxLayout()
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItem("Прямой проход", ProcessingMode.PASSTHROUGH.value)
        self.processing_mode_combo.addItem("Тишина", ProcessingMode.MUTE.value)
        self.processing_mode_combo.addItem("Тестовый тон", ProcessingMode.TEST_TONE.value)
        self.apply_mode_button = QPushButton("Применить режим")
        mode_layout.addWidget(QLabel("Режим обработки:"))
        mode_layout.addWidget(self.processing_mode_combo)
        mode_layout.addWidget(self.apply_mode_button)
        layout.addLayout(mode_layout)

        buttons_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Обновить устройства")
        self.save_config_button = QPushButton("Сохранить текущий конфиг")
        self.start_button = QPushButton("Запустить пайплайн")
        self.stop_button = QPushButton("Остановить пайплайн")
        self.stop_button.setEnabled(False)

        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.save_config_button)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        layout.addLayout(buttons_layout)
        return group

    def _bind_events(self) -> None:
        self.refresh_button.clicked.connect(self.load_devices)
        self.save_config_button.clicked.connect(self.save_current_config)
        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.apply_mode_button.clicked.connect(self.apply_mode)
        self.apply_preset_button.clicked.connect(self.apply_selected_preset)
        self.save_profile_button.clicked.connect(self.save_profile)
        self.load_profile_button.clicked.connect(self.load_selected_profile)

    def load_devices(self) -> None:
        try:
            ensure_translator_sink_exists()

            inputs = list_input_devices()
            outputs = list_output_devices()
            enrich_default_flags(inputs, outputs)

            self.input_devices = inputs
            self.output_devices = outputs

            self.input_combo.clear()
            self.output_combo.clear()

            default_input_index = 0
            default_output_index = 0

            for index, device in enumerate(self.input_devices):
                label = self._format_device_label(device)
                self.input_combo.addItem(label, device.name)
                if device.is_default:
                    default_input_index = index

            for index, device in enumerate(self.output_devices):
                label = self._format_device_label(device)
                self.output_combo.addItem(label, device.name)

                if device.name == TRANSLATOR_SINK_NAME:
                    default_output_index = index
                elif device.is_default and default_output_index == 0:
                    default_output_index = index

            if self.input_devices:
                self.input_combo.setCurrentIndex(default_input_index)

            if self.output_devices:
                self.output_combo.setCurrentIndex(default_output_index)

            self._emit_log(
                f"Loaded devices: inputs={len(self.input_devices)}, outputs={len(self.output_devices)}"
            )

        except Exception as error:
            QMessageBox.critical(self, "Ошибка устройств", str(error))

    def start_pipeline(self) -> None:
        selected_input_name = self.input_combo.currentData()
        selected_output_name = self.output_combo.currentData()

        if not selected_input_name:
            QMessageBox.warning(self, "Ошибка входа", "Выберите входной микрофон.")
            return

        if not selected_output_name:
            QMessageBox.warning(self, "Ошибка выхода", "Выберите выходное устройство.")
            return

        config = self._collect_config_from_ui()
        self._set_app_config(config)
        save_app_config(config, self.app_config_path)

        input_index = find_sounddevice_device_index_by_name(
            selected_input_name,
            min_input_channels=1,
        )
        output_index = find_sounddevice_device_index_by_name(
            selected_output_name,
            min_output_channels=1,
        )

        input_sd_name = None
        output_sd_name = None

        if input_index is not None:
            input_sd_name = sd.query_devices(input_index)["name"]

        if output_index is not None:
            output_sd_name = sd.query_devices(output_index)["name"]

        self._emit_log(f"Selected pactl input: {selected_input_name}")
        self._emit_log(f"Selected pactl output: {selected_output_name}")
        self._emit_log(f"Mapped sounddevice input index: {input_index}, name: {input_sd_name}")
        self._emit_log(f"Mapped sounddevice output index: {output_index}, name: {output_sd_name}")

        if input_index is None:
            QMessageBox.critical(
                self,
                "Ошибка аудиомаршрутизации",
                f"Не удалось сопоставить входное устройство:\n{selected_input_name}",
            )
            return

        if output_index is None:
            QMessageBox.critical(
                self,
                "Ошибка аудиомаршрутизации",
                f"Не удалось сопоставить выходное устройство:\n{selected_output_name}",
            )
            return

        try:
            stt_window_seconds = float(self.stt_window_combo.currentData())

            ensure_stt_runtime_for_app_config(self.app_config)

            self.engine.start(
                input_device_index=input_index,
                output_device_index=output_index,
                selected_pactl_input_name=selected_input_name,
                selected_pactl_output_name=selected_output_name,
                samplerate=self.app_config.audio.samplerate,
                channels=self.app_config.audio.channels,
                blocksize=self.app_config.audio.blocksize,
                stt_window_seconds=stt_window_seconds,
            )

            self._apply_selected_mode_to_engine()

            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self._emit_log(
                f"Pipeline started: input='{selected_input_name}' -> output='{selected_output_name}'"
            )

            if self.app_config.runtime.conversation_mode != self.MODE_ALL:
                self._emit_log(
                    f"Conversation mode saved: {self.app_config.runtime.conversation_mode} "
                    "(полная логика будет подключена позже)"
                )

        except Exception as error:
            QMessageBox.critical(self, "Ошибка запуска", str(error))

    def stop_pipeline(self) -> None:
        try:
            self.engine.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self._emit_log("Pipeline stopped")
        except Exception as error:
            QMessageBox.critical(self, "Ошибка остановки", str(error))

    def save_current_config(self) -> None:
        config = self._collect_config_from_ui()
        self._set_app_config(config)
        save_app_config(config, self.app_config_path)
        self._emit_log(f"Config saved: {self.app_config_path}")

    def apply_mode(self) -> None:
        if not self.engine.running:
            self._emit_log("Engine is not running yet. Mode will be applied after start.")
            return

        self._apply_selected_mode_to_engine()

    def apply_selected_preset(self) -> None:
        preset = self.preset_combo.currentData()
        config = self._collect_config_from_ui()

        if preset == self.PRESET_LOW_LATENCY:
            config = replace(
                config,
                stt=replace(
                    config.stt,
                    commit_interval_sec=0.35,
                    final_debounce_sec=0.45,
                    partial_emit_enabled=True,
                    partial_stability_sec=0.30,
                    partial_min_words=3,
                ),
                tts=replace(config.tts, max_queue_latency_sec=0.50),
            )
        elif preset == self.PRESET_HIGH_QUALITY:
            config = replace(
                config,
                stt=replace(
                    config.stt,
                    commit_interval_sec=0.55,
                    final_debounce_sec=0.75,
                    partial_emit_enabled=True,
                    partial_stability_sec=0.55,
                    partial_min_words=5,
                ),
                tts=replace(config.tts, max_queue_latency_sec=1.00),
            )
        else:
            config = replace(
                config,
                stt=replace(
                    config.stt,
                    commit_interval_sec=0.50,
                    final_debounce_sec=0.60,
                    partial_emit_enabled=True,
                    partial_stability_sec=0.45,
                    partial_min_words=4,
                ),
                tts=replace(config.tts, max_queue_latency_sec=0.75),
            )

        self._apply_config_to_ui(config)
        self._set_app_config(config)
        self._emit_log(f"Preset applied: {preset}")

    def save_profile(self) -> None:
        profile_name, accepted = QInputDialog.getText(
            self,
            "Сохранение профиля",
            "Имя профиля:",
        )
        if not accepted or not profile_name.strip():
            return

        safe_name = "".join(ch for ch in profile_name.strip() if ch.isalnum() or ch in ("-", "_", " "))
        safe_name = safe_name.replace(" ", "_")
        if not safe_name:
            QMessageBox.warning(self, "Ошибка профиля", "Имя профиля пустое или некорректное.")
            return

        profile_path = self.profiles_dir / f"{safe_name}.json"
        config = self._collect_config_from_ui()
        self._set_app_config(config)
        save_app_config(config, profile_path)
        self._reload_profiles(selected_path=profile_path)
        self._emit_log(f"Profile saved: {profile_path}")

    def load_selected_profile(self) -> None:
        profile_path_value = self.profile_combo.currentData()
        if not profile_path_value:
            QMessageBox.warning(self, "Ошибка профиля", "Выберите профиль для загрузки.")
            return

        profile_path = Path(profile_path_value)
        config = load_app_config(profile_path)
        self._apply_config_to_ui(config)
        self._set_app_config(config)
        self._emit_log(
            f"Profile loaded: {profile_path} "
            f"(backend={config.stt.backend}, direction={config.translation.direction})"
        )

    def _reload_profiles(self, selected_path: Path | None = None) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.profile_combo.clear()

        for profile_path in list_profile_paths():
            self.profile_combo.addItem(profile_path.stem, str(profile_path))

        if selected_path is not None:
            index = self.profile_combo.findData(str(selected_path))
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)

    def _set_app_config(self, config: AppConfig) -> None:
        self.app_config = config
        self.engine.app_config = config

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

    def _collect_config_from_ui(self) -> AppConfig:
        current_stt = self.app_config.stt
        current_primary = self.app_config.branches.primary
        return AppConfig(
            runtime=AppRuntimeConfig(
                conversation_mode=str(self.conversation_mode_combo.currentData()),
            ),
            audio=AudioConfig(
                samplerate=int(self.samplerate_spin.value()),
                channels=int(self.channels_combo.currentData()),
                blocksize=int(self.blocksize_spin.value()),
            ),
            stt=STTConfig(
                backend=str(self.stt_backend_combo.currentData()),
                base_url=current_stt.base_url,
                ws_url=current_stt.ws_url,
                riva_uri=current_stt.riva_uri,
                riva_use_ssl=current_stt.riva_use_ssl,
                riva_ssl_cert_path=current_stt.riva_ssl_cert_path,
                language=current_stt.language,
                sample_rate_hz=current_stt.sample_rate_hz,
                num_channels=current_stt.num_channels,
                timeout=current_stt.timeout,
                commit_interval_sec=float(self.commit_interval_spin.value()),
                enable_automatic_punctuation=current_stt.enable_automatic_punctuation,
                final_debounce_sec=float(self.final_debounce_spin.value()),
                partial_emit_enabled=bool(self.partial_emit_checkbox.isChecked()),
                partial_stability_sec=float(self.partial_stability_spin.value()),
                partial_min_words=int(self.partial_min_words_spin.value()),
                noise_gate_threshold=float(self.noise_gate_threshold_spin.value()),
                noise_gate_hangover_sec=float(self.noise_gate_hangover_spin.value()),
                canary_container_id=current_stt.canary_container_id,
                canary_tags_selector=current_stt.canary_tags_selector,
                canary_http_port=current_stt.canary_http_port,
                canary_grpc_port=current_stt.canary_grpc_port,
                canary_startup_timeout_sec=current_stt.canary_startup_timeout_sec,
                canary_poll_interval_sec=current_stt.canary_poll_interval_sec,
                canary_min_window_sec=current_stt.canary_min_window_sec,
                canary_finalize_silence_sec=current_stt.canary_finalize_silence_sec,
                silero_partial_interval_sec=current_stt.silero_partial_interval_sec,
                silero_min_window_sec=current_stt.silero_min_window_sec,
                silero_max_window_sec=current_stt.silero_max_window_sec,
                silero_min_silence_ms=current_stt.silero_min_silence_ms,
                silero_speech_pad_ms=current_stt.silero_speech_pad_ms,
                silero_preroll_sec=current_stt.silero_preroll_sec,
                silero_speech_threshold=current_stt.silero_speech_threshold,
                whisper_model_size=current_stt.whisper_model_size,
                whisper_compute_type=current_stt.whisper_compute_type,
                whisper_beam_size=current_stt.whisper_beam_size,
                whisper_best_of=current_stt.whisper_best_of,
                whisper_patience=current_stt.whisper_patience,
            ),
            branches=BranchesConfig(
                primary=TranslationBranchConfig(
                    branch_id="primary",
                    label=self._resolve_branch_label(str(self.translation_direction_combo.currentData())),
                    enabled=bool(self.translation_enabled_checkbox.isChecked()),
                    stt_language=self._resolve_stt_language(str(self.translation_direction_combo.currentData())),
                    translation_direction=str(self.translation_direction_combo.currentData()),
                    tts_voice_name=str(self.voice_combo.currentData()),
                    nim_container_id=self._resolve_nim_container_id(str(self.translation_direction_combo.currentData())),
                    nim_tags_selector=self._resolve_nim_tags_selector(str(self.translation_direction_combo.currentData())),
                    nim_startup_timeout_sec=current_primary.nim_startup_timeout_sec,
                ),
                secondary=self.app_config.branches.secondary,
            ),
            translation=TranslationRuntimeConfig(
                direction=str(self.translation_direction_combo.currentData()),
                enabled=bool(self.translation_enabled_checkbox.isChecked()),
            ),
            tts=TTSRuntimeConfig(
                voice_name=str(self.voice_combo.currentData()),
                data_dir=self.app_config.tts.data_dir,
                use_cuda=self.app_config.tts.use_cuda,
                max_queue_latency_sec=float(self.max_queue_latency_spin.value()),
            ),
        )

    def _apply_config_to_ui(self, config: AppConfig) -> None:
        primary_branch = get_primary_branch_config(config)
        self._set_combo_data(self.conversation_mode_combo, config.runtime.conversation_mode)
        self._set_combo_data(self.translation_direction_combo, primary_branch.translation_direction)
        self.translation_enabled_checkbox.setChecked(primary_branch.enabled)
        self.partial_emit_checkbox.setChecked(config.stt.partial_emit_enabled)
        self._set_combo_data(self.stt_backend_combo, config.stt.backend)

        self.commit_interval_spin.setValue(config.stt.commit_interval_sec)
        self.final_debounce_spin.setValue(config.stt.final_debounce_sec)
        self.partial_stability_spin.setValue(config.stt.partial_stability_sec)
        self.partial_min_words_spin.setValue(config.stt.partial_min_words)
        self.noise_gate_threshold_spin.setValue(config.stt.noise_gate_threshold)
        self.noise_gate_hangover_spin.setValue(config.stt.noise_gate_hangover_sec)

        self._set_combo_data(self.voice_combo, primary_branch.tts_voice_name)
        self.max_queue_latency_spin.setValue(config.tts.max_queue_latency_sec)

        self.samplerate_spin.setValue(config.audio.samplerate)
        self.blocksize_spin.setValue(config.audio.blocksize)
        self._set_combo_data(self.channels_combo, config.audio.channels)

        self._set_combo_data(self.stt_window_combo, 1.0)
        self._set_combo_data(self.processing_mode_combo, ProcessingMode.PASSTHROUGH.value)

    def _apply_selected_mode_to_engine(self) -> None:
        selected_mode_value = self.processing_mode_combo.currentData()

        try:
            mode = ProcessingMode(selected_mode_value)
        except Exception:
            QMessageBox.warning(self, "Ошибка режима", "Выбран некорректный режим обработки.")
            return

        self.engine.set_mode(mode)

    def _append_log_to_ui(self, message: str) -> None:
        self.log_output.append(message)

    def _append_error_to_ui(self, message: str) -> None:
        self.log_output.append(f"ОШИБКА: {message}")

    def _handle_engine_log(self, message: str) -> None:
        self._emit_log(message)

    def _handle_engine_error(self, message: str) -> None:
        self._emit_error(message)

    def _emit_log(self, message: str) -> None:
        self.log_signal.emit(message)
        self.file_logger.info(message)

    def _emit_error(self, message: str) -> None:
        self.error_signal.emit(message)
        self.file_logger.error(message)

    @staticmethod
    def _build_double_spin(minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        return widget

    @staticmethod
    def _build_int_spin(minimum: int, maximum: int, step: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        return widget

    @staticmethod
    def _set_combo_data(combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _resolve_stt_language(direction: str) -> str:
        if direction == "ru_to_en":
            return "ru-RU"
        return "en-US"

    @staticmethod
    def _resolve_branch_label(direction: str) -> str:
        if direction == "ru_to_en":
            return "RU -> EN"
        return "EN -> RU"

    @staticmethod
    def _resolve_nim_container_id(direction: str) -> str:
        if direction == "ru_to_en":
            return "parakeet-1-1b-rnnt-multilingual"
        return "parakeet-1-1b-ctc-en-us"

    @staticmethod
    def _resolve_nim_tags_selector(direction: str) -> str:
        if direction == "ru_to_en":
            return "mode=str"
        return "name=parakeet-1-1b-ctc-en-us,mode=str,diarizer=disabled,vad=default"

    @staticmethod
    def _format_device_label(device: AudioDevice) -> str:
        default_suffix = " [по умолчанию]" if device.is_default else ""
        return f"{device.description} ({device.name}){default_suffix}"
