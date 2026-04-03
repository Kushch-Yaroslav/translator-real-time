from __future__ import annotations

from typing import List
import sounddevice as sd

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QMessageBox,
)

from core.audio_engine import AudioEngine
from core.app_config import get_default_config_path, load_app_config
from core.audio_service import (
    AudioDevice,
    list_input_devices,
    list_output_devices,
    enrich_default_flags,
    ensure_translator_sink_exists,
    TRANSLATOR_SINK_NAME,
)
from core.audio_utils import find_sounddevice_device_index_by_name
from core.chunk_processor import ProcessingMode


class MainWindow(QWidget):
    log_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Translator Audio App")
        self.resize(860, 620)

        self.app_config_path = get_default_config_path()
        self.app_config = load_app_config(self.app_config_path)
        self.engine = AudioEngine(self.app_config)

        self.input_devices: List[AudioDevice] = []
        self.output_devices: List[AudioDevice] = []

        self._build_ui()
        self._bind_events()

        self.log_signal.connect(self._append_log_to_ui)
        self.error_signal.connect(self._append_error_to_ui)

        # ВАЖНО: теперь engine пишет не прямо в QTextEdit,
        # а через thread-safe Qt signals
        self.engine.on_log = self.log_signal.emit
        self.engine.on_error = self.error_signal.emit

        self.load_devices()

    def _build_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Pipeline сейчас: real mic -> Python AudioEngine -> TranslatorMic\n"
            "В Telegram выбирай устройство записи: Monitor of TranslatorMic\n"
            f"Config: {self.app_config_path}"
        )
        self.main_layout.addWidget(self.info_label)

        input_layout = QHBoxLayout()
        input_label = QLabel("Input microphone:")
        self.input_combo = QComboBox()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_combo)
        self.main_layout.addLayout(input_layout)

        output_layout = QHBoxLayout()
        output_label = QLabel("Output sink for app:")
        self.output_combo = QComboBox()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_combo)
        self.main_layout.addLayout(output_layout)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Processing mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Passthrough", ProcessingMode.PASSTHROUGH.value)
        self.mode_combo.addItem("Mute", ProcessingMode.MUTE.value)
        self.mode_combo.addItem("Test tone", ProcessingMode.TEST_TONE.value)

        self.apply_mode_button = QPushButton("Apply mode")

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(self.apply_mode_button)
        self.main_layout.addLayout(mode_layout)

        stt_layout = QHBoxLayout()
        stt_label = QLabel("STT window:")
        self.stt_window_combo = QComboBox()
        self.stt_window_combo.addItem("0.5 sec", 0.5)
        self.stt_window_combo.addItem("1.0 sec", 1.0)
        self.stt_window_combo.addItem("2.0 sec", 2.0)
        self.stt_window_combo.setCurrentIndex(1)

        stt_layout.addWidget(stt_label)
        stt_layout.addWidget(self.stt_window_combo)
        self.main_layout.addLayout(stt_layout)

        buttons_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh devices")
        self.start_button = QPushButton("Start pipeline")
        self.stop_button = QPushButton("Stop pipeline")
        self.stop_button.setEnabled(False)

        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)

        self.main_layout.addLayout(buttons_layout)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.main_layout.addWidget(self.log_output)

    def _bind_events(self) -> None:
        self.refresh_button.clicked.connect(self.load_devices)
        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.apply_mode_button.clicked.connect(self.apply_mode)

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

            self.log_signal.emit(
                f"Loaded devices: inputs={len(self.input_devices)}, outputs={len(self.output_devices)}"
            )

        except Exception as error:
            QMessageBox.critical(self, "Device error", str(error))

    def start_pipeline(self) -> None:
        selected_input_name = self.input_combo.currentData()
        selected_output_name = self.output_combo.currentData()

        if not selected_input_name:
            QMessageBox.warning(self, "Input error", "Select input device")
            return

        if not selected_output_name:
            QMessageBox.warning(self, "Output error", "Select output device")
            return

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

        self.log_signal.emit(f"Selected pactl input: {selected_input_name}")
        self.log_signal.emit(f"Selected pactl output: {selected_output_name}")
        self.log_signal.emit(f"Mapped sounddevice input index: {input_index}, name: {input_sd_name}")
        self.log_signal.emit(f"Mapped sounddevice output index: {output_index}, name: {output_sd_name}")

        if input_index is None:
            QMessageBox.critical(
                self,
                "Audio mapping error",
                f"Could not map pactl input device to sounddevice:\n{selected_input_name}",
            )
            return

        if output_index is None:
            QMessageBox.critical(
                self,
                "Audio mapping error",
                f"Could not map pactl output device to sounddevice:\n{selected_output_name}",
            )
            return

        try:
            stt_window_seconds = float(self.stt_window_combo.currentData())

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

            self.log_signal.emit(
                f"Pipeline started: input='{selected_input_name}' -> output='{selected_output_name}'"
            )

        except Exception as error:
            QMessageBox.critical(self, "Start error", str(error))

    def stop_pipeline(self) -> None:
        try:
            self.engine.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.log_signal.emit("Pipeline stopped")
        except Exception as error:
            QMessageBox.critical(self, "Stop error", str(error))

    def apply_mode(self) -> None:
        if not self.engine.running:
            self.log_signal.emit("Engine is not running yet. Mode will be applied after start.")
            return

        self._apply_selected_mode_to_engine()

    def _apply_selected_mode_to_engine(self) -> None:
        selected_mode_value = self.mode_combo.currentData()

        try:
            mode = ProcessingMode(selected_mode_value)
        except Exception:
            QMessageBox.warning(self, "Mode error", "Invalid processing mode selected")
            return

        self.engine.set_mode(mode)

    def _append_log_to_ui(self, message: str) -> None:
        self.log_output.append(message)

    def _append_error_to_ui(self, message: str) -> None:
        self.log_output.append(f"ERROR: {message}")

    @staticmethod
    def _format_device_label(device: AudioDevice) -> str:
        default_suffix = " [default]" if device.is_default else ""
        return f"{device.description} ({device.name}){default_suffix}"
