from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.benchmark.paths import BENCHMARK_AUDIO_PATH, ensure_benchmark_dirs


class BenchmarkRecorderWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("RU=>EN Audio Benchmark Recorder")
        self.resize(560, 220)

        self._recording_stream = None
        self._recording_lock = threading.Lock()
        self._recording_chunks: list[np.ndarray] = []
        self._recording_started_at = 0.0
        self._recording_samplerate = 48000
        self._recording_channels = 1
        self._recording_blocksize = 1024
        self._recording_error: str | None = None

        ensure_benchmark_dirs()
        self._build_ui()
        self._bind_events()
        self._refresh_labels()

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_timer_label)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.info_label = QLabel(
            "Окно записывает один эталонный аудиофайл для внутреннего RU=>EN benchmark.\n"
            "Каждая новая запись заменяет предыдущий файл."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        controls_layout = QHBoxLayout()
        self.record_button = QPushButton("Запись")
        self.stop_button = QPushButton("Стоп")
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch(1)
        layout.addLayout(controls_layout)

        self.status_label = QLabel("Статус: idle")
        layout.addWidget(self.status_label)

        self.timer_label = QLabel("Таймер: 00:00.0")
        layout.addWidget(self.timer_label)

        self.device_label = QLabel("Устройство ввода: —")
        self.device_label.setWordWrap(True)
        layout.addWidget(self.device_label)

        self.file_label = QLabel("Актуальный файл: —")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        self.duration_label = QLabel("Длина файла: —")
        layout.addWidget(self.duration_label)

        layout.addStretch(1)

    def _bind_events(self) -> None:
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

    def _refresh_labels(self) -> None:
        self.file_label.setText(f"Актуальный файл: {BENCHMARK_AUDIO_PATH}")
        self.duration_label.setText(
            f"Длина файла: {self._format_seconds(self._probe_duration_sec(BENCHMARK_AUDIO_PATH))}"
            if BENCHMARK_AUDIO_PATH.exists()
            else "Длина файла: файл ещё не записан"
        )
        self.device_label.setText(f"Устройство ввода: {self._get_default_input_device_label()}")

    def _get_default_input_device_label(self) -> str:
        try:
            default_input, _default_output = sd.default.device
            if default_input is None or int(default_input) < 0:
                return "default input не найден"
            device = sd.query_devices(int(default_input))
            return str(device.get("name", "unknown"))
        except Exception as error:
            return f"ошибка определения устройства: {error}"

    def _input_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._recording_error = str(status)

        with self._recording_lock:
            self._recording_chunks.append(np.copy(indata))

    def start_recording(self) -> None:
        if self._recording_stream is not None:
            return

        ensure_benchmark_dirs()
        self._recording_error = None
        self._recording_chunks = []
        self._recording_started_at = time.monotonic()

        try:
            self._recording_stream = sd.InputStream(
                samplerate=self._recording_samplerate,
                channels=self._recording_channels,
                dtype="float32",
                blocksize=self._recording_blocksize,
                callback=self._input_callback,
            )
            self._recording_stream.start()
        except Exception as error:
            self._recording_stream = None
            QMessageBox.critical(self, "Ошибка записи", f"Не удалось начать запись:\n{error}")
            return

        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Статус: запись идёт")
        self.timer_label.setText("Таймер: 00:00.0")
        self._timer.start()

    def stop_recording(self) -> None:
        stream = self._recording_stream
        if stream is None:
            return

        self._recording_stream = None
        self._timer.stop()

        try:
            stream.stop()
        finally:
            stream.close()

        with self._recording_lock:
            chunks = [chunk for chunk in self._recording_chunks if chunk.size > 0]
            self._recording_chunks = []

        if not chunks:
            self.record_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Статус: запись не сохранилась")
            self._refresh_labels()
            QMessageBox.warning(self, "Пустая запись", "Аудио не было записано.")
            return

        audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        self._save_recording(audio, self._recording_samplerate)

        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Статус: файл обновлён")
        self._update_timer_label()
        self._refresh_labels()

        if self._recording_error:
            QMessageBox.warning(
                self,
                "Запись завершена с предупреждением",
                f"Файл сохранён, но во время записи было сообщение:\n{self._recording_error}",
            )

    def closeEvent(self, event) -> None:
        if self._recording_stream is not None:
            self.stop_recording()
        super().closeEvent(event)

    def _save_recording(self, audio: np.ndarray, samplerate: int) -> None:
        ensure_benchmark_dirs()
        tmp_path = BENCHMARK_AUDIO_PATH.with_suffix(".tmp.wav")
        pcm16 = np.clip(audio.reshape(-1), -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)

        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm16.tobytes())

        if BENCHMARK_AUDIO_PATH.exists():
            BENCHMARK_AUDIO_PATH.unlink()
        tmp_path.replace(BENCHMARK_AUDIO_PATH)

    def _update_timer_label(self) -> None:
        if self._recording_started_at <= 0.0:
            self.timer_label.setText("Таймер: 00:00.0")
            return

        elapsed = time.monotonic() - self._recording_started_at
        self.timer_label.setText(f"Таймер: {self._format_seconds(elapsed)}")

    @staticmethod
    def _probe_duration_sec(path: Path) -> float:
        if not path.exists():
            return 0.0
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            samplerate = wf.getframerate()
            if samplerate <= 0:
                return 0.0
            return frames / float(samplerate)

    @staticmethod
    def _format_seconds(value: float) -> str:
        total = max(0.0, float(value))
        minutes = int(total // 60)
        seconds = total - minutes * 60
        return f"{minutes:02d}:{seconds:04.1f}"
