from __future__ import annotations

import json
import sys
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.benchmark.paths import (
    BENCHMARK_AUDIO_PATH,
    BENCHMARK_RUNS_DIR,
    BENCHMARK_SOURCE_DIR,
    CATEGORY_MAPPING,
    ensure_benchmark_dirs,
    get_category_source_dir,
    get_test_paths,
)


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
            "Окно для записи и запуска RU=>EN бенчмарков.\n"
            "Можно записывать новые тесты в категории или запускать существующие."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Form for recording new test
        form_group = QVBoxLayout()
        form_layout = QFormLayout()
        
        self.category_combo = QComboBox()
        self.category_combo.setEditable(False)
        for cat_id, cat_name in CATEGORY_MAPPING.items():
            self.category_combo.addItem(cat_name, cat_id)
        # self.category_combo.addItem("Custom", "custom") # Removed per simplified UX

        self.test_name_input = QLineEdit()
        self.test_name_input.setReadOnly(True)
        self.test_name_input.setStyleSheet("background-color: #f0f0f0;")
        
        self._update_test_name_from_category()
        
        self.expected_text_input = QTextEdit()
        self.expected_text_input.setPlaceholderText("Введите эталонный текст здесь...")
        self.expected_text_input.setMaximumHeight(80)

        form_layout.addRow("Категория:", self.category_combo)
        form_layout.addRow("Имя теста:", self.test_name_input)
        form_layout.addRow("Ожидаемый текст:", self.expected_text_input)
        form_group.addLayout(form_layout)
        layout.addLayout(form_group)

        # Recording controls
        record_controls = QHBoxLayout()
        self.record_button = QPushButton("Запись")
        self.stop_button = QPushButton("Стоп")
        self.stop_button.setEnabled(False)
        record_controls.addWidget(self.record_button)
        record_controls.addWidget(self.stop_button)
        record_controls.addStretch(1)
        layout.addLayout(record_controls)

        # Status and info
        self.status_label = QLabel("Статус: idle")
        layout.addWidget(self.status_label)

        self.timer_label = QLabel("Таймер: 00:00.0")
        layout.addWidget(self.timer_label)

        # Run controls
        run_group = QVBoxLayout()
        run_group.addWidget(QLabel("Запуск бенчмарков:"))
        
        run_buttons_layout = QHBoxLayout()
        self.run_selected_button = QPushButton("Запустить текущий")
        self.run_category_button = QPushButton("Запустить категорию")
        self.run_all_button = QPushButton("Запустить все")
        
        run_buttons_layout.addWidget(self.run_selected_button)
        run_buttons_layout.addWidget(self.run_category_button)
        run_buttons_layout.addWidget(self.run_all_button)
        run_group.addLayout(run_buttons_layout)
        layout.addLayout(run_group)

        # File info
        self.file_label = QLabel("Последний файл: —")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        
        self.summary_label = QLabel("Последний прогон: —")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.comparison_path_label = QLabel("Последнее сравнение: —")
        self.comparison_path_label.setWordWrap(True)
        layout.addWidget(self.comparison_path_label)

        self.comparison_status_label = QLabel("Сравнение: нет предыдущего прогона")
        self.comparison_status_label.setWordWrap(True)
        self.comparison_status_label.setStyleSheet("font-weight: bold; color: #555;")
        layout.addWidget(self.comparison_status_label)

        layout.addStretch(1)

    def _bind_events(self) -> None:
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.run_selected_button.clicked.connect(self.run_selected_test)
        self.run_category_button.clicked.connect(self.run_category_tests)
        self.run_all_button.clicked.connect(self.run_all_benchmarks)
        self.category_combo.currentIndexChanged.connect(self._update_test_name_from_category)

    def _update_test_name_from_category(self) -> None:
        category = self.category_combo.currentData()
        if category:
            self.test_name_input.setText(category)

    def _refresh_labels(self) -> None:
        # We don't have a single current file anymore in the same way, but we can show the last recorded
        pass

    def _get_target_paths(self) -> tuple[Path, Path]:
        category = self.category_combo.currentData() or "custom"
        test_name = self.test_name_input.text().strip()
        if not test_name:
            test_name = f"rec_{int(time.time())}"
        
        return get_test_paths(category, test_name)

    def run_selected_test(self) -> None:
        category = self.category_combo.currentData()
        test_name = self.test_name_input.text().strip()
        
        if not test_name:
            # Fallback to legacy
            self._run_benchmark_cmd([])
        else:
            audio_path, _ = get_test_paths(category, test_name)
            if not audio_path.exists():
                QMessageBox.warning(self, "Файл не найден", f"Файл {audio_path} не существует.")
                return
            self._run_benchmark_cmd([str(audio_path)])

    def run_category_tests(self) -> None:
        category = self.category_combo.currentData()
        if not category:
            return
        self._run_benchmark_cmd(["--category", category])

    def run_all_benchmarks(self) -> None:
        self._run_benchmark_cmd(["--all"])

    def _run_benchmark_cmd(self, args: list[str]) -> None:
        self.status_label.setText("Статус: запуск бенчмарка...")
        self.repaint()
        
        cmd = [str(Path(sys.executable)), "tests/run_ru_to_en_offline_benchmark.py"] + args
        
        try:
            # We run it in a separate thread to not freeze UI, but since it's a tool, we might want to just see result
            # For simplicity of this task, let's run it and then update UI
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.status_label.setText("Статус: прогон завершён успешно")
                # Try to find the last run dir
                runs = sorted(BENCHMARK_RUNS_DIR.glob("run_*"), reverse=True)
                if runs:
                    last_run = runs[0]
                    self.summary_label.setText(f"Последний прогон: {last_run}")
                    
                    # Update comparison info
                    comp_file = last_run / "comparison.json"
                    if comp_file.exists():
                        self.comparison_path_label.setText(f"Последнее сравнение: {comp_file}")
                        try:
                            comp_data = json.loads(comp_file.read_text(encoding="utf-8"))
                            avg_diff = comp_data.get("average_diff", {})
                            
                            status_parts = []
                            for metric in ["realtime_factor", "duplicate_queued_count", "long_translation_gaps_count"]:
                                if metric in avg_diff:
                                    status = avg_diff[metric].get("status", "unknown")
                                    status_parts.append(f"{metric} {status}")
                            
                            if status_parts:
                                self.comparison_status_label.setText(f"Сравнение: {', '.join(status_parts)}")
                            else:
                                self.comparison_status_label.setText("Сравнение: данные получены")
                        except Exception as e:
                            self.comparison_status_label.setText(f"Сравнение: ошибка чтения ({e})")
                    else:
                        summary_file = last_run / "summary.json"
                        if summary_file.exists():
                            try:
                                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                                if summary_data.get("comparison_path"):
                                    self.comparison_path_label.setText(f"Последнее сравнение: {summary_data['comparison_path']}")
                                    # We could also try to load it from there
                                else:
                                    self.comparison_path_label.setText("Последнее сравнение: —")
                                    self.comparison_status_label.setText("Сравнение: нет предыдущего прогона")
                            except Exception:
                                pass
            else:
                self.status_label.setText(f"Статус: ошибка при прогоне (код {process.returncode})")
                print(stderr)
                QMessageBox.critical(self, "Ошибка бенчмарка", f"Ошибка при выполнении:\n{stderr}")
        except Exception as e:
            self.status_label.setText(f"Статус: системная ошибка")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить бенчмарк:\n{e}")

    def _save_recording(self, audio: np.ndarray, samplerate: int) -> None:
        ensure_benchmark_dirs()
        audio_path, expected_path = self._get_target_paths()
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp_path = audio_path.with_suffix(".tmp.wav")
        pcm16 = np.clip(audio.reshape(-1), -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)

        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm16.tobytes())

        if audio_path.exists():
            audio_path.unlink()
        tmp_path.replace(audio_path)
        
        # Save expected text
        expected_text = self.expected_text_input.toPlainText().strip()
        if expected_text:
            expected_path.write_text(expected_text, encoding="utf-8")
        
        self.file_label.setText(f"Последний файл: {audio_path}")

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
