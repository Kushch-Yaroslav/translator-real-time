from __future__ import annotations

import io
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import requests
from scipy.signal import resample
from silero_vad import VADIterator, load_silero_vad


@dataclass
class WhisperCppRealtimeSTTConfig:
    base_url: str = "http://127.0.0.1:8178"
    language: str = "en"
    sample_rate_hz: int = 16000
    partial_interval_sec: float = 0.35
    min_window_sec: float = 0.9
    max_window_sec: float = 8.0
    min_silence_duration_ms: int = 180
    speech_pad_ms: int = 80
    speech_threshold: float = 0.55
    timeout: float = 10.0
    prompt: str = ""
    on_log: Optional[Callable[[str], None]] = None


class WhisperCppRealtimeSTTService:
    _VAD_FRAME_SAMPLES = 512

    def __init__(self, config: Optional[WhisperCppRealtimeSTTConfig] = None):
        self.config = config or WhisperCppRealtimeSTTConfig()
        self._vad_model = load_silero_vad(onnx=False)
        self._vad_iterator = VADIterator(
            self._vad_model,
            threshold=self.config.speech_threshold,
            sampling_rate=self.config.sample_rate_hz,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )

        self._partial_callback: Optional[Callable[[str], None]] = None
        self._final_callback: Optional[Callable[[str], None]] = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._session = requests.Session()

        self._speech_active = False
        self._pending_audio = np.zeros((0,), dtype=np.float32)
        self._current_utterance_chunks: list[np.ndarray] = []
        self._current_utterance_frames = 0
        self._last_partial_job_at = 0.0
        self._partial_job_audio: Optional[np.ndarray] = None
        self._final_job_audios: list[np.ndarray] = []
        self._last_partial_text = ""
        self._last_final_text = ""

    def start(
        self,
        partial_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._partial_callback = partial_callback
        self._final_callback = final_callback
        self._stop_event.clear()
        self.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="whispercpp-stt-worker",
        )
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        self._worker = None
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self.clear()

    def restart(self) -> None:
        self.stop()
        time.sleep(0.15)
        self.start(
            partial_callback=self._partial_callback,
            final_callback=self._final_callback,
        )

    def send_audio_chunk(self, audio: np.ndarray, samplerate: int) -> None:
        mono = self._prepare_audio(audio)
        mono = self._resample_if_needed(mono, samplerate, self.config.sample_rate_hz)
        if mono.size == 0:
            return

        with self._lock:
            self._pending_audio = np.concatenate([self._pending_audio, mono]).astype(
                np.float32,
                copy=False,
            )

            while self._pending_audio.shape[0] >= self._VAD_FRAME_SAMPLES:
                frame = self._pending_audio[:self._VAD_FRAME_SAMPLES]
                self._pending_audio = self._pending_audio[self._VAD_FRAME_SAMPLES:]
                self._process_vad_frame(frame)

            if self._speech_active:
                self._current_utterance_chunks.append(mono.astype(np.float32, copy=True))
                self._current_utterance_frames += int(mono.shape[0])
                max_frames = int(self.config.max_window_sec * self.config.sample_rate_hz)
                while self._current_utterance_frames > max_frames and self._current_utterance_chunks:
                    removed = self._current_utterance_chunks.pop(0)
                    self._current_utterance_frames -= int(removed.shape[0])
                self._maybe_schedule_partial_job()

    def commit(self) -> None:
        with self._lock:
            self._maybe_schedule_partial_job(force=True)

    def clear(self) -> None:
        with self._lock:
            self._vad_iterator.reset_states()
            self._speech_active = False
            self._pending_audio = np.zeros((0,), dtype=np.float32)
            self._current_utterance_chunks = []
            self._current_utterance_frames = 0
            self._last_partial_job_at = 0.0
            self._partial_job_audio = None
            self._final_job_audios = []
            self._last_partial_text = ""
            self._last_final_text = ""

    def send_done(self) -> None:
        with self._lock:
            utterance = self._build_current_utterance_audio()
            if utterance.size > 0:
                self._final_job_audios.append(utterance)
            self._speech_active = False
            self._current_utterance_chunks = []
            self._current_utterance_frames = 0

    def get_last_partial_text(self) -> str:
        return self._last_partial_text

    def get_last_final_text(self) -> str:
        return self._last_final_text

    def _process_vad_frame(self, frame: np.ndarray) -> None:
        event = self._vad_iterator(frame)
        if event is None:
            return

        if "start" in event and not self._speech_active:
            self._speech_active = True
            self._current_utterance_chunks = []
            self._current_utterance_frames = 0
            self._last_partial_job_at = 0.0
            self._log("whisper.cpp VAD start")
            return

        if "end" in event and self._speech_active:
            utterance = self._build_current_utterance_audio()
            if utterance.size > 0:
                self._final_job_audios.append(utterance)
                self._log(
                    f"whisper.cpp VAD end | duration={utterance.shape[0] / self.config.sample_rate_hz:.2f}s"
                )
            self._speech_active = False
            self._current_utterance_chunks = []
            self._current_utterance_frames = 0

    def _maybe_schedule_partial_job(self, force: bool = False) -> None:
        utterance = self._build_current_utterance_audio()
        if utterance.size == 0:
            return

        duration_sec = utterance.shape[0] / float(self.config.sample_rate_hz)
        if duration_sec < self.config.min_window_sec:
            return

        now = time.monotonic()
        if not force and (now - self._last_partial_job_at) < self.config.partial_interval_sec:
            return

        self._partial_job_audio = utterance
        self._last_partial_job_at = now

    def _build_current_utterance_audio(self) -> np.ndarray:
        if not self._current_utterance_chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(self._current_utterance_chunks, axis=0).astype(np.float32, copy=False)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            job_kind = ""
            audio = np.zeros((0,), dtype=np.float32)

            with self._lock:
                if self._final_job_audios:
                    audio = self._final_job_audios.pop(0)
                    job_kind = "final"
                elif self._partial_job_audio is not None:
                    audio = self._partial_job_audio
                    self._partial_job_audio = None
                    job_kind = "partial"

            if not job_kind:
                time.sleep(0.03)
                continue

            try:
                text = self._transcribe(audio)
            except Exception as error:
                self._log(f"whisper.cpp STT error: {error}")
                continue

            if not text:
                continue

            if job_kind == "partial":
                if self._normalize_compare_text(text) == self._normalize_compare_text(self._last_partial_text):
                    continue
                self._last_partial_text = text
                self._log(f"whisper.cpp partial: {text}")
                if self._partial_callback is not None:
                    self._partial_callback(text)
                continue

            self._last_final_text = text
            self._last_partial_text = ""
            self._log(f"whisper.cpp final: {text}")
            if self._final_callback is not None:
                self._final_callback(text)

    def _transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = self._to_wav_bytes(audio, self.config.sample_rate_hz)
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
            "language": (None, self.config.language),
            "response_format": (None, "json"),
            "temperature": (None, "0.0"),
            "temperature_inc": (None, "0.0"),
            "no_timestamps": (None, "true"),
            "suppress_nst": (None, "true"),
        }
        if self.config.prompt:
            files["prompt"] = (None, self.config.prompt)

        response = self._session.post(
            f"{self.config.base_url.rstrip('/')}/inference",
            files=files,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return self._normalize_text(payload.get("text", "") or "")

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sample_rate_hz: int) -> bytes:
        pcm16 = WhisperCppRealtimeSTTService._float32_to_pcm16(audio)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        return audio.astype(np.float32, copy=False).flatten()

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        if from_sr == to_sr or audio.size == 0:
            return audio
        target_samples = int(round(audio.shape[0] * float(to_sr) / float(from_sr)))
        if target_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        return resample(audio, target_samples).astype(np.float32, copy=False)

    @staticmethod
    def _float32_to_pcm16(audio: np.ndarray) -> np.ndarray:
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").replace("\n", " ").split()).strip()

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        return WhisperCppRealtimeSTTService._normalize_text(text).casefold()

    def _log(self, message: str) -> None:
        if self.config.on_log:
            self.config.on_log(message)
