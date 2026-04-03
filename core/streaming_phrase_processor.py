from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional

import numpy as np


@dataclass
class StreamingPhraseProcessorConfig:
    samplerate: int = 48000
    channels: int = 1
    blocksize: int = 1024

    # Размер окна, которое будем отправлять в STT
    window_seconds: float = 1.2

    # Как часто разрешаем новый запуск STT
    trigger_interval_seconds: float = 0.4

    # Минимальная "озвученность" окна, чтобы не гонять тишину
    min_voiced_ratio: float = 0.35

    # Порог RMS для определения, что chunk не тишина
    speech_threshold: float = 0.008

    # Сколько тишины подряд считаем окончанием активности
    silence_reset_seconds: float = 1.0


class StreamingPhraseProcessor:
    def __init__(self, config: StreamingPhraseProcessorConfig):
        self.config = config

        self.chunk_duration = self.config.blocksize / float(self.config.samplerate)

        self.window_chunks = max(
            1, int(round(self.config.window_seconds / self.chunk_duration))
        )
        self.trigger_interval_chunks = max(
            1, int(round(self.config.trigger_interval_seconds / self.chunk_duration))
        )
        self.silence_reset_chunks = max(
            1, int(round(self.config.silence_reset_seconds / self.chunk_duration))
        )

        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window_chunks)

        self._chunks_since_last_trigger = 0
        self._consecutive_silence_chunks = 0
        self._had_recent_speech = False

    def reset(self) -> None:
        self.buffer.clear()
        self._chunks_since_last_trigger = 0
        self._consecutive_silence_chunks = 0
        self._had_recent_speech = False

    def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        chunk = self._ensure_2d_float32(chunk)
        self.buffer.append(chunk)
        self._chunks_since_last_trigger += 1

        rms = self._calculate_rms(chunk)
        is_voiced = rms >= self.config.speech_threshold

        if is_voiced:
            self._consecutive_silence_chunks = 0
            self._had_recent_speech = True
        else:
            self._consecutive_silence_chunks += 1

        if self._consecutive_silence_chunks >= self.silence_reset_chunks:
            self._had_recent_speech = False

        if len(self.buffer) < self.window_chunks:
            return None

        if not self._had_recent_speech:
            return None

        if self._chunks_since_last_trigger < self.trigger_interval_chunks:
            return None

        voiced_ratio = self._get_voiced_ratio()
        if voiced_ratio < self.config.min_voiced_ratio:
            return None

        self._chunks_since_last_trigger = 0
        return self._build_window()

    def _build_window(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros((0, self.config.channels), dtype=np.float32)

        return np.vstack(list(self.buffer)).astype(np.float32, copy=False)

    def _get_voiced_ratio(self) -> float:
        if not self.buffer:
            return 0.0

        voiced = 0
        total = 0

        for chunk in self.buffer:
            total += 1
            if self._calculate_rms(chunk) >= self.config.speech_threshold:
                voiced += 1

        if total == 0:
            return 0.0

        return voiced / float(total)

    @staticmethod
    def _calculate_rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    @staticmethod
    def _ensure_2d_float32(chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.astype(np.float32, copy=False)

        if chunk.ndim == 1:
            return chunk.reshape(-1, 1)

        return chunk