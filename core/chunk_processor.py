from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class ProcessingMode(str, Enum):
    PASSTHROUGH = "passthrough"
    MUTE = "mute"
    TEST_TONE = "test_tone"


@dataclass
class ChunkProcessorConfig:
    samplerate: int = 48000
    channels: int = 1
    tone_frequency: float = 440.0
    tone_amplitude: float = 0.15


class ChunkProcessor:
    def __init__(self, config: ChunkProcessorConfig):
        self.config = config
        self.mode: ProcessingMode = ProcessingMode.PASSTHROUGH
        self._tone_phase: float = 0.0

    def set_mode(self, mode: ProcessingMode) -> None:
        self.mode = mode

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self.mode == ProcessingMode.PASSTHROUGH:
            return self._process_passthrough(chunk)

        if self.mode == ProcessingMode.MUTE:
            return self._process_mute(chunk)

        if self.mode == ProcessingMode.TEST_TONE:
            return self._process_test_tone(chunk)

        return self._process_passthrough(chunk)

    def _process_passthrough(self, chunk: np.ndarray) -> np.ndarray:
        return chunk.astype(np.float32, copy=True)

    def _process_mute(self, chunk: np.ndarray) -> np.ndarray:
        return np.zeros_like(chunk, dtype=np.float32)

    def _process_test_tone(self, chunk: np.ndarray) -> np.ndarray:
        frames = chunk.shape[0]
        channels = chunk.shape[1] if chunk.ndim > 1 else 1

        t = (
                    np.arange(frames, dtype=np.float32) + self._tone_phase
            ) / float(self.config.samplerate)

        wave = self.config.tone_amplitude * np.sin(
            2.0 * math.pi * self.config.tone_frequency * t
        ).astype(np.float32)

        self._tone_phase += frames
        if self._tone_phase > self.config.samplerate:
            self._tone_phase %= self.config.samplerate

        if channels == 1:
            return wave.reshape(-1, 1)

        return np.repeat(wave.reshape(-1, 1), channels, axis=1)