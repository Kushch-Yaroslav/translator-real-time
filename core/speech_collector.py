from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np


@dataclass
class SpeechCollectorConfig:
    samplerate: int = 48000
    silence_threshold: float = 0.008
    max_silence_chunks: int = 4
    min_speech_chunks: int = 1
    min_phrase_seconds: float = 0.8
    max_phrase_seconds: float = 12.0
    preroll_chunks: int = 1
    keep_silence_at_end: bool = True
    debug: bool = False


@dataclass
class SpeechCollector:
    config: SpeechCollectorConfig
    speech_chunks: List[np.ndarray] = field(default_factory=list)
    silence_chunks: List[np.ndarray] = field(default_factory=list)
    preroll_buffer: Deque[np.ndarray] = field(default_factory=deque)
    is_recording: bool = False
    silence_counter: int = 0
    total_samples: int = 0

    def __post_init__(self) -> None:
        self.preroll_buffer = deque(maxlen=self.config.preroll_chunks)

    def reset(self) -> None:
        self.speech_chunks.clear()
        self.silence_chunks.clear()
        self.is_recording = False
        self.silence_counter = 0
        self.total_samples = 0

    def process_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        chunk = self._normalize_chunk(chunk)
        if chunk.size == 0:
            return None

        is_speech = self._is_speech(chunk)

        if not self.is_recording:
            if is_speech:
                self.is_recording = True

                if self.preroll_buffer:
                    self.speech_chunks.extend(self.preroll_buffer)
                    self.total_samples += sum(len(c) for c in self.preroll_buffer)

                self.speech_chunks.append(chunk)
                self.total_samples += len(chunk)

                if self.config.debug:
                    print("[SpeechCollector] Speech started")
            else:
                self.preroll_buffer.append(chunk)

            return None

        if is_speech:
            if self.silence_chunks:
                self.speech_chunks.extend(self.silence_chunks)
                self.total_samples += sum(len(c) for c in self.silence_chunks)
                self.silence_chunks.clear()

            self.speech_chunks.append(chunk)
            self.total_samples += len(chunk)
            self.silence_counter = 0
        else:
            self.silence_chunks.append(chunk)
            self.silence_counter += 1

        phrase_too_long = (
                self.total_samples / self.config.samplerate
                >= self.config.max_phrase_seconds
        )

        if self.silence_counter >= self.config.max_silence_chunks or phrase_too_long:
            return self._finalize_phrase()

        return None

    def flush(self) -> Optional[np.ndarray]:
        if not self.speech_chunks:
            self.reset()
            return None

        return self._finalize_phrase()

    def _finalize_phrase(self) -> Optional[np.ndarray]:
        if len(self.speech_chunks) < self.config.min_speech_chunks:
            if self.config.debug:
                print("[SpeechCollector] Phrase discarded: too short by chunks")
            self.reset()
            return None

        chunks = list(self.speech_chunks)

        if self.config.keep_silence_at_end and self.silence_chunks:
            chunks.extend(self.silence_chunks)

        if not chunks:
            self.reset()
            return None

        phrase = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        duration = len(phrase) / self.config.samplerate

        if duration < self.config.min_phrase_seconds:
            if self.config.debug:
                print(f"[SpeechCollector] Phrase discarded: too short by time ({duration:.2f}s)")
            self.reset()
            return None

        if self.config.debug:
            print(f"[SpeechCollector] Phrase finalized: {duration:.2f}s")

        self.reset()
        return phrase

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)

        if chunk.ndim == 2:
            if chunk.shape[1] == 1:
                chunk = chunk[:, 0]
            else:
                chunk = np.mean(chunk, axis=1)

        return chunk

    def _is_speech(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-10))

        if self.config.debug:
            print(f"[SpeechCollector] RMS={rms:.6f}")

        return rms >= self.config.silence_threshold