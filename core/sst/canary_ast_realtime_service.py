from __future__ import annotations

import io
import re
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import requests
from scipy.signal import resample


@dataclass
class CanaryASTRealtimeConfig:
    base_url: str = "http://localhost:9000"
    source_language: str = "ru-RU"
    target_language: str = "en-US"
    sample_rate_hz: int = 16000
    timeout: float = 20.0
    poll_interval_sec: float = 0.5
    min_window_sec: float = 1.2
    finalize_silence_sec: float = 0.55
    min_emit_words: int = 3
    speech_rms_threshold: float = 0.003
    min_hypothesis_words: int = 4
    on_log: Optional[Callable[[str], None]] = None


class CanaryASTRealtimeService:
    def __init__(self, config: Optional[CanaryASTRealtimeConfig] = None):
        self.config = config or CanaryASTRealtimeConfig()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._partial_callback: Optional[Callable[[str], None]] = None
        self._final_callback: Optional[Callable[[str], None]] = None

        self._audio_chunks: list[np.ndarray] = []
        self._audio_total_frames = 0
        self._utterance_started_at = 0.0
        self._last_speech_at = 0.0
        self._last_inference_at = 0.0
        self._force_commit = False
        self._force_finalize = False
        self._utterance_active = False

        self._last_partial_text = ""
        self._last_final_text = ""
        self._last_hypothesis_text = ""
        self._emitted_sentence_count = 0

    def start(
        self,
        partial_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._partial_callback = partial_callback
        self._final_callback = final_callback
        self._stop_event.clear()
        self.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="canary-ast-realtime-loop",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
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

        now = time.monotonic()
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-10))
        has_speech = rms >= self.config.speech_rms_threshold

        with self._lock:
            if has_speech and not self._utterance_active:
                self._utterance_active = True
                self._utterance_started_at = now
                self._last_speech_at = now
                self._log("Canary AST utterance started")

            if not self._utterance_active:
                return

            self._audio_chunks.append(mono.astype(np.float32, copy=True))
            self._audio_total_frames += int(mono.shape[0])
            max_frames = int(self.config.sample_rate_hz * 20.0)
            while self._audio_total_frames > max_frames and self._audio_chunks:
                removed = self._audio_chunks.pop(0)
                self._audio_total_frames -= int(removed.shape[0])

            if has_speech:
                self._last_speech_at = now

    def commit(self) -> None:
        with self._lock:
            self._force_commit = True

    def clear(self) -> None:
        with self._lock:
            self._audio_chunks = []
            self._audio_total_frames = 0
            self._utterance_started_at = 0.0
            self._last_speech_at = 0.0
            self._last_inference_at = 0.0
            self._force_commit = False
            self._force_finalize = False
            self._utterance_active = False
            self._last_partial_text = ""
            self._last_final_text = ""
            self._last_hypothesis_text = ""
            self._emitted_sentence_count = 0

    def send_done(self) -> None:
        with self._lock:
            self._force_finalize = True

    def get_last_partial_text(self) -> str:
        return self._last_partial_text

    def get_last_final_text(self) -> str:
        return self._last_final_text

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            job = self._pull_job()
            if job is None:
                time.sleep(0.05)
                continue

            audio, finalize = job
            try:
                hypothesis = self._normalize_text(self._translate_audio(audio))
            except Exception as error:
                self._log(f"Canary AST request error: {error}")
                if finalize:
                    self._reset_after_finalize()
                time.sleep(0.2)
                continue

            if not hypothesis:
                if finalize:
                    self._reset_after_finalize()
                continue

            if len(hypothesis.split()) < self.config.min_hypothesis_words:
                if finalize:
                    self._reset_after_finalize()
                continue

            self._last_partial_text = hypothesis
            self._log(f"Canary AST hypothesis: {hypothesis}")
            if self._partial_callback is not None:
                self._partial_callback(hypothesis)

            if finalize:
                self._emit_final_hypothesis(hypothesis)
                self._reset_after_finalize()
                continue

            self._emit_stable_prefix(hypothesis, audio_duration_sec=audio.shape[0] / self.config.sample_rate_hz)

    def _pull_job(self) -> Optional[tuple[np.ndarray, bool]]:
        with self._lock:
            if not self._utterance_active or not self._audio_chunks:
                return None

            now = time.monotonic()
            audio_duration_sec = self._audio_total_frames / float(self.config.sample_rate_hz)
            if audio_duration_sec < self.config.min_window_sec:
                return None

            silence_sec = now - self._last_speech_at if self._last_speech_at > 0.0 else 0.0
            should_finalize = self._force_finalize or silence_sec >= self.config.finalize_silence_sec
            should_poll = (
                self._force_commit
                or (now - self._last_inference_at) >= self.config.poll_interval_sec
            )

            if not should_finalize and not should_poll:
                return None

            audio = np.concatenate(self._audio_chunks, axis=0).astype(np.float32, copy=False)
            self._last_inference_at = now
            self._force_commit = False
            self._force_finalize = False
            return audio, should_finalize

    def _emit_stable_prefix(self, hypothesis: str, audio_duration_sec: float) -> None:
        completed_sentences, _ = self._split_sentences(hypothesis)
        previous_sentences, _ = self._split_sentences(self._last_hypothesis_text)

        stable_sentences: list[str] = []
        for previous, current in zip(previous_sentences, completed_sentences):
            if self._normalize_compare_text(previous) != self._normalize_compare_text(current):
                break
            stable_sentences.append(current)

        if not stable_sentences and completed_sentences and audio_duration_sec >= 2.4:
            stable_sentences = completed_sentences[:1]

        self._emit_sentence_list(stable_sentences)
        self._last_hypothesis_text = hypothesis

    def _emit_final_hypothesis(self, hypothesis: str) -> None:
        completed_sentences, trailing_fragment = self._split_sentences(hypothesis)
        self._emit_sentence_list(completed_sentences)

        trailing_fragment = self._normalize_text(trailing_fragment)
        if trailing_fragment and len(trailing_fragment.split()) >= self.config.min_emit_words:
            if self._final_callback is not None:
                self._last_final_text = trailing_fragment
                self._log(f"Canary AST final tail: {trailing_fragment}")
                self._final_callback(trailing_fragment)

    def _emit_sentence_list(self, sentences: list[str]) -> None:
        for sentence in sentences[self._emitted_sentence_count:]:
            normalized = self._normalize_text(sentence)
            if len(normalized.split()) < self.config.min_emit_words:
                self._emitted_sentence_count += 1
                continue

            if self._final_callback is not None:
                self._last_final_text = normalized
                self._log(f"Canary AST queued: {normalized}")
                self._final_callback(normalized)

            self._emitted_sentence_count += 1

    def _reset_after_finalize(self) -> None:
        with self._lock:
            self._audio_chunks = []
            self._audio_total_frames = 0
            self._utterance_started_at = 0.0
            self._last_speech_at = 0.0
            self._last_inference_at = 0.0
            self._force_commit = False
            self._force_finalize = False
            self._utterance_active = False
            self._last_hypothesis_text = ""
            self._emitted_sentence_count = 0

    def _translate_audio(self, audio: np.ndarray) -> str:
        wav_bytes = self._to_wav_bytes(audio, self.config.sample_rate_hz)
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/v1/audio/translations",
            files={"file": ("phrase.wav", wav_bytes, "audio/wav")},
            data={
                "language": self.config.source_language,
                "target_language": self.config.target_language,
            },
            timeout=self.config.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        payload = response.json()
        return payload.get("text", "").strip()

    def _log(self, message: str) -> None:
        if self.config.on_log is not None:
            self.config.on_log(message)

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio = audio.astype(np.float32, copy=False)
        peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
        if peak > 1.0:
            audio = audio / 32768.0
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio.astype(np.float32, copy=False)
        target_len = int(len(audio) * dst_sr / src_sr)
        if target_len <= 0:
            return np.zeros((0,), dtype=np.float32)
        return resample(audio, target_len).astype(np.float32, copy=False)

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, samplerate: int) -> bytes:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        normalized = CanaryASTRealtimeService._normalize_text(text).lower()
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch.isspace())
        return " ".join(normalized.split())

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        text = CanaryASTRealtimeService._normalize_text(text)
        if not text:
            return [], ""

        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]", text))
        completed_sentences = [
            match.group(0).strip()
            for match in sentence_matches
            if match.group(0).strip()
        ]

        trailing_fragment = text[sentence_matches[-1].end():].strip() if sentence_matches else text
        return completed_sentences, trailing_fragment
