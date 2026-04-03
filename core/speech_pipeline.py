from __future__ import annotations

import time
from typing import Optional, Callable

import numpy as np


class SpeechPipeline:
    def __init__(
            self,
            name: str,
            samplerate: int,
            stt_service,
            translation_service=None,
            tts_service=None,
            on_log: Optional[Callable[[str], None]] = None,
            on_error: Optional[Callable[[str], None]] = None,
            on_tts_audio: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.name = name
        self.samplerate = samplerate
        self.stt_service = stt_service
        self.translation_service = translation_service
        self.tts_service = tts_service
        self.on_log = on_log
        self.on_error = on_error
        self.on_tts_audio = on_tts_audio

        self.last_recognized_text = ""
        self.last_translated_text = ""

    def process_phrase(self, audio_phrase: np.ndarray) -> None:
        if audio_phrase.size == 0:
            return

        stt_started_at = time.perf_counter()

        try:
            recognized_text = self.stt_service.transcribe(audio_phrase, self.samplerate)
            stt_elapsed = time.perf_counter() - stt_started_at
            self._log(f"[{self.name}] STT time: {stt_elapsed:.3f} sec")
        except Exception as error:
            self._error(f"STT error: {error}")
            return

        recognized_text = self._normalize_text(recognized_text)

        if not recognized_text:
            self._log(f"[{self.name}] STT: <empty>")
            return

        self._log(f"[{self.name}] STT: {recognized_text}")

        incremental_text = self._extract_incremental_text(
            self.last_recognized_text,
            recognized_text,
        )

        if not incremental_text:
            self._log(f"[{self.name}] STT skipped: duplicate/overlap")
            return

        self.last_recognized_text = recognized_text
        self._log(f"[{self.name}] STT incremental: {incremental_text}")

        text_for_output = incremental_text

        if self.translation_service is not None:
            translation_started_at = time.perf_counter()

            try:
                translated_text = self.translation_service.translate(incremental_text)
                translation_elapsed = time.perf_counter() - translation_started_at
                self._log(f"[{self.name}] Translation time: {translation_elapsed:.3f} sec")
            except Exception as error:
                self._error(f"Translation error: {error}")
                return

            translated_text = self._normalize_text(translated_text)

            if not translated_text:
                self._log(f"[{self.name}] TRANSLATED: <empty>")
                return

            self.last_translated_text = translated_text
            text_for_output = translated_text
            self._log(f"[{self.name}] TRANSLATED: {translated_text}")

        if self.tts_service is not None:
            tts_started_at = time.perf_counter()

            try:
                tts_audio = self.tts_service.synthesize(
                    text_for_output,
                    target_samplerate=self.samplerate,
                )
                tts_elapsed = time.perf_counter() - tts_started_at
                self._log(f"[{self.name}] TTS time: {tts_elapsed:.3f} sec")

                duration = (
                    tts_audio.shape[0] / float(self.samplerate)
                    if tts_audio.size > 0
                    else 0.0
                )
                self._log(f"[{self.name}] TTS audio ready: {duration:.2f} sec")

                if self.on_tts_audio is not None:
                    self.on_tts_audio(tts_audio)

            except Exception as error:
                self._error(f"TTS error: {error}")
                return

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        text = SpeechPipeline._normalize_text(text).lower()
        text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        return " ".join(text.split()).strip()

    @staticmethod
    def _extract_incremental_text(previous: str, current: str) -> str:
        previous_norm = SpeechPipeline._normalize_compare_text(previous)
        current_norm = SpeechPipeline._normalize_compare_text(current)

        if not current_norm:
            return ""

        if not previous_norm:
            return current.strip()

        if current_norm == previous_norm:
            return ""

        if current_norm.startswith(previous_norm):
            extra_raw = current[len(previous):].strip()
            if extra_raw:
                return extra_raw

        prev_words = previous_norm.split()
        curr_words = current_norm.split()

        common_prefix_len = 0
        for prev_word, curr_word in zip(prev_words, curr_words):
            if prev_word != curr_word:
                break
            common_prefix_len += 1

        if common_prefix_len > 0 and common_prefix_len < len(curr_words):
            raw_words = current.strip().split()
            if common_prefix_len < len(raw_words):
                return " ".join(raw_words[common_prefix_len:]).strip()

        return current.strip()

    def _log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    def _error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)
        else:
            self._log(message)