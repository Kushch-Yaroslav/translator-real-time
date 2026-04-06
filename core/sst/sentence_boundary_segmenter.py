from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SentenceBoundarySegmenterConfig:
    stability_sec: float = 0.45
    min_words: int = 4
    on_log: Optional[Callable[[str], None]] = None


class SentenceBoundarySegmenter:
    def __init__(self, config: SentenceBoundarySegmenterConfig | None = None):
        self.config = config or SentenceBoundarySegmenterConfig()
        self._sentences: list[str] = []
        self._sentences_changed_at: float = 0.0
        self._emitted_count: int = 0
        self._last_partial_text: str = ""

    def reset(self) -> None:
        self._sentences = []
        self._sentences_changed_at = 0.0
        self._emitted_count = 0
        self._last_partial_text = ""

    def push_partial(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        self._last_partial_text = normalized
        sentences, trailing_fragment = self._split_sentences(normalized)
        if self._normalized_sentence_list(sentences) != self._normalized_sentence_list(self._sentences):
            self._sentences = sentences
            self._sentences_changed_at = time.monotonic()

        if not sentences:
            return []

        if trailing_fragment and len(trailing_fragment.split()) < 2:
            return []

        if self._sentences_changed_at <= 0.0:
            return []

        age = time.monotonic() - self._sentences_changed_at
        if age < self.config.stability_sec:
            return []

        return self._emit_new_sentences(sentences, is_final=False)

    def push_final(self, text: str) -> list[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        sentences, trailing_fragment = self._split_sentences(normalized)
        emitted = self._emit_new_sentences(sentences, is_final=True)

        trailing_fragment = self._normalize_text(trailing_fragment)
        if trailing_fragment and len(trailing_fragment.split()) >= self.config.min_words:
            emitted.append(trailing_fragment)
            self._emitted_count += 1
            self._log(f"Boundary final tail emitted: {trailing_fragment}")

        return emitted

    def _emit_new_sentences(self, sentences: list[str], is_final: bool) -> list[str]:
        ready: list[str] = []
        start_index = min(self._emitted_count, len(sentences))

        for sentence in sentences[start_index:]:
            normalized = self._normalize_text(sentence)
            if len(normalized.split()) < self.config.min_words:
                self._emitted_count += 1
                continue

            ready.append(normalized)
            self._emitted_count += 1
            self._log(
                f"Boundary {'final' if is_final else 'partial'} emitted: {normalized}"
            )

        return ready

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        text = SentenceBoundarySegmenter._normalize_text(text)
        if not text:
            return [], ""

        matches = list(re.finditer(r"[^.!?]+[.!?]", text))
        sentences = [match.group(0).strip() for match in matches if match.group(0).strip()]

        trailing_fragment = ""
        if matches:
            trailing_fragment = text[matches[-1].end():].strip()
        else:
            trailing_fragment = text

        return sentences, trailing_fragment

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        text = SentenceBoundarySegmenter._normalize_text(text).lower()
        text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        return " ".join(text.split()).strip()

    @classmethod
    def _normalized_sentence_list(cls, sentences: list[str]) -> list[str]:
        return [
            cls._normalize_compare_text(sentence)
            for sentence in sentences
            if cls._normalize_compare_text(sentence)
        ]

    def _log(self, message: str) -> None:
        if self.config.on_log is not None:
            self.config.on_log(message)
