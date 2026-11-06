from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EnToRuSegmentDecision:
    emit: bool
    reason: str


class EnToRuStrategy:
    FUNCTION_WORDS = frozenset({
        "the", "a", "an", "to", "of", "in", "on", "at", "with", "for", "from", "by",
        "and", "but", "or", "if", "was", "were", "is", "are", "did", "do", "does", "not",
    })
    AUXILIARIES = frozenset({
        "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "have", "has", "had", "will", "would", "can", "could", "shall", "should", "may",
        "might", "must",
    })
    NEGATIONS = frozenset({"not", "n't", "never", "no"})
    PRONOUN_SUBJECTS = frozenset({"i", "you", "he", "she", "it", "we", "they"})
    SHORT_STABLE_UTTERANCES = frozenset({
        "hello", "hi", "thanks", "thank you", "yes", "no", "okay", "ok", "sure",
    })

    def should_emit_segment(
        self,
        text: str,
        *,
        is_final: bool,
        seen_count: int,
        stable_for_sec: float,
        min_words: int,
        stable_partial_min_sec: float = 0.9,
        stable_partial_min_words: int = 5,
    ) -> EnToRuSegmentDecision:
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return EnToRuSegmentDecision(False, "empty")

        normalized = self._normalize_compare_text(normalized_text)
        words = normalized.split()
        if not words:
            return EnToRuSegmentDecision(False, "empty")

        punctuated = normalized_text[-1] in ".!?,;:"
        if self._ends_with_function_word(words):
            return EnToRuSegmentDecision(False, "ends_with_function_word")

        if self._looks_like_subject_aux_negation(words):
            return EnToRuSegmentDecision(False, "subject_aux_without_content")

        if not punctuated and self._looks_like_subject_aux_negation_prefix(words):
            return EnToRuSegmentDecision(False, "subject_aux_without_content")

        if not punctuated and len(words) < min_words:
            if normalized in self.SHORT_STABLE_UTTERANCES and (is_final or seen_count >= 2):
                reason = "final_flush" if is_final else "stable_partial"
                return EnToRuSegmentDecision(True, reason)
            return EnToRuSegmentDecision(False, "too_short_unpunctuated")

        if punctuated:
            return EnToRuSegmentDecision(True, "final_flush" if is_final else "punctuation")

        if is_final:
            return EnToRuSegmentDecision(True, "final_flush")

        if len(words) >= stable_partial_min_words and stable_for_sec >= stable_partial_min_sec:
            return EnToRuSegmentDecision(True, "stable_partial")

        return EnToRuSegmentDecision(False, "unstable_partial")

    def _looks_like_subject_aux_negation(self, words: list[str]) -> bool:
        if len(words) < 2 or len(words) > 4:
            return False

        first = words[0]
        second = words[1]
        if first not in self.PRONOUN_SUBJECTS and not self._looks_like_name(first):
            return False

        if second not in self.AUXILIARIES:
            return False

        if len(words) == 2:
            return True

        if len(words) == 3 and words[2] in self.NEGATIONS:
            return True

        if len(words) == 4 and words[2] in self.NEGATIONS and words[3] in self.FUNCTION_WORDS:
            return True

        return False

    def _looks_like_subject_aux_negation_prefix(self, words: list[str]) -> bool:
        if len(words) < 3 or len(words) > 4:
            return False

        first = words[0]
        second = words[1]
        if first not in self.PRONOUN_SUBJECTS and not self._looks_like_name(first):
            return False

        return second in self.AUXILIARIES and words[2] in self.NEGATIONS

    def _ends_with_function_word(self, words: list[str]) -> bool:
        return bool(words) and words[-1] in self.FUNCTION_WORDS

    @staticmethod
    def _looks_like_name(word: str) -> bool:
        # Normalized text is lowercase; treat a non-pronoun alphabetic first token as a possible proper-name subject.
        return bool(re.fullmatch(r"[a-z][a-z'-]{1,}", word))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        text = EnToRuStrategy._normalize_text(text).lower()
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        return " ".join(text.split()).strip()
