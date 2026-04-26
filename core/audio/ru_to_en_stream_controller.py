from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RuToEnChunk:
    text: str
    is_final: bool
    is_preview: bool = False


class RuToEnStreamController:
    def __init__(
        self,
        *,
        normalize_text: Callable[[str], str],
        normalize_compare_text: Callable[[str], str],
        extract_incremental_text: Callable[[str, str], str],
        merge_final_texts: Callable[[str, str], str],
    ) -> None:
        self._normalize_text = normalize_text
        self._normalize_compare_text = normalize_compare_text
        self._extract_incremental_text = extract_incremental_text
        self._merge_final_texts = merge_final_texts
        self.reset()

    def reset(self) -> None:
        self._committed_source_text = ""
        self._preview_emitted = False
        self._emitted_norms: set[str] = set()
        self._emitted_segments: list[str] = []
        self._pending_preview_text = ""
        self._pending_preview_norm = ""
        self._pending_preview_seen_count = 0
        self._pending_preview_first_seen_at = 0.0
        self._pending_preview_last_seen_at = 0.0

    @property
    def committed_source_text(self) -> str:
        return self._committed_source_text

    def push_partial(self, text: str, now: float) -> list[RuToEnChunk]:
        text = self._normalize_text(text)
        if not text:
            return []

        incremental = self._normalize_text(
            self._extract_incremental_text(self._committed_source_text, text)
        )
        if not incremental:
            self._expire_pending_preview(now)
            return []

        if self._preview_emitted:
            self._expire_pending_preview(now)
            return []

        candidate = self._select_preview_candidate(incremental)
        if not candidate:
            self._expire_pending_preview(now)
            return []

        if not self._mark_preview_candidate(candidate, now):
            return []

        return [self._remember_chunk(candidate, is_final=False, is_preview=True)]

    def push_final(self, text: str, now: float) -> list[RuToEnChunk]:
        text = self._normalize_text(text)
        if not text:
            return []

        incremental = self._normalize_text(
            self._extract_incremental_text(self._committed_source_text, text)
        )
        if not incremental:
            return []

        self._expire_pending_preview(now)

        clauses = self._split_final_incremental(incremental)
        prepared_clauses: list[str] = []
        for clause in clauses:
            prepared = self._prepare_clause_for_emit(
                clause,
                min_words=4,
                allow_short_exclamation=True,
            )
            if not prepared or self._is_duplicate(prepared):
                continue
            prepared_clauses.append(prepared)

        if not prepared_clauses:
            fallback = self._prepare_clause_for_emit(
                incremental,
                min_words=3,
                allow_short_exclamation=False,
            )
            if fallback and not self._is_duplicate(fallback):
                prepared_clauses.append(fallback)

        chunks: list[RuToEnChunk] = []
        for index, clause in enumerate(prepared_clauses):
            chunks.append(
                self._remember_chunk(
                    clause,
                    is_final=index == len(prepared_clauses) - 1,
                    is_preview=False,
                )
            )
        return chunks

    def _remember_chunk(self, text: str, *, is_final: bool, is_preview: bool) -> RuToEnChunk:
        text = self._normalize_text(text)
        normalized = self._normalize_compare_text(text)
        if normalized:
            self._emitted_norms.add(normalized)
            self._emitted_segments.append(text)
            if len(self._emitted_segments) > 24:
                self._emitted_segments = self._emitted_segments[-24:]
        self._committed_source_text = self._merge_final_texts(self._committed_source_text, text)
        if is_preview:
            self._preview_emitted = True
        self._reset_pending_preview()
        return RuToEnChunk(text=text, is_final=is_final, is_preview=is_preview)

    def _is_duplicate(self, text: str) -> bool:
        normalized = self._normalize_compare_text(text)
        if not normalized:
            return True
        if normalized in self._emitted_norms:
            return True

        for emitted in reversed(self._emitted_segments[-8:]):
            emitted_norm = self._normalize_compare_text(emitted)
            if not emitted_norm:
                continue
            if normalized == emitted_norm:
                return True
            if len(normalized.split()) <= 5 and f" {normalized} " in f" {emitted_norm} ":
                return True
            if len(emitted_norm.split()) <= 5 and f" {emitted_norm} " in f" {normalized} ":
                return True
        return False

    def _select_preview_candidate(self, incremental: str) -> str:
        preview_clauses = self._split_preview_incremental(incremental)
        if preview_clauses:
            return preview_clauses[0]

        fallback = self._prepare_clause_for_emit(
            incremental,
            min_words=5,
            allow_short_exclamation=False,
        )
        if not fallback:
            return ""
        if not self._is_preview_candidate_strong_enough(fallback):
            return ""
        return fallback

    def _split_preview_incremental(self, text: str) -> list[str]:
        clauses = self._split_incremental_clauses(
            text,
            min_words_for_soft_break=5,
        )
        prepared: list[str] = []
        for clause in clauses:
            candidate = self._prepare_clause_for_emit(
                clause,
                min_words=4,
                allow_short_exclamation=True,
            )
            if candidate and self._is_preview_candidate_strong_enough(candidate):
                prepared.append(candidate)
                break
        return prepared

    def _split_final_incremental(self, text: str) -> list[str]:
        clauses = self._split_incremental_clauses(
            text,
            min_words_for_soft_break=8,
        )
        if not clauses:
            return []

        merged: list[str] = []
        pending = ""
        for clause in clauses:
            clause = self._normalize_text(clause)
            if not clause:
                continue
            candidate = self._normalize_text(f"{pending} {clause}" if pending else clause)
            if self._is_strong_final_clause(candidate):
                merged.append(candidate)
                pending = ""
            else:
                pending = candidate

        if pending:
            merged.append(pending)

        return merged

    def _split_incremental_clauses(self, text: str, *, min_words_for_soft_break: int) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        clauses: list[str] = []
        buffer = ""
        for character in text:
            buffer += character
            if character in ".!?":
                clause = self._normalize_text(buffer)
                if clause:
                    clauses.append(clause)
                buffer = ""
                continue

            if character in ",;:":
                candidate = self._normalize_text(buffer)
                if self._count_words(candidate) >= min_words_for_soft_break and not self._ends_with_weak_token(candidate):
                    clauses.append(candidate)
                    buffer = ""

        trailing = self._normalize_text(buffer)
        if trailing:
            clauses.append(trailing)
        return clauses

    def _prepare_clause_for_emit(
        self,
        clause: str,
        *,
        min_words: int,
        allow_short_exclamation: bool,
    ) -> str:
        clause = self._normalize_text(clause)
        if not clause:
            return ""

        normalized = self._normalize_compare_text(clause)
        if not normalized:
            return ""

        words = normalized.split()
        if len(words) < min_words:
            punctuation = clause[-1] if clause and clause[-1] in ".!?,;:" else ""
            if not (
                allow_short_exclamation
                and punctuation in {"!", "?"}
                and len(words) >= 2
            ):
                return ""

        if self._ends_with_weak_token(clause):
            return ""

        if self._starts_with_weak_noise(clause):
            return ""

        if normalized in {"..", ".", ","}:
            return ""

        return clause

    def _is_preview_candidate_strong_enough(self, clause: str) -> bool:
        normalized = self._normalize_compare_text(clause)
        words = normalized.split()
        if len(words) < 4:
            return False
        if self._starts_with_weak_noise(clause):
            return False
        return not self._ends_with_weak_token(clause)

    def _is_strong_final_clause(self, clause: str) -> bool:
        normalized = self._normalize_compare_text(clause)
        words = normalized.split()
        if len(words) >= 5 and not self._ends_with_weak_token(clause):
            return True

        punctuation = clause[-1] if clause and clause[-1] in ".!?,;:" else ""
        if punctuation in {"!", "?"} and len(words) >= 2:
            return True
        return punctuation == "." and len(words) >= 3 and not self._ends_with_weak_token(clause)

    def _mark_preview_candidate(self, candidate: str, now: float) -> bool:
        candidate = self._normalize_text(candidate)
        candidate_norm = self._normalize_compare_text(candidate)
        if not candidate_norm:
            return False

        stale_after_sec = 0.55
        if (
            candidate_norm != self._pending_preview_norm
            or (now - self._pending_preview_last_seen_at) > stale_after_sec
        ):
            self._pending_preview_text = candidate
            self._pending_preview_norm = candidate_norm
            self._pending_preview_seen_count = 1
            self._pending_preview_first_seen_at = now
        else:
            self._pending_preview_text = candidate
            self._pending_preview_seen_count += 1

        self._pending_preview_last_seen_at = now
        stable_for = now - self._pending_preview_first_seen_at

        punctuation = candidate[-1] if candidate and candidate[-1] in ".!?,;:" else ""
        if punctuation in {"!", "?"}:
            return self._pending_preview_seen_count >= 1

        if punctuation in {".", ",", ";", ":"} and self._count_words(candidate) >= 4:
            return self._pending_preview_seen_count >= 1 or stable_for >= 0.12

        if self._pending_preview_seen_count >= 2:
            return True

        return stable_for >= 0.22

    def _expire_pending_preview(self, now: float) -> None:
        if not self._pending_preview_norm:
            return
        if (now - self._pending_preview_last_seen_at) > 0.55:
            self._reset_pending_preview()

    def _reset_pending_preview(self) -> None:
        self._pending_preview_text = ""
        self._pending_preview_norm = ""
        self._pending_preview_seen_count = 0
        self._pending_preview_first_seen_at = 0.0
        self._pending_preview_last_seen_at = 0.0

    def _starts_with_weak_noise(self, text: str) -> bool:
        normalized = self._normalize_compare_text(text)
        if not normalized:
            return True

        starts_with_noise = (
            "лет я из",
            "проект же у",
            "для помощи в преодолении",
            "преодоление языкового",
            "называют картинки",
        )
        return normalized.startswith(starts_with_noise)

    def _ends_with_weak_token(self, text: str) -> bool:
        normalized = self._normalize_compare_text(text)
        if not normalized:
            return True

        weak_tail_tokens = {
            "а",
            "без",
            "в",
            "во",
            "где",
            "да",
            "для",
            "до",
            "если",
            "и",
            "из",
            "или",
            "как",
            "когда",
            "ко",
            "к",
            "мне",
            "меня",
            "мой",
            "моя",
            "мои",
            "моё",
            "на",
            "но",
            "о",
            "об",
            "от",
            "по",
            "под",
            "при",
            "про",
            "с",
            "со",
            "то",
            "у",
            "что",
            "чтобы",
            "эта",
            "это",
            "этот",
            "эти",
            "я",
        }
        words = normalized.split()
        if not words:
            return True
        return words[-1] in weak_tail_tokens

    def _count_words(self, text: str) -> int:
        normalized = self._normalize_compare_text(text)
        if not normalized:
            return 0
        return len(normalized.split())
