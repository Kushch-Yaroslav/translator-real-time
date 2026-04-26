from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re


_WEAK_LEADING_WORDS = {
    "и",
    "а",
    "но",
    "или",
    "да",
}

_CONTINUATION_PREFIXES = (
    "меня зовут",
    "мне ",
    "я из ",
    "и ",
    "но ",
    "а ",
    "или ",
    "так как ",
    "который ",
    "которым ",
    "это ",
)

_HARD_SENTENCE_PREFIXES = (
    "я работал",
    "я работаю",
    "я решил",
    "также ",
    "сейчас ",
    "мы ",
    "когда ",
)


@dataclass(frozen=True)
class PhraseFragmentReconstruction:
    phrases: list[str]
    text: str


@dataclass
class _FragmentCluster:
    variants: list[str] = field(default_factory=list)
    canonical_text: str = ""

    def add(self, text: str) -> None:
        self.variants.append(text)
        self.canonical_text = text


class PhraseFragmentReconstructor:
    def reconstruct(self, fragments: list[str]) -> PhraseFragmentReconstruction:
        clusters: list[_FragmentCluster] = []

        for raw_fragment in fragments:
            fragment = self._normalize_text(raw_fragment)
            if not fragment:
                continue

            cluster_index = self._find_cluster_index(fragment, clusters)
            if cluster_index is None:
                cluster = _FragmentCluster()
                cluster.add(fragment)
                clusters.append(cluster)
                continue

            cluster = clusters[cluster_index]
            cluster.variants.append(fragment)
            cluster.canonical_text = self._choose_better_phrase(
                cluster.canonical_text,
                fragment,
            )

        phrases = self._finalize_clusters(clusters)
        return PhraseFragmentReconstruction(
            phrases=phrases,
            text=self._compose_text(phrases),
        )

    def reconstruct_text(self, fragments: list[str]) -> str:
        return self.reconstruct(fragments).text

    def _find_cluster_index(
        self,
        fragment: str,
        clusters: list[_FragmentCluster],
    ) -> int | None:
        best_index: int | None = None
        best_score = 0.0

        for index in range(max(0, len(clusters) - 4), len(clusters)):
            cluster = clusters[index]
            score = self._score_cluster_match(fragment, cluster)
            if score > best_score:
                best_score = score
                best_index = index

        if best_score >= 0.58:
            return best_index
        return None

    def _score_cluster_match(self, fragment: str, cluster: _FragmentCluster) -> float:
        best_score = self._score_phrase_match(fragment, cluster.canonical_text)
        for variant in cluster.variants[-3:]:
            best_score = max(best_score, self._score_phrase_match(fragment, variant))
        return best_score

    def _score_phrase_match(self, left: str, right: str) -> float:
        left_tokens = self._tokenize_compare(left)
        right_tokens = self._tokenize_compare(right)
        if not left_tokens or not right_tokens:
            return 0.0

        prefix = self._common_prefix_len(left_tokens, right_tokens)
        overlap = self._suffix_prefix_overlap(left_tokens, right_tokens)
        similarity = SequenceMatcher(
            None,
            " ".join(left_tokens),
            " ".join(right_tokens),
        ).ratio()

        prefix_score = prefix / max(1, min(len(left_tokens), len(right_tokens)))
        overlap_score = overlap / max(1, min(len(left_tokens), len(right_tokens)))

        if self._numbers_compatible(left_tokens, right_tokens):
            similarity = max(similarity, 0.82)

        return max(similarity, prefix_score, overlap_score)

    def _finalize_clusters(self, clusters: list[_FragmentCluster]) -> list[str]:
        phrases: list[str] = []
        for cluster in clusters:
            phrase = self._resolve_cluster_phrase(cluster)
            if not phrase:
                continue
            if phrases and self._normalize_compare_text(phrases[-1]) == self._normalize_compare_text(phrase):
                continue
            phrases.append(phrase)
        return phrases

    def _resolve_cluster_phrase(self, cluster: _FragmentCluster) -> str:
        best = cluster.canonical_text
        if not best:
            return ""

        best = self._normalize_spaced_digits(best)
        variants = [self._normalize_spaced_digits(variant) for variant in cluster.variants]

        for variant in variants:
            best = self._choose_better_phrase(best, variant)

        return best

    def _choose_better_phrase(self, current: str, candidate: str) -> str:
        current = self._normalize_text(current)
        candidate = self._normalize_text(candidate)
        if not current:
            return candidate
        if not candidate:
            return current

        current_tokens = self._tokenize_compare(current)
        candidate_tokens = self._tokenize_compare(candidate)
        prefix = self._common_prefix_len(current_tokens, candidate_tokens)
        suffix = self._common_suffix_len(current_tokens, candidate_tokens)

        if self._is_phrase_refinement(current_tokens, candidate_tokens, prefix, suffix):
            return candidate

        if prefix >= min(3, len(current_tokens), len(candidate_tokens)):
            current_tail_score = self._score_variant_tail(current, prefix)
            candidate_tail_score = self._score_variant_tail(candidate, prefix)
            if candidate_tail_score > current_tail_score:
                current = candidate
                current_tokens = candidate_tokens
            elif current_tail_score > candidate_tail_score:
                candidate = current
                candidate_tokens = current_tokens

        if prefix >= min(3, len(current_tokens), len(candidate_tokens)):
            current = self._upgrade_phrase(current, candidate)
            candidate = self._upgrade_phrase(candidate, current)

        current_score = self._score_phrase_quality(current)
        candidate_score = self._score_phrase_quality(candidate)

        if candidate_score > current_score:
            return candidate
        return current

    def _is_phrase_refinement(
        self,
        current_tokens: list[str],
        candidate_tokens: list[str],
        prefix: int,
        suffix: int,
    ) -> bool:
        if not current_tokens or not candidate_tokens:
            return False

        shared_edge = max(prefix, suffix)
        if shared_edge < min(3, len(current_tokens), len(candidate_tokens)):
            return False

        if len(candidate_tokens) + 2 < len(current_tokens):
            return False

        current_middle = current_tokens[prefix: len(current_tokens) - suffix if suffix else len(current_tokens)]
        candidate_middle = candidate_tokens[prefix: len(candidate_tokens) - suffix if suffix else len(candidate_tokens)]
        middle_size_delta = abs(len(candidate_middle) - len(current_middle))
        if middle_size_delta > 2:
            return False

        differing_pairs = 0
        for current_token, candidate_token in zip(current_middle, candidate_middle):
            if current_token == candidate_token:
                continue
            similarity = SequenceMatcher(None, current_token, candidate_token).ratio()
            if similarity < 0.55:
                return False
            differing_pairs += 1

        if len(candidate_middle) > len(current_middle):
            differing_pairs += len(candidate_middle) - len(current_middle)

        if differing_pairs == 0 and len(candidate_tokens) == len(current_tokens):
            return False

        return differing_pairs <= 2

    def _score_variant_tail(self, text: str, prefix_tokens: int) -> tuple[float, float, float]:
        raw_tokens = self._tokenize_raw(text)
        if prefix_tokens >= len(raw_tokens):
            return (0.0, 0.0, 0.0)

        tail_tokens = raw_tokens[prefix_tokens:]
        tail_text = " ".join(tail_tokens)
        compact_entities = sum(1 for token in tail_tokens if self._looks_like_compact_entity(token))
        apostrophes = tail_text.count("'") + tail_text.count("’")
        latin_noise = len(re.findall(r"\b(?:im|i'm)\b", tail_text.lower()))

        return (
            float(compact_entities) * 2.0 - float(apostrophes) - float(latin_noise),
            -float(len(tail_tokens)),
            float(len("".join(tail_tokens))),
        )

    def _upgrade_phrase(self, base: str, candidate: str) -> str:
        base_tokens = self._tokenize_raw(base)
        candidate_tokens = self._tokenize_raw(candidate)
        if not base_tokens or not candidate_tokens:
            return base

        prefix = self._common_prefix_len(
            self._tokenize_compare(base),
            self._tokenize_compare(candidate),
        )
        if prefix <= 0:
            return base

        if len(candidate_tokens) <= prefix:
            return base

        candidate_tail = candidate_tokens[prefix:]
        if not candidate_tail:
            return base

        if len(base_tokens) == prefix:
            return self._normalize_text(" ".join(base_tokens + candidate_tail))
        return base

    def _score_phrase_quality(self, text: str) -> tuple[float, float, float, float]:
        normalized = self._normalize_text(text)
        compare = self._normalize_compare_text(normalized)
        compare_tokens = compare.split()
        raw_tokens = self._tokenize_raw(normalized)

        has_bad_apostrophe = "'" in normalized or "’" in normalized
        latin_noise = len(re.findall(r"\b(?:im|i'm|days)\b", normalized.lower()))
        compact_tokens = sum(1 for token in raw_tokens if self._looks_like_compact_entity(token))
        zero_noise = sum(
            1
            for token in raw_tokens
            if token.isdigit() and len(token) >= 3 and "0" in token
        )

        return (
            float(len(compare_tokens)),
            float(compact_tokens) * 1.2 - float(latin_noise) * 1.1 - float(has_bad_apostrophe),
            -float(zero_noise),
            float(len(normalized)),
        )

    def _compose_text(self, phrases: list[str]) -> str:
        if not phrases:
            return ""

        sentences: list[list[str]] = []
        current_sentence: list[str] = []

        for phrase in phrases:
            if not current_sentence:
                current_sentence.append(self._capitalize_phrase(phrase))
                continue

            previous = current_sentence[-1]
            if self._starts_new_sentence(previous, phrase, len(current_sentence)):
                sentences.append(current_sentence)
                current_sentence = [self._capitalize_phrase(phrase)]
                continue

            current_sentence.append(phrase)

        if current_sentence:
            sentences.append(current_sentence)

        rendered_sentences: list[str] = []
        for sentence in sentences:
            rendered = self._render_sentence(sentence)
            if rendered:
                rendered_sentences.append(rendered)

        return " ".join(rendered_sentences)

    def _starts_new_sentence(self, previous: str, current: str, clause_count: int) -> bool:
        current_norm = self._normalize_compare_text(current)
        if not current_norm:
            return False

        if current_norm.startswith(_HARD_SENTENCE_PREFIXES):
            return True

        if current_norm.startswith(_CONTINUATION_PREFIXES):
            return False

        previous_norm = self._normalize_compare_text(previous)
        if previous_norm.startswith("всем привет") or previous_norm.startswith("добрый"):
            return False

        if clause_count >= 4:
            return True

        return False

    def _render_sentence(self, clauses: list[str]) -> str:
        if not clauses:
            return ""

        rendered = self._capitalize_phrase(clauses[0])
        for clause in clauses[1:]:
            clause = self._normalize_text(clause)
            if not clause:
                continue

            clause_norm = self._normalize_compare_text(clause)
            if clause_norm.startswith(tuple(f"{word} " for word in _WEAK_LEADING_WORDS)):
                rendered = f"{rendered} {clause}"
            else:
                rendered = f"{rendered}, {clause}"

        rendered = rendered.rstrip(" ,.;:")
        if rendered and rendered[-1] not in ".!?":
            rendered += "."
        return rendered

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip()
        text = text.replace("ё", "е").replace("Ё", "Е")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.strip(" \t\r\n,;")

    def _normalize_compare_text(self, text: str) -> str:
        text = self._normalize_text(text).lower()
        text = re.sub(r"[\"'`’]", "", text)
        text = re.sub(r"[^0-9a-zа-я\s-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _tokenize_compare(self, text: str) -> list[str]:
        tokens = self._normalize_compare_text(text).split()
        return [self._normalize_numeric_token(token) for token in tokens if token]

    def _tokenize_raw(self, text: str) -> list[str]:
        return [token for token in re.split(r"\s+", self._normalize_text(text)) if token]

    def _normalize_numeric_token(self, token: str) -> str:
        if token.isdigit():
            stripped = token.lstrip("0")
            return stripped or "0"
        return token

    def _normalize_spaced_digits(self, text: str) -> str:
        tokens = self._tokenize_raw(text)
        normalized: list[str] = []
        index = 0

        while index < len(tokens):
            token = tokens[index]
            if token.isdigit():
                digit_run = [token]
                lookahead = index + 1
                while lookahead < len(tokens) and tokens[lookahead].isdigit():
                    digit_run.append(tokens[lookahead])
                    lookahead += 1

                if len(digit_run) > 1:
                    joined = "".join(digit_run)
                    compact = joined.replace("0", "")
                    if compact and len(compact) < len(joined):
                        normalized.append(compact)
                    else:
                        normalized.append(joined)
                    index = lookahead
                    continue

            normalized.append(token)
            index += 1

        return self._normalize_text(" ".join(normalized))

    def _numbers_compatible(self, left_tokens: list[str], right_tokens: list[str]) -> bool:
        if len(left_tokens) != len(right_tokens):
            return False

        found_numeric = False
        for left, right in zip(left_tokens, right_tokens):
            if left.isdigit() and right.isdigit():
                found_numeric = True
                if left == right:
                    continue
                if left.replace("0", "") == right.replace("0", ""):
                    continue
                return False
            elif left != right:
                return False

        return found_numeric

    def _common_prefix_len(self, left_tokens: list[str], right_tokens: list[str]) -> int:
        prefix = 0
        for left, right in zip(left_tokens, right_tokens):
            if left == right:
                prefix += 1
                continue
            if left.isdigit() and right.isdigit() and left.replace("0", "") == right.replace("0", ""):
                prefix += 1
                continue
            break
        return prefix

    def _common_suffix_len(self, left_tokens: list[str], right_tokens: list[str]) -> int:
        suffix = 0
        for left, right in zip(reversed(left_tokens), reversed(right_tokens)):
            if left == right:
                suffix += 1
                continue
            if left.isdigit() and right.isdigit() and left.replace("0", "") == right.replace("0", ""):
                suffix += 1
                continue
            break
        return suffix

    def _suffix_prefix_overlap(self, left_tokens: list[str], right_tokens: list[str]) -> int:
        max_overlap = min(len(left_tokens), len(right_tokens))
        for overlap in range(max_overlap, 1, -1):
            if left_tokens[-overlap:] == right_tokens[:overlap]:
                return overlap
        return 0

    @staticmethod
    def _looks_like_compact_entity(token: str) -> bool:
        token = token.strip(",.;:!?")
        if len(token) < 4:
            return False
        has_upper = any(ch.isupper() for ch in token)
        has_lower = any(ch.islower() for ch in token)
        has_letters = any(ch.isalpha() for ch in token)
        return has_letters and (has_upper or has_lower) and " " not in token

    @staticmethod
    def _capitalize_phrase(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        return text[0].upper() + text[1:]
