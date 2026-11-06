"""Word-span repetition detection for STT/TTS degeneration loops.

Whisper (and, less often, MT) can degenerate into repeating the same
phrase many times:

    "The University of California was founded in 1868. The University of
     California was founded in 1868. The University of California..."

or, with inflection drift (observed in the app logs):

    "В Калифорнийском университете Калифорнийского университета Калифорнийском
     университете..."

``collapse_repeated_spans`` finds a word span (>= ``min_ngram`` words) that
repeats >= ``min_copies`` times back-to-back and keeps only the first copy.
Tokens match strictly, or fuzzily when they share a long common prefix
(>= 75% of the shorter token, min 6 chars). This catches Russian case
variants: "калифорнийском" ~ "калифорнийского", "университета" ~ "университете".
"""

from __future__ import annotations

import re

MIN_FUZZY_TOKEN_LEN = 6
FUZZY_PREFIX_RATIO = 0.75

_TOKEN_CLEAN_RE = re.compile(r"[^0-9a-zа-яё]")


def _normalize_token(token: str) -> str:
    return _TOKEN_CLEAN_RE.sub("", token.lower())


def _common_prefix_len(a: str, b: str) -> int:
    length = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        length += 1
    return length


def _token_match(a: str, b: str) -> bool:
    """Strict match, or stem-like match for longer tokens.

    Russian case variants are not prefix pairs ("университета" vs
    "университете"), so compare the common prefix length against a ratio
    of the shorter token instead of requiring a full prefix.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < MIN_FUZZY_TOKEN_LEN:
        return False
    threshold = max(MIN_FUZZY_TOKEN_LEN, min(len(a), len(b)) * FUZZY_PREFIX_RATIO)
    return _common_prefix_len(a, b) >= threshold


def _span_match(head: list[str], other: list[str]) -> bool:
    return all(_token_match(h, o) for h, o in zip(head, other))


def find_repeated_spans(
    tokens: list[str],
    *,
    min_ngram: int = 2,
    min_copies: int = 3,
    max_span_words: int = 40,
) -> list[tuple[int, int]]:
    """Return (drop_start, drop_end) ranges to remove.

    For a span at index ``i`` with ``copies`` repetitions, the first copy
    (tokens[i:i+span]) is kept; tokens[i+span : i+span*copies] are dropped.
    """
    n = len(tokens)
    if n < min_ngram * min_copies:
        return []

    max_span = min(max_span_words, n // min_copies)
    removals: list[tuple[int, int, int]] = []  # (drop_start, drop_end, span)

    for span in range(min_ngram, max_span + 1):
        i = 0
        while i + span * min_copies <= n:
            head = tokens[i : i + span]
            if any(not t for t in head):
                i += 1
                continue

            copies = 1
            j = i + span
            while j + span <= n and _span_match(head, tokens[j : j + span]):
                copies += 1
                j += span

            if copies >= min_copies:
                removals.append((i + span, j, span))
                i = j
            else:
                i += 1

    if not removals:
        return []

    # Prefer more specific (longer) spans when two candidates overlap.
    removals.sort(key=lambda r: (r[0], -r[2]))
    resolved: list[tuple[int, int]] = []
    for drop_start, drop_end, _span in removals:
        if resolved and drop_start < resolved[-1][1]:
            continue
        resolved.append((drop_start, drop_end))
    return resolved


def collapse_repeated_spans(
    text: str,
    *,
    min_ngram: int = 2,
    min_copies: int = 3,
    max_span_words: int = 40,
) -> tuple[str, bool]:
    """Collapse consecutive repeated word spans; keep the first copy.

    Returns (collapsed_text, changed).
    """
    words = text.split()
    n = len(words)
    if n < min_ngram * min_copies:
        return text, False

    tokens = [_normalize_token(w) for w in words]
    removals = find_repeated_spans(
        tokens,
        min_ngram=min_ngram,
        min_copies=min_copies,
        max_span_words=max_span_words,
    )
    if not removals:
        return text, False

    drop = [False] * n
    for drop_start, drop_end in removals:
        for k in range(drop_start, drop_end):
            drop[k] = True

    kept = [w for w, d in zip(words, drop) if not d]
    return " ".join(kept), True
