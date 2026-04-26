from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DICTIONARY_PATH = _REPO_ROOT / "dictionary" / "dictionary.md"


@dataclass(frozen=True)
class TermReplacement:
    source: str
    target: str


class TermDictionary:
    def __init__(self, path: Path | None = None):
        self.path = path or _DEFAULT_DICTIONARY_PATH
        self._cached_mtime_ns: int | None = None
        self._cached_replacements: tuple[TermReplacement, ...] = ()

    def normalize_ru_to_en_source_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        replacements = self._load_replacements()
        normalized = text
        for replacement in replacements:
            pattern = self._build_term_pattern(replacement.source)
            normalized = re.sub(
                pattern,
                replacement.target,
                normalized,
                flags=re.IGNORECASE,
            )

        return " ".join(normalized.split())

    def _load_replacements(self) -> tuple[TermReplacement, ...]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._cached_mtime_ns = None
            self._cached_replacements = ()
            return self._cached_replacements

        if self._cached_mtime_ns == stat.st_mtime_ns:
            return self._cached_replacements

        replacements: dict[str, str] = {}
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line == "---":
                continue
            if line.startswith("- "):
                continue
            if re.match(r"^\d+\.\s+", line):
                continue

            source, target = self._parse_dictionary_line(line)
            if not source or not target:
                continue

            replacements[source.lower()] = target

        ordered = sorted(
            replacements.items(),
            key=lambda item: (-len(item[0]), item[0]),
        )
        self._cached_mtime_ns = stat.st_mtime_ns
        self._cached_replacements = tuple(
            TermReplacement(source=source, target=target)
            for source, target in ordered
        )
        return self._cached_replacements

    @staticmethod
    def _parse_dictionary_line(line: str) -> tuple[str, str]:
        if "->" in line:
            source, target = line.split("->", 1)
            return source.strip(), target.strip()
        if "→" in line:
            source, target = line.split("→", 1)
            return source.strip(), target.strip()
        return line.strip(), line.strip()

    @staticmethod
    def _build_term_pattern(term: str) -> str:
        escaped = re.escape(term.strip())
        return rf"(?<![0-9A-Za-zА-Яа-яЁё]){escaped}(?![0-9A-Za-zА-Яа-яЁё])"
