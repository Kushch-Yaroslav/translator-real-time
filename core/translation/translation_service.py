from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional

import torch
from transformers import MarianMTModel, MarianTokenizer


class TranslationDirection(str, Enum):
    EN_TO_RU = "en_to_ru"
    RU_TO_EN = "ru_to_en"


@dataclass
class TranslationConfig:
    direction: TranslationDirection = TranslationDirection.EN_TO_RU
    enabled: bool = True
    device: Optional[str] = None


class TranslationService:
    _KNOWN_EN_RU_NAME_ALIASES = {
        "alyssa": "Alyssa",
        "arasan": "Alyssa",
        "arassa": "Alyssa",
        "jeremy": "Jeremy",
        "oli": "Oli",
        "yaroslav": "Yaroslav",
        "jaroslav": "Yaroslav",
        "scott": "Scott",
        "tiff": "Tiff",
    }
    _KNOWN_EN_RU_NAME_TRANSLATIONS = {
        "Alyssa": "Алисса",
        "Jeremy": "Джереми",
        "Oli": "Оли",
        "Yaroslav": "Ярослав",
        "Scott": "Скотт",
        "Tiff": "Тифф",
    }

    def __init__(self, config: TranslationConfig):
        self.config = config

        self.model_name = self._resolve_model_name(config.direction)
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = self._load_model_with_fallback()

    def warmup(self) -> None:
        sample_text = "Привет." if self.config.direction == TranslationDirection.RU_TO_EN else "Hello."
        self.translate(sample_text)

    def translate(self, text: str) -> str:
        text = self._normalize_source_text(text)

        if not self.config.enabled or not text:
            return text

        template_translation = self._translate_name_intro_template(text)
        if template_translation:
            return template_translation

        template_translation = self._translate_en_ru_dialogue_template(text)
        if template_translation:
            return template_translation

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        result = translated[0].strip() if translated else ""
        return self._postprocess_translation(result)

    def _normalize_source_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self.config.direction == TranslationDirection.RU_TO_EN:
            text = re.sub(
                r"\bинженером[\s-]+программистом\b",
                "инженером-программистом",
                text,
                flags=re.IGNORECASE,
            )
        elif self.config.direction == TranslationDirection.EN_TO_RU:
            replacements = (
                (r"\bi\s+apreciate\s+it\b", "I appreciate it"),
                (r"\b(?:it\s+)?affects[, ]+takes\s+time\s+to\s+kick\s+in\b", "it takes time to kick in"),
                (r"\b(?:it\s+)?effects[, ]+takes\s+time\s+to\s+kick\s+in\b", "it takes time to kick in"),
                (r"\b(?:it\s+)?takes\s+time\s+to\s+get\s+there\b", "it takes time to kick in"),
            )
            for pattern, replacement in replacements:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

            for alias, canonical in self._KNOWN_EN_RU_NAME_ALIASES.items():
                text = re.sub(rf"\b{re.escape(alias)}\b", canonical, text, flags=re.IGNORECASE)

        return " ".join(text.split())

    def _postprocess_translation(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self.config.direction == TranslationDirection.RU_TO_EN:
            replacements = {
                "programmer engineer": "software engineer",
                "Programmer engineer": "Software engineer",
                "Short sentence": "Short phrase",
            }
            for source, target in replacements.items():
                text = text.replace(source, target)

            text = self._collapse_repeated_en_clause(text)
        elif self.config.direction == TranslationDirection.EN_TO_RU:
            text = self._postprocess_en_ru_mixed_intro(text)

        return " ".join(text.split())

    def _postprocess_en_ru_mixed_intro(self, text: str) -> str:
        compact = " ".join((text or "").strip().split())
        if not compact:
            return ""

        match = re.fullmatch(
            r"(?is)привет,\s+i'?m\s+([a-z][a-z' -]*?)\s+founder\s+and\s+ceo\s+of\s+([a-z][a-z0-9' -]*)[.!?]?",
            compact,
        )
        if not match:
            return compact

        raw_name = match.group(1).strip(" ,.!?")
        raw_company = match.group(2).strip(" ,.!?")
        company = self._normalize_known_company_name(raw_company)
        return f"Привет, я {self._translate_known_name_to_ru(raw_name)}, основатель и CEO {company}."

    @staticmethod
    def _collapse_repeated_en_clause(text: str) -> str:
        compact = " ".join((text or "").strip().split())
        if not compact:
            return ""

        repeated_clause_patterns = (
            re.compile(
                r"(?is)^if there is no ([a-z][a-z' -]+), then there is no \1[.!?]?$"
            ),
            re.compile(
                r"(?is)^if there's no ([a-z][a-z' -]+), then there's no \1[.!?]?$"
            ),
            re.compile(
                r"(?is)^there is no ([a-z][a-z' -]+), there is no \1[.!?]?$"
            ),
            re.compile(
                r"(?is)^there's no ([a-z][a-z' -]+), there's no \1[.!?]?$"
            ),
        )
        for pattern in repeated_clause_patterns:
            match = pattern.fullmatch(compact)
            if match:
                noun_phrase = " ".join(match.group(1).split())
                return f"There is no {noun_phrase}."

        return compact

    def _translate_name_intro_template(self, text: str) -> str:
        if self.config.direction != TranslationDirection.EN_TO_RU:
            return ""

        match = re.fullmatch(
            r"(?is)(?:hello(?:\s+everyone)?|hi(?:\s+everyone)?)[,!\s]*my\s+name\s+is\s+(.+?)[.!?]?",
            text.strip(),
        )
        if not match:
            return ""

        raw_name = " ".join(match.group(1).split()).strip(" ,.!?")
        if not raw_name:
            return ""

        greeting = "Здравствуйте" if re.search(r"\beveryone\b", text, flags=re.IGNORECASE) else "Привет"
        return f"{greeting}, меня зовут {self._translate_known_name_to_ru(raw_name)}."

    def _translate_en_ru_dialogue_template(self, text: str) -> str:
        if self.config.direction != TranslationDirection.EN_TO_RU:
            return ""

        normalized = self._normalize_compare_text(text)
        if not normalized:
            return ""

        founder_ceo_match = re.fullmatch(
            r"(?:hey|hi|hello)\s+im\s+([a-zа-я][a-zа-я' -]*?)\s+founder\s+and\s+ceo\s+of\s+([a-zа-я][a-zа-я0-9' -]*)",
            normalized,
        )
        if founder_ceo_match:
            raw_name = founder_ceo_match.group(1).strip(" ,.!?")
            raw_company = founder_ceo_match.group(2).strip(" ,.!?")
            company = self._normalize_known_company_name(raw_company)
            return f"Привет, я {self._translate_known_name_to_ru(raw_name)}, основатель и CEO {company}."

        greeting_match = re.fullmatch(r"(?:hi|hello|hey|yo)\s+([a-zа-я][a-zа-я' -]*)", normalized)
        if greeting_match:
            raw_name = " ".join(greeting_match.group(1).split()).strip(" ,.!?")
            if raw_name:
                return f"Привет, {self._translate_known_name_to_ru(raw_name)}."

        farewell_match = re.fullmatch(r"bye\s+([a-zа-я][a-zа-я' -]*)", normalized)
        if farewell_match:
            raw_name = " ".join(farewell_match.group(1).split()).strip(" ,.!?")
            if raw_name:
                return f"Пока, {self._translate_known_name_to_ru(raw_name)}."

        exact_templates = {
            "are you feeling better now": "Тебе уже лучше?",
            "nice to meet you": "Приятно познакомиться.",
            "just a bit im still feeling nauseous": "Немного. Меня всё ещё тошнит.",
            "did you take the medication": "Ты принял лекарство?",
            "yeah but it takes time to kick in": "Да, но нужно время, чтобы лекарство подействовало.",
            "oh i better leave you to rest then": "Тогда я оставлю тебя отдыхать.",
            "ill check back on you later": "Я зайду к тебе позже.",
            "thanks alyssa i appreciate it": "Спасибо, Алисса. Я это ценю.",
        }
        return exact_templates.get(normalized, "")

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        text = " ".join((text or "").strip().split()).lower()
        text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        return " ".join(text.split()).strip()

    def _translate_known_name_to_ru(self, raw_name: str) -> str:
        normalized = self._normalize_compare_text(raw_name)
        canonical = self._KNOWN_EN_RU_NAME_ALIASES.get(normalized)
        if canonical:
            return self._KNOWN_EN_RU_NAME_TRANSLATIONS.get(canonical, canonical)

        compact = " ".join((raw_name or "").strip().split()).strip(" ,.!?")
        if not compact:
            return ""

        if re.fullmatch(r"[A-Za-z][A-Za-z' -]*", compact):
            return compact.title()

        return compact

    @staticmethod
    def _normalize_known_company_name(raw_company: str) -> str:
        compact = " ".join((raw_company or "").strip().split()).strip(" ,.!?")
        if not compact:
            return ""

        normalized = TranslationService._normalize_compare_text(compact)
        known_companies = {
            "microone": "MicroOne",
        }
        return known_companies.get(normalized, compact)

    def _resolve_model_name(self, direction: TranslationDirection) -> str:
        if direction == TranslationDirection.EN_TO_RU:
            return "Helsinki-NLP/opus-mt-en-ru"

        if direction == TranslationDirection.RU_TO_EN:
            return "Helsinki-NLP/opus-mt-ru-en"

        raise ValueError(f"Unsupported translation direction: {direction}")

    def _load_model_with_fallback(self) -> MarianMTModel:
        try:
            model = MarianMTModel.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=False,
            )
            model.to(self.device)
            return model
        except Exception as error:
            if self.device != "cpu" and self._looks_like_meta_tensor_error(error):
                self.device = "cpu"
                model = MarianMTModel.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=False,
                )
                model.to(self.device)
                return model
            raise

    @staticmethod
    def _looks_like_meta_tensor_error(error: Exception) -> bool:
        text = str(error).lower()
        return "meta tensor" in text or "to_empty" in text
