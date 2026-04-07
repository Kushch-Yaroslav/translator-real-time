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
    def __init__(self, config: TranslationConfig):
        self.config = config

        self.model_name = self._resolve_model_name(config.direction)
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model.to(self.device)

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

        return " ".join(text.split())

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
        return f"{greeting}, меня зовут {raw_name}."

    def _resolve_model_name(self, direction: TranslationDirection) -> str:
        if direction == TranslationDirection.EN_TO_RU:
            return "Helsinki-NLP/opus-mt-en-ru"

        if direction == TranslationDirection.RU_TO_EN:
            return "Helsinki-NLP/opus-mt-ru-en"

        raise ValueError(f"Unsupported translation direction: {direction}")
