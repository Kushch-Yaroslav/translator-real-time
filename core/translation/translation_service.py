from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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

    def translate(self, text: str) -> str:
        text = (text or "").strip()

        if not self.config.enabled or not text:
            return text

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

        return translated[0].strip() if translated else ""

    def _resolve_model_name(self, direction: TranslationDirection) -> str:
        if direction == TranslationDirection.EN_TO_RU:
            return "Helsinki-NLP/opus-mt-en-ru"

        if direction == TranslationDirection.RU_TO_EN:
            return "Helsinki-NLP/opus-mt-ru-en"

        raise ValueError(f"Unsupported translation direction: {direction}")