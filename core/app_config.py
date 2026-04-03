from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Type, TypeVar


T = TypeVar("T")


@dataclass
class AudioConfig:
    samplerate: int = 48000
    channels: int = 1
    blocksize: int = 1024


@dataclass
class STTConfig:
    base_url: str = "http://localhost:9000"
    ws_url: str = "ws://localhost:9000/v1/realtime?intent=transcription"
    language: str = "en-US"
    sample_rate_hz: int = 16000
    num_channels: int = 1
    timeout: float = 10.0
    commit_interval_sec: float = 0.5
    enable_automatic_punctuation: bool = True
    final_debounce_sec: float = 0.6
    partial_emit_enabled: bool = True
    partial_stability_sec: float = 0.45
    partial_min_words: int = 4
    noise_gate_threshold: float = 0.009
    noise_gate_hangover_sec: float = 0.35


@dataclass
class TranslationRuntimeConfig:
    direction: str = "en_to_ru"
    enabled: bool = True


@dataclass
class TTSRuntimeConfig:
    voice_name: str = "ru_RU-dmitri-medium"
    data_dir: str = "/media/yaroslav/DATA/ai_models/piper"
    use_cuda: bool | None = None
    max_queue_latency_sec: float = 0.75


@dataclass
class AppConfig:
    audio: AudioConfig
    stt: STTConfig
    translation: TranslationRuntimeConfig
    tts: TTSRuntimeConfig


DEFAULT_CONFIG = AppConfig(
    audio=AudioConfig(),
    stt=STTConfig(),
    translation=TranslationRuntimeConfig(),
    tts=TTSRuntimeConfig(),
)


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "app_config.json"


def load_app_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else get_default_config_path()
    if not config_path.exists():
        return DEFAULT_CONFIG

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(
        audio=_load_section(AudioConfig, payload.get("audio")),
        stt=_load_section(STTConfig, payload.get("stt")),
        translation=_load_section(TranslationRuntimeConfig, payload.get("translation")),
        tts=_load_section(TTSRuntimeConfig, payload.get("tts")),
    )


def _load_section(section_type: Type[T], payload: Any) -> T:
    payload = payload or {}
    allowed_keys = {field.name for field in fields(section_type)}
    normalized_payload = {
        key: value
        for key, value in payload.items()
        if key in allowed_keys
    }
    return section_type(**normalized_payload)
