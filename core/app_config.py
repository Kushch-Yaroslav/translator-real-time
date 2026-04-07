from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
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
    backend: str = "nim"
    base_url: str = "http://localhost:9000"
    ws_url: str = "ws://localhost:9000/v1/realtime?intent=transcription"
    whispercpp_base_url: str = "http://127.0.0.1:8178"
    riva_uri: str = "localhost:50051"
    riva_use_ssl: bool = False
    riva_ssl_cert_path: str = ""
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
    canary_container_id: str = "riva-asr"
    canary_tags_selector: str = "name=canary-0-6b-turbo,mode=ofl"
    canary_http_port: int = 9010
    canary_grpc_port: int = 50061
    canary_startup_timeout_sec: float = 300.0
    canary_poll_interval_sec: float = 0.35
    canary_min_window_sec: float = 0.85
    canary_finalize_silence_sec: float = 0.40
    silero_partial_interval_sec: float = 0.35
    silero_min_window_sec: float = 0.9
    silero_max_window_sec: float = 6.0
    silero_min_silence_ms: int = 180
    silero_speech_pad_ms: int = 80
    silero_preroll_sec: float = 0.25
    silero_speech_threshold: float = 0.55
    whisper_model_size: str = "large-v3-turbo"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 1
    whisper_best_of: int = 1
    whisper_patience: float = 1.0


@dataclass
class TranslationRuntimeConfig:
    direction: str = "en_to_ru"
    enabled: bool = True


@dataclass
class TranslationBranchConfig:
    branch_id: str = "primary"
    label: str = "EN -> RU"
    enabled: bool = True
    stt_language: str = "en-US"
    translation_direction: str = "en_to_ru"
    tts_voice_name: str = "ru_RU-dmitri-medium"
    nim_container_id: str = "parakeet-1-1b-ctc-en-us"
    nim_tags_selector: str = "name=parakeet-1-1b-ctc-en-us,mode=str,diarizer=disabled,vad=default"
    nim_startup_timeout_sec: float = 60.0


@dataclass
class BranchesConfig:
    primary: TranslationBranchConfig
    secondary: TranslationBranchConfig


@dataclass
class AppRuntimeConfig:
    conversation_mode: str = "all"


@dataclass
class TTSRuntimeConfig:
    voice_name: str = "ru_RU-dmitri-medium"
    data_dir: str = "/media/yaroslav/DATA/ai_models/piper"
    use_cuda: bool | None = None
    max_queue_latency_sec: float = 0.75


@dataclass
class AppConfig:
    runtime: AppRuntimeConfig
    audio: AudioConfig
    stt: STTConfig
    branches: BranchesConfig
    translation: TranslationRuntimeConfig
    tts: TTSRuntimeConfig


DEFAULT_CONFIG = AppConfig(
    runtime=AppRuntimeConfig(),
    audio=AudioConfig(),
    stt=STTConfig(),
    branches=BranchesConfig(
        primary=TranslationBranchConfig(),
        secondary=TranslationBranchConfig(
            branch_id="secondary",
            label="RU -> EN",
            enabled=False,
            stt_language="ru-RU",
            translation_direction="ru_to_en",
            tts_voice_name="en_US-ryan-medium",
            nim_container_id="parakeet-1-1b-rnnt-multilingual",
            nim_tags_selector="mode=str",
            nim_startup_timeout_sec=180.0,
        ),
    ),
    translation=TranslationRuntimeConfig(),
    tts=TTSRuntimeConfig(),
)


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "app_config.json"


def get_profiles_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "profiles"


def load_app_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else get_default_config_path()
    if not config_path.exists():
        return DEFAULT_CONFIG

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    stt_config = _load_section(STTConfig, payload.get("stt"))
    translation_config = _load_section(TranslationRuntimeConfig, payload.get("translation"))
    tts_config = _load_section(TTSRuntimeConfig, payload.get("tts"))

    branches_payload = payload.get("branches") or {}
    primary_branch = _load_section(
        TranslationBranchConfig,
        branches_payload.get("primary") or {
            "branch_id": "primary",
            "label": _resolve_branch_label(translation_config.direction),
            "enabled": translation_config.enabled,
            "stt_language": stt_config.language,
            "translation_direction": translation_config.direction,
            "tts_voice_name": tts_config.voice_name,
        },
    )
    secondary_branch = _load_section(
        TranslationBranchConfig,
        branches_payload.get("secondary") or asdict(DEFAULT_CONFIG.branches.secondary),
    )

    return AppConfig(
        runtime=_load_section(AppRuntimeConfig, payload.get("runtime")),
        audio=_load_section(AudioConfig, payload.get("audio")),
        stt=stt_config,
        branches=BranchesConfig(
            primary=primary_branch,
            secondary=secondary_branch,
        ),
        translation=translation_config,
        tts=tts_config,
    )


def save_app_config(config: AppConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path) if path else get_default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path


def list_profile_paths() -> list[Path]:
    profiles_dir = get_profiles_dir()
    if not profiles_dir.exists():
        return []

    return sorted(profiles_dir.glob("*.json"))


def get_primary_branch_config(config: AppConfig) -> TranslationBranchConfig:
    return config.branches.primary


def _resolve_branch_label(direction: str) -> str:
    if direction == "ru_to_en":
        return "RU -> EN"
    return "EN -> RU"


def _load_section(section_type: Type[T], payload: Any) -> T:
    payload = payload or {}
    allowed_keys = {field.name for field in fields(section_type)}
    normalized_payload = {
        key: value
        for key, value in payload.items()
        if key in allowed_keys
    }
    return section_type(**normalized_payload)
