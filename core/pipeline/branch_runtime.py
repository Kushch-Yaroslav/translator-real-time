from __future__ import annotations

from dataclasses import dataclass, replace

from core.config.app_config import AppConfig, TranslationBranchConfig, get_branch_config


@dataclass(frozen=True)
class BranchRuntimeProfile:
    source_branch_id: str
    runtime_branch_id: str = "primary"
    force_enabled: bool = True
    stt_backend: str | None = None
    stt_commit_interval_sec: float | None = None
    stt_final_debounce_sec: float | None = None
    stt_partial_emit_enabled: bool | None = None
    stt_partial_stability_sec: float | None = None
    stt_partial_min_words: int | None = None
    silero_partial_interval_sec: float | None = None
    silero_min_window_sec: float | None = None
    silero_max_window_sec: float | None = None
    silero_min_silence_ms: int | None = None
    silero_speech_pad_ms: int | None = None
    silero_preroll_sec: float | None = None


LISTEN_BRANCH_PROFILE = BranchRuntimeProfile(
    source_branch_id="listen",
    runtime_branch_id="primary",
    force_enabled=True,
)


SPEAK_BRANCH_PROFILE = BranchRuntimeProfile(
    source_branch_id="speak",
    runtime_branch_id="primary",
    force_enabled=True,
    stt_backend="faster_whisper",
    stt_commit_interval_sec=0.18,
    stt_final_debounce_sec=0.25,
    stt_partial_emit_enabled=True,
    stt_partial_stability_sec=0.18,
    stt_partial_min_words=2,
    silero_partial_interval_sec=0.12,
    silero_min_window_sec=0.45,
    silero_max_window_sec=5.0,
    silero_min_silence_ms=90,
    silero_speech_pad_ms=40,
    silero_preroll_sec=0.15,
)


def resolve_runtime_branch_config(
    base_config: AppConfig,
    profile: BranchRuntimeProfile,
) -> TranslationBranchConfig:
    branch = get_branch_config(base_config, profile.source_branch_id)
    if profile.force_enabled and not branch.enabled:
        branch = replace(branch, enabled=True)

    return replace(branch, branch_id=profile.runtime_branch_id)


def build_branch_runtime_config(
    base_config: AppConfig,
    profile: BranchRuntimeProfile,
) -> AppConfig:
    runtime_branch = resolve_runtime_branch_config(base_config, profile)

    stt = base_config.stt
    if profile.stt_backend is not None:
        stt = replace(stt, backend=profile.stt_backend)

    stt = replace(
        stt,
        language=runtime_branch.stt_language,
        commit_interval_sec=_coalesce(profile.stt_commit_interval_sec, stt.commit_interval_sec),
        final_debounce_sec=_coalesce(profile.stt_final_debounce_sec, stt.final_debounce_sec),
        partial_emit_enabled=_coalesce(profile.stt_partial_emit_enabled, stt.partial_emit_enabled),
        partial_stability_sec=_coalesce(profile.stt_partial_stability_sec, stt.partial_stability_sec),
        partial_min_words=_coalesce(profile.stt_partial_min_words, stt.partial_min_words),
        silero_partial_interval_sec=_coalesce(
            profile.silero_partial_interval_sec,
            stt.silero_partial_interval_sec,
        ),
        silero_min_window_sec=_coalesce(profile.silero_min_window_sec, stt.silero_min_window_sec),
        silero_max_window_sec=_coalesce(profile.silero_max_window_sec, stt.silero_max_window_sec),
        silero_min_silence_ms=_coalesce(profile.silero_min_silence_ms, stt.silero_min_silence_ms),
        silero_speech_pad_ms=_coalesce(profile.silero_speech_pad_ms, stt.silero_speech_pad_ms),
        silero_preroll_sec=_coalesce(profile.silero_preroll_sec, stt.silero_preroll_sec),
    )

    return replace(
        base_config,
        stt=stt,
        translation=replace(
            base_config.translation,
            direction=runtime_branch.translation_direction,
            enabled=runtime_branch.enabled,
        ),
        tts=replace(
            base_config.tts,
            voice_name=runtime_branch.tts_voice_name,
        ),
    )


def _coalesce(value, fallback):
    return fallback if value is None else value
