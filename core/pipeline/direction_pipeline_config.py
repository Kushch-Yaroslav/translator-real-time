from __future__ import annotations

from dataclasses import dataclass

from core.config.app_config import AppConfig, TranslationBranchConfig


@dataclass(frozen=True)
class DirectionPipelineConfig:
    direction: str
    source_language: str
    target_language: str
    log_label: str
    low_latency_enabled: bool
    preserve_ordered_chunks: bool
    collapse_queue_to_latest: bool
    final_flushes_tail: bool
    segment_delimiters: str
    segment_min_words: int
    partial_emit_enabled: bool
    partial_interval_sec: float
    partial_min_words: int
    min_window_sec: float
    min_silence_ms: int
    speech_pad_ms: int
    final_debounce_sec: float
    max_pending_partial_chunks: int = 4
    stale_partial_after_sec: float = 1.2
    partial_tts_grace_sec: float = 0.0
    stable_partial_min_sec: float = 0.45
    stable_partial_min_words: int = 3
    flow_logging_enabled: bool = False


def resolve_direction_pipeline_config(
    app_config: AppConfig,
    branch_config: TranslationBranchConfig,
) -> DirectionPipelineConfig:
    direction = (branch_config.translation_direction or "en_to_ru").strip().lower()
    if direction == "ru_to_en":
        return DirectionPipelineConfig(
            direction="ru_to_en",
            source_language="ru",
            target_language="en",
            log_label="RU→EN",
            low_latency_enabled=bool(app_config.stt.partial_emit_enabled),
            preserve_ordered_chunks=True,
            collapse_queue_to_latest=False,
            final_flushes_tail=True,
            segment_delimiters=".!?,;:",
            segment_min_words=2,
            partial_emit_enabled=bool(app_config.stt.partial_emit_enabled),
            partial_interval_sec=app_config.stt.silero_partial_interval_sec,
            partial_min_words=app_config.stt.partial_min_words,
            min_window_sec=app_config.stt.silero_min_window_sec,
            min_silence_ms=app_config.stt.silero_min_silence_ms,
            speech_pad_ms=app_config.stt.silero_speech_pad_ms,
            final_debounce_sec=app_config.stt.final_debounce_sec,
            max_pending_partial_chunks=6,
            stale_partial_after_sec=1.4,
            partial_tts_grace_sec=0.0,
            stable_partial_min_sec=0.24,
            stable_partial_min_words=2,
            flow_logging_enabled=False,
        )

    return DirectionPipelineConfig(
        direction="en_to_ru",
        source_language="en",
        target_language="ru",
        log_label="EN→RU",
        low_latency_enabled=bool(app_config.stt.partial_emit_enabled),
        preserve_ordered_chunks=True,
        collapse_queue_to_latest=False,
        final_flushes_tail=True,
        segment_delimiters=".!?,;:",
        segment_min_words=4,
        partial_emit_enabled=bool(app_config.stt.partial_emit_enabled),
        partial_interval_sec=app_config.stt.silero_partial_interval_sec,
        partial_min_words=app_config.stt.partial_min_words,
        min_window_sec=app_config.stt.silero_min_window_sec,
        min_silence_ms=app_config.stt.silero_min_silence_ms,
        speech_pad_ms=app_config.stt.silero_speech_pad_ms,
        final_debounce_sec=app_config.stt.final_debounce_sec,
        max_pending_partial_chunks=4,
        stale_partial_after_sec=1.0,
        partial_tts_grace_sec=0.40,
        stable_partial_min_sec=0.90,
        stable_partial_min_words=5,
        flow_logging_enabled=True,
    )
