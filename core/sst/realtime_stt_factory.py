from __future__ import annotations

from core.app_config import AppConfig, TranslationBranchConfig
from core.sst.canary_ast_realtime_service import (
    CanaryASTRealtimeConfig,
    CanaryASTRealtimeService,
)
from core.sst.faster_whisper_realtime_stt_service import (
    FasterWhisperRealtimeSTTConfig,
    FasterWhisperRealtimeSTTService,
)
from core.sst.nim_realtime_stt_service import (
    NIMRealtimeSTTConfig,
    NIMRealtimeSTTService,
)
from core.sst.realtime_stt_protocol import RealtimeSTTService
from core.sst.riva_realtime_stt_service import (
    RivaRealtimeSTTConfig,
    RivaRealtimeSTTService,
)


def create_realtime_stt_service(
    app_config: AppConfig,
    branch_config: TranslationBranchConfig,
    on_log=None,
) -> tuple[RealtimeSTTService, str]:
    backend = (app_config.stt.backend or "nim").strip().lower()

    if backend == "faster_whisper":
        return (
            FasterWhisperRealtimeSTTService(
                FasterWhisperRealtimeSTTConfig(
                    language="ru" if branch_config.stt_language.lower().startswith("ru") else "en",
                    sample_rate_hz=app_config.stt.sample_rate_hz,
                    partial_interval_sec=app_config.stt.silero_partial_interval_sec,
                    min_window_sec=app_config.stt.silero_min_window_sec,
                    max_window_sec=app_config.stt.silero_max_window_sec,
                    min_silence_duration_ms=app_config.stt.silero_min_silence_ms,
                    speech_pad_ms=app_config.stt.silero_speech_pad_ms,
                    speech_threshold=app_config.stt.silero_speech_threshold,
                    whisper_model_size=app_config.stt.whisper_model_size,
                    compute_type=app_config.stt.whisper_compute_type,
                    beam_size=app_config.stt.whisper_beam_size,
                    best_of=app_config.stt.whisper_best_of,
                    patience=app_config.stt.whisper_patience,
                    on_log=on_log,
                )
            ),
            "Silero VAD + faster-whisper",
        )

    if backend == "canary_ast":
        target_language = "en-US"
        if (branch_config.translation_direction or "").strip().lower() == "en_to_ru":
            target_language = "ru-RU"

        return (
            CanaryASTRealtimeService(
                CanaryASTRealtimeConfig(
                    base_url=f"http://localhost:{app_config.stt.canary_http_port}",
                    source_language=branch_config.stt_language,
                    target_language=target_language,
                    sample_rate_hz=app_config.stt.sample_rate_hz,
                    timeout=max(20.0, app_config.stt.timeout),
                    poll_interval_sec=app_config.stt.canary_poll_interval_sec,
                    min_window_sec=app_config.stt.canary_min_window_sec,
                    finalize_silence_sec=app_config.stt.canary_finalize_silence_sec,
                    on_log=on_log,
                )
            ),
            "NVIDIA Canary AST",
        )

    if backend == "riva":
        return (
            RivaRealtimeSTTService(
                RivaRealtimeSTTConfig(
                    uri=app_config.stt.riva_uri,
                    language=branch_config.stt_language,
                    sample_rate_hz=app_config.stt.sample_rate_hz,
                    num_channels=app_config.stt.num_channels,
                    timeout=app_config.stt.timeout,
                    enable_automatic_punctuation=app_config.stt.enable_automatic_punctuation,
                    use_ssl=app_config.stt.riva_use_ssl,
                    ssl_cert_path=app_config.stt.riva_ssl_cert_path,
                    on_log=on_log,
                )
            ),
            "NVIDIA Riva gRPC",
        )

    return (
        NIMRealtimeSTTService(
            NIMRealtimeSTTConfig(
                base_url=app_config.stt.base_url,
                ws_url=app_config.stt.ws_url,
                language=branch_config.stt_language,
                sample_rate_hz=app_config.stt.sample_rate_hz,
                num_channels=app_config.stt.num_channels,
                timeout=app_config.stt.timeout,
                commit_interval_sec=app_config.stt.commit_interval_sec,
                enable_automatic_punctuation=app_config.stt.enable_automatic_punctuation,
                on_log=on_log,
            )
        ),
        "NVIDIA NIM WebSocket",
    )
