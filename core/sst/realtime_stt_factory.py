from __future__ import annotations

from core.app_config import AppConfig, TranslationBranchConfig
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
