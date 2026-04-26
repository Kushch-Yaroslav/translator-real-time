from __future__ import annotations

from core.config.app_config import AppConfig, TranslationBranchConfig
from core.sst.confirm_stt_service import ConfirmSTTService
from core.sst.stt_service import STTService


def create_confirm_stt_service(
    app_config: AppConfig,
    branch_config: TranslationBranchConfig,
    on_log=None,
) -> ConfirmSTTService:
    backend = (app_config.stt.backend or "nim").strip().lower()

    if backend == "riva":
        # Riva streaming is used only for low-latency boundaries here.
        # Confirm-pass runs through the HTTP NIM endpoint because the current
        # local Riva deployment does not expose a Russian offline ASR model.
        return STTService(
            base_url=app_config.stt.base_url,
            language=branch_config.stt_language,
            timeout=max(20.0, app_config.stt.timeout),
            target_samplerate=app_config.stt.sample_rate_hz,
        )

    return STTService(
        base_url=app_config.stt.base_url,
        language=branch_config.stt_language,
        timeout=max(20.0, app_config.stt.timeout),
        target_samplerate=app_config.stt.sample_rate_hz,
    )
