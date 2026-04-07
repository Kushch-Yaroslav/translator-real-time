from __future__ import annotations

from dataclasses import replace

from core.app_config import DEFAULT_CONFIG
from core.audio_engine import AudioEngine
from core.sst.nemotron_realtime_stt_service import NemotronRealtimeSTTService
from core.sst.realtime_stt_factory import create_realtime_stt_service


def test_create_realtime_stt_service_returns_nemotron_backend():
    config = replace(
        DEFAULT_CONFIG,
        stt=replace(DEFAULT_CONFIG.stt, backend="nemotron"),
    )
    branch = replace(
        DEFAULT_CONFIG.branches.primary,
        translation_direction="en_to_ru",
        stt_language="en-US",
    )

    service, label = create_realtime_stt_service(config, branch)

    assert isinstance(service, NemotronRealtimeSTTService)
    assert label == "NVIDIA Nemotron streaming"


def test_audio_engine_uses_low_latency_pipeline_for_nemotron():
    config = replace(
        DEFAULT_CONFIG,
        stt=replace(DEFAULT_CONFIG.stt, backend="nemotron"),
    )

    engine = AudioEngine(config)

    assert engine._uses_low_latency_direct_pipeline() is True
