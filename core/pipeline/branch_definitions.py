from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.config.app_config import LISTEN_BRANCH_ID, SPEAK_BRANCH_ID
from core.pipeline.branch_runtime import LISTEN_BRANCH_PROFILE, SPEAK_BRANCH_PROFILE, BranchRuntimeProfile


BranchInputRouteRole = Literal["real_source", "listen_monitor"]
BranchOutputRouteRole = Literal["real_sink", "translator_sink"]


@dataclass(frozen=True)
class PipelineLaneDefinition:
    lane_key: str
    default_branch_id: str
    title: str
    runtime_profile: BranchRuntimeProfile
    log_prefix: str
    stream_tag: str
    input_route_role: BranchInputRouteRole
    output_route_role: BranchOutputRouteRole
    active_status_text: str
    paused_status_text: str
    paused_log_text: str
    resumed_log_text: str
    backend_id: str | None = None
    backend_title: str | None = None


BranchDefinition = PipelineLaneDefinition


LISTEN_LANE_DEFINITION = PipelineLaneDefinition(
    lane_key="listen",
    default_branch_id=LISTEN_BRANCH_ID,
    title="Слушать EN=>RU",
    runtime_profile=LISTEN_BRANCH_PROFILE,
    log_prefix="[LISTEN]",
    stream_tag="TranslatorListenEngine",
    input_route_role="listen_monitor",
    output_route_role="real_sink",
    active_status_text="EN=>RU активно",
    paused_status_text="EN=>RU на паузе",
    paused_log_text="Listen EN=>RU paused",
    resumed_log_text="Listen EN=>RU resumed",
    backend_id="en_to_ru",
    backend_title="EN=>RU whisper.cpp",
)


SPEAK_LANE_DEFINITION = PipelineLaneDefinition(
    lane_key="speak",
    default_branch_id=SPEAK_BRANCH_ID,
    title="Говорить RU=>EN",
    runtime_profile=SPEAK_BRANCH_PROFILE,
    log_prefix="[SPEAK]",
    stream_tag="TranslatorSpeakEngine",
    input_route_role="real_source",
    output_route_role="translator_sink",
    active_status_text="RU=>EN активно",
    paused_status_text="RU=>EN на паузе",
    paused_log_text="Speak RU=>EN paused",
    resumed_log_text="Speak RU=>EN resumed",
    backend_id="ru_to_en",
    backend_title="RU=>EN faster-whisper",
)


DEFAULT_LANE_DEFINITIONS = (
    LISTEN_LANE_DEFINITION,
    SPEAK_LANE_DEFINITION,
)


LISTEN_BRANCH_DEFINITION = LISTEN_LANE_DEFINITION
SPEAK_BRANCH_DEFINITION = SPEAK_LANE_DEFINITION


def get_default_lane_definitions() -> tuple[PipelineLaneDefinition, ...]:
    return DEFAULT_LANE_DEFINITIONS


def get_default_branch_definitions() -> tuple[PipelineLaneDefinition, ...]:
    return get_default_lane_definitions()
