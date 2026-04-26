from core.config.app_config import DEFAULT_CONFIG
from core.pipeline.branch_definitions import get_default_lane_definitions
from core.pipeline.branch_registry import BranchRegistry
from core.pipeline.pipeline_orchestrator import PipelineOrchestrator


class FakeEngine:
    def __init__(self, lane_key: str):
        self.lane_key = lane_key
        self.running = False
        self.on_log = None
        self.on_error = None
        self.on_input_level = None

    def stop(self) -> None:
        self.running = False


def test_pipeline_orchestrator_reassigns_lane_and_rebinds_callbacks() -> None:
    registry = BranchRegistry(DEFAULT_CONFIG, get_default_lane_definitions())
    created_engines: list[FakeEngine] = []

    def engine_factory(lane_key: str) -> FakeEngine:
        engine = FakeEngine(lane_key)
        created_engines.append(engine)
        return engine

    orchestrator = PipelineOrchestrator(registry, engine_factory)
    messages: list[str] = []
    errors: list[str] = []
    levels: list[float] = []

    orchestrator.bind_lane_callbacks(
        "listen",
        emit_log=messages.append,
        emit_error=errors.append,
        emit_input_level=levels.append,
    )

    original = orchestrator.get_controller("listen")
    original.active = False
    replacement = orchestrator.reassign_lane_branch("listen", "listen")

    assert replacement is orchestrator.get_controller("listen")
    assert replacement is not original
    assert replacement.active is False

    replacement.engine.on_log("ready")
    replacement.engine.on_error("fail")
    replacement.engine.on_input_level(0.42)

    assert messages == ["[LISTEN] ready"]
    assert errors == ["[LISTEN] fail"]
    assert levels == [0.42]
    assert len(created_engines) == 3
