from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from core.pipeline.branch_controller import BranchController
from core.pipeline.branch_registry import BranchRegistry

if TYPE_CHECKING:
    from core.audio.audio_engine import AudioEngine


class PipelineOrchestrator:
    def __init__(
        self,
        branch_registry: BranchRegistry,
        engine_factory: Callable[[str], AudioEngine],
    ):
        self.branch_registry = branch_registry
        self._engine_factory = engine_factory
        self._controllers = {
            lane_definition.lane_key: BranchController(
                lane_definition,
                engine_factory(lane_definition.lane_key),
            )
            for lane_definition in branch_registry.list_lane_definitions()
        }
        self._callback_bindings: dict[str, tuple[Callable[[str], None], Callable[[str], None], Callable[[float], None]]] = {}

    def list_controllers(self) -> tuple[BranchController, ...]:
        return tuple(self._controllers.values())

    def get_controller(self, lane_key: str) -> BranchController:
        return self._controllers[lane_key]

    def bind_lane_callbacks(
        self,
        lane_key: str,
        *,
        emit_log: Callable[[str], None],
        emit_error: Callable[[str], None],
        emit_input_level: Callable[[float], None],
    ) -> None:
        self._callback_bindings[lane_key] = (emit_log, emit_error, emit_input_level)
        self.get_controller(lane_key).bind_callbacks(
            emit_log=emit_log,
            emit_error=emit_error,
            emit_input_level=emit_input_level,
        )

    def reassign_lane_branch(
        self,
        lane_key: str,
        branch_id: str,
    ) -> BranchController:
        controller = self.get_controller(lane_key)
        was_running = controller.engine.running
        was_active = controller.active

        if was_running:
            controller.stop()

        self.branch_registry.assign_branch(lane_key, branch_id)
        replacement = BranchController(
            self.branch_registry.get_lane_definition(lane_key),
            self._engine_factory(lane_key),
        )
        replacement.active = was_active
        self._controllers[lane_key] = replacement

        callback_binding = self._callback_bindings.get(lane_key)
        if callback_binding is not None:
            emit_log, emit_error, emit_input_level = callback_binding
            replacement.bind_callbacks(
                emit_log=emit_log,
                emit_error=emit_error,
                emit_input_level=emit_input_level,
            )

        return replacement
