from __future__ import annotations

from dataclasses import replace

from core.config.app_config import AppConfig, TranslationBranchConfig, get_branch_config
from core.pipeline.branch_definitions import PipelineLaneDefinition
from core.pipeline.branch_runtime import (
    BranchRuntimeProfile,
    build_branch_runtime_config,
    resolve_runtime_branch_config,
)


class BranchRegistry:
    def __init__(
        self,
        base_config: AppConfig,
        lane_definitions: tuple[PipelineLaneDefinition, ...],
    ):
        self.base_config = base_config
        self._lane_definitions = {
            definition.lane_key: definition
            for definition in lane_definitions
        }
        self._lane_assignments = {
            definition.lane_key: definition.default_branch_id
            for definition in lane_definitions
        }

    def list_lane_definitions(self) -> tuple[PipelineLaneDefinition, ...]:
        return tuple(self._lane_definitions.values())

    def list_branch_configs(self) -> tuple[TranslationBranchConfig, ...]:
        return self.base_config.branches

    def get_lane_definition(self, lane_key: str) -> PipelineLaneDefinition:
        return self._lane_definitions[lane_key]

    def get_assigned_branch_id(self, lane_key: str) -> str:
        return self._lane_assignments[lane_key]

    def get_assigned_branch_config(self, lane_key: str) -> TranslationBranchConfig:
        return get_branch_config(self.base_config, self.get_assigned_branch_id(lane_key))

    def assign_branch(self, lane_key: str, branch_id: str) -> TranslationBranchConfig:
        branch_config = get_branch_config(self.base_config, branch_id)
        self._lane_assignments[lane_key] = branch_config.branch_id
        return branch_config

    def build_runtime_profile(self, lane_key: str) -> BranchRuntimeProfile:
        lane_definition = self.get_lane_definition(lane_key)
        return replace(
            lane_definition.runtime_profile,
            source_branch_id=self.get_assigned_branch_id(lane_key),
        )

    def build_runtime_config(self, lane_key: str) -> AppConfig:
        return build_branch_runtime_config(
            self.base_config,
            self.build_runtime_profile(lane_key),
        )

    def resolve_runtime_branch_config(self, lane_key: str) -> TranslationBranchConfig:
        return resolve_runtime_branch_config(
            self.base_config,
            self.build_runtime_profile(lane_key),
        )
