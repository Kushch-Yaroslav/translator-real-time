from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from core.pipeline.branch_definitions import PipelineLaneDefinition

if TYPE_CHECKING:
    from core.audio.audio_engine import AudioEngine


class BranchController:
    def __init__(self, definition: PipelineLaneDefinition, engine: AudioEngine):
        self.definition = definition
        self.engine = engine
        self.active = True
        self._emit_log: Callable[[str], None] | None = None
        self._emit_error: Callable[[str], None] | None = None

    def bind_callbacks(
        self,
        *,
        emit_log: Callable[[str], None],
        emit_error: Callable[[str], None],
        emit_input_level: Callable[[float], None],
    ) -> None:
        self._emit_log = emit_log
        self._emit_error = emit_error
        self.engine.on_log = lambda message: emit_log(f"{self.definition.log_prefix} {message}")
        self.engine.on_error = lambda message: emit_error(f"{self.definition.log_prefix} {message}")
        self.engine.on_input_level = emit_input_level

    def start(self, *, input_name: str, output_name: str) -> None:
        import sounddevice as sd

        from core.audio.audio_service import temporary_pulse_stream_properties
        from core.audio.audio_utils import find_sounddevice_device_index_by_name

        input_index = find_sounddevice_device_index_by_name(
            input_name,
            min_input_channels=1,
            prefer_pulse=True,
        )
        output_index = find_sounddevice_device_index_by_name(
            output_name,
            min_output_channels=1,
            prefer_pulse=True,
        )

        if input_index is None:
            raise RuntimeError(
                f"Не удалось сопоставить {self.definition.lane_key} input: {input_name}"
            )
        if output_index is None:
            raise RuntimeError(
                f"Не удалось сопоставить {self.definition.lane_key} output: {output_name}"
            )

        input_sd_name = sd.query_devices(input_index)["name"]
        output_sd_name = sd.query_devices(output_index)["name"]

        self._log(f"Selected pactl input: {input_name}")
        self._log(f"Selected pactl output: {output_name}")
        self._log(f"Mapped sounddevice input index: {input_index}, name: {input_sd_name}")
        self._log(f"Mapped sounddevice output index: {output_index}, name: {output_sd_name}")

        with temporary_pulse_stream_properties(
            application_name=self.definition.stream_tag,
            media_name=self.definition.stream_tag,
        ):
            self.engine.start(
                input_device_index=input_index,
                output_device_index=output_index,
                selected_pactl_input_name=input_name,
                selected_pactl_output_name=output_name,
                samplerate=self.engine.app_config.audio.samplerate,
                channels=self.engine.app_config.audio.channels,
                blocksize=self.engine.app_config.audio.blocksize,
                stt_window_seconds=1.0,
                pulse_stream_tag=self.definition.stream_tag,
            )

    def stop(self) -> None:
        self.engine.stop()

    def set_paused(self, paused: bool) -> None:
        self.engine.set_translation_paused(paused)
        self.active = not paused

    def prewarm_runtime(self) -> None:
        self.engine.prewarm_runtime()

    def _log(self, message: str) -> None:
        if self._emit_log is not None:
            self._emit_log(f"{self.definition.log_prefix} {message}")
