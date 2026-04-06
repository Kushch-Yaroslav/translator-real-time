from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


TRANSLATOR_SINK_NAME = "translator_mic"
TRANSLATOR_SINK_DESCRIPTION = "TranslatorMic"


@dataclass
class AudioDevice:
    id: str
    name: str
    description: str
    is_default: bool = False


def _run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


def _safe_get_props(item: dict) -> dict:
    return item.get("properties", {}) if isinstance(item, dict) else {}


def _extract_description(item: dict) -> str:
    props = _safe_get_props(item)
    return (
            props.get("device.description")
            or item.get("description")
            or item.get("name")
            or "Unknown device"
    )


def list_input_devices() -> List[AudioDevice]:
    output = _run_command(["pactl", "-f", "json", "list", "sources"])
    raw_devices = json.loads(output)

    devices: List[AudioDevice] = []
    for item in raw_devices:
        name = item.get("name", "")
        description = _extract_description(item)

        devices.append(
            AudioDevice(
                id=name,
                name=name,
                description=description,
                is_default=False,
            )
        )

    return devices


def list_output_devices() -> List[AudioDevice]:
    output = _run_command(["pactl", "-f", "json", "list", "sinks"])
    raw_devices = json.loads(output)

    devices: List[AudioDevice] = []
    for item in raw_devices:
        name = item.get("name", "")
        description = _extract_description(item)

        devices.append(
            AudioDevice(
                id=name,
                name=name,
                description=description,
                is_default=False,
            )
        )

    return devices


def get_default_source_name() -> Optional[str]:
    try:
        return _run_command(["pactl", "get-default-source"]).strip()
    except Exception:
        return None


def get_default_sink_name() -> Optional[str]:
    try:
        return _run_command(["pactl", "get-default-sink"]).strip()
    except Exception:
        return None


def enrich_default_flags(inputs: List[AudioDevice], outputs: List[AudioDevice]) -> None:
    default_source = get_default_source_name()
    default_sink = get_default_sink_name()

    for device in inputs:
        if device.name == default_source:
            device.is_default = True

    for device in outputs:
        if device.name == default_sink:
            device.is_default = True


def ensure_translator_sink_exists() -> None:
    sinks = list_output_devices()
    exists = any(device.name == TRANSLATOR_SINK_NAME for device in sinks)

    if exists:
        return

    subprocess.run(
        [
            "pactl",
            "load-module",
            "module-null-sink",
            f"sink_name={TRANSLATOR_SINK_NAME}",
            f"sink_properties=device.description={TRANSLATOR_SINK_DESCRIPTION}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def get_translator_sink_name() -> str:
    ensure_translator_sink_exists()
    return TRANSLATOR_SINK_NAME


def get_monitor_source_name_for_sink(sink_name: str) -> str:
    return f"{sink_name}.monitor"


def _list_sink_inputs() -> list[dict]:
    output = _run_command(["pactl", "-f", "json", "list", "sink-inputs"])
    return json.loads(output)


def _list_source_outputs() -> list[dict]:
    output = _run_command(["pactl", "-f", "json", "list", "source-outputs"])
    return json.loads(output)


def _format_stream_debug_line(item: dict) -> str:
    props = _safe_get_props(item)
    index = item.get("index", "?")
    name = item.get("name", "") or props.get("node.name", "")
    app_name = props.get("application.name", "")
    media_name = props.get("media.name", "")
    binary_name = props.get("application.process.binary", "")
    source_name = item.get("source", "") or item.get("source_name", "")
    sink_name = item.get("sink", "") or item.get("sink_name", "")
    return (
        f"index={index} name='{name}' app='{app_name}' media='{media_name}' "
        f"binary='{binary_name}' source='{source_name}' sink='{sink_name}'"
    )


def _find_our_pulse_sink_input_id() -> Optional[str]:
    """
    Ищем playback stream нашего Python-приложения.
    """
    sink_inputs = _list_sink_inputs()

    for item in sink_inputs:
        props = _safe_get_props(item)
        app_name = str(props.get("application.name", "")).lower()
        media_name = str(props.get("media.name", "")).lower()
        binary_name = str(props.get("application.process.binary", "")).lower()

        if "python" in app_name or "python" in binary_name or "portaudio" in media_name:
            return str(item.get("index"))

    return None


def _find_our_pulse_source_output_id() -> Optional[str]:
    """
    Ищем recording stream нашего Python-приложения.
    """
    source_outputs = _list_source_outputs()

    for item in source_outputs:
        props = _safe_get_props(item)
        app_name = str(props.get("application.name", "")).lower()
        media_name = str(props.get("media.name", "")).lower()
        binary_name = str(props.get("application.process.binary", "")).lower()

        if "python" in app_name or "python" in binary_name or "portaudio" in media_name:
            return str(item.get("index"))

    return None


def log_audio_routing_snapshot(
    logger: Callable[[str], None],
    *,
    include_sink_inputs: bool = True,
    include_source_outputs: bool = True,
) -> None:
    try:
        if include_sink_inputs:
            sink_inputs = _list_sink_inputs()
            if not sink_inputs:
                logger("Audio routing snapshot | sink-inputs: <none>")
            else:
                logger(f"Audio routing snapshot | sink-inputs={len(sink_inputs)}")
                for item in sink_inputs:
                    logger(f"  sink-input { _format_stream_debug_line(item) }")
    except Exception as error:
        logger(f"Audio routing snapshot | sink-inputs error: {error}")

    try:
        if include_source_outputs:
            source_outputs = _list_source_outputs()
            if not source_outputs:
                logger("Audio routing snapshot | source-outputs: <none>")
            else:
                logger(f"Audio routing snapshot | source-outputs={len(source_outputs)}")
                for item in source_outputs:
                    logger(f"  source-output { _format_stream_debug_line(item) }")
    except Exception as error:
        logger(f"Audio routing snapshot | source-outputs error: {error}")


def move_app_playback_to_sink(
    target_sink_name: str,
    retries: int = 20,
    delay: float = 0.15,
    logger: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    После старта OutputStream перемещаем stream приложения в нужный sink.
    """
    for _ in range(retries):
        try:
            sink_inputs = _list_sink_inputs()
            sink_input_id = _find_our_pulse_sink_input_id()
            if sink_input_id:
                if logger is not None:
                    matching = next(
                        (item for item in sink_inputs if str(item.get("index")) == sink_input_id),
                        None,
                    )
                    if matching is not None:
                        logger(
                            "Audio routing | moving sink-input "
                            f"{_format_stream_debug_line(matching)} -> '{target_sink_name}'"
                        )
                subprocess.run(
                    ["pactl", "move-sink-input", sink_input_id, target_sink_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return True
            elif logger is not None:
                logger("Audio routing | sink-input not found yet")
                for item in sink_inputs:
                    logger(f"  sink-input candidate { _format_stream_debug_line(item) }")
        except Exception:
            if logger is not None:
                logger("Audio routing | move-sink-input attempt failed")

        time.sleep(delay)

    return False


def move_app_recording_to_source(
    target_source_name: str,
    retries: int = 20,
    delay: float = 0.15,
    logger: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    После старта InputStream перемещаем recording stream приложения на нужный source.
    """
    for _ in range(retries):
        try:
            source_outputs = _list_source_outputs()
            source_output_id = _find_our_pulse_source_output_id()
            if source_output_id:
                if logger is not None:
                    matching = next(
                        (item for item in source_outputs if str(item.get("index")) == source_output_id),
                        None,
                    )
                    if matching is not None:
                        logger(
                            "Audio routing | moving source-output "
                            f"{_format_stream_debug_line(matching)} -> '{target_source_name}'"
                        )
                result = subprocess.run(
                    ["pactl", "move-source-output", source_output_id, target_source_name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return True
                if logger is not None:
                    logger(
                        "Audio routing | move-source-output failed "
                        f"(code={result.returncode}): {result.stderr.strip()}"
                    )
            elif logger is not None:
                logger("Audio routing | source-output not found yet")
                for item in source_outputs:
                    logger(f"  source-output candidate { _format_stream_debug_line(item) }")
        except Exception as error:
            if logger is not None:
                logger(f"Audio routing | move-source-output attempt error: {error}")

        time.sleep(delay)

    return False
