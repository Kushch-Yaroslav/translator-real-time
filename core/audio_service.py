from __future__ import annotations

import json
import subprocess
import time
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional


TRANSLATOR_SINK_NAME = "translator_mic"
TRANSLATOR_SINK_DESCRIPTION = "TranslatorMic"
TRANSLATOR_LISTEN_SINK_NAME = "translator_listen"
TRANSLATOR_LISTEN_SINK_DESCRIPTION = "TranslatorListen"
TRANSLATOR_MIC_SOURCE_NAME = "translator_microphone"
TRANSLATOR_MIC_SOURCE_DESCRIPTION = "TranslatorMicrophone"


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


def set_default_source_name(source_name: str) -> bool:
    try:
        subprocess.run(
            ["pactl", "set-default-source", source_name],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def set_default_sink_name(sink_name: str) -> bool:
    try:
        subprocess.run(
            ["pactl", "set-default-sink", sink_name],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


@contextmanager
def temporary_pulse_stream_properties(
    *,
    application_name: Optional[str] = None,
    media_name: Optional[str] = None,
) -> Iterator[None]:
    previous_values = {
        "PULSE_PROP_application.name": os.environ.get("PULSE_PROP_application.name"),
        "PULSE_PROP_media.name": os.environ.get("PULSE_PROP_media.name"),
    }

    try:
        if application_name is not None:
            os.environ["PULSE_PROP_application.name"] = application_name
        if media_name is not None:
            os.environ["PULSE_PROP_media.name"] = media_name
        yield
    finally:
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def is_virtual_translator_device(name: str) -> bool:
    normalized = (name or "").strip().lower()
    return normalized in {
        TRANSLATOR_SINK_NAME,
        f"{TRANSLATOR_SINK_NAME}.monitor",
        TRANSLATOR_LISTEN_SINK_NAME,
        f"{TRANSLATOR_LISTEN_SINK_NAME}.monitor",
        TRANSLATOR_MIC_SOURCE_NAME,
    }


def _score_real_input_device(device: AudioDevice, default_source: Optional[str]) -> int:
    name = (device.name or "").lower()
    description = (device.description or "").lower()
    haystack = f"{name} {description}"

    score = 0

    preferred_keywords = (
        "jbl",
        "quantum",
        "wireless",
        "headset",
        "headphone",
        "microphone",
        "mic",
        "usb audio",
    )
    discouraged_keywords = (
        "camera",
        "webcam",
        "ugreen",
        "hd webcam",
        "integrated",
        "monitor of ",
    )

    if default_source and device.name == default_source:
        score += 100

    for keyword in preferred_keywords:
        if keyword in haystack:
            score += 20

    for keyword in discouraged_keywords:
        if keyword in haystack:
            score -= 50

    if ".mono-fallback" in name:
        score += 10

    return score


def get_default_real_source_name() -> Optional[str]:
    default_source = get_default_source_name()
    inputs = list_input_devices()

    if default_source and not default_source.endswith(".monitor") and not is_virtual_translator_device(default_source):
        default_device = next((device for device in inputs if device.name == default_source), None)
        if default_device and _score_real_input_device(default_device, default_source) >= 0:
            return default_source

    candidates = [
        device
        for device in inputs
        if not device.name.endswith(".monitor") and not is_virtual_translator_device(device.name)
    ]
    if candidates:
        best = max(candidates, key=lambda device: _score_real_input_device(device, default_source))
        return best.name

    return default_source


def get_default_real_sink_name() -> Optional[str]:
    default_sink = get_default_sink_name()
    outputs = list_output_devices()

    if default_sink and not is_virtual_translator_device(default_sink):
        return default_sink

    for device in outputs:
        if is_virtual_translator_device(device.name):
            continue
        return device.name

    return default_sink


def repair_default_audio_devices() -> list[str]:
    """
    Keep Translator* devices available for explicit routing, but do not leave
    them as desktop-wide defaults. GNOME/PipeWire can behave badly when a
    monitor source becomes the default input.
    """
    changes: list[str] = []

    default_source = get_default_source_name()
    if default_source and (
        default_source.endswith(".monitor") or is_virtual_translator_device(default_source)
    ):
        real_source = get_default_real_source_name()
        if real_source and real_source != default_source and set_default_source_name(real_source):
            changes.append(f"default source: {default_source} -> {real_source}")

    default_sink = get_default_sink_name()
    if default_sink and is_virtual_translator_device(default_sink):
        real_sink = get_default_real_sink_name()
        if real_sink and real_sink != default_sink and set_default_sink_name(real_sink):
            changes.append(f"default sink: {default_sink} -> {real_sink}")

    return changes


def enrich_default_flags(inputs: List[AudioDevice], outputs: List[AudioDevice]) -> None:
    default_source = get_default_source_name()
    default_sink = get_default_sink_name()

    for device in inputs:
        if device.name == default_source:
            device.is_default = True

    for device in outputs:
        if device.name == default_sink:
            device.is_default = True


def _ensure_null_sink_exists(sink_name: str, description: str) -> None:
    sinks = list_output_devices()
    exists = any(device.name == sink_name for device in sinks)

    if exists:
        return

    subprocess.run(
        [
            "pactl",
            "load-module",
            "module-null-sink",
            f"sink_name={sink_name}",
            f"sink_properties=device.description={description}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def ensure_translator_sink_exists() -> None:
    _ensure_null_sink_exists(TRANSLATOR_SINK_NAME, TRANSLATOR_SINK_DESCRIPTION)


def ensure_translator_listen_sink_exists() -> None:
    _ensure_null_sink_exists(TRANSLATOR_LISTEN_SINK_NAME, TRANSLATOR_LISTEN_SINK_DESCRIPTION)


def ensure_translator_mic_source_exists() -> None:
    sources = list_input_devices()
    exists = any(device.name == TRANSLATOR_MIC_SOURCE_NAME for device in sources)
    if exists:
        return

    ensure_translator_sink_exists()
    subprocess.run(
        [
            "pactl",
            "load-module",
            "module-remap-source",
            f"source_name={TRANSLATOR_MIC_SOURCE_NAME}",
            f"master={TRANSLATOR_SINK_NAME}.monitor",
            f"source_properties=device.description={TRANSLATOR_MIC_SOURCE_DESCRIPTION}",
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


def get_sink_volume_percent(sink_name: str) -> Optional[int]:
    try:
        output = _run_command(["pactl", "get-sink-volume", sink_name])
    except Exception:
        return None

    percentages = [
        int(match.rstrip("%"))
        for match in output.replace("/", " ").split()
        if match.endswith("%") and match[:-1].isdigit()
    ]
    if not percentages:
        return None

    return int(round(sum(percentages) / len(percentages)))


def set_sink_volume_percent(sink_name: str, percent: int) -> bool:
    percent = max(0, min(150, int(percent)))
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", sink_name, f"{percent}%"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def load_source_loopback(
    source_name: str,
    sink_name: str,
    *,
    latency_msec: int = 30,
) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "pactl",
                "load-module",
                "module-loopback",
                f"source={source_name}",
                f"sink={sink_name}",
                f"latency_msec={int(latency_msec)}",
                "source_dont_move=true",
                "sink_dont_move=true",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    module_id = result.stdout.strip()
    return module_id or None


def unload_pulse_module(module_id: str) -> bool:
    if not module_id or module_id == "None":
        return False
    try:
        subprocess.run(
            ["pactl", "unload-module", str(module_id)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def _find_sink_input_id_by_owner_module(module_id: str) -> Optional[str]:
    for item in _list_sink_inputs():
        if str(item.get("owner_module", "")) == str(module_id):
            return str(item.get("index"))
    return None


def set_loopback_volume_percent(
    module_id: str,
    percent: int,
    *,
    retries: int = 20,
    delay: float = 0.1,
) -> bool:
    percent = max(0, min(150, int(percent)))

    for _ in range(retries):
        sink_input_id = _find_sink_input_id_by_owner_module(module_id)
        if sink_input_id:
            try:
                mute_value = "1" if percent <= 0 else "0"
                subprocess.run(
                    ["pactl", "set-sink-input-mute", sink_input_id, mute_value],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if percent <= 0:
                    return True
                subprocess.run(
                    ["pactl", "set-sink-input-volume", sink_input_id, f"{percent}%"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return True
            except Exception:
                pass
        time.sleep(delay)

    return False


def _list_sink_inputs() -> list[dict]:
    output = _run_command(["pactl", "-f", "json", "list", "sink-inputs"])
    return json.loads(output)


def _list_source_outputs() -> list[dict]:
    output = _run_command(["pactl", "-f", "json", "list", "source-outputs"])
    return json.loads(output)


def _list_modules() -> list[dict]:
    output = _run_command(["pactl", "-f", "json", "list", "modules"])
    return json.loads(output)


def _list_modules_short() -> list[tuple[str, str, str]]:
    try:
        output = _run_command(["pactl", "list", "short", "modules"])
    except Exception:
        return []

    result: list[tuple[str, str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        module_id, name, args = parts[0].strip(), parts[1].strip(), parts[2].strip()
        result.append((module_id, name, args))
    return result


def cleanup_translator_loopbacks(
    logger: Optional[Callable[[str], None]] = None,
) -> int:
    translator_tokens = {
        TRANSLATOR_SINK_NAME,
        f"{TRANSLATOR_SINK_NAME}.monitor",
        TRANSLATOR_LISTEN_SINK_NAME,
        f"{TRANSLATOR_LISTEN_SINK_NAME}.monitor",
        TRANSLATOR_MIC_SOURCE_NAME,
    }

    unloaded = 0

    short_modules = _list_modules_short()
    if short_modules:
        for module_id, name, args in short_modules:
            if name != "module-loopback":
                continue
            if not any(token in args for token in translator_tokens):
                continue

            success = unload_pulse_module(module_id)
            if success:
                unloaded += 1
            if logger is not None:
                logger(
                    "Audio routing | cleanup loopback "
                    f"module_id={module_id} args='{args}' -> {'OK' if success else 'FAILED'}"
                )
        return unloaded

    try:
        modules = _list_modules()
    except Exception as error:
        if logger is not None:
            logger(f"Audio routing | module list failed: {error}")
        return 0

    for item in modules:
        name = str(item.get("name", ""))
        if name != "module-loopback":
            continue

        args = str(item.get("argument", "") or item.get("arguments", "") or "")
        if not any(token in args for token in translator_tokens):
            continue

        module_id = str(item.get("index", item.get("id", "")))
        success = unload_pulse_module(module_id)
        if success:
            unloaded += 1
        if logger is not None:
            logger(
                "Audio routing | cleanup loopback "
                f"module_id={module_id} args='{args}' -> {'OK' if success else 'FAILED'}"
            )

    return unloaded


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


def _matches_our_stream(item: dict, stream_tag: Optional[str] = None) -> bool:
    props = _safe_get_props(item)
    app_name = str(props.get("application.name", "")).lower()
    media_name = str(props.get("media.name", "")).lower()
    binary_name = str(props.get("application.process.binary", "")).lower()

    if stream_tag:
        tag = stream_tag.lower()
        if tag in app_name or tag in media_name:
            return True

    return "python" in app_name or "python" in binary_name or "portaudio" in media_name


def _find_our_pulse_sink_input_id(stream_tag: Optional[str] = None) -> Optional[str]:
    """
    Ищем playback stream нашего Python-приложения.
    """
    sink_inputs = _list_sink_inputs()

    for item in sink_inputs:
        if _matches_our_stream(item, stream_tag):
            return str(item.get("index"))

    return None


def _find_our_pulse_source_output_id(stream_tag: Optional[str] = None) -> Optional[str]:
    """
    Ищем recording stream нашего Python-приложения.
    """
    source_outputs = _list_source_outputs()

    for item in source_outputs:
        if _matches_our_stream(item, stream_tag):
            return str(item.get("index"))

    return None


def snapshot_sink_input_ids() -> set[str]:
    return {str(item.get("index")) for item in _list_sink_inputs()}


def snapshot_source_output_ids() -> set[str]:
    return {str(item.get("index")) for item in _list_source_outputs()}


def _find_new_our_pulse_sink_input_id(existing_ids: set[str], stream_tag: Optional[str] = None) -> Optional[str]:
    sink_inputs = _list_sink_inputs()

    for item in sink_inputs:
        item_id = str(item.get("index"))
        if item_id in existing_ids:
            continue

        if _matches_our_stream(item, stream_tag):
            return item_id

    return None


def _find_new_our_pulse_source_output_id(existing_ids: set[str], stream_tag: Optional[str] = None) -> Optional[str]:
    source_outputs = _list_source_outputs()

    for item in source_outputs:
        item_id = str(item.get("index"))
        if item_id in existing_ids:
            continue

        if _matches_our_stream(item, stream_tag):
            return item_id

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
    existing_ids: Optional[set[str]] = None,
    stream_tag: Optional[str] = None,
) -> bool:
    """
    После старта OutputStream перемещаем stream приложения в нужный sink.
    """
    for _ in range(retries):
        try:
            sink_inputs = _list_sink_inputs()
            sink_input_id = (
                _find_new_our_pulse_sink_input_id(existing_ids, stream_tag)
                if existing_ids is not None
                else _find_our_pulse_sink_input_id(stream_tag)
            )
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
    existing_ids: Optional[set[str]] = None,
    stream_tag: Optional[str] = None,
) -> bool:
    """
    После старта InputStream перемещаем recording stream приложения на нужный source.
    """
    for _ in range(retries):
        try:
            source_outputs = _list_source_outputs()
            source_output_id = (
                _find_new_our_pulse_source_output_id(existing_ids, stream_tag)
                if existing_ids is not None
                else _find_our_pulse_source_output_id(stream_tag)
            )
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
