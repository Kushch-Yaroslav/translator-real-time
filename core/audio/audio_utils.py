from __future__ import annotations

from typing import Optional, List, Dict, Any
import re


PULSE_DEVICE_NAME = "pulse"


def _get_sounddevice():
    import sounddevice as sd

    return sd


def stop_stream(stream) -> None:
    if stream is None:
        return

    try:
        if hasattr(stream, "active") and stream.active:
            stream.stop()
    except Exception:
        pass

    try:
        if hasattr(stream, "closed") and not stream.closed:
            stream.close()
    except Exception:
        pass


def query_sounddevice_devices() -> List[Dict[str, Any]]:
    sd = _get_sounddevice()
    return sd.query_devices()


def _normalize_device_name(value: str) -> str:
    value = value.lower().strip()

    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\.\d+$", "", value)

    return value


def _extract_search_tokens(device_name: str) -> List[str]:
    normalized = _normalize_device_name(device_name)

    tokens: List[str] = [normalized]

    parts = re.split(r"[._\-]", normalized)
    parts = [part for part in parts if len(part) >= 4]

    joined_parts = " ".join(parts)
    if joined_parts:
        tokens.append(joined_parts)

    preferred_keywords = [
        "jbl",
        "quantum",
        "wireless",
        "harman",
        "usb",
        "mono",
        "stereo",
        "monitor",
        "translator",
        "mic",
        "input",
        "output",
    ]

    matched_keywords = [word for word in preferred_keywords if word in normalized]
    if matched_keywords:
        tokens.append(" ".join(matched_keywords))

    unique_tokens: List[str] = []
    for token in tokens:
        token = token.strip()
        if token and token not in unique_tokens:
            unique_tokens.append(token)

    return unique_tokens


def find_pulse_device_index(is_input: bool) -> Optional[int]:
    sd = _get_sounddevice()
    devices = sd.query_devices()

    for index, device in enumerate(devices):
        name = str(device.get("name", "")).lower()
        max_input_channels = int(device.get("max_input_channels", 0))
        max_output_channels = int(device.get("max_output_channels", 0))

        if "pulse" not in name:
            continue

        if is_input and max_input_channels > 0:
            return index

        if not is_input and max_output_channels > 0:
            return index

    return None


def find_sounddevice_device_index_by_name(
        name_substring: str,
        min_input_channels: int = 0,
        min_output_channels: int = 0,
        prefer_pulse: bool = False,
) -> Optional[int]:
    sd = _get_sounddevice()
    if prefer_pulse:
        pulse_index = find_pulse_device_index(is_input=min_input_channels > 0)
        if pulse_index is not None:
            pulse_device = sd.query_devices(pulse_index)
            max_input_channels = int(pulse_device.get("max_input_channels", 0))
            max_output_channels = int(pulse_device.get("max_output_channels", 0))
            if max_input_channels >= min_input_channels and max_output_channels >= min_output_channels:
                return pulse_index

    devices = sd.query_devices()
    search_tokens = _extract_search_tokens(name_substring)

    for token in search_tokens:
        for index, device in enumerate(devices):
            device_name = str(device.get("name", "")).lower()
            max_input_channels = int(device.get("max_input_channels", 0))
            max_output_channels = int(device.get("max_output_channels", 0))

            if token in device_name:
                if max_input_channels >= min_input_channels and max_output_channels >= min_output_channels:
                    return index

    for index, device in enumerate(devices):
        device_name = str(device.get("name", "")).lower()
        max_input_channels = int(device.get("max_input_channels", 0))
        max_output_channels = int(device.get("max_output_channels", 0))

        if not (max_input_channels >= min_input_channels and max_output_channels >= min_output_channels):
            continue

        score = 0
        for token in search_tokens:
            words = [word for word in token.split() if len(word) >= 3]
            for word in words:
                if word in device_name:
                    score += 1

        if score >= 2:
            return index

    if min_input_channels > 0:
        return find_pulse_device_index(is_input=True)

    if min_output_channels > 0:
        return find_pulse_device_index(is_input=False)

    return None


def create_input_stream(device_index: int, samplerate: int, channels: int, blocksize: int, callback):
    sd = _get_sounddevice()
    return sd.InputStream(
        device=device_index,
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )


def create_output_stream(device_index: int, samplerate: int, channels: int, blocksize: int, callback):
    sd = _get_sounddevice()
    return sd.OutputStream(
        device=device_index,
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )


def create_duplex_stream(
        input_device_index: int,
        output_device_index: int,
        samplerate: int,
        channels: int,
        blocksize: int,
        callback,
):
    sd = _get_sounddevice()
    return sd.Stream(
        device=(input_device_index, output_device_index),
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
