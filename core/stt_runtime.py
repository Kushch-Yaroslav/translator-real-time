from __future__ import annotations

import socket

from core.app_config import AppConfig, load_app_config
from core.nim_runtime import ensure_nim_runtime_for_app_config


def ensure_stt_runtime_for_app_config(app_config: AppConfig | None = None) -> None:
    app_config = app_config or load_app_config()
    backend = (app_config.stt.backend or "nim").strip().lower()

    if backend == "riva":
        ensure_riva_runtime_for_app_config(app_config)
        return

    ensure_nim_runtime_for_app_config(app_config)


def ensure_riva_runtime_for_app_config(app_config: AppConfig) -> None:
    uri = (app_config.stt.riva_uri or "").strip()
    if not uri:
        raise RuntimeError("Riva URI is empty. Set `stt.riva_uri`, for example `localhost:50051`.")

    host, port = _parse_host_port(uri)
    if not _is_tcp_port_open(host, port):
        raise RuntimeError(
            f"Riva server is not reachable at {host}:{port}. "
            "Start the Riva server first, then install the Python client dependencies."
        )


def _parse_host_port(uri: str) -> tuple[str, int]:
    if ":" not in uri:
        return uri, 50051

    host, raw_port = uri.rsplit(":", 1)
    return host or "localhost", int(raw_port)


def _is_tcp_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False
