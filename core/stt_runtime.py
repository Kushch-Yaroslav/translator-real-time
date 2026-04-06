from __future__ import annotations

import socket

from core.app_config import AppConfig, get_primary_branch_config, load_app_config
from core.nim_runtime import config_from_branch, ensure_nim_runtime, ensure_nim_runtime_for_app_config


def ensure_stt_runtime_for_app_config(app_config: AppConfig | None = None) -> None:
    app_config = app_config or load_app_config()
    backend = (app_config.stt.backend or "nim").strip().lower()

    if backend == "faster_whisper":
        return

    if backend == "canary_ast":
        ensure_canary_runtime_for_app_config(app_config)
        return

    if backend == "riva":
        ensure_riva_runtime_for_app_config(app_config)
        ensure_confirm_runtime_for_app_config(app_config)
        return

    ensure_nim_runtime_for_app_config(app_config)


def ensure_canary_runtime_for_app_config(app_config: AppConfig) -> None:
    branch_config = get_primary_branch_config(app_config)
    nim_config = config_from_branch(branch_config)
    nim_config.container_id = app_config.stt.canary_container_id
    nim_config.nim_tags_selector = app_config.stt.canary_tags_selector
    nim_config.http_port = app_config.stt.canary_http_port
    nim_config.grpc_port = app_config.stt.canary_grpc_port
    nim_config.startup_timeout_sec = max(
        app_config.stt.canary_startup_timeout_sec,
        branch_config.nim_startup_timeout_sec,
    )
    ensure_nim_runtime(nim_config)


def ensure_confirm_runtime_for_app_config(app_config: AppConfig) -> None:
    branch_config = get_primary_branch_config(app_config)
    if (branch_config.translation_direction or "").strip().lower() != "ru_to_en":
        return

    nim_config = config_from_branch(branch_config)
    nim_config.grpc_port = 50061
    try:
        ensure_nim_runtime(nim_config)
    except Exception:
        # Confirm-pass can degrade to boundary fallback if the auxiliary NIM
        # runtime is not available yet.
        return


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
