from __future__ import annotations

import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from core.app_config import AppConfig, TranslationBranchConfig, get_primary_branch_config, load_app_config


DEFAULT_CONTAINER_ID = "parakeet-1-1b-ctc-en-us"
DEFAULT_NIM_TAGS_SELECTOR = "name=parakeet-1-1b-ctc-en-us,mode=str,diarizer=disabled,vad=default"
DEFAULT_LOCAL_NIM_CACHE = "/media/yaroslav/DATA/nim-cache"
DEFAULT_HTTP_PORT = 9000
DEFAULT_GRPC_PORT = 50051
DEFAULT_ENV_FILE = str(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class NIMRuntimeConfig:
    container_id: str = DEFAULT_CONTAINER_ID
    nim_tags_selector: str = DEFAULT_NIM_TAGS_SELECTOR
    local_nim_cache: str = DEFAULT_LOCAL_NIM_CACHE
    http_port: int = DEFAULT_HTTP_PORT
    grpc_port: int = DEFAULT_GRPC_PORT
    gpu_device: str = "0"
    startup_timeout_sec: float = 60.0
    poll_interval_sec: float = 1.0
    env_file: str = DEFAULT_ENV_FILE


def ensure_nim_runtime(config: NIMRuntimeConfig | None = None) -> None:
    config = config or NIMRuntimeConfig()

    docker_prefix = _resolve_docker_prefix()

    if _is_container_running(docker_prefix, config.container_id) and _is_nim_http_ready(config.http_port):
        return

    if _is_nim_http_ready(config.http_port):
        raise RuntimeError(
            f"На порту {config.http_port} уже отвечает другой NIM или другой сервис. "
            f"Автоматическая замена контейнера отключена. "
            f"Останови старый контейнер вручную и перезапусти приложение."
        )

    if _is_container_running(docker_prefix, config.container_id):
        _wait_until_ready(config)
        return

    ngc_api_key = _resolve_ngc_api_key(Path(config.env_file))
    if not ngc_api_key:
        raise RuntimeError(
            "NGC_API_KEY is not set. Export it or add NGC_API_KEY to .env."
        )

    _remove_container_if_exists(docker_prefix, config.container_id)
    _start_container(docker_prefix, config, ngc_api_key)
    _wait_until_ready(config)


def ensure_nim_runtime_for_app_config(app_config: AppConfig | None = None) -> None:
    app_config = app_config or load_app_config()
    branch_config = get_primary_branch_config(app_config)
    ensure_nim_runtime(config_from_branch(branch_config))


def config_from_branch(branch_config: TranslationBranchConfig) -> NIMRuntimeConfig:
    return NIMRuntimeConfig(
        container_id=branch_config.nim_container_id,
        nim_tags_selector=branch_config.nim_tags_selector,
        startup_timeout_sec=branch_config.nim_startup_timeout_sec,
    )


def _resolve_ngc_api_key(env_file: Path) -> str:
    key = (os.environ.get("NGC_API_KEY") or "").strip()
    if key:
        return key

    env_values = _read_env_file(env_file)
    for env_name in ("NGC_API_KEY", "NV_API_KEY"):
        candidate = (env_values.get(env_name) or "").strip()
        if candidate:
            os.environ["NGC_API_KEY"] = candidate
            return candidate

    return ""


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")

    return values


def _resolve_docker_prefix() -> list[str]:
    if _can_run_command(["docker", "ps"]):
        return []

    if _can_run_command(["sudo", "-n", "docker", "ps"]):
        return ["sudo", "-n"]

    return ["sudo"]


def _can_run_command(command: list[str]) -> bool:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def _is_container_running(docker_prefix: list[str], container_id: str) -> bool:
    command = [
        *docker_prefix,
        "docker",
        "inspect",
        "-f",
        "{{.State.Running}}",
        container_id,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == "true"


def _remove_container_if_exists(docker_prefix: list[str], container_id: str) -> None:
    command = [*docker_prefix, "docker", "rm", "-f", container_id]
    subprocess.run(command, capture_output=True, text=True)


def _start_container(
    docker_prefix: list[str],
    config: NIMRuntimeConfig,
    ngc_api_key: str,
) -> None:
    cache_dir = Path(config.local_nim_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    command = [
        *docker_prefix,
        "docker",
        "run",
        "-d",
        "--rm",
        f"--name={config.container_id}",
        "--runtime=nvidia",
        "--gpus",
        f"device={config.gpu_device}",
        "--shm-size=8GB",
        "--ulimit",
        "nofile=2048:2048",
        "-e",
        f"NGC_API_KEY={ngc_api_key}",
        "-e",
        f"NIM_HTTP_API_PORT={config.http_port}",
        "-e",
        f"NIM_GRPC_API_PORT={config.grpc_port}",
        "-e",
        f"NIM_TAGS_SELECTOR={config.nim_tags_selector}",
        "-v",
        f"{config.local_nim_cache}:/opt/nim/.cache",
        "-p",
        f"{config.http_port}:{config.http_port}",
        "-p",
        f"{config.grpc_port}:{config.grpc_port}",
        f"nvcr.io/nim/nvidia/{config.container_id}:latest",
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        rendered_command = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            "Failed to start NIM container.\n"
            f"Command: {rendered_command}\n"
            f"stderr: {result.stderr.strip()}"
        )


def _wait_until_ready(config: NIMRuntimeConfig) -> None:
    deadline = time.monotonic() + config.startup_timeout_sec

    while time.monotonic() < deadline:
        if _is_nim_http_ready(config.http_port):
            return
        time.sleep(config.poll_interval_sec)

    raise RuntimeError(
        f"NIM did not become ready on http://localhost:{config.http_port} "
        f"within {config.startup_timeout_sec:.0f} seconds."
    )


def _is_nim_http_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/v1/health/ready",
            timeout=2.0,
        ) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as error:
        return 200 <= error.code < 500
    except Exception:
        return False
