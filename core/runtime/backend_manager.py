from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from core.config.app_config import AppConfig
from core.pipeline.branch_definitions import (
    LISTEN_LANE_DEFINITION,
    SPEAK_LANE_DEFINITION,
)


@dataclass
class BackendStatus:
    backend_id: str
    title: str
    state: str
    detail: str
    managed: bool = False
    pid: int | None = None

class BackendManager:
    def __init__(self, base_config: AppConfig, root_dir: str | Path):
        self.base_config = base_config
        self.root_dir = Path(root_dir)
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._whispercpp_process: subprocess.Popen | None = None
        self._whispercpp_log_handle = None
        self._whispercpp_starting = False
        self._ru_warming = False
        self._ru_stop_requested = False

        self._statuses: dict[str, BackendStatus] = {
            LISTEN_LANE_DEFINITION.backend_id: BackendStatus(
                backend_id=LISTEN_LANE_DEFINITION.backend_id,
                title=LISTEN_LANE_DEFINITION.backend_title or LISTEN_LANE_DEFINITION.title,
                state="checking",
                detail="Проверка backend-а...",
            ),
            SPEAK_LANE_DEFINITION.backend_id: BackendStatus(
                backend_id=SPEAK_LANE_DEFINITION.backend_id,
                title=SPEAK_LANE_DEFINITION.backend_title or SPEAK_LANE_DEFINITION.title,
                state="checking",
                detail="Проверка встроенного runtime...",
            ),
        }

    def ensure_started_async(self) -> None:
        self.start_en_to_ru_async()
        self.start_ru_to_en_async()

    def get_status_snapshot(self) -> list[BackendStatus]:
        with self._lock:
            return [replace(status) for status in self._statuses.values()]

    def start_en_to_ru_async(self) -> None:
        with self._lock:
            if self._whispercpp_starting:
                return
            self._whispercpp_starting = True
            self._set_status(
                LISTEN_LANE_DEFINITION.backend_id,
                state="starting",
                detail="Поднимается whisper.cpp сервер...",
            )

        threading.Thread(
            target=self._start_en_to_ru_worker,
            daemon=True,
            name="backend-start-en-to-ru",
        ).start()

    def restart_en_to_ru_async(self) -> None:
        threading.Thread(
            target=self._restart_en_to_ru_worker,
            daemon=True,
            name="backend-restart-en-to-ru",
        ).start()

    def stop_en_to_ru(self) -> None:
        with self._lock:
            process = self._whispercpp_process
            self._whispercpp_process = None
            self._whispercpp_starting = False

        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except Exception:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except Exception:
                    pass
            self._close_whispercpp_log_handle()
            self._set_status(
                LISTEN_LANE_DEFINITION.backend_id,
                state="stopped",
                detail="whisper.cpp сервер остановлен.",
            )
            return

        if self._is_whispercpp_reachable():
            self._set_status(
                LISTEN_LANE_DEFINITION.backend_id,
                state="running",
                detail="whisper.cpp сервер уже работает вне приложения.",
            )
            return

        self._set_status(
            LISTEN_LANE_DEFINITION.backend_id,
            state="stopped",
            detail="whisper.cpp сервер остановлен.",
        )

    def start_ru_to_en_async(self) -> None:
        with self._lock:
            if self._ru_warming:
                return
            self._ru_warming = True
            self._ru_stop_requested = False
            self._set_status(
                SPEAK_LANE_DEFINITION.backend_id,
                state="starting",
                detail="Подготовка встроенного faster-whisper runtime...",
            )

        threading.Thread(
            target=self._start_ru_to_en_worker,
            daemon=True,
            name="backend-start-ru-to-en",
        ).start()

    def restart_ru_to_en_async(self) -> None:
        threading.Thread(
            target=self._restart_ru_to_en_worker,
            daemon=True,
            name="backend-restart-ru-to-en",
        ).start()

    def stop_ru_to_en(self) -> None:
        with self._lock:
            self._ru_stop_requested = True
            self._ru_warming = False

        self._set_status(
            SPEAK_LANE_DEFINITION.backend_id,
            state="stopped",
            detail="Встроенный faster-whisper runtime остановлен.",
        )

    def _start_en_to_ru_worker(self) -> None:
        try:
            if self._is_whispercpp_reachable():
                self._set_status(
                    LISTEN_LANE_DEFINITION.backend_id,
                    state="ready",
                    detail="whisper.cpp сервер уже доступен.",
                )
                return

            script_path = self.root_dir / "scripts" / "start_whispercpp_server.sh"
            log_path = self.logs_dir / "backend_whispercpp.log"
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                [str(script_path)],
                cwd=str(self.root_dir),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )

            with self._lock:
                self._whispercpp_process = process
                self._whispercpp_log_handle = log_handle

            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                if self._is_whispercpp_reachable():
                    self._set_status(
                        LISTEN_LANE_DEFINITION.backend_id,
                        state="ready",
                        detail="whisper.cpp сервер готов.",
                        managed=True,
                        pid=process.pid,
                    )
                    return

                if process.poll() is not None:
                    raise RuntimeError(f"whisper.cpp process exited with code {process.returncode}")

                time.sleep(0.5)

            raise RuntimeError("timeout while waiting for whisper.cpp server")
        except Exception as error:
            self._set_status(
                LISTEN_LANE_DEFINITION.backend_id,
                state="error",
                detail=f"Ошибка запуска: {error}",
            )
        finally:
            with self._lock:
                self._whispercpp_starting = False

    def _restart_en_to_ru_worker(self) -> None:
        self.stop_en_to_ru()
        time.sleep(0.3)
        self.start_en_to_ru_async()

    def _start_ru_to_en_worker(self) -> None:
        try:
            self._set_status(
                SPEAK_LANE_DEFINITION.backend_id,
                state="ready",
                detail="Встроенный faster-whisper runtime готов к запуску пайплайна.",
            )
        except Exception as error:
            self._set_status(
                SPEAK_LANE_DEFINITION.backend_id,
                state="error",
                detail=f"Ошибка прогрева: {error}",
            )
        finally:
            with self._lock:
                self._ru_warming = False

    def _restart_ru_to_en_worker(self) -> None:
        with self._lock:
            self._ru_stop_requested = False
            self._ru_warming = True
            self._set_status(
                SPEAK_LANE_DEFINITION.backend_id,
                state="starting",
                detail="Обновление встроенного faster-whisper runtime...",
            )

        try:
            self._set_status(
                SPEAK_LANE_DEFINITION.backend_id,
                state="ready",
                detail="Встроенный faster-whisper runtime готов к запуску пайплайна.",
            )
        except Exception as error:
            self._set_status(
                SPEAK_LANE_DEFINITION.backend_id,
                state="error",
                detail=f"Ошибка перезапуска: {error}",
            )
        finally:
            with self._lock:
                self._ru_warming = False

    def _set_status(
        self,
        backend_id: str,
        *,
        state: str,
        detail: str,
        managed: bool | None = None,
        pid: int | None = None,
    ) -> None:
        with self._lock:
            current = self._statuses[backend_id]
            self._statuses[backend_id] = BackendStatus(
                backend_id=backend_id,
                title=current.title,
                state=state,
                detail=detail,
                managed=current.managed if managed is None else managed,
                pid=current.pid if pid is None else pid,
            )

    def _is_whispercpp_reachable(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 8178), timeout=1.5):
                return True
        except OSError:
            return False

    def _close_whispercpp_log_handle(self) -> None:
        with self._lock:
            handle = self._whispercpp_log_handle
            self._whispercpp_log_handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
