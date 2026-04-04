from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "app.log"
DEFAULT_MAX_BYTES = 512 * 1024
DEFAULT_BACKUP_COUNT = 5


class AppFileLogger:
    def __init__(self) -> None:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger("translator_app")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = RotatingFileHandler(
                DEFAULT_LOG_FILE,
                maxBytes=DEFAULT_MAX_BYTES,
                backupCount=DEFAULT_BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(handler)

    @property
    def log_path(self) -> Path:
        return DEFAULT_LOG_FILE

    def session_started(self) -> None:
        self._logger.info("=" * 24 + " session started " + "=" * 24)

    def info(self, message: str) -> None:
        if self._should_persist_info(message):
            self._logger.info(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    @staticmethod
    def _should_persist_info(message: str) -> bool:
        if not message:
            return False

        excluded_prefixes = (
            "PARTIAL:",
            "Realtime partial:",
        )
        if message.startswith(excluded_prefixes):
            return False

        included_prefixes = (
            "Loaded devices:",
            "Selected pactl input:",
            "Selected pactl output:",
            "Mapped sounddevice input index:",
            "Mapped sounddevice output index:",
            "Loading translation model:",
            "Translation model loaded",
            "Loading TTS voice:",
            "TTS voice loaded",
            "Connecting realtime STT:",
            "Realtime STT session prepared",
            "Realtime STT connected",
            "Realtime STT websocket connected",
            "Move recording stream",
            "Move playback stream",
            "AudioEngine started",
            "Processing mode changed",
            "Pipeline started:",
            "Pipeline stopped",
            "Stopping AudioEngine",
            "AudioEngine stopped",
            "Config saved:",
            "Profile saved:",
            "Profile loaded:",
            "Preset applied:",
            "Conversation mode saved:",
            "FINAL:",
            "FINAL incremental:",
            "FINAL merged:",
            "FINAL skipped:",
            "PARTIAL promoted:",
            "Translation time:",
            "TRANSLATED:",
            "TRANSLATED skipped:",
            "TTS time:",
            "TTS audio ready:",
            "Dropped stale TTS audio queue",
            "Engine is not running yet.",
        )
        return message.startswith(included_prefixes)
