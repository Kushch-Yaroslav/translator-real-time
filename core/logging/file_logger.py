from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "app.log"
DEFAULT_SPEAK_LOG_FILE = DEFAULT_LOG_DIR / "speak_ru_to_en.log"
DEFAULT_MAX_BYTES = 512 * 1024
DEFAULT_BACKUP_COUNT = 5


class AppFileLogger:
    def __init__(self) -> None:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger("translator_app")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        self._speak_logger = logging.getLogger("translator_speak_ru_to_en")
        self._speak_logger.setLevel(logging.INFO)
        self._speak_logger.propagate = False

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

        if not self._speak_logger.handlers:
            handler = RotatingFileHandler(
                DEFAULT_SPEAK_LOG_FILE,
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
            self._speak_logger.addHandler(handler)

    @property
    def log_path(self) -> Path:
        return DEFAULT_LOG_FILE

    @property
    def speak_log_path(self) -> Path:
        return DEFAULT_SPEAK_LOG_FILE

    def session_started(self) -> None:
        self._logger.info("=" * 24 + " session started " + "=" * 24)
        self._speak_logger.info("=" * 20 + " speak session started " + "=" * 20)

    def info(self, message: str) -> None:
        if message and message.startswith("[SPEAK] "):
            # Always persist RU=>EN branch logs for debugging (can be noisy by design).
            self._speak_logger.info(message)
        if self._should_persist_info(message):
            self._logger.info(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    @staticmethod
    def _should_persist_info(message: str) -> bool:
        if not message:
            return False

        normalized = message
        for prefix in ("[LISTEN] ", "[SPEAK] "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break

        excluded_prefixes = (
            "PARTIAL:",
            "Realtime partial:",
        )
        if normalized.startswith(excluded_prefixes):
            return False

        included_prefixes = (
            "Loaded devices:",
            "Selected pactl input:",
            "Selected pactl output:",
            "Mapped sounddevice input index:",
            "Mapped sounddevice output index:",
            "Audio routing |",
            "Audio routing snapshot |",
            "Original audio loopback",
            "Speak passthrough loopback",
            "Original audio mode:",
            "Translator loopbacks cleaned before start:",
            "Loading translation model:",
            "Translation model loaded",
            "Loading TTS voice:",
            "TTS voice loaded",
            "Connecting realtime STT:",
            "Realtime STT session prepared",
            "Realtime STT connected",
            "Realtime STT websocket connected",
            "whisper.cpp partial:",
            "whisper.cpp final:",
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
            "LOWLAT partial queued:",
            "LOWLAT final queued:",
            "LOWLAT sentence queued:",
            "LOWLAT sentence stream:",
            "LOWLAT final skipped:",
            "LOWLAT skipped:",
            "Translation time:",
            "TRANSLATED:",
            "TRANSLATED skipped:",
            "TTS time:",
            "TTS audio ready:",
            "TTS item created:",
            "TTS item queued:",
            "PLAYBACK queued:",
            "PLAYBACK started:",
            "PLAYBACK finished:",
            "PLAYBACK skipped:",
            "Dropped stale TTS audio queue",
            "Engine is not running yet.",
        )
        return normalized.startswith(included_prefixes)
