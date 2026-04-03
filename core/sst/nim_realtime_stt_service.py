from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import requests
import websocket
from scipy.signal import resample


@dataclass
class NIMRealtimeSTTConfig:
    base_url: str = "http://localhost:9000"
    ws_url: str = "ws://localhost:9000/v1/realtime?intent=transcription"
    language: str = "en-US"
    sample_rate_hz: int = 16000
    num_channels: int = 1
    timeout: float = 10.0
    commit_interval_sec: float = 0.5
    enable_automatic_punctuation: bool = True
    enable_word_time_offsets: bool = False
    enable_verbatim_transcripts: bool = False
    on_log: Optional[Callable[[str], None]] = None


class NIMRealtimeSTTService:
    def __init__(self, config: Optional[NIMRealtimeSTTConfig] = None):
        self.config = config or NIMRealtimeSTTConfig()

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._connected = threading.Event()
        self._stop_requested = False

        self._last_commit_at = 0.0
        self._last_delta_text = ""
        self._last_completed_text = ""

        self._partial_callback: Optional[Callable[[str], None]] = None
        self._final_callback: Optional[Callable[[str], None]] = None
        self._pending_session_update: Optional[dict] = None

    def start(
            self,
            partial_callback: Optional[Callable[[str], None]] = None,
            final_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._partial_callback = partial_callback
        self._final_callback = final_callback
        self._stop_requested = False
        self._connected.clear()
        self._last_commit_at = 0.0
        self._last_delta_text = ""
        self._last_completed_text = ""

        self._create_session()
        self._connect_ws()

        if not self._connected.wait(timeout=self.config.timeout):
            raise RuntimeError("NIM realtime STT websocket connection timeout")

    def stop(self) -> None:
        self._stop_requested = True

        ws = self._ws
        self._ws = None

        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        thread = self._ws_thread
        self._ws_thread = None

        if (
                thread
                and thread.is_alive()
                and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

        self._connected.clear()

    def restart(self) -> None:
        self.stop()
        time.sleep(0.15)
        self.start(
            partial_callback=self._partial_callback,
            final_callback=self._final_callback,
        )

    def send_audio_chunk(self, audio: np.ndarray, samplerate: int) -> None:
        if audio.size == 0 or not self._connected.is_set():
            return

        mono = self._prepare_audio(audio)
        mono = self._resample_if_needed(mono, samplerate, self.config.sample_rate_hz)
        pcm16 = self._float32_to_pcm16_bytes(mono)

        if not pcm16:
            return

        event = {
            "event_id": self._event_id(),
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }
        self._send_json(event)

        now = time.monotonic()
        if now - self._last_commit_at >= self.config.commit_interval_sec:
            self.commit()
            self._last_commit_at = now

    def commit(self) -> None:
        if not self._connected.is_set():
            return

        self._send_json({
            "event_id": self._event_id(),
            "type": "input_audio_buffer.commit",
        })

    def clear(self) -> None:
        if not self._connected.is_set():
            return

        self._send_json({
            "event_id": self._event_id(),
            "type": "input_audio_buffer.clear",
        })

    def send_done(self) -> None:
        if not self._connected.is_set():
            return

        self._send_json({
            "event_id": self._event_id(),
            "type": "input_audio_buffer.done",
        })

    def get_last_partial_text(self) -> str:
        return self._last_delta_text

    def get_last_final_text(self) -> str:
        return self._last_completed_text

    def _create_session(self) -> None:
        url = f"{self.config.base_url.rstrip('/')}/v1/realtime/transcription_sessions"
        response = requests.post(url, timeout=self.config.timeout)
        response.raise_for_status()

        session = response.json()
        session["input_audio_transcription"]["language"] = self.config.language
        session["input_audio_params"]["sample_rate_hz"] = self.config.sample_rate_hz
        session["input_audio_params"]["num_channels"] = self.config.num_channels
        session["recognition_config"]["enable_automatic_punctuation"] = (
            self.config.enable_automatic_punctuation
        )
        session["recognition_config"]["enable_word_time_offsets"] = (
            self.config.enable_word_time_offsets
        )
        session["recognition_config"]["enable_verbatim_transcripts"] = (
            self.config.enable_verbatim_transcripts
        )

        self._pending_session_update = {
            "event_id": self._event_id(),
            "type": "transcription_session.update",
            "session": session,
        }

        self._log("Realtime STT session prepared")

    def _connect_ws(self) -> None:
        self._ws = websocket.WebSocketApp(
            self.config.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            daemon=True,
            name="nim-realtime-stt-ws",
        )
        self._ws_thread.start()

    def _on_open(self, ws) -> None:
        self._log("Realtime STT websocket connected")
        if self._pending_session_update is not None:
            self._send_json(self._pending_session_update)
        self._connected.set()

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            self._log(f"Realtime STT raw message: {message}")
            return

        event_type = payload.get("type", "")

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = self._normalize_text(payload.get("delta", "") or "")
            if delta:
                self._last_delta_text = delta
                self._log(f"Realtime partial: {delta}")
                if self._partial_callback:
                    self._partial_callback(delta)
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            text = self._normalize_text(
                payload.get("transcript", "") or payload.get("text", "") or ""
            )
            if text:
                self._last_completed_text = text
                self._log(f"Realtime final: {text}")
                if self._final_callback:
                    self._final_callback(text)
            return

        if event_type == "error":
            self._log(f"Realtime STT server error: {payload}")
            return

    def _on_error(self, ws, error) -> None:
        self._log(f"Realtime STT websocket error: {error}")

    def _on_close(self, ws, status_code, message) -> None:
        self._connected.clear()
        if self._stop_requested:
            self._log("Realtime STT websocket closed")
        else:
            self._log(f"Realtime STT websocket closed: code={status_code}, message={message}")

    def _send_json(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return

        with self._lock:
            ws.send(json.dumps(payload))

    def _log(self, message: str) -> None:
        if self.config.on_log:
            self.config.on_log(message)

    @staticmethod
    def _event_id() -> str:
        return f"event_{uuid.uuid4().hex}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio[:, 0]

        audio = audio.astype(np.float32, copy=False)

        peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
        if peak > 1.0:
            audio = audio / 32768.0

        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio.astype(np.float32, copy=False)

        target_len = int(len(audio) * dst_sr / src_sr)
        if target_len <= 0:
            return np.zeros((0,), dtype=np.float32)

        return resample(audio, target_len).astype(np.float32, copy=False)

    @staticmethod
    def _float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()