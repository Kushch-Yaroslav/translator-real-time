from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import websocket
from scipy.signal import resample


@dataclass
class NemotronRealtimeSTTConfig:
    ws_url: str = "ws://localhost:8765/stream"
    sample_rate_hz: int = 16000
    timeout: float = 15.0
    on_log: Optional[Callable[[str], None]] = None


class NemotronRealtimeSTTService:
    def __init__(self, config: Optional[NemotronRealtimeSTTConfig] = None):
        self.config = config or NemotronRealtimeSTTConfig()

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._connected = threading.Event()
        self._stop_requested = False

        self._partial_callback: Optional[Callable[[str], None]] = None
        self._final_callback: Optional[Callable[[str], None]] = None

        self._last_partial_text = ""
        self._last_final_text = ""

    def start(
        self,
        partial_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._partial_callback = partial_callback
        self._final_callback = final_callback
        self._stop_requested = False
        self._connected.clear()
        self._last_partial_text = ""
        self._last_final_text = ""

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
            name="nemotron-realtime-stt-ws",
        )
        self._ws_thread.start()

        if not self._connected.wait(timeout=self.config.timeout):
            raise RuntimeError("Nemotron realtime STT websocket connection timeout")

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
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        self._connected.clear()
        self._last_partial_text = ""
        self._last_final_text = ""

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

        self._send_json({
            "type": "audio",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        })

    def commit(self) -> None:
        if not self._connected.is_set():
            return
        self._send_json({"type": "commit"})

    def clear(self) -> None:
        if not self._connected.is_set():
            self._last_partial_text = ""
            self._last_final_text = ""
            return

        self._send_json({"type": "clear"})
        self._last_partial_text = ""
        self._last_final_text = ""

    def send_done(self) -> None:
        if not self._connected.is_set():
            return
        self._send_json({"type": "done"})

    def get_last_partial_text(self) -> str:
        return self._last_partial_text

    def get_last_final_text(self) -> str:
        return self._last_final_text

    def _on_open(self, ws) -> None:
        self._send_json({
            "type": "start",
            "sample_rate_hz": self.config.sample_rate_hz,
        })
        self._connected.set()

    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            self._log(f"Nemotron raw message: {message}")
            return

        event_type = str(payload.get("type") or "").strip().lower()
        text = self._normalize_text(payload.get("text", "") or "")

        if event_type == "ready":
            self._log("Nemotron realtime STT ready")
            return

        if event_type == "partial":
            if not text or self._normalize_compare_text(text) == self._normalize_compare_text(self._last_partial_text):
                return
            self._last_partial_text = text
            self._log(f"Nemotron partial: {text}")
            if self._partial_callback is not None:
                self._partial_callback(text)
            return

        if event_type == "final":
            if not text:
                return
            self._last_final_text = text
            self._last_partial_text = ""
            self._log(f"Nemotron final: {text}")
            if self._final_callback is not None:
                self._final_callback(text)
            return

        if event_type == "log":
            details = self._normalize_text(payload.get("message", "") or "")
            if details:
                self._log(f"Nemotron server: {details}")
            return

        if event_type == "error":
            details = self._normalize_text(payload.get("message", "") or "")
            self._log(f"Nemotron realtime STT error: {details or payload}")

    def _on_error(self, ws, error) -> None:
        self._log(f"Nemotron realtime STT websocket error: {error}")

    def _on_close(self, ws, status_code, message) -> None:
        self._connected.clear()
        if self._stop_requested:
            self._log("Nemotron realtime STT websocket closed")
        else:
            self._log(
                f"Nemotron realtime STT websocket closed: code={status_code}, message={message}"
            )

    def _send_json(self, payload: dict) -> None:
        ws = self._ws
        if ws is None:
            return

        with self._lock:
            ws.send(json.dumps(payload))

    def _log(self, message: str) -> None:
        if self.config.on_log is not None:
            self.config.on_log(message)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        normalized = NemotronRealtimeSTTService._normalize_text(text).lower()
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch.isspace())
        return " ".join(normalized.split())

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
