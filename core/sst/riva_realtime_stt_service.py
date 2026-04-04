from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.signal import resample


@dataclass
class RivaRealtimeSTTConfig:
    uri: str = "localhost:50051"
    language: str = "en-US"
    sample_rate_hz: int = 16000
    num_channels: int = 1
    timeout: float = 10.0
    enable_automatic_punctuation: bool = True
    use_ssl: bool = False
    ssl_cert_path: str = ""
    interim_results: bool = True
    on_log: Optional[Callable[[str], None]] = None


class RivaRealtimeSTTService:
    def __init__(self, config: Optional[RivaRealtimeSTTConfig] = None):
        self.config = config or RivaRealtimeSTTConfig()

        self._audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=128)
        self._stream_thread: Optional[threading.Thread] = None
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
        self._ensure_riva_sdk_available()

        self._partial_callback = partial_callback
        self._final_callback = final_callback
        self._stop_requested = False
        self._last_partial_text = ""
        self._last_final_text = ""
        self._connected.clear()
        self._clear_audio_queue()

        self._stream_thread = threading.Thread(
            target=self._streaming_loop,
            daemon=True,
            name="riva-realtime-stt",
        )
        self._stream_thread.start()

        if not self._connected.wait(timeout=self.config.timeout):
            raise RuntimeError("Riva realtime STT connection timeout")

    def stop(self) -> None:
        self._stop_requested = True
        self._connected.clear()
        self._signal_stream_end()

        thread = self._stream_thread
        self._stream_thread = None
        if (
            thread
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

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

        try:
            self._audio_queue.put_nowait(pcm16)
        except queue.Full:
            self._log("Riva audio queue overflow, dropping chunk")

    def commit(self) -> None:
        return

    def clear(self) -> None:
        self._clear_audio_queue()

    def send_done(self) -> None:
        self._signal_stream_end()

    def get_last_partial_text(self) -> str:
        return self._last_partial_text

    def get_last_final_text(self) -> str:
        return self._last_final_text

    def _streaming_loop(self) -> None:
        try:
            import riva.client
        except Exception as error:
            self._log(f"Riva client import failed: {error}")
            return

        try:
            auth = riva.client.Auth(
                uri=self.config.uri,
                use_ssl=self.config.use_ssl,
                ssl_root_cert=self.config.ssl_cert_path or None,
            )
            asr_service = riva.client.ASRService(auth)
            streaming_config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hertz=self.config.sample_rate_hz,
                    language_code=self.config.language,
                    max_alternatives=1,
                    enable_automatic_punctuation=self.config.enable_automatic_punctuation,
                    audio_channel_count=self.config.num_channels,
                ),
                interim_results=self.config.interim_results,
            )
        except Exception as error:
            self._log(f"Riva client setup failed: {error}")
            return

        try:
            self._connected.set()
            self._log(
                f"Riva realtime STT connected ({self.config.language}, uri={self.config.uri})"
            )
            responses = asr_service.streaming_response_generator(
                audio_chunks=self._iter_audio_chunks(),
                streaming_config=streaming_config,
            )
            for response in responses:
                if self._stop_requested:
                    break
                self._handle_response(response)
        except Exception as error:
            self._log(f"Riva streaming error: {error}")
        finally:
            self._connected.clear()

    def _handle_response(self, response) -> None:
        results = getattr(response, "results", None) or []
        for result in results:
            alternatives = getattr(result, "alternatives", None) or []
            if not alternatives:
                continue

            transcript = self._normalize_text(
                getattr(alternatives[0], "transcript", "") or ""
            )
            if not transcript:
                continue

            is_final = bool(getattr(result, "is_final", False))
            if is_final:
                self._last_final_text = transcript
                self._log(f"Riva final: {transcript}")
                if self._final_callback:
                    self._final_callback(transcript)
            else:
                self._last_partial_text = transcript
                self._log(f"Riva partial: {transcript}")
                if self._partial_callback:
                    self._partial_callback(transcript)

    def _iter_audio_chunks(self):
        while not self._stop_requested:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if chunk is None:
                break

            yield chunk

    def _signal_stream_end(self) -> None:
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass

    def _clear_audio_queue(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _ensure_riva_sdk_available(self) -> None:
        try:
            import riva.client  # noqa: F401
        except Exception as error:
            raise RuntimeError(
                "Riva Python SDK is not installed. Install `nvidia-riva-client` and `grpcio`, "
                "then point `stt.riva_uri` to a running Riva server."
            ) from error

    def _log(self, message: str) -> None:
        if self.config.on_log is not None:
            self.config.on_log(message)

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
        if audio.size == 0:
            return b""

        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())
