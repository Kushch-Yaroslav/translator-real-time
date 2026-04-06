from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.signal import resample


@dataclass
class RivaConfirmSTTConfig:
    uri: str = "localhost:50051"
    language: str = "en-US"
    sample_rate_hz: int = 16000
    num_channels: int = 1
    timeout: float = 10.0
    enable_automatic_punctuation: bool = True
    use_ssl: bool = False
    ssl_cert_path: str = ""
    on_log: Optional[Callable[[str], None]] = None


class RivaConfirmSTTService:
    def __init__(self, config: Optional[RivaConfirmSTTConfig] = None):
        self.config = config or RivaConfirmSTTConfig()
        self._ensure_riva_sdk_available()

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        import riva.client

        mono = self._prepare_audio(audio)
        mono = self._resample_if_needed(mono, samplerate, self.config.sample_rate_hz)
        audio_bytes = self._float32_to_pcm16_bytes(mono)
        if not audio_bytes:
            return ""

        auth = riva.client.Auth(
            uri=self.config.uri,
            use_ssl=self.config.use_ssl,
            ssl_root_cert=self.config.ssl_cert_path or None,
        )
        asr_service = riva.client.ASRService(auth)
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=self.config.sample_rate_hz,
            language_code=self.config.language,
            max_alternatives=1,
            enable_automatic_punctuation=self.config.enable_automatic_punctuation,
            audio_channel_count=self.config.num_channels,
        )
        response = asr_service.offline_recognize(audio_bytes=audio_bytes, config=config)

        segments: list[str] = []
        for result in getattr(response, "results", None) or []:
            alternatives = getattr(result, "alternatives", None) or []
            if not alternatives:
                continue
            transcript = self._normalize_text(
                getattr(alternatives[0], "transcript", "") or ""
            )
            if transcript:
                segments.append(transcript)

        text = " ".join(segments).strip()
        if text:
            self._log(f"Confirm STT text: {text}")
        return text

    def _log(self, message: str) -> None:
        if self.config.on_log is not None:
            self.config.on_log(message)

    @staticmethod
    def _ensure_riva_sdk_available() -> None:
        try:
            import riva.client  # noqa: F401
        except Exception as error:
            raise RuntimeError(
                "Riva Python SDK is not installed. Install `nvidia-riva-client` and `grpcio`."
            ) from error

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
