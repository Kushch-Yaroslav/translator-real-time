from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np
import requests
from scipy.signal import resample


@dataclass
class STTServiceConfig:
    base_url: str = "http://localhost:9000"
    language: str = "en-US"
    timeout: float = 20.0
    target_samplerate: int = 16000


class STTService:
    def __init__(
            self,
            base_url: str = "http://localhost:9000",
            language: str = "en-US",
            timeout: float = 20.0,
            target_samplerate: int = 16000,
    ):
        self.config = STTServiceConfig(
            base_url=base_url.rstrip("/"),
            language=language,
            timeout=timeout,
            target_samplerate=target_samplerate,
        )

    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        mono = self._prepare_audio(audio)
        mono = self._resample_if_needed(mono, samplerate, self.config.target_samplerate)
        wav_bytes = self._to_wav_bytes(mono, self.config.target_samplerate)

        files = {
            "file": ("phrase.wav", wav_bytes, "audio/wav"),
        }
        data = {
            "language": self.config.language,
        }

        response = requests.post(
            f"{self.config.base_url}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=self.config.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        payload = response.json()
        return payload.get("text", "").strip()

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio[:, 0]

        audio = audio.astype(np.float32, copy=False)

        peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
        if peak > 1.0:
            audio = audio / 32768.0

        audio = np.clip(audio, -1.0, 1.0)
        return audio

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio.astype(np.float32, copy=False)

        target_len = int(len(audio) * dst_sr / src_sr)
        if target_len <= 0:
            return np.zeros((0,), dtype=np.float32)

        return resample(audio, target_len).astype(np.float32, copy=False)

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, samplerate: int) -> bytes:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm16.tobytes())

        return buffer.getvalue()