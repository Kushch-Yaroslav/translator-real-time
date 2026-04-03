from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download
from piper import PiperVoice


@dataclass
class TTSConfig:
    voice_name: str = "en_US-lessac-medium"
    data_dir: str = "/media/yaroslav/DATA/ai_models/piper"
    use_cuda: Optional[bool] = None


class TTSService:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.relative_dir = self._resolve_relative_dir(self.config.voice_name)
        self.use_cuda = self._resolve_use_cuda(self.config.use_cuda)

        self.model_path = self._download_voice_file(f"{self.config.voice_name}.onnx")
        self.config_path = self._download_voice_file(f"{self.config.voice_name}.onnx.json")

        self.voice = PiperVoice.load(str(self.model_path), use_cuda=self.use_cuda)

    def synthesize(self, text: str, target_samplerate: int) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros((0, 1), dtype=np.float32)

        audio_chunks: list[np.ndarray] = []
        source_samplerate: int | None = None

        for chunk in self.voice.synthesize(text):
            if source_samplerate is None:
                source_samplerate = int(chunk.sample_rate)

            pcm_int16 = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            if pcm_int16.size == 0:
                continue

            pcm_float32 = pcm_int16.astype(np.float32) / 32768.0
            audio_chunks.append(pcm_float32)

        if not audio_chunks:
            return np.zeros((0, 1), dtype=np.float32)

        audio = np.concatenate(audio_chunks, axis=0).astype(np.float32, copy=False)

        if source_samplerate is None:
            source_samplerate = target_samplerate

        if source_samplerate != target_samplerate:
            audio = self._resample(audio, source_samplerate, target_samplerate)

        return audio.reshape(-1, 1).astype(np.float32, copy=False)

    def _download_voice_file(self, short_filename: str) -> Path:
        filename = f"{self.relative_dir}/{short_filename}"

        local_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=filename,
            local_dir=str(self.data_dir),
        )

        return Path(local_path)

    def _resolve_relative_dir(self, voice_name: str) -> str:
        voice_map = {
            "ru_RU-dmitri-medium": "ru/ru_RU/dmitri/medium",
            "en_US-lessac-medium": "en/en_US/lessac/medium",
            "en_US-ryan-medium": "en/en_US/ryan/medium",
        }

        if voice_name not in voice_map:
            raise ValueError(f"Unsupported Piper voice: {voice_name}")

        return voice_map[voice_name]

    @staticmethod
    def _resolve_use_cuda(requested: Optional[bool]) -> bool:
        if requested is not None:
            return requested and TTSService.has_cuda_provider()

        return TTSService.has_cuda_provider()

    @staticmethod
    def has_cuda_provider() -> bool:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
        except Exception:
            return False

        return "CUDAExecutionProvider" in providers

    @staticmethod
    def get_runtime_backend_label() -> str:
        return "cuda" if TTSService.has_cuda_provider() else "cpu"

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        if from_sr == to_sr or audio.size == 0:
            return audio.astype(np.float32, copy=False)

        duration_seconds = len(audio) / float(from_sr)
        target_length = max(1, int(duration_seconds * to_sr))

        old_indices = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_indices = np.linspace(0.0, 1.0, num=target_length, endpoint=False)

        resampled = np.interp(new_indices, old_indices, audio)
        return resampled.astype(np.float32, copy=False)
