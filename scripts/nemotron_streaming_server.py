from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from aiohttp import WSMsgType, web
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


LOG_MEL_ZERO = -16.635


class AudioBufferer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)

    def reset(self) -> None:
        self.sample_buffer.zero_()

    def update(self, audio: np.ndarray) -> None:
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)

        audio_size = audio.shape[0]
        if audio_size > self.buffer_size:
            raise ValueError(f"Frame size ({audio_size}) exceeds buffer size ({self.buffer_size})")

        shift = audio_size
        self.sample_buffer[:-shift] = self.sample_buffer[shift:].clone()
        self.sample_buffer[-shift:] = audio.clone()


class CacheFeatureBufferer:
    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg,
        device: torch.device,
        fill_value: float = LOG_MEL_ZERO,
    ):
        if buffer_size_in_secs < chunk_size_in_secs:
            raise ValueError(
                f"Buffer size ({buffer_size_in_secs}s) should be no less than chunk size ({chunk_size_in_secs}s)"
            )

        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.device = device

        if hasattr(preprocessor_cfg, "log") and preprocessor_cfg.log:
            self.zero_level_spec_db_val = LOG_MEL_ZERO
        else:
            self.zero_level_spec_db_val = fill_value

        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.sample_buffer = AudioBufferer(sample_rate, buffer_size_in_secs)

        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.zero_level_spec_db_val,
            dtype=torch.float32,
            device=self.device,
        )

        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(self.device)

    def reset(self) -> None:
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.zero_level_spec_db_val)

    def preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        audio_signal = audio_signal.unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(input_signal=audio_signal, length=audio_signal_len)
        return features.squeeze()

    def update(self, audio: np.ndarray) -> None:
        self.sample_buffer.update(audio)

        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            samples = self.sample_buffer.sample_buffer.clone()
        else:
            samples = self.sample_buffer.sample_buffer[-(self.n_chunk_look_back + self.chunk_size):]

        features = self.preprocess(samples)
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]

        self.feature_buffer[:, :-self.feature_chunk_len] = self.feature_buffer[:, self.feature_chunk_len:].clone()
        self.feature_buffer[:, -self.feature_chunk_len:] = features[:, -self.feature_chunk_len:].clone()

    def get_feature_buffer(self) -> torch.Tensor:
        return self.feature_buffer.clone()


@dataclass
class ASRResult:
    text: str


class NemoStreamingASRService:
    def __init__(
        self,
        model: str,
        att_context_size: list[int],
        device: str,
        sample_rate: int,
        use_amp: bool,
        chunk_size_in_secs: float,
    ):
        self.model = model
        self.device = device
        self.att_context_size = att_context_size
        self.sample_rate = sample_rate
        self.use_amp = use_amp
        self.chunk_size_in_secs = chunk_size_in_secs
        self.decoder_type = None
        self.chunk_size = -1
        self.shift_size = -1
        self.left_chunks = 2

        self.asr_model = self._load_model(model)
        self.tokenizer = self.asr_model.tokenizer
        self.blank_id = self.get_blank_id()

        window_stride_in_secs = self.asr_model.cfg.preprocessor.window_stride
        model_stride = self.asr_model.cfg.encoder.subsampling_factor
        self.model_chunk_size = self.asr_model.encoder.streaming_cfg.chunk_size
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * window_stride_in_secs

        self.tokens_per_frame = math.ceil(np.trunc(self.chunk_size_in_secs / window_stride_in_secs) / model_stride)
        self.asr_model.encoder.setup_streaming_params(
            chunk_size=self.model_chunk_size // model_stride,
            shift_size=self.tokens_per_frame,
        )

        model_chunk_size_in_secs = self.model_chunk_size * window_stride_in_secs
        self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs
        self._audio_buffer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.chunk_size_in_secs,
            preprocessor_cfg=self.asr_model.cfg.preprocessor,
            device=torch.device(self.device),
        )
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()

    def _reset_cache(self) -> None:
        (
            self._cache_last_channel,
            self._cache_last_time,
            self._cache_last_channel_len,
        ) = self.asr_model.encoder.get_initial_cache_state(1)

    def _get_blank_hypothesis(self) -> list[Hypothesis]:
        return [Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)]

    @property
    def drop_extra_pre_encoded(self):
        return self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

    def get_blank_id(self) -> int:
        return len(self.tokenizer.vocab)

    def get_text_from_tokens(self, tokens: list[int]) -> str:
        separator = "\u2581"
        tokens = [int(token) for token in tokens if token != self.blank_id]
        if not tokens:
            return ""

        pieces = self.tokenizer.ids_to_tokens(tokens)
        return "".join(
            [piece.replace(separator, " ") if piece.startswith(separator) else piece for piece in pieces]
        )

    def _load_model(self, model: str):
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model,
            map_location=torch.device(self.device),
        )

        if isinstance(asr_model, nemo_asr.models.EncDecCTCModel):
            self.decoder_type = "ctc"
        elif isinstance(asr_model, nemo_asr.models.EncDecRNNTModel):
            self.decoder_type = "rnnt"
        else:
            raise ValueError("Decoder type not supported for this model.")

        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
        else:
            raise ValueError("Model does not support multiple lookaheads.")

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = True
            if hasattr(asr_model, "joint"):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
            asr_model.change_decoding_strategy(decoding_cfg)

        asr_model.eval()
        return asr_model

    def _get_best_hypothesis(self, encoded, encoded_len, partial_hypotheses=None):
        if self.decoder_type == "ctc":
            return self.asr_model.decoding.ctc_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
            )

        return self.asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=True,
            partial_hypotheses=partial_hypotheses,
        )

    def _get_tokens_from_alignments(self, alignments) -> list[int]:
        tokens: list[int] = []
        if self.decoder_type == "ctc":
            all_tokens = alignments[1]
            for token_id in all_tokens:
                token_id = int(token_id)
                if token_id != self.blank_id:
                    tokens.append(token_id)
            return tokens

        for timestep_alignments in alignments:
            for _logits, token_id in timestep_alignments:
                token_id = int(token_id)
                if token_id != self.blank_id:
                    tokens.append(token_id)
        return tokens

    def transcribe(self, audio: bytes, stream_id: str = "default") -> ASRResult:
        del stream_id
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.update(audio_array)

        features = self._audio_buffer.get_feature_buffer()
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)
        features = features.unsqueeze(0)

        with torch.no_grad():
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.asr_model.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop_extra_pre_encoded,
            )

        best_hyp = self._get_best_hypothesis(
            encoded,
            encoded_len,
            partial_hypotheses=self._previous_hypotheses,
        )
        self._previous_hypotheses = best_hyp
        self._cache_last_channel = cache_last_channel
        self._cache_last_time = cache_last_time
        self._cache_last_channel_len = cache_last_channel_len

        tokens = self._get_tokens_from_alignments(best_hyp[0].alignments)
        return ASRResult(text=self.get_text_from_tokens(tokens))

    def reset_state(self, stream_id: str = "default") -> None:
        del stream_id
        self._audio_buffer.reset()
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class ServerConfig:
    host: str = os.getenv("NEMOTRON_HOST", "0.0.0.0")
    port: int = _env_int("NEMOTRON_PORT", 8765)
    model_name: str = os.getenv(
        "NEMOTRON_MODEL",
        "nvidia/nemotron-speech-streaming-en-0.6b",
    )
    sample_rate_hz: int = _env_int("NEMOTRON_SAMPLE_RATE_HZ", 16000)
    chunk_size_ms: int = _env_int("NEMOTRON_CHUNK_SIZE_MS", 160)
    att_right_context: int = _env_int("NEMOTRON_ATT_RIGHT_CONTEXT", 1)
    speech_rms_threshold: float = _env_float("NEMOTRON_SPEECH_RMS_THRESHOLD", 0.003)
    finalize_silence_sec: float = _env_float("NEMOTRON_FINALIZE_SILENCE_SEC", 0.45)
    use_amp: bool = os.getenv("NEMOTRON_USE_AMP", "1").strip() not in {"0", "false", "False"}


class NemotronConnection:
    def __init__(self, ws: web.WebSocketResponse, streamer: NemoStreamingASRService, config: ServerConfig):
        self.ws = ws
        self.streamer = streamer
        self.config = config
        self.chunk_size_samples = int(config.sample_rate_hz * (config.chunk_size_ms / 1000.0))

        self._speech_active = False
        self._last_speech_at = 0.0
        self._pending_audio = np.zeros((0,), dtype=np.float32)
        self._last_partial_text = ""

    async def handle(self) -> None:
        await self._send({"type": "ready"})

        async for msg in self.ws:
            if msg.type == WSMsgType.TEXT:
                await self._handle_text(msg.data)
                continue

            if msg.type == WSMsgType.ERROR:
                break

        await self._finalize(force=False)
        self._reset()

    async def _handle_text(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send({"type": "error", "message": "invalid JSON"})
            return

        message_type = str(payload.get("type") or "").strip().lower()
        if message_type == "start":
            self._reset()
            return

        if message_type == "audio":
            encoded = payload.get("audio", "") or ""
            if not encoded:
                return
            audio = np.frombuffer(base64.b64decode(encoded), dtype=np.int16).astype(np.float32) / 32768.0
            await self._process_audio(audio)
            return

        if message_type == "commit":
            return

        if message_type == "clear":
            self._reset()
            return

        if message_type == "done":
            await self._finalize(force=True)
            return

    async def _process_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return

        now = asyncio.get_running_loop().time()
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-10))
        has_speech = rms >= self.config.speech_rms_threshold

        if has_speech and not self._speech_active:
            self._speech_active = True
            self._last_speech_at = now
            self._pending_audio = np.zeros((0,), dtype=np.float32)
            self._last_partial_text = ""
            await asyncio.to_thread(self.streamer.reset_state)

        if not self._speech_active:
            return

        if has_speech:
            self._last_speech_at = now

        self._pending_audio = np.concatenate([self._pending_audio, audio]).astype(np.float32, copy=False)
        while self._pending_audio.shape[0] >= self.chunk_size_samples:
            chunk = self._pending_audio[:self.chunk_size_samples]
            self._pending_audio = self._pending_audio[self.chunk_size_samples:]
            await self._transcribe_chunk(chunk)

        if not has_speech and (now - self._last_speech_at) >= self.config.finalize_silence_sec:
            await self._finalize(force=True)

    async def _transcribe_chunk(self, chunk: np.ndarray) -> None:
        pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        result = await asyncio.to_thread(self.streamer.transcribe, pcm16, "default")
        text = self._normalize_text(result.text)
        if not text:
            return

        if text == self._last_partial_text:
            return

        self._last_partial_text = text
        await self._send({"type": "partial", "text": text})

    async def _finalize(self, force: bool) -> None:
        if not self._speech_active:
            return

        if force and self._pending_audio.size > 0:
            padded = np.pad(
                self._pending_audio,
                (0, max(0, self.chunk_size_samples - self._pending_audio.shape[0])),
                mode="constant",
            ).astype(np.float32, copy=False)
            self._pending_audio = np.zeros((0,), dtype=np.float32)
            await self._transcribe_chunk(padded)

        if self._last_partial_text:
            await self._send({"type": "final", "text": self._last_partial_text})

        self._reset()

    async def _send(self, payload: dict) -> None:
        await self.ws.send_str(json.dumps(payload))

    def _reset(self) -> None:
        self._speech_active = False
        self._last_speech_at = 0.0
        self._pending_audio = np.zeros((0,), dtype=np.float32)
        self._last_partial_text = ""
        self.streamer.reset_state()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())


def create_app(config: ServerConfig) -> web.Application:
    app = web.Application()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    streamer = NemoStreamingASRService(
        model=config.model_name,
        att_context_size=[70, config.att_right_context],
        device=device,
        sample_rate=config.sample_rate_hz,
        use_amp=config.use_amp and device == "cuda",
        chunk_size_in_secs=config.chunk_size_ms / 1000.0,
    )
    app["streamer"] = streamer
    app["config"] = config

    async def stream_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)

        connection = NemotronConnection(ws, request.app["streamer"], request.app["config"])
        await connection.handle()
        return ws

    app.router.add_get("/stream", stream_handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("NEMOTRON_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("NEMOTRON_PORT", 8765))
    args = parser.parse_args()

    config = ServerConfig(host=args.host, port=args.port)
    app = create_app(config)
    web.run_app(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
