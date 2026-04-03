from __future__ import annotations

import threading
import queue
import time
from typing import Optional, Callable

import numpy as np

from core.audio_session import AudioSession, AudioSessionConfig
from core.chunk_processor import (
    ChunkProcessor,
    ChunkProcessorConfig,
    ProcessingMode,
)
from core.audio_service import (
    move_app_playback_to_sink,
    move_app_recording_to_source,
)
from core.sst.nim_realtime_stt_service import (
    NIMRealtimeSTTService,
    NIMRealtimeSTTConfig,
)
from core.translation.translation_service import (
    TranslationService,
    TranslationConfig,
    TranslationDirection,
)
from core.tts.tts_service import TTSService, TTSConfig


class AudioEngine:
    def __init__(self):
        self.session: Optional[AudioSession] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        self.on_log: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        self.processor: Optional[ChunkProcessor] = None

        self.current_samplerate: int = 48000
        self.current_channels: int = 1
        self.current_blocksize: int = 1024

        self.realtime_stt: Optional[NIMRealtimeSTTService] = None
        self.translation_service: Optional[TranslationService] = None
        self.tts_service: Optional[TTSService] = None

        self.last_final_text: str = ""
        self.last_translated_text: str = ""

        self._translation_lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._restart_in_progress = False

    def start(
            self,
            input_device_index: int,
            output_device_index: int,
            selected_pactl_input_name: str,
            selected_pactl_output_name: str,
            samplerate: int = 48000,
            channels: int = 1,
            blocksize: int = 1024,
            stt_window_seconds: float = 1.0,
    ) -> None:
        if self.running:
            self._log("AudioEngine already running")
            return

        config = AudioSessionConfig(
            input_device_index=input_device_index,
            output_device_index=output_device_index,
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
        )

        self.current_samplerate = samplerate
        self.current_channels = channels
        self.current_blocksize = blocksize

        self.last_final_text = ""
        self.last_translated_text = ""
        self._restart_in_progress = False

        self.processor = ChunkProcessor(
            ChunkProcessorConfig(
                samplerate=samplerate,
                channels=channels,
            )
        )

        self._log("Loading translation model: EN->RU ...")
        self.translation_service = TranslationService(
            TranslationConfig(
                direction=TranslationDirection.EN_TO_RU,
                enabled=True,
            )
        )
        self._log("Translation model loaded")

        self._log("Loading TTS voice: ru_RU-dmitri-medium ...")
        self.tts_service = TTSService(
            TTSConfig(
                voice_name="ru_RU-dmitri-medium",
                data_dir="/media/yaroslav/DATA/ai_models/piper",
                use_cuda=False,
            )
        )
        self._log("TTS voice loaded")

        self._log("Connecting realtime STT: NVIDIA NIM WebSocket (en-US) ...")
        self.realtime_stt = NIMRealtimeSTTService(
            NIMRealtimeSTTConfig(
                base_url="http://localhost:9000",
                ws_url="ws://localhost:9000/v1/realtime?intent=transcription",
                language="en-US",
                sample_rate_hz=16000,
                num_channels=1,
                timeout=10.0,
                commit_interval_sec=0.5,
                enable_automatic_punctuation=True,
                on_log=self._log,
            )
        )
        self.realtime_stt.start(
            partial_callback=self._on_realtime_partial,
            final_callback=self._on_realtime_final,
        )
        self._log("Realtime STT connected")

        self.session = AudioSession(config)
        self.session.on_error = self._handle_error
        self.session.start()

        moved_input = move_app_recording_to_source(selected_pactl_input_name)
        self._log(
            f"Move recording stream to source '{selected_pactl_input_name}': {'OK' if moved_input else 'FAILED'}"
        )

        moved_output = move_app_playback_to_sink(selected_pactl_output_name)
        self._log(
            f"Move playback stream to sink '{selected_pactl_output_name}': {'OK' if moved_output else 'FAILED'}"
        )

        self.running = True

        self.worker_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="audio-processing-loop",
        )
        self.worker_thread.start()

        self._log(
            "AudioEngine started | "
            f"samplerate={samplerate}, channels={channels}, blocksize={blocksize}, "
            "pipeline=realtime EN->RU"
        )

    def stop(self) -> None:
        if not self.running and self.session is None:
            return

        self._log("Stopping AudioEngine...")
        self.running = False

        worker_thread = self.worker_thread
        session = self.session
        realtime_stt = self.realtime_stt

        self.worker_thread = None
        self.session = None
        self.realtime_stt = None

        if realtime_stt is not None:
            try:
                realtime_stt.stop()
            except Exception as error:
                self._handle_error(f"Realtime STT stop error: {error}")

        if session is not None:
            session.stop()

        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=2.0)

        self.translation_service = None
        self.tts_service = None

        self.last_final_text = ""
        self.last_translated_text = ""
        self._restart_in_progress = False

        self._log("AudioEngine stopped")

    def set_mode(self, mode: ProcessingMode) -> None:
        if self.processor is None:
            self._log(f"Processor is not initialized yet, deferred mode={mode.value}")
            return

        self.processor.set_mode(mode)
        self._log(f"Processing mode changed to: {mode.value}")

    def get_mode(self) -> str:
        if self.processor is None:
            return ProcessingMode.PASSTHROUGH.value

        return self.processor.mode.value

    def _processing_loop(self) -> None:
        while self.running:
            session = self.session
            if session is None or session.stopping:
                break

            try:
                chunk = session.input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.running:
                break

            try:
                processed_chunk = self.process_chunk(chunk)

                if not self.running:
                    break

                if self._should_output_processed_audio():
                    try:
                        session.output_queue.put_nowait(processed_chunk)
                    except queue.Full:
                        pass

                if self.realtime_stt is not None:
                    self.realtime_stt.send_audio_chunk(
                        processed_chunk,
                        self.current_samplerate,
                    )

            except Exception as error:
                self._handle_error(f"Processing error: {error}")

    def _on_realtime_partial(self, text: str) -> None:
        if not text:
            return
        self._log(f"PARTIAL: {text}")

    def _on_realtime_final(self, text: str) -> None:
        if not self.running:
            return

        text = self._normalize_text(text)
        if not text:
            self._log("FINAL: <empty>")
            return

        with self._translation_lock:
            if not self.running:
                return

            incremental_text = self._extract_incremental_text(self.last_final_text, text)
            if not incremental_text:
                self._log("FINAL skipped: duplicate/overlap")
                self._restart_realtime_stt_async()
                return

            self.last_final_text = text
            self._log(f"FINAL: {text}")
            self._log(f"FINAL incremental: {incremental_text}")

            if self.translation_service is None:
                self._restart_realtime_stt_async()
                return

            translation_started_at = time.perf_counter()
            try:
                translated_text = self.translation_service.translate(incremental_text)
            except Exception as error:
                self._handle_error(f"Translation error: {error}")
                self._restart_realtime_stt_async()
                return

            translated_text = self._normalize_text(translated_text)
            translation_elapsed = time.perf_counter() - translation_started_at
            self._log(f"Translation time: {translation_elapsed:.3f} sec")

            if not translated_text:
                self._log("TRANSLATED: <empty>")
                self._restart_realtime_stt_async()
                return

            if self._should_skip_translated_text(translated_text):
                self._log("TRANSLATED skipped: duplicate")
                self._restart_realtime_stt_async()
                return

            self.last_translated_text = translated_text
            self._log(f"TRANSLATED: {translated_text}")

            if self.tts_service is None:
                self._restart_realtime_stt_async()
                return

            tts_started_at = time.perf_counter()
            try:
                tts_audio = self.tts_service.synthesize(
                    translated_text,
                    target_samplerate=self.current_samplerate,
                )
            except Exception as error:
                self._handle_error(f"TTS error: {error}")
                self._restart_realtime_stt_async()
                return

            tts_elapsed = time.perf_counter() - tts_started_at
            self._log(f"TTS time: {tts_elapsed:.3f} sec")

            duration = (
                tts_audio.shape[0] / float(self.current_samplerate)
                if tts_audio.size > 0
                else 0.0
            )
            self._log(f"TTS audio ready: {duration:.2f} sec")

            self._enqueue_tts_audio(tts_audio)
            self._restart_realtime_stt_async()

    def _restart_realtime_stt_async(self) -> None:
        if self.realtime_stt is None or not self.running:
            return

        with self._restart_lock:
            if self._restart_in_progress:
                return
            self._restart_in_progress = True

        def worker():
            try:
                if self.realtime_stt is not None and self.running:
                    self._log("Restarting realtime STT session...")
                    self.realtime_stt.restart()
                    self._log("Realtime STT session restarted")
            except Exception as error:
                self._handle_error(f"Realtime STT restart error: {error}")
            finally:
                with self._restart_lock:
                    self._restart_in_progress = False

        threading.Thread(
            target=worker,
            daemon=True,
            name="realtime-stt-restart",
        ).start()

    def _should_output_processed_audio(self) -> bool:
        if self.processor is None:
            return False

        return self.processor.mode in {ProcessingMode.MUTE, ProcessingMode.TEST_TONE}

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if self.processor is None:
            return chunk.astype(np.float32, copy=True)

        return self.processor.process(chunk)

    def _enqueue_tts_audio(self, audio: np.ndarray) -> None:
        session = self.session
        if session is None or not self.running or session.stopping:
            return

        if audio.size == 0:
            return

        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        total_frames = audio.shape[0]
        offset = 0

        while self.running and offset < total_frames:
            session = self.session
            if session is None or session.stopping:
                return

            piece = audio[offset: offset + self.current_blocksize]

            if piece.shape[0] < self.current_blocksize:
                padding = np.zeros(
                    (self.current_blocksize - piece.shape[0], self.current_channels),
                    dtype=np.float32,
                )
                piece = np.vstack([piece, padding])

            try:
                session.output_queue.put(piece.astype(np.float32, copy=False), timeout=0.2)
            except queue.Full:
                if not self.running:
                    return
                continue

            offset += self.current_blocksize

    def _should_skip_translated_text(self, text: str) -> bool:
        previous = self._normalize_compare_text(self.last_translated_text)
        current = self._normalize_compare_text(text)

        if not current:
            return True

        if not previous:
            return False

        if current == previous:
            return True

        if current in previous:
            return True

        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def _normalize_compare_text(text: str) -> str:
        text = AudioEngine._normalize_text(text).lower()
        text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        return " ".join(text.split()).strip()

    @staticmethod
    def _extract_incremental_text(previous: str, current: str) -> str:
        previous_norm = AudioEngine._normalize_compare_text(previous)
        current_norm = AudioEngine._normalize_compare_text(current)

        if not current_norm:
            return ""

        if not previous_norm:
            return current.strip()

        if current_norm == previous_norm:
            return ""

        if current_norm.startswith(previous_norm):
            extra_raw = current[len(previous):].strip()
            if extra_raw:
                return extra_raw

        prev_words = previous_norm.split()
        curr_words = current_norm.split()

        common_prefix_len = 0
        for prev_word, curr_word in zip(prev_words, curr_words):
            if prev_word != curr_word:
                break
            common_prefix_len += 1

        if common_prefix_len > 0 and common_prefix_len < len(curr_words):
            raw_words = current.strip().split()
            if common_prefix_len < len(raw_words):
                return " ".join(raw_words[common_prefix_len:]).strip()

        return current.strip()

    def _handle_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)
        else:
            self._log(message)

    def _log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)