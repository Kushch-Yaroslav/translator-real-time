from __future__ import annotations

import threading
import queue
import re
import time
from difflib import SequenceMatcher
from typing import Optional, Callable

import numpy as np

from core.audio.audio_session import AudioSession, AudioSessionConfig
from core.audio.chunk_processor import (
    ChunkProcessor,
    ChunkProcessorConfig,
    ProcessingMode,
)
from core.audio.audio_service import (
    log_audio_routing_snapshot,
    move_app_playback_to_sink,
    move_app_recording_to_source,
    snapshot_sink_input_ids,
    snapshot_source_output_ids,
)
from core.config.app_config import AppConfig, DEFAULT_CONFIG, TranslationBranchConfig, get_default_branch_config
from core.sst.confirm_stt_factory import create_confirm_stt_service
from core.sst.confirm_stt_service import ConfirmSTTService
from core.sst.realtime_stt_factory import create_realtime_stt_service
from core.sst.realtime_stt_protocol import RealtimeSTTService
from core.sst.sentence_boundary_segmenter import (
    SentenceBoundarySegmenter,
    SentenceBoundarySegmenterConfig,
)
from core.translation.translation_service import (
    TranslationService,
    TranslationConfig,
    TranslationDirection,
)
from core.tts.tts_service import TTSService, TTSConfig


class AudioEngine:
    _KNOWN_STANDALONE_STT_HALLUCINATIONS = frozenset({
        "продолжение следует",
    })
    _KNOWN_EN_STANDALONE_STT_HALLUCINATIONS = frozenset({
        "welcome to the american league of legends",
    })

    def __init__(
        self,
        app_config: AppConfig | None = None,
        active_branch_config: TranslationBranchConfig | None = None,
    ):
        self.session: Optional[AudioSession] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.phrase_worker_thread: Optional[threading.Thread] = None
        self.final_worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.app_config = app_config or DEFAULT_CONFIG
        self.active_branch_config = active_branch_config or get_default_branch_config(self.app_config)

        self.on_log: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_input_level: Optional[Callable[[float], None]] = None

        self.processor: Optional[ChunkProcessor] = None

        self.current_samplerate: int = 48000
        self.current_channels: int = 1
        self.current_blocksize: int = 1024

        self.realtime_stt: Optional[RealtimeSTTService] = None
        self.sentence_segmenter: Optional[SentenceBoundarySegmenter] = None
        self.confirm_stt_service: Optional[ConfirmSTTService] = None
        self.translation_service: Optional[TranslationService] = None
        self.tts_service: Optional[TTSService] = None
        self._service_cache_lock = threading.RLock()
        self._cached_translation_key: tuple | None = None
        self._cached_translation_service: Optional[TranslationService] = None
        self._cached_tts_key: tuple | None = None
        self._cached_tts_service: Optional[TTSService] = None
        self._cached_realtime_stt_key: tuple | None = None
        self._cached_realtime_stt_service: Optional[RealtimeSTTService] = None
        self._cached_realtime_stt_label: str = ""

        self.last_final_text: str = ""
        self.last_translated_text: str = ""
        self.last_enqueued_final_text: str = ""
        self.last_emitted_source_text: str = ""
        self.last_translated_at: float = 0.0
        self._last_stt_activity_at: float = 0.0

        self.final_text_queue: queue.Queue[str] = queue.Queue(maxsize=32)
        self.low_latency_text_queue: queue.Queue[tuple[str, bool, int]] = queue.Queue(maxsize=32)
        self.tts_text_queue: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=64)
        self.tts_worker_thread: Optional[threading.Thread] = None
        self._tts_stop_event = threading.Event()
        self._pending_final_text: str = ""
        self._pending_final_updated_at: float = 0.0
        self._pending_final_is_partial: bool = False
        self._pending_final_lock = threading.Lock()
        self._last_partial_text: str = ""
        self._last_partial_changed_at: float = 0.0
        self._last_partial_emitted_text: str = ""
        self._stable_complete_sentences: list[str] = []
        self._stable_complete_sentences_changed_at: float = 0.0
        self._partial_lock = threading.Lock()
        self._partial_promoted_since_last_final = False
        self._emitted_complete_sentence_count: int = 0
        self._confirmed_sentence_count: int = 0
        self._utterance_audio_chunks: list[np.ndarray] = []
        self._utterance_audio_total_frames: int = 0
        self._utterance_audio_lock = threading.Lock()
        self._stt_speech_hangover_chunks = 0
        self._stt_speech_hangover_max_chunks = 1
        self._low_latency_generation = 0
        self._low_latency_last_partial_text = ""
        self._low_latency_partial_repeat_count = 0
        self._low_latency_emitted_text = ""
        self._low_latency_last_queued_text = ""
        self._low_latency_partial_queued = False
        self._low_latency_deferred_partial_text = ""
        self._ru_to_en_last_emitted_at: float = 0.0
        self._ru_to_en_emitted_phrase_norms: set[str] = set()
        self._ru_to_en_emitted_phrases: list[str] = []
        self._ru_to_en_pending_phrase_text: str = ""
        self._ru_to_en_pending_phrase_norm: str = ""
        self._ru_to_en_pending_phrase_seen_count: int = 0
        self._ru_to_en_pending_phrase_first_seen_at: float = 0.0
        self._ru_to_en_pending_phrase_last_seen_at: float = 0.0
        self._translation_paused = False
        self._last_input_level = 0.0

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
            pulse_stream_tag: str | None = None,
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
        self._stt_speech_hangover_chunks = 0
        self._stt_speech_hangover_max_chunks = max(
            1,
            int(
                round(
                    self.app_config.stt.noise_gate_hangover_sec
                    / (blocksize / float(samplerate))
                )
            ),
        )

        self.last_final_text = ""
        self.last_translated_text = ""
        self.last_enqueued_final_text = ""
        self.last_emitted_source_text = ""
        self.last_translated_at = 0.0
        self._last_stt_activity_at = 0.0
        self._clear_final_text_queue()
        self._clear_low_latency_queue()
        self._clear_tts_text_queue()
        self._clear_pending_final()
        self._clear_partial_state()
        self.sentence_segmenter = None
        self.confirm_stt_service = None

        branch_config = self._get_active_branch_config()

        self.processor = ChunkProcessor(
            ChunkProcessorConfig(
                samplerate=samplerate,
                channels=channels,
            )
        )

        if self._stt_backend_outputs_translated_text():
            self.translation_service = None
            self._log("Translation model bypassed: backend returns translated text")
        else:
            self.translation_service, translation_cached = self._get_or_create_translation_service(
                branch_config
            )
            self._log(
                "Translation model loaded"
                if not translation_cached
                else "Translation model ready (cached)"
            )

        if self._uses_boundary_layer_streaming():
            self.sentence_segmenter = SentenceBoundarySegmenter(
                SentenceBoundarySegmenterConfig(
                    stability_sec=max(0.45, self.app_config.stt.partial_stability_sec),
                    min_words=max(4, self.app_config.stt.partial_min_words),
                    on_log=self._log,
                )
            )
            self.confirm_stt_service = create_confirm_stt_service(
                self.app_config,
                branch_config,
                on_log=self._log,
            )

        self.tts_service, tts_cached = self._get_or_create_tts_service(branch_config)
        self._log(
            ("TTS voice loaded " if not tts_cached else "TTS voice ready (cached) ")
            + f"(backend={self.tts_service.get_runtime_backend_label()})"
        )

        self.realtime_stt, stt_backend_label, stt_cached = self._get_or_create_realtime_stt_service(
            branch_config
        )
        self._log(
            f"{'Connecting' if not stt_cached else 'Reusing'} realtime STT: "
            f"{stt_backend_label} ({branch_config.stt_language}) ..."
        )
        self.realtime_stt.start(
            partial_callback=self._on_realtime_partial,
            final_callback=self._on_realtime_final,
        )
        self._log("Realtime STT connected")

        self.running = True
        self._tts_stop_event.clear()
        self.tts_worker_thread = threading.Thread(
            target=self._tts_worker_loop,
            daemon=True,
            name="tts-worker-loop",
        )
        self.tts_worker_thread.start()

        sink_input_ids_before = snapshot_sink_input_ids()
        source_output_ids_before = snapshot_source_output_ids()

        self.session = AudioSession(config)
        self.session.on_error = self._handle_error
        self.session.start()

        log_audio_routing_snapshot(self._log)

        moved_input = move_app_recording_to_source(
            selected_pactl_input_name,
            logger=self._log,
            existing_ids=source_output_ids_before,
            stream_tag=pulse_stream_tag,
        )
        self._log(
            f"Move recording stream to source '{selected_pactl_input_name}': {'OK' if moved_input else 'FAILED'}"
        )

        moved_output = move_app_playback_to_sink(
            selected_pactl_output_name,
            logger=self._log,
            existing_ids=sink_input_ids_before,
            stream_tag=pulse_stream_tag,
        )
        self._log(
            f"Move playback stream to sink '{selected_pactl_output_name}': {'OK' if moved_output else 'FAILED'}"
        )

        log_audio_routing_snapshot(self._log)

        self.worker_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="audio-processing-loop",
        )
        self.worker_thread.start()

        self.phrase_worker_thread = threading.Thread(
            target=self._phrase_processing_loop,
            daemon=True,
            name="phrase-processing-loop",
        )
        self.phrase_worker_thread.start()

        self.final_worker_thread = threading.Thread(
            target=self._final_debounce_loop,
            daemon=True,
            name="final-debounce-loop",
        )
        self.final_worker_thread.start()

        self._log(
            "AudioEngine started | "
            f"samplerate={samplerate}, channels={channels}, blocksize={blocksize}, "
            f"pipeline=realtime {branch_config.label}"
        )

    def stop(self) -> None:
        if not self.running and self.session is None:
            return

        self._log("Stopping AudioEngine...")
        self.running = False

        worker_thread = self.worker_thread
        phrase_worker_thread = self.phrase_worker_thread
        final_worker_thread = self.final_worker_thread
        tts_worker_thread = self.tts_worker_thread
        session = self.session
        realtime_stt = self.realtime_stt

        self.worker_thread = None
        self.phrase_worker_thread = None
        self.final_worker_thread = None
        self.session = None
        self.realtime_stt = None
        self.sentence_segmenter = None
        self.confirm_stt_service = None
        self.tts_worker_thread = None
        self._tts_stop_event.set()

        if realtime_stt is not None:
            try:
                realtime_stt.stop()
            except Exception as error:
                self._handle_error(f"Realtime STT stop error: {error}")

        if session is not None:
            session.stop()

        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=2.0)

        if phrase_worker_thread and phrase_worker_thread.is_alive():
            phrase_worker_thread.join(timeout=2.0)

        if final_worker_thread and final_worker_thread.is_alive():
            final_worker_thread.join(timeout=2.0)

        if tts_worker_thread and tts_worker_thread.is_alive():
            tts_worker_thread.join(timeout=2.0)

        self.translation_service = None
        self.tts_service = None

        self.last_final_text = ""
        self.last_translated_text = ""
        self.last_enqueued_final_text = ""
        self.last_emitted_source_text = ""
        self.last_translated_at = 0.0
        self._last_stt_activity_at = 0.0
        self._clear_final_text_queue()
        self._clear_low_latency_queue()
        self._clear_tts_text_queue()
        self._clear_pending_final()
        self._clear_partial_state()

        self._emit_input_level(0.0)
        self._log("AudioEngine stopped")

    def prewarm_runtime(self) -> None:
        branch_config = self._get_active_branch_config()
        self._log("Runtime prewarm started")

        if not self._stt_backend_outputs_translated_text():
            translation_service, translation_cached = self._get_or_create_translation_service(branch_config)
            if not translation_cached:
                self._log(f"Prewarming translation model: {branch_config.label}")
                translation_service.warmup()

        tts_service, tts_cached = self._get_or_create_tts_service(branch_config)
        if not tts_cached:
            self._log(f"Prewarming TTS voice: {branch_config.tts_voice_name}")
            tts_service.warmup(target_samplerate=self.app_config.audio.samplerate)

        realtime_stt, stt_backend_label, stt_cached = self._get_or_create_realtime_stt_service(
            branch_config
        )
        if not stt_cached and hasattr(realtime_stt, "warmup"):
            self._log(f"Prewarming realtime STT: {stt_backend_label}")
            realtime_stt.warmup()

        self._log("Runtime prewarm ready")

    def set_mode(self, mode: ProcessingMode) -> None:
        if self.processor is None:
            self._log(f"Processor is not initialized yet, deferred mode={mode.value}")
            return

        self.processor.set_mode(mode)
        self._log(f"Processing mode changed to: {mode.value}")

    def set_translation_paused(self, paused: bool) -> None:
        self._translation_paused = bool(paused)

        if self.processor is not None:
            self.processor.set_mode(
                ProcessingMode.MUTE if self._translation_paused else ProcessingMode.PASSTHROUGH
            )

        self._clear_final_text_queue()
        self._clear_low_latency_queue()
        self._clear_tts_text_queue()
        self._clear_pending_final()
        self._clear_partial_state()
        self._reset_output_audio_queue()

        state = "paused" if self._translation_paused else "resumed"
        self._log(f"Translation stream {state}")

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
                self._emit_input_level(self._calculate_input_level(chunk))
                processed_chunk = self.process_chunk(chunk)

                if not self.running:
                    break

                if self._should_output_processed_audio():
                    try:
                        session.output_queue.put_nowait(processed_chunk)
                    except queue.Full:
                        pass

                if self.realtime_stt is not None:
                    stt_chunk = self._prepare_chunk_for_stt(processed_chunk)
                    self._append_utterance_audio_chunk(stt_chunk)
                    self.realtime_stt.send_audio_chunk(
                        stt_chunk,
                        self.current_samplerate,
                    )

            except Exception as error:
                self._handle_error(f"Processing error: {error}")

    @staticmethod
    def _calculate_input_level(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0

        audio = chunk.astype(np.float32, copy=False)
        mono = audio[:, 0] if audio.ndim == 2 else audio
        if mono.size == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-10))
        normalized = min(1.0, max(0.0, rms * 8.0))
        return normalized

    def _emit_input_level(self, level: float) -> None:
        smoothed = max(float(level), self._last_input_level * 0.82)
        self._last_input_level = smoothed
        if self.on_input_level is not None:
            self.on_input_level(smoothed)

    def _on_realtime_partial(self, text: str) -> None:
        if self._translation_paused:
            return

        if not text:
            return

        now = time.monotonic()
        self._maybe_reset_utterance_state(now)

        text = self._normalize_text(text)
        if not text:
            return

        sanitized_text = self._strip_known_stt_hallucination_tail(text)
        if sanitized_text != text:
            self._log(f"PARTIAL sanitized: removed hallucinated tail ({text} -> {sanitized_text or '<empty>'})")
        text = sanitized_text
        if not text:
            return

        if self._uses_low_latency_direct_pipeline():
            self._last_stt_activity_at = now
            self._log(f"PARTIAL: {text}")
            self._handle_low_latency_partial(text)
            return

        if self._stt_backend_outputs_translated_text():
            self._last_stt_activity_at = now
            self._log(f"PARTIAL: {text}")
            return

        if self._uses_boundary_layer_streaming():
            self._last_stt_activity_at = now
            self._log(f"PARTIAL: {text}")
            segmenter = self.sentence_segmenter
            if segmenter is not None:
                new_sentences = segmenter.push_partial(text)
                if new_sentences:
                    target_sentence_count = self._confirmed_sentence_count + len(new_sentences)
                    self._confirm_and_emit_buffered_sentences(
                        target_sentence_count,
                        fallback_sentences=new_sentences,
                        is_final=False,
                    )
            return

        with self._partial_lock:
            if self._normalize_compare_text(text) != self._normalize_compare_text(self._last_partial_text):
                self._last_partial_text = text
                self._last_partial_changed_at = time.monotonic()

            if self._uses_sentence_partial_streaming():
                complete_sentences, _ = self._split_sentences(text)
                if (
                    self._normalize_sentence_list(complete_sentences)
                    != self._normalize_sentence_list(self._stable_complete_sentences)
                ):
                    self._stable_complete_sentences = complete_sentences
                    self._stable_complete_sentences_changed_at = time.monotonic()

        self._last_stt_activity_at = now
        self._log(f"PARTIAL: {text}")

    def _on_realtime_final(self, text: str) -> None:
        if not self.running or self._translation_paused:
            return

        now = time.monotonic()
        self._maybe_reset_utterance_state(now)

        text = self._normalize_text(text)
        text = self._collapse_immediate_repetitions(text)
        if not text:
            self._log("FINAL: <empty>")
            return

        sanitized_text = self._strip_known_stt_hallucination_tail(text)
        if sanitized_text != text:
            self._log(f"FINAL sanitized: removed hallucinated tail ({text} -> {sanitized_text or '<empty>'})")
        text = sanitized_text
        if not text:
            return

        self._partial_promoted_since_last_final = False
        self._last_stt_activity_at = now

        if self._uses_low_latency_direct_pipeline():
            self.last_final_text = text
            self._log(f"FINAL: {text}")
            self._handle_low_latency_final(text)
            return

        if self._uses_boundary_layer_streaming():
            self.last_final_text = text
            self._log(f"FINAL: {text}")
            segmenter = self.sentence_segmenter
            if segmenter is not None:
                new_sentences = segmenter.push_final(text)
                if new_sentences:
                    target_sentence_count = self._confirmed_sentence_count + len(new_sentences)
                    self._confirm_and_emit_buffered_sentences(
                        target_sentence_count,
                        fallback_sentences=new_sentences,
                        is_final=True,
                    )
            return

        if self._uses_sentence_partial_streaming():
            self.last_final_text = text
            self._log(f"FINAL: {text}")
            self._emit_sentence_stream_segments(text, is_final=True)
            return

        incremental_text = self._extract_incremental_text(self.last_emitted_source_text, text)
        if not incremental_text:
            self._log("FINAL skipped: duplicate/overlap")
            self.last_final_text = text
            return

        self.last_final_text = text
        self._log(f"FINAL: {text}")
        self._log(f"FINAL incremental: {incremental_text}")
        self._stage_final_text(incremental_text)

    def _phrase_processing_loop(self) -> None:
        while self.running:
            if self._uses_low_latency_direct_pipeline():
                try:
                    item = self.low_latency_text_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if not self.running:
                    break

                # For RU=>EN we enqueue multiple ordered phrase chunks.
                # Collapsing the queue would drop earlier chunks and cause "missing tail" speech.
                if self._get_active_branch_config().translation_direction == "ru_to_en":
                    text, is_final, generation = item
                    self._process_low_latency_text(text, is_final, generation)
                else:
                    text, is_final, generation = self._collapse_low_latency_queue(item)
                    self._process_low_latency_text(text, is_final, generation)
                continue

            try:
                text = self.final_text_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.running:
                break

            self._process_final_text(text)

    def _final_debounce_loop(self) -> None:
        while self.running:
            if self._uses_low_latency_direct_pipeline():
                time.sleep(0.05)
                continue

            if self._uses_boundary_layer_streaming():
                time.sleep(0.05)
                continue

            if self._uses_sentence_partial_streaming():
                self._maybe_emit_stable_partial()
                time.sleep(0.05)
                continue

            ready_text = ""

            with self._pending_final_lock:
                if self._pending_final_text:
                    age = time.monotonic() - self._pending_final_updated_at
                    debounce_sec = self.app_config.stt.final_debounce_sec
                    if self._pending_final_is_partial:
                        debounce_sec = self._get_partial_merge_window_sec()
                    if age >= debounce_sec:
                        ready_text = self._pending_final_text
                        self._pending_final_text = ""
                        self._pending_final_updated_at = 0.0
                        self._pending_final_is_partial = False

            if ready_text:
                self._enqueue_final_text(ready_text)
                continue

            self._maybe_emit_stable_partial()

            time.sleep(0.05)

    def _process_final_text(self, text: str) -> None:
        if not self.running or self._translation_paused:
            return

        if self._stt_backend_outputs_translated_text():
            translated_text = self._normalize_text(text)
            translation_elapsed = 0.0
        else:
            translation_service = self.translation_service
            if translation_service is None:
                self._log("Translation skipped: service is unavailable")
                return

            translation_started_at = time.perf_counter()
            try:
                translated_text = translation_service.translate(text)
            except Exception as error:
                self._handle_error(f"Translation error: {error}")
                return

            translated_text = self._normalize_text(translated_text)
            translation_elapsed = time.perf_counter() - translation_started_at

        self._log(f"Translation time: {translation_elapsed:.3f} sec")

        if not translated_text:
            self._log("TRANSLATED: <empty>")
            return

        if self._should_skip_translated_text(translated_text):
            self._log("TRANSLATED skipped: duplicate")
            return

        self.last_translated_text = translated_text
        self.last_translated_at = time.monotonic()
        self._log(f"TRANSLATED: {translated_text}")
        self._enqueue_tts_text(translated_text)

    def _enqueue_tts_text(self, text: str) -> None:
        if not self.running or self._translation_paused:
            return
        text = self._normalize_text(text)
        if not text:
            return
        try:
            self.tts_text_queue.put_nowait((text, self.current_samplerate))
        except queue.Full:
            self._handle_error("TTS text queue is full, dropping chunk")

    def _tts_worker_loop(self) -> None:
        while not self._tts_stop_event.is_set():
            if not self.running:
                break
            try:
                text, samplerate = self.tts_text_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.running or self._translation_paused:
                continue

            tts_service = self.tts_service
            if tts_service is None:
                continue

            tts_started_at = time.perf_counter()
            try:
                tts_audio = tts_service.synthesize(
                    text,
                    target_samplerate=samplerate,
                )
            except Exception as error:
                self._handle_error(f"TTS error: {error}")
                continue

            tts_elapsed = time.perf_counter() - tts_started_at
            self._log(f"TTS time: {tts_elapsed:.3f} sec")

            duration = (
                tts_audio.shape[0] / float(self.current_samplerate)
                if tts_audio.size > 0
                else 0.0
            )
            self._log(f"TTS audio ready: {duration:.2f} sec")
            self._enqueue_tts_audio(tts_audio)

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

        audio = self._trim_tts_audio_silence(audio)
        if audio.size == 0:
            return

        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        self._trim_output_queue_if_needed(session)

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

    def _trim_tts_audio_silence(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        mono = audio[:, 0] if audio.ndim == 2 else audio
        if mono.size == 0:
            return audio

        threshold = 0.0035
        active_indices = np.flatnonzero(np.abs(mono) >= threshold)
        if active_indices.size == 0:
            return audio

        pad_frames = int(0.03 * self.current_samplerate)
        start = max(0, int(active_indices[0]) - pad_frames)
        end = min(int(mono.shape[0]), int(active_indices[-1]) + pad_frames + 1)
        if audio.ndim == 2:
            return audio[start:end, :]
        return audio[start:end]

    def _trim_output_queue_if_needed(self, session: AudioSession) -> None:
        if self._uses_low_latency_direct_pipeline():
            queued_audio_seconds = session.get_output_queue_duration_seconds()
            threshold = max(3.0, self.app_config.tts.max_queue_latency_sec)
            if queued_audio_seconds <= threshold:
                return
        else:
            queued_audio_seconds = session.get_output_queue_duration_seconds()
            if queued_audio_seconds <= self.app_config.tts.max_queue_latency_sec:
                return

        cleared_blocks = session.clear_output_queue()
        self._log(
            "Dropped stale TTS audio queue "
            f"({queued_audio_seconds:.2f} sec, blocks={cleared_blocks})"
        )

    def _prepare_chunk_for_stt(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return chunk

        audio = chunk.astype(np.float32, copy=False)
        mono = audio[:, 0] if audio.ndim == 2 else audio
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-10))

        if rms >= self.app_config.stt.noise_gate_threshold:
            self._stt_speech_hangover_chunks = self._stt_speech_hangover_max_chunks
            return audio

        if self._stt_speech_hangover_chunks > 0:
            self._stt_speech_hangover_chunks -= 1
            return audio

        return np.zeros_like(audio, dtype=np.float32)

    def _clear_final_text_queue(self) -> None:
        while not self.final_text_queue.empty():
            try:
                self.final_text_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_low_latency_queue(self) -> None:
        while not self.low_latency_text_queue.empty():
            try:
                self.low_latency_text_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_tts_text_queue(self) -> None:
        while not self.tts_text_queue.empty():
            try:
                self.tts_text_queue.get_nowait()
            except queue.Empty:
                break

    def _clear_pending_final(self) -> None:
        with self._pending_final_lock:
            self._pending_final_text = ""
            self._pending_final_updated_at = 0.0
            self._pending_final_is_partial = False

    def _clear_partial_state(self) -> None:
        with self._partial_lock:
            self._last_partial_text = ""
            self._last_partial_changed_at = 0.0
            self._last_partial_emitted_text = ""
            self._stable_complete_sentences = []
            self._stable_complete_sentences_changed_at = 0.0
        self._partial_promoted_since_last_final = False
        self._emitted_complete_sentence_count = 0
        self._confirmed_sentence_count = 0
        self._low_latency_generation = 0
        self._low_latency_last_partial_text = ""
        self._low_latency_partial_repeat_count = 0
        self._low_latency_emitted_text = ""
        self._low_latency_last_queued_text = ""
        self._low_latency_partial_queued = False
        self._ru_to_en_last_emitted_at = 0.0
        self._ru_to_en_emitted_phrase_norms = set()
        self._ru_to_en_emitted_phrases = []
        self._reset_ru_to_en_pending_phrase()
        with self._utterance_audio_lock:
            self._utterance_audio_chunks = []
            self._utterance_audio_total_frames = 0

    def _reset_output_audio_queue(self) -> None:
        session = self.session
        if session is None:
            return
        session.clear_output_queue()

    def _reset_utterance_state(self) -> None:
        self.last_final_text = ""
        self.last_enqueued_final_text = ""
        self.last_emitted_source_text = ""
        self._clear_pending_final()
        self._clear_partial_state()
        if self.sentence_segmenter is not None:
            self.sentence_segmenter.reset()

    def _maybe_reset_utterance_state(self, now: float) -> None:
        if self._last_stt_activity_at <= 0.0:
            return

        inactivity_sec = now - self._last_stt_activity_at
        if inactivity_sec < 2.0:
            return

        self._reset_utterance_state()

    def _stage_final_text(self, text: str, is_partial: bool = False) -> None:
        text = self._normalize_text(text)
        if not text:
            return

        with self._pending_final_lock:
            pending_text = self._pending_final_text

            if pending_text:
                merged_text = self._merge_final_texts(pending_text, text)
                if merged_text != pending_text:
                    self._log(f"FINAL merged: {merged_text}")
                self._pending_final_text = merged_text
                self._pending_final_is_partial = self._pending_final_is_partial and is_partial
            else:
                self._pending_final_text = text
                self._pending_final_is_partial = is_partial

            self._pending_final_updated_at = time.monotonic()

    def _enqueue_final_text(self, text: str, source_text: str | None = None) -> None:
        normalized_incremental = self._normalize_compare_text(text)
        normalized_last_enqueued = self._normalize_compare_text(self.last_enqueued_final_text)

        if (
            normalized_incremental == normalized_last_enqueued
            and not self._uses_sentence_partial_streaming()
            and not self._uses_boundary_layer_streaming()
        ):
            self._log("FINAL skipped: already queued")
            return

        try:
            self.final_text_queue.put_nowait(text)
            self.last_enqueued_final_text = text
            emitted_source = source_text if source_text is not None else text
            self.last_emitted_source_text = self._merge_final_texts(
                self.last_emitted_source_text,
                emitted_source,
            )
        except queue.Full:
            self._handle_error("Final text queue is full, dropping phrase")

    def _maybe_emit_stable_partial(self) -> None:
        if not self.app_config.stt.partial_emit_enabled:
            return

        if self._uses_sentence_partial_streaming():
            self._maybe_emit_stable_partial_sentences()
            return

        with self._partial_lock:
            partial_text = self._last_partial_text
            partial_changed_at = self._last_partial_changed_at
            last_partial_emitted_text = self._last_partial_emitted_text

        if not partial_text or partial_changed_at <= 0.0:
            return

        age = time.monotonic() - partial_changed_at
        if age < self._get_partial_stability_window_sec():
            return

        if len(partial_text.split()) < self.app_config.stt.partial_min_words:
            return

        if self._normalize_compare_text(partial_text) == self._normalize_compare_text(last_partial_emitted_text):
            return

        incremental_text = self._extract_incremental_text(self.last_emitted_source_text, partial_text)
        incremental_text = self._normalize_text(incremental_text)
        if not incremental_text:
            return

        incremental_text = self._extract_emittable_partial_text(incremental_text)
        incremental_text = self._normalize_text(incremental_text)
        if not incremental_text:
            return

        if len(incremental_text.split()) < self.app_config.stt.partial_min_words:
            return

        if not self._should_emit_partial_text(partial_text, incremental_text):
            return

        self._log(f"PARTIAL promoted: {incremental_text}")
        self._stage_final_text(incremental_text, is_partial=True)
        self.last_emitted_source_text = self._merge_final_texts(
            self.last_emitted_source_text,
            incremental_text,
        )
        if (
            self._get_active_branch_config().translation_direction == "ru_to_en"
            and not self._uses_sentence_partial_streaming()
        ):
            self._partial_promoted_since_last_final = True

        with self._partial_lock:
            self._last_partial_emitted_text = partial_text

    def _maybe_emit_stable_partial_sentences(self) -> None:
        with self._partial_lock:
            stable_complete_sentences = list(self._stable_complete_sentences)
            stable_complete_sentences_changed_at = self._stable_complete_sentences_changed_at

        if not stable_complete_sentences or stable_complete_sentences_changed_at <= 0.0:
            return

        age = time.monotonic() - stable_complete_sentences_changed_at
        if age < self._get_partial_stability_window_sec():
            return

        self._emit_sentence_stream_segments(
            " ".join(stable_complete_sentences).strip(),
            is_final=False,
        )

    def _should_emit_partial_text(self, partial_text: str, incremental_text: str) -> bool:
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "ru_to_en":
            return True

        if self._partial_promoted_since_last_final and not self._uses_sentence_partial_streaming():
            return False

        normalized_partial = self._normalize_text(partial_text)
        normalized_incremental = self._normalize_text(incremental_text)
        if not normalized_partial or not normalized_incremental:
            return False

        incremental_word_count = len(normalized_incremental.split())
        if incremental_word_count < max(4, self.app_config.stt.partial_min_words):
            return False

        if normalized_incremental[-1] not in ".!?":
            return False

        partial_norm = self._normalize_compare_text(normalized_partial)
        if partial_norm.endswith("меня зовут я"):
            return False

        tail_word = "".join(ch for ch in normalized_incremental.split()[-1].lower() if ch.isalnum())
        weak_tail_words = {
            "и",
            "в",
            "во",
            "на",
            "с",
            "со",
            "к",
            "ко",
            "у",
            "из",
            "от",
            "до",
            "для",
            "по",
            "под",
            "над",
            "же",
            "а",
            "но",
            "или",
        }
        if tail_word in weak_tail_words:
            return False

        return True

    def _extract_emittable_partial_text(self, incremental_text: str) -> str:
        text = self._normalize_text(incremental_text)
        if not text:
            return ""

        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "ru_to_en":
            return text

        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend != "riva":
            return text

        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]", text))
        if not sentence_matches:
            return ""

        candidate_match = sentence_matches[0]
        candidate = candidate_match.group(0).strip()
        if not candidate:
            return ""

        trailing_text = text[candidate_match.end():].strip()
        complete_sentence_count = len(sentence_matches)
        trailing_word_count = len(trailing_text.split())

        if complete_sentence_count < 2 and trailing_word_count < 2:
            return ""

        candidate_tail_word = "".join(
            ch for ch in candidate.split()[-1].lower() if ch.isalnum()
        )
        weak_sentence_tail_words = {
            "меня",
            "тебя",
            "себя",
            "его",
            "ее",
            "её",
            "их",
            "мой",
            "моя",
            "мое",
            "моё",
            "мои",
            "наш",
            "наша",
            "наше",
            "наши",
            "этот",
            "эта",
            "это",
            "эти",
        }
        if candidate_tail_word in weak_sentence_tail_words:
            return ""

        return candidate

    def _uses_sentence_partial_streaming(self) -> bool:
        if self._uses_boundary_layer_streaming():
            return False

        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "ru_to_en":
            return False

        backend = (self.app_config.stt.backend or "nim").strip().lower()
        return backend == "riva"

    def _uses_boundary_layer_streaming(self) -> bool:
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "ru_to_en":
            return False

        backend = (self.app_config.stt.backend or "nim").strip().lower()
        return backend == "riva"

    def _stt_backend_outputs_translated_text(self) -> bool:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        return backend == "canary_ast"

    def _supports_cached_realtime_stt(self) -> bool:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        return backend == "faster_whisper"

    def _get_or_create_translation_service(
        self,
        branch_config: TranslationBranchConfig,
    ) -> tuple[TranslationService, bool]:
        translation_direction = self._resolve_translation_direction(branch_config)
        translation_device = None
        key = (
            translation_direction.value,
            bool(branch_config.enabled),
            translation_device or "auto",
        )

        with self._service_cache_lock:
            if self._cached_translation_service is not None and self._cached_translation_key == key:
                return self._cached_translation_service, True

            self._log(f"Loading translation model: {branch_config.label} ...")
            service = TranslationService(
                TranslationConfig(
                    direction=translation_direction,
                    enabled=branch_config.enabled,
                    device=translation_device,
                )
            )
            self._cached_translation_service = service
            self._cached_translation_key = key
            return service, False

    def _get_or_create_tts_service(
        self,
        branch_config: TranslationBranchConfig,
    ) -> tuple[TTSService, bool]:
        key = (
            branch_config.tts_voice_name,
            self.app_config.tts.data_dir,
            self.app_config.tts.use_cuda,
        )

        with self._service_cache_lock:
            if self._cached_tts_service is not None and self._cached_tts_key == key:
                return self._cached_tts_service, True

            self._log(f"Loading TTS voice: {branch_config.tts_voice_name} ...")
            service = TTSService(
                TTSConfig(
                    voice_name=branch_config.tts_voice_name,
                    data_dir=self.app_config.tts.data_dir,
                    use_cuda=self.app_config.tts.use_cuda,
                )
            )
            self._cached_tts_service = service
            self._cached_tts_key = key
            return service, False

    def _get_or_create_realtime_stt_service(
        self,
        branch_config: TranslationBranchConfig,
    ) -> tuple[RealtimeSTTService, str, bool]:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if not self._supports_cached_realtime_stt():
            service, label = create_realtime_stt_service(
                self.app_config,
                branch_config,
                on_log=self._log,
            )
            return service, label, False

        key = (
            backend,
            branch_config.stt_language,
            self.app_config.stt.sample_rate_hz,
            self.app_config.stt.silero_partial_interval_sec,
            self.app_config.stt.silero_min_window_sec,
            self.app_config.stt.silero_max_window_sec,
            self.app_config.stt.silero_min_silence_ms,
            self.app_config.stt.silero_speech_pad_ms,
            self.app_config.stt.silero_speech_threshold,
            self.app_config.stt.whisper_model_size,
            self.app_config.stt.whisper_compute_type,
            self.app_config.stt.whisper_beam_size,
            self.app_config.stt.whisper_best_of,
            self.app_config.stt.whisper_patience,
        )

        with self._service_cache_lock:
            if self._cached_realtime_stt_service is not None and self._cached_realtime_stt_key == key:
                return self._cached_realtime_stt_service, self._cached_realtime_stt_label, True

            service, label = create_realtime_stt_service(
                self.app_config,
                branch_config,
                on_log=self._log,
            )
            self._cached_realtime_stt_service = service
            self._cached_realtime_stt_key = key
            self._cached_realtime_stt_label = label
            return service, label, False

    def _uses_low_latency_direct_pipeline(self) -> bool:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend == "faster_whisper":
            if not self.app_config.stt.partial_emit_enabled:
                return False
            return True
        if backend == "whisper_cpp":
            branch_config = self._get_active_branch_config()
            if (
                branch_config.translation_direction == "en_to_ru"
                and not self.app_config.stt.partial_emit_enabled
            ):
                return False
            return True
        return False

    def _handle_low_latency_partial(self, text: str) -> None:
        normalized = self._normalize_text(text)
        if not normalized:
            return

        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction == "ru_to_en":
            now = time.monotonic()

            incremental = self._extract_incremental_text(self._low_latency_emitted_text, normalized)
            incremental = self._normalize_text(incremental)
            if not incremental:
                self._expire_ru_to_en_pending_phrase(now)
                return

            completed_phrases, _trailing_fragment = self._split_ru_to_en_phrases(incremental)
            if not completed_phrases:
                self._expire_ru_to_en_pending_phrase(now)
                return

            for phrase in completed_phrases:
                candidate = self._normalize_text(phrase)
                if not self._should_emit_ru_to_en_partial_phrase(candidate):
                    continue
                if not self._mark_ru_to_en_partial_phrase_seen(candidate, now):
                    return

                candidate_norm = self._normalize_compare_text(candidate)
                self._ru_to_en_last_emitted_at = now
                self._ru_to_en_emitted_phrase_norms.add(candidate_norm)
                self._low_latency_generation += 1
                self._low_latency_last_queued_text = candidate
                self._log(f"LOWLAT sentence queued: {candidate}")
                self._enqueue_low_latency_text(
                    candidate,
                    is_final=False,
                    generation=self._low_latency_generation,
                )
                self._reset_ru_to_en_pending_phrase()
                return
            return

        if self._normalize_compare_text(normalized) == self._normalize_compare_text(self._low_latency_last_partial_text):
            self._low_latency_partial_repeat_count += 1
        else:
            self._low_latency_last_partial_text = normalized
            self._low_latency_partial_repeat_count = 1

        if self._low_latency_partial_queued:
            if self._supports_low_latency_followup_partial():
                anchor_text = self._get_low_latency_incremental_anchor()
                deferred_candidate = self._select_low_latency_followup_partial_candidate_from_anchor(
                    anchor_text,
                    normalized,
                )
                deferred_candidate = self._sanitize_low_latency_partial_candidate(deferred_candidate)
                if self._should_defer_whispercpp_age_partial(deferred_candidate):
                    return
                min_words = self._get_low_latency_partial_min_words()
                if (
                    deferred_candidate
                    and len(deferred_candidate.split()) >= min_words
                    and self._normalize_compare_text(deferred_candidate)
                    != self._normalize_compare_text(self._low_latency_last_queued_text)
                ):
                    self._low_latency_deferred_partial_text = deferred_candidate
            return

        if self._low_latency_emitted_text:
            if not self._supports_low_latency_followup_partial():
                return
            candidate = self._select_low_latency_followup_partial_candidate_from_anchor(
                self._get_low_latency_incremental_anchor(),
                normalized,
            )
        else:
            candidate = self._select_low_latency_partial_candidate(normalized)

        candidate = self._sanitize_low_latency_partial_candidate(candidate)
        if self._should_defer_whispercpp_age_partial(candidate):
            return
        min_words = self._get_low_latency_partial_min_words()
        if len(candidate.split()) < min_words:
            return

        if self._normalize_compare_text(candidate) == self._normalize_compare_text(self._low_latency_last_queued_text):
            return

        self._low_latency_generation += 1
        self._low_latency_last_queued_text = candidate
        self._low_latency_partial_queued = True
        self._log(f"LOWLAT partial queued: {candidate}")
        self._enqueue_low_latency_text(candidate, is_final=False, generation=self._low_latency_generation)

    def _handle_low_latency_final(self, text: str) -> None:
        final_text = self._normalize_text(text)
        if not final_text:
            return

        branch_config = self._get_active_branch_config()
        if self._should_skip_known_low_latency_source_hallucination(final_text):
            self._log(f"LOWLAT final skipped: known hallucination ({final_text})")
            return

        if branch_config.translation_direction == "ru_to_en":
            # RU=>EN: flush remaining tail of the utterance.
            self._reset_ru_to_en_pending_phrase()
            self._enqueue_low_latency_sentence_stream_segments(final_text, is_final=True, force=True)
            return

        incremental = self._extract_incremental_text(self._get_low_latency_incremental_anchor(), final_text)
        incremental = self._normalize_text(incremental)
        if not incremental:
            self._log("LOWLAT final skipped: duplicate/overlap")
            return

        sanitized_incremental = self._sanitize_low_latency_partial_candidate(incremental)
        if not sanitized_incremental:
            self._log(f"LOWLAT final skipped: weak followup ({incremental})")
            return
        incremental = sanitized_incremental

        if self._should_skip_low_latency_final_tail(incremental):
            self._log(f"LOWLAT final skipped: weak tail ({incremental})")
            return

        if self._normalize_compare_text(incremental) == self._normalize_compare_text(self._low_latency_last_queued_text):
            return

        self._low_latency_generation += 1
        self._low_latency_last_queued_text = incremental
        self._log(f"LOWLAT final queued: {incremental}")
        self._enqueue_low_latency_text(incremental, is_final=True, generation=self._low_latency_generation)

    def _enqueue_low_latency_sentence_stream_segments(self, text: str, *, is_final: bool, force: bool = False) -> None:
        text = self._normalize_text(text)
        if not text:
            return
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction == "ru_to_en":
            completed_sentences, trailing_fragment = self._split_ru_to_en_phrases(text)
            phrase_min_words = 2
        else:
            completed_sentences, trailing_fragment = self._split_sentences(text)
            sentence_min_words = self._get_sentence_stream_min_words()

        emitted_any = False
        if branch_config.translation_direction == "ru_to_en":
            # RU=>EN: faster-whisper partials are unstable and can rewrite earlier text.
            # Using a simple index causes us to "miss" newly-appearing phrases.
            # Emit any phrase we haven't emitted yet (dedup by normalized phrase).
            for sentence in completed_sentences:
                normalized_sentence = self._prepare_ru_to_en_phrase_for_queue(sentence)
                if len(normalized_sentence.split()) < phrase_min_words:
                    continue
                phrase_norm = self._normalize_compare_text(normalized_sentence)
                if (
                    not phrase_norm
                    or phrase_norm in self._ru_to_en_emitted_phrase_norms
                    or self._should_skip_ru_to_en_overlap_phrase(normalized_sentence)
                ):
                    continue

                now = time.monotonic()
                if not force and (now - self._ru_to_en_last_emitted_at) < 0.12:
                    break
                self._ru_to_en_last_emitted_at = now
                self._remember_ru_to_en_emitted_phrase(normalized_sentence)

                self._low_latency_generation += 1
                self._low_latency_last_queued_text = normalized_sentence
                self._log(f"LOWLAT sentence queued: {normalized_sentence}")
                self._enqueue_low_latency_text(
                    normalized_sentence,
                    is_final=False,
                    generation=self._low_latency_generation,
                )
                emitted_any = True
        else:
            start_index = min(self._emitted_complete_sentence_count, len(completed_sentences))
            for sentence in completed_sentences[start_index:]:
                normalized_sentence = self._normalize_text(sentence)
                self._emitted_complete_sentence_count += 1
                if len(normalized_sentence.split()) < sentence_min_words:
                    continue

                self._low_latency_generation += 1
                self._low_latency_last_queued_text = normalized_sentence
                self._log(f"LOWLAT sentence queued: {normalized_sentence}")
                self._enqueue_low_latency_text(
                    normalized_sentence,
                    is_final=False,
                    generation=self._low_latency_generation,
                )
                emitted_any = True

        if is_final:
            trailing_fragment = self._prepare_ru_to_en_phrase_for_queue(trailing_fragment)
            if branch_config.translation_direction == "ru_to_en":
                if trailing_fragment and len(trailing_fragment.split()) >= phrase_min_words:
                    trailing_norm = self._normalize_compare_text(trailing_fragment)
                    last_queued_norm = self._normalize_compare_text(self._low_latency_last_queued_text)
                    if (
                        trailing_norm
                        and trailing_norm != last_queued_norm
                        and trailing_norm not in self._ru_to_en_emitted_phrase_norms
                        and not self._should_skip_ru_to_en_overlap_phrase(trailing_fragment)
                    ):
                        self._low_latency_generation += 1
                        self._low_latency_last_queued_text = trailing_fragment
                        self._ru_to_en_last_emitted_at = time.monotonic()
                        self._remember_ru_to_en_emitted_phrase(trailing_fragment)
                        self._log(f"LOWLAT final tail queued: {trailing_fragment}")
                        self._enqueue_low_latency_text(
                            trailing_fragment,
                            is_final=True,
                            generation=self._low_latency_generation,
                        )
                        emitted_any = True
            elif trailing_fragment and len(trailing_fragment.split()) >= sentence_min_words:
                trailing_norm = self._normalize_compare_text(trailing_fragment)
                last_queued_norm = self._normalize_compare_text(self._low_latency_last_queued_text)
                if trailing_norm and trailing_norm != last_queued_norm:
                    self._low_latency_generation += 1
                    self._low_latency_last_queued_text = trailing_fragment
                    self._log(f"LOWLAT final tail queued: {trailing_fragment}")
                    self._enqueue_low_latency_text(
                        trailing_fragment,
                        is_final=True,
                        generation=self._low_latency_generation,
                    )
                    emitted_any = True

        if not emitted_any:
            self._log(f"LOWLAT sentence stream: no new segments (final={is_final})")

    def _enqueue_low_latency_text(self, text: str, is_final: bool, generation: int) -> None:
        try:
            self.low_latency_text_queue.put_nowait((text, is_final, generation))
        except queue.Full:
            self._handle_error("Low-latency text queue is full, dropping chunk")

    def _collapse_low_latency_queue(
        self,
        item: tuple[str, bool, int],
    ) -> tuple[str, bool, int]:
        latest = item

        while True:
            try:
                candidate = self.low_latency_text_queue.get_nowait()
            except queue.Empty:
                break

            if candidate[2] > latest[2] or (candidate[2] == latest[2] and candidate[1] and not latest[1]):
                latest = candidate

        return latest

    def _process_low_latency_text(self, text: str, is_final: bool, generation: int) -> None:
        branch_config = self._get_active_branch_config()
        if (
            branch_config.translation_direction != "ru_to_en"
            and generation < self._low_latency_generation
        ):
            self._log("LOWLAT skipped: stale chunk")
            return

        self._process_final_text(text)
        self._low_latency_emitted_text = self._merge_final_texts(
            self._low_latency_emitted_text,
            text,
        )

        if is_final:
            self._low_latency_last_partial_text = ""
            self._low_latency_partial_repeat_count = 0
            self._low_latency_last_queued_text = ""
            self._low_latency_partial_queued = False
            self._low_latency_deferred_partial_text = ""
            return

        if self._supports_low_latency_followup_partial():
            self._low_latency_partial_queued = False
            deferred_candidate = self._normalize_text(self._low_latency_deferred_partial_text)
            self._low_latency_deferred_partial_text = ""
            if (
                deferred_candidate
                and self._normalize_compare_text(deferred_candidate)
                != self._normalize_compare_text(self._low_latency_last_queued_text)
            ):
                deferred_candidate = self._sanitize_low_latency_partial_candidate(deferred_candidate)
                if not deferred_candidate:
                    return
                self._low_latency_generation += 1
                self._low_latency_last_queued_text = deferred_candidate
                self._low_latency_partial_queued = True
                self._log(f"LOWLAT partial queued: {deferred_candidate}")
                self._enqueue_low_latency_text(
                    deferred_candidate,
                    is_final=False,
                    generation=self._low_latency_generation,
                )

    def _select_low_latency_partial_candidate(self, normalized: str) -> str:
        completed_sentences, _ = self._split_sentences(normalized)
        if completed_sentences:
            return self._normalize_text(completed_sentences[0])

        if self._low_latency_partial_repeat_count >= 3 and len(normalized.split()) >= 8:
            return normalized

        return ""

    def _supports_low_latency_followup_partial(self) -> bool:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        return backend in {"whisper_cpp"}

    def _get_low_latency_partial_min_words(self) -> int:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend == "whisper_cpp":
            return max(4, self.app_config.stt.partial_min_words)
        return max(6, self.app_config.stt.partial_min_words)

    def _select_low_latency_followup_partial_candidate(self, normalized: str) -> str:
        return self._select_low_latency_followup_partial_candidate_from_anchor(
            self._low_latency_emitted_text,
            normalized,
        )

    def _select_low_latency_followup_partial_candidate_from_anchor(
        self,
        anchor_text: str,
        normalized: str,
    ) -> str:
        incremental = self._extract_incremental_text(anchor_text, normalized)
        incremental = self._normalize_text(incremental)
        if not incremental:
            return ""

        completed_sentences, trailing_fragment = self._split_sentences(incremental)
        if completed_sentences:
            return self._normalize_text(" ".join(completed_sentences))

        followup_min_words = self._get_low_latency_followup_partial_min_words()
        if len(incremental.split()) >= followup_min_words:
            return incremental

        if trailing_fragment and len(trailing_fragment.split()) >= followup_min_words:
            return trailing_fragment

        return ""

    def _sanitize_low_latency_partial_candidate(self, candidate: str) -> str:
        candidate = self._normalize_text(candidate)
        if not candidate:
            return ""

        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend != "whisper_cpp":
            return candidate

        candidate = self._strip_whispercpp_repeated_followup_context(candidate)
        if not candidate:
            return ""

        emitted_norm = self._normalize_compare_text(self._low_latency_emitted_text)
        candidate_norm = self._normalize_compare_text(candidate)

        if emitted_norm and "my name is" in emitted_norm and "my name is" in candidate_norm:
            continuation = self._extract_intro_continuation(self._low_latency_emitted_text, candidate)
            continuation = self._normalize_text(continuation)
            if continuation:
                candidate = continuation

        if self._is_weak_whispercpp_followup_partial(candidate):
            return ""

        return candidate

    def _strip_whispercpp_repeated_followup_context(self, candidate: str) -> str:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend != "whisper_cpp":
            return candidate

        candidate = self._normalize_text(candidate)
        if not candidate:
            return ""

        anchor = self._normalize_compare_text(self._get_low_latency_incremental_anchor())
        candidate_norm = self._normalize_compare_text(candidate)

        if "from ukraine" not in anchor or "from ukraine" not in candidate_norm:
            return candidate

        if "from ukraine and" in candidate_norm:
            tail = re.split(r"\bfrom ukraine and\b", candidate, maxsplit=1, flags=re.IGNORECASE)[-1]
            tail = self._normalize_text(tail)
            if not tail:
                return ""
            candidate = tail
            candidate_norm = self._normalize_compare_text(candidate)

        if candidate_norm in {"i am from ukraine", "im from ukraine"}:
            return ""

        if "from ukraine" in candidate_norm and "years old" not in candidate_norm:
            return ""

        return candidate

    def _get_low_latency_incremental_anchor(self) -> str:
        emitted = self._normalize_text(self._low_latency_emitted_text)
        queued = self._normalize_text(self._low_latency_last_queued_text)
        if len(queued.split()) > len(emitted.split()):
            return queued
        return emitted or queued

    def _get_low_latency_followup_partial_min_words(self) -> int:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend == "whisper_cpp":
            return max(4, self.app_config.stt.partial_min_words)
        return max(5, self.app_config.stt.partial_min_words)

    @staticmethod
    def _is_weak_whispercpp_followup_partial(text: str) -> bool:
        normalized = AudioEngine._normalize_compare_text(text)
        if not normalized:
            return True

        if any(
            fragment in normalized
            for fragment in (
                "russian country",
                "meet with my friend",
                "mid20th century",
                "mid twentieth century",
                "from ukraine and my name is",
            )
        ):
            return True

        if "i am from ukraine and i am from ukraine" in normalized:
            return True
        if "im from ukraine and im from ukraine" in normalized:
            return True

        if re.search(r"\bmid\s*\d{1,3}\s+years\s+old\b", normalized):
            return True

        weak_trailing_tokens = {
            "a",
            "an",
            "and",
            "at",
            "for",
            "meet",
            "of",
            "the",
            "to",
            "with",
        }
        tokens = normalized.split()
        if not tokens:
            return True

        suspicious_token_pattern = re.compile(r"^(?:pn|[a-z]\d{1,3}|[a-z]{1,2}\d{1,3})$")
        if any(suspicious_token_pattern.match(token) for token in tokens):
            return True

        if tokens[-1] in weak_trailing_tokens:
            return True

        weak_suffixes = (
            "meet the",
            "meet with",
            "meet at",
            "me too",
            "and me too",
        )
        return any(normalized.endswith(suffix) for suffix in weak_suffixes)

    def _should_defer_whispercpp_age_partial(self, text: str) -> bool:
        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend != "whisper_cpp":
            return False

        normalized = self._normalize_compare_text(text)
        if not normalized:
            return False

        if not re.fullmatch(r"(?:and\s+)?(?:me|im|i am)\s+\d{1,3}\s+years\s+old", normalized):
            return False

        anchor = self._normalize_compare_text(self._get_low_latency_incremental_anchor())
        return "from ukraine" in anchor

    def _should_skip_low_latency_final_tail(self, text: str) -> bool:
        if not self._low_latency_emitted_text:
            return False

        words = text.split()
        if len(words) >= 2:
            return False

        normalized = self._normalize_compare_text(text)
        if not normalized:
            return True

        weak_single_word_tails = {
            "you",
            "thankyou",
            "thanks",
            "your",
            "yours",
            "tu",
            "ty",
            "ты",
        }
        return normalized in weak_single_word_tails

    def _should_skip_known_low_latency_source_hallucination(self, text: str) -> bool:
        if self._low_latency_emitted_text:
            return False

        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "en_to_ru":
            return False

        normalized = self._normalize_compare_text(text)
        if not normalized:
            return True

        return normalized in self._KNOWN_EN_STANDALONE_STT_HALLUCINATIONS

    def _get_partial_stability_window_sec(self) -> float:
        if self._uses_sentence_partial_streaming():
            return max(0.35, self.app_config.stt.partial_stability_sec)
        return self.app_config.stt.partial_stability_sec

    def _emit_sentence_stream_segments(self, text: str, is_final: bool) -> None:
        text = self._normalize_text(text)
        if not text:
            return

        completed_sentences, trailing_fragment = self._split_sentences(text)
        emitted_any = False

        sentence_min_words = self._get_sentence_stream_min_words()
        start_index = min(self._emitted_complete_sentence_count, len(completed_sentences))

        for sentence in completed_sentences[start_index:]:
            normalized_sentence = self._normalize_text(sentence)
            self._emitted_complete_sentence_count += 1
            if len(normalized_sentence.split()) < sentence_min_words:
                continue

            self._log(
                f"{'FINAL' if is_final else 'PARTIAL'} sentence queued: {normalized_sentence}"
            )
            self._enqueue_final_text(normalized_sentence, source_text=normalized_sentence)
            emitted_any = True

        if is_final:
            trailing_fragment = self._normalize_text(trailing_fragment)
            if trailing_fragment:
                trailing_norm = self._normalize_compare_text(trailing_fragment)
                last_completed_norm = ""
                if completed_sentences:
                    last_completed_norm = self._normalize_compare_text(completed_sentences[-1])

                if (
                    trailing_norm
                    and trailing_norm != last_completed_norm
                    and trailing_norm != self._normalize_compare_text(self.last_enqueued_final_text)
                    and len(trailing_fragment.split()) >= sentence_min_words
                ):
                    self._log(f"FINAL tail queued: {trailing_fragment}")
                    self._enqueue_final_text(trailing_fragment, source_text=trailing_fragment)
                    emitted_any = True

        if not emitted_any:
            self._log(
                f"{'FINAL' if is_final else 'PARTIAL'} sentence stream: no new segments"
            )

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        text = AudioEngine._normalize_text(text)
        if not text:
            return [], ""

        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]", text))
        completed_sentences = [
            match.group(0).strip()
            for match in sentence_matches
            if match.group(0).strip()
        ]

        trailing_fragment = ""
        if sentence_matches:
            trailing_fragment = text[sentence_matches[-1].end():].strip()
        else:
            trailing_fragment = text

        return completed_sentences, trailing_fragment

    @staticmethod
    def _split_ru_to_en_phrases(text: str) -> tuple[list[str], str]:
        text = AudioEngine._normalize_text(text)
        if not text:
            return [], ""

        phrase_matches = list(re.finditer(r"[^.!?,;:]+[.!?,;:]", text))
        completed_phrases = [
            match.group(0).strip()
            for match in phrase_matches
            if match.group(0).strip()
        ]

        trailing_fragment = ""
        if phrase_matches:
            trailing_fragment = text[phrase_matches[-1].end():].strip()
        else:
            trailing_fragment = text

        return completed_phrases, trailing_fragment

    def _should_emit_ru_to_en_partial_phrase(self, phrase: str) -> bool:
        phrase = self._normalize_text(phrase)
        if not phrase:
            return False

        normalized = self._normalize_compare_text(phrase)
        if not normalized or normalized in self._ru_to_en_emitted_phrase_norms:
            return False

        words = normalized.split()
        if len(words) < 2:
            return False

        punctuation = phrase[-1] if phrase and phrase[-1] in ".!?,;:" else ""
        if punctuation in {",", ";", ":"} and len(words) < 3:
            return False

        allowed_short_phrases = {
            "всем привет",
            "добрый день",
            "добрый вечер",
            "доброе утро",
            "как дела",
        }
        if len(words) == 2 and punctuation not in {"!", "?"} and normalized not in allowed_short_phrases:
            return False

        weak_tail_tokens = {
            "а",
            "без",
            "в",
            "во",
            "где",
            "да",
            "для",
            "до",
            "если",
            "и",
            "из",
            "или",
            "как",
            "когда",
            "ко",
            "к",
            "мне",
            "меня",
            "мой",
            "моя",
            "мои",
            "моё",
            "на",
            "но",
            "о",
            "об",
            "от",
            "по",
            "под",
            "при",
            "про",
            "с",
            "со",
            "то",
            "у",
            "что",
            "чтобы",
            "эта",
            "это",
            "этот",
            "эти",
            "я",
        }
        if words[-1] in weak_tail_tokens:
            return False

        if normalized in {"меня зовут", "я из", "я язык"}:
            return False

        if normalized.startswith("меня зовут") and len(words) < 3:
            return False

        if normalized.startswith("я из") and len(words) < 3:
            return False

        if re.fullmatch(r"мне\s+\d{1,3}", normalized):
            return False

        if len(words) <= 3 and re.search(r"\b\d{1,3}\b", normalized) and "лет" not in words and "год" not in words:
            return False

        return True

    def _mark_ru_to_en_partial_phrase_seen(self, phrase: str, now: float) -> bool:
        phrase = self._normalize_text(phrase)
        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return False

        stale_after_sec = 0.55
        if (
            phrase_norm != self._ru_to_en_pending_phrase_norm
            or (now - self._ru_to_en_pending_phrase_last_seen_at) > stale_after_sec
        ):
            self._ru_to_en_pending_phrase_text = phrase
            self._ru_to_en_pending_phrase_norm = phrase_norm
            self._ru_to_en_pending_phrase_seen_count = 1
            self._ru_to_en_pending_phrase_first_seen_at = now
        else:
            self._ru_to_en_pending_phrase_text = phrase
            self._ru_to_en_pending_phrase_seen_count += 1

        self._ru_to_en_pending_phrase_last_seen_at = now
        stable_for = now - self._ru_to_en_pending_phrase_first_seen_at
        return (
            self._ru_to_en_pending_phrase_seen_count >= 2
            or stable_for >= self._get_ru_to_en_partial_phrase_stability_sec(phrase)
        )

    def _get_ru_to_en_partial_phrase_stability_sec(self, phrase: str) -> float:
        normalized = self._normalize_compare_text(phrase)
        punctuation = phrase[-1] if phrase and phrase[-1] in ".!?,;:" else ""
        if normalized in {
            "всем привет",
            "добрый день",
            "добрый вечер",
            "доброе утро",
        } and punctuation in {"!", ","}:
            return 0.0
        if punctuation in {"!", "?"}:
            return 0.18
        if punctuation in {",", ";", ":"}:
            return 0.28
        return 0.24

    def _prepare_ru_to_en_phrase_for_queue(self, phrase: str) -> str:
        phrase = self._normalize_text(phrase)
        if not phrase:
            return ""

        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return ""
        phrase_words = phrase.split()
        phrase_norm_words = phrase_norm.split()

        for emitted_phrase in reversed(self._ru_to_en_emitted_phrases[-8:]):
            emitted_norm = self._normalize_compare_text(emitted_phrase)
            if not emitted_norm:
                continue
            emitted_norm_words = emitted_norm.split()

            if self._is_ru_to_en_near_duplicate_phrase(phrase_norm, emitted_norm):
                return ""

            common_prefix_len = 0
            for emitted_word, phrase_word in zip(emitted_norm_words, phrase_norm_words):
                if emitted_word != phrase_word:
                    break
                common_prefix_len += 1

            if 0 < common_prefix_len < len(phrase_norm_words):
                remaining_raw_words = phrase_words[common_prefix_len:]
                if remaining_raw_words:
                    return self._normalize_text(" ".join(remaining_raw_words))

            incremental = self._normalize_text(
                self._extract_incremental_text(emitted_phrase, phrase)
            )
            incremental_norm = self._normalize_compare_text(incremental)
            if incremental_norm and incremental_norm != phrase_norm:
                return incremental

            if phrase_norm == emitted_norm:
                return ""

            if len(phrase_norm.split()) <= 6 and f" {phrase_norm} " in f" {emitted_norm} ":
                return ""

        return phrase

    def _remember_ru_to_en_emitted_phrase(self, phrase: str) -> None:
        phrase = self._normalize_text(phrase)
        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return

        self._ru_to_en_emitted_phrase_norms.add(phrase_norm)
        self._ru_to_en_emitted_phrases.append(phrase)
        if len(self._ru_to_en_emitted_phrases) > 24:
            self._ru_to_en_emitted_phrases = self._ru_to_en_emitted_phrases[-24:]

    @staticmethod
    def _is_ru_to_en_near_duplicate_phrase(phrase_norm: str, emitted_norm: str) -> bool:
        if not phrase_norm or not emitted_norm or phrase_norm == emitted_norm:
            return phrase_norm == emitted_norm

        phrase_words = phrase_norm.split()
        emitted_words = emitted_norm.split()
        if len(phrase_words) < 4 or len(emitted_words) < 4:
            return False

        common_prefix_len = 0
        for left, right in zip(phrase_words, emitted_words):
            if left != right:
                break
            common_prefix_len += 1

        if common_prefix_len < 4:
            return False

        if abs(len(phrase_words) - len(emitted_words)) > 1:
            return False

        similarity = SequenceMatcher(None, phrase_norm, emitted_norm).ratio()
        return similarity >= 0.84

    def _should_skip_ru_to_en_overlap_phrase(self, phrase: str) -> bool:
        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return True

        emitted_norm = self._normalize_compare_text(self._low_latency_emitted_text)
        if not emitted_norm:
            return False

        if phrase_norm == emitted_norm:
            return True

        phrase_words = phrase_norm.split()
        if len(phrase_words) <= 5 and f" {phrase_norm} " in f" {emitted_norm} ":
            return True

        for emitted_phrase in reversed(self._ru_to_en_emitted_phrases[-8:]):
            emitted_phrase_norm = self._normalize_compare_text(emitted_phrase)
            if (
                emitted_phrase_norm
                and (
                    (len(phrase_words) <= 6 and f" {phrase_norm} " in f" {emitted_phrase_norm} ")
                    or self._is_ru_to_en_near_duplicate_phrase(phrase_norm, emitted_phrase_norm)
                )
            ):
                return True

        emitted_words = emitted_norm.split()
        max_overlap = min(len(phrase_words), len(emitted_words), 6)
        for overlap_len in range(max_overlap, 1, -1):
            if emitted_words[-overlap_len:] == phrase_words[:overlap_len]:
                remaining_words = phrase_words[overlap_len:]
                if not remaining_words or len(remaining_words) <= 2:
                    return True

        return False

    def _expire_ru_to_en_pending_phrase(self, now: float) -> None:
        if not self._ru_to_en_pending_phrase_norm:
            return
        if (now - self._ru_to_en_pending_phrase_last_seen_at) > 0.55:
            self._reset_ru_to_en_pending_phrase()

    def _reset_ru_to_en_pending_phrase(self) -> None:
        self._ru_to_en_pending_phrase_text = ""
        self._ru_to_en_pending_phrase_norm = ""
        self._ru_to_en_pending_phrase_seen_count = 0
        self._ru_to_en_pending_phrase_first_seen_at = 0.0
        self._ru_to_en_pending_phrase_last_seen_at = 0.0

    def _get_sentence_stream_min_words(self) -> int:
        return max(4, self.app_config.stt.partial_min_words)

    @staticmethod
    def _normalize_sentence_list(sentences: list[str]) -> list[str]:
        return [
            AudioEngine._normalize_compare_text(sentence)
            for sentence in sentences
            if AudioEngine._normalize_compare_text(sentence)
        ]

    def _get_partial_merge_window_sec(self) -> float:
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction == "ru_to_en":
            return min(0.35, self.app_config.stt.final_debounce_sec)
        return min(0.20, self.app_config.stt.final_debounce_sec)

    @staticmethod
    def _merge_final_texts(previous: str, current: str) -> str:
        previous = AudioEngine._normalize_text(previous)
        current = AudioEngine._normalize_text(current)

        if not previous:
            return current

        if not current:
            return previous

        previous_norm = AudioEngine._normalize_compare_text(previous)
        current_norm = AudioEngine._normalize_compare_text(current)

        if previous_norm == current_norm:
            return previous

        if current_norm.startswith(previous_norm):
            return current

        extra = AudioEngine._extract_incremental_text(previous, current)
        if extra:
            extra_norm = AudioEngine._normalize_compare_text(extra)
            if extra_norm and extra_norm != current_norm:
                return f"{previous} {extra}".strip()

        previous_words = previous.split()
        current_words = current.split()
        if len(previous_words) >= 4 and len(current_words) <= 4:
            return f"{previous} {current}".strip()

        trailing_connectors = {
            "a",
            "an",
            "and",
            "but",
            "for",
            "or",
            "so",
            "the",
            "to",
            "with",
        }
        if previous_words:
            last_word = "".join(ch for ch in previous_words[-1].lower() if ch.isalnum())
            if last_word in trailing_connectors:
                return f"{previous} {current}".strip()

        return current

    def _should_skip_translated_text(self, text: str) -> bool:
        if self._uses_sentence_partial_streaming() or self._uses_boundary_layer_streaming():
            return False

        previous = self._normalize_compare_text(self.last_translated_text)
        current = self._normalize_compare_text(text)

        if not current:
            return True

        if not previous:
            return False

        if (time.monotonic() - self.last_translated_at) > 4.0:
            return False

        if current == previous:
            return True

        if current in previous:
            return True

        return False

    def _strip_known_stt_hallucination_tail(self, text: str) -> str:
        if self._get_active_branch_config().translation_direction != "ru_to_en":
            return self._normalize_text(text)

        text = self._normalize_text(text)
        if not text:
            return ""

        completed_sentences, trailing_fragment = self._split_sentences(text)

        while completed_sentences and self._is_known_standalone_stt_hallucination(completed_sentences[-1]):
            completed_sentences.pop()

        if trailing_fragment and self._is_known_standalone_stt_hallucination(trailing_fragment):
            trailing_fragment = ""

        sanitized_parts = completed_sentences
        if trailing_fragment:
            sanitized_parts = [*sanitized_parts, trailing_fragment]

        return self._normalize_text(" ".join(sanitized_parts))

    def _is_known_standalone_stt_hallucination(self, text: str) -> bool:
        normalized = self._normalize_compare_text(text)
        if not normalized:
            return False

        if len(normalized.split()) > 3:
            return False

        return normalized in self._KNOWN_STANDALONE_STT_HALLUCINATIONS

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def _collapse_immediate_repetitions(text: str) -> str:
        text = AudioEngine._normalize_text(text)
        if not text:
            return ""

        text = AudioEngine._collapse_repeated_sentences(text)

        raw_words = text.split()
        normalized_words = [
            "".join(ch for ch in word.lower() if ch.isalnum())
            for word in raw_words
        ]

        word_count = len(raw_words)
        for chunk_len in range(word_count // 2, 0, -1):
            if word_count % chunk_len != 0:
                continue

            repetition_count = word_count // chunk_len
            if repetition_count < 2:
                continue

            first_chunk = normalized_words[:chunk_len]
            if not first_chunk or any(not token for token in first_chunk):
                continue

            if all(
                normalized_words[offset: offset + chunk_len] == first_chunk
                for offset in range(chunk_len, word_count, chunk_len)
            ):
                return " ".join(raw_words[:chunk_len]).strip()

        return text

    @staticmethod
    def _collapse_repeated_sentences(text: str) -> str:
        sentence_pattern = re.compile(r"[^.!?]+[.!?]?")
        raw_sentences = [segment.strip() for segment in sentence_pattern.findall(text) if segment.strip()]

        if len(raw_sentences) < 2:
            return text

        collapsed_sentences: list[str] = []
        last_norm = ""

        for sentence in raw_sentences:
            sentence_norm = AudioEngine._normalize_compare_text(sentence)
            if not sentence_norm:
                continue

            if sentence_norm == last_norm:
                continue

            collapsed_sentences.append(sentence)
            last_norm = sentence_norm

        collapsed_text = " ".join(collapsed_sentences).strip()
        if collapsed_text:
            return collapsed_text

        return text

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

        intro_continuation = AudioEngine._extract_intro_continuation(previous, current)
        if intro_continuation:
            return intro_continuation

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

        previous_tokens = AudioEngine._tokenize_compare_words(previous)
        current_tokens = AudioEngine._tokenize_compare_words(current)
        previous_token_values = [token for _, token in previous_tokens]
        current_token_values = [token for _, token in current_tokens]

        max_overlap = min(len(previous_token_values), len(current_token_values))
        for overlap_len in range(max_overlap, 2, -1):
            suffix = previous_token_values[-overlap_len:]
            for start_idx in range(0, len(current_token_values) - overlap_len + 1):
                candidate = current_token_values[start_idx:start_idx + overlap_len]
                if candidate != suffix:
                    continue

                overlap_end = start_idx + overlap_len
                if overlap_end >= len(current_tokens):
                    return ""

                return " ".join(raw for raw, _ in current_tokens[overlap_end:]).strip()

        return current.strip()

    @staticmethod
    def _extract_intro_continuation(previous: str, current: str) -> str:
        previous_norm = AudioEngine._normalize_compare_text(previous)
        current_norm = AudioEngine._normalize_compare_text(current)
        if "my name is" not in previous_norm or "my name is" not in current_norm:
            return ""

        current_text = AudioEngine._normalize_text(current)
        continuation_match = re.search(
            r"\b(I am|I'm|I’m|from|me\s+\d+)\b",
            current_text,
            flags=re.IGNORECASE,
        )
        if not continuation_match:
            return ""

        continuation = current_text[continuation_match.start():].strip(" ,.")
        return continuation

    @staticmethod
    def _tokenize_compare_words(text: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        for raw_word in text.strip().split():
            normalized_word = "".join(ch for ch in raw_word.lower() if ch.isalnum())
            if not normalized_word:
                continue
            tokens.append((raw_word, normalized_word))
        return tokens

    def _handle_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)
        else:
            self._log(message)

    def _get_active_branch_config(self) -> TranslationBranchConfig:
        return self.active_branch_config

    def _enqueue_boundary_sentence(self, text: str) -> None:
        text = self._normalize_text(text)
        if not text:
            return

        self._log(f"BOUNDARY queued: {text}")
        self._enqueue_final_text(text, source_text=text)

    def _append_utterance_audio_chunk(self, audio: np.ndarray) -> None:
        if not self._uses_boundary_layer_streaming():
            return

        mono = audio[:, 0] if audio.ndim == 2 else audio
        if mono.size == 0:
            return

        with self._utterance_audio_lock:
            self._utterance_audio_chunks.append(mono.astype(np.float32, copy=True))
            self._utterance_audio_total_frames += int(mono.shape[0])

            max_frames = int(self.current_samplerate * 20.0)
            while self._utterance_audio_total_frames > max_frames and self._utterance_audio_chunks:
                removed = self._utterance_audio_chunks.pop(0)
                self._utterance_audio_total_frames -= int(removed.shape[0])

    def _get_utterance_audio_snapshot(self) -> np.ndarray:
        with self._utterance_audio_lock:
            if not self._utterance_audio_chunks:
                return np.zeros((0,), dtype=np.float32)
            return np.concatenate(self._utterance_audio_chunks, axis=0).astype(np.float32, copy=False)

    def _confirm_and_emit_buffered_sentences(
        self,
        target_sentence_count: int,
        fallback_sentences: list[str],
        is_final: bool,
    ) -> None:
        confirm_stt_service = self.confirm_stt_service
        if confirm_stt_service is None:
            self._emit_fallback_boundary_sentences(fallback_sentences)
            return

        audio = self._get_utterance_audio_snapshot()
        if audio.size == 0:
            self._emit_fallback_boundary_sentences(fallback_sentences)
            return

        try:
            confirmed_text = self._normalize_text(
                confirm_stt_service.transcribe(audio, self.current_samplerate)
            )
        except Exception as error:
            self._handle_error(f"Confirm STT error: {error}")
            self._emit_fallback_boundary_sentences(fallback_sentences)
            return

        if not confirmed_text:
            self._emit_fallback_boundary_sentences(fallback_sentences)
            return

        confirmed_sentences, trailing_fragment = self._split_sentences(confirmed_text)
        sentence_min_words = self._get_sentence_stream_min_words()
        safe_target = min(target_sentence_count, len(confirmed_sentences))

        for sentence in confirmed_sentences[self._confirmed_sentence_count:safe_target]:
            normalized_sentence = self._normalize_text(sentence)
            self._confirmed_sentence_count += 1
            if len(normalized_sentence.split()) < sentence_min_words:
                continue
            self._log(f"CONFIRM queued: {normalized_sentence}")
            self._enqueue_boundary_sentence(normalized_sentence)

        if is_final and self._confirmed_sentence_count >= len(confirmed_sentences):
            trailing_fragment = self._normalize_text(trailing_fragment)
            if trailing_fragment and len(trailing_fragment.split()) >= sentence_min_words:
                self._log(f"CONFIRM final tail: {trailing_fragment}")
                self._enqueue_boundary_sentence(trailing_fragment)

    def _emit_fallback_boundary_sentences(self, sentences: list[str]) -> None:
        for sentence in sentences:
            normalized_sentence = self._normalize_text(sentence)
            if not normalized_sentence:
                continue
            self._confirmed_sentence_count += 1
            self._log(f"CONFIRM fallback queued: {normalized_sentence}")
            self._enqueue_boundary_sentence(normalized_sentence)

    def _resolve_translation_direction(self, branch_config: TranslationBranchConfig) -> TranslationDirection:
        direction = (branch_config.translation_direction or "").strip().lower()
        if direction == TranslationDirection.RU_TO_EN.value:
            return TranslationDirection.RU_TO_EN
        return TranslationDirection.EN_TO_RU

    @staticmethod
    def _should_emit_log_message(message: str) -> bool:
        noisy_prefixes = (
            "PARTIAL:",
            "Realtime partial:",
            "Faster-whisper partial:",
            "Faster-whisper final:",
            "Faster-whisper VAD start",
            "Faster-whisper VAD end",
            "whisper.cpp partial:",
            "whisper.cpp final:",
            "whisper.cpp VAD start",
            "whisper.cpp VAD end",
            "FINAL incremental:",
            "FINAL merged:",
            "FINAL skipped:",
            # keep low-latency + timing logs visible for tuning latency/debugging
            "Dropped stale TTS audio queue",
        )
        return not message.startswith(noisy_prefixes)

    def _log(self, message: str) -> None:
        if self.on_log and self._should_emit_log_message(message):
            self.on_log(message)
