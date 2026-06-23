from __future__ import annotations

import threading
import queue
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Callable

import numpy as np

from core.audio.audio_session import AudioSession, AudioSessionConfig, PlaybackAudioBlock
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


@dataclass(frozen=True)
class TTSRequest:
    text: str
    samplerate: int
    source_type: str
    generation: int = 0


class AudioEngine:
    _RU_TO_EN_DEPENDENT_SUFFIX_ADMISSION_FILTER_ENABLED = False
    RU_TO_EN_SOURCE_GLOSSARY_REPLACEMENTS = (
        (r"\bперевозчик\b", "переводчик"),
        (r"\bпэт\s+проект\b", "пет-проект"),
        (r"\bпедпроект\b", "пет-проект"),
        (r"\bлайтинги\b", "лендинги"),
        (r"\bлейтинги\b", "лендинги"),
        (r"\bлейдинги\b", "лендинги"),
        (r"\bлейдингов\b", "лендингов"),
        (r"\bлайндинги\b", "лендинги"),
        (r"\bландинги\b", "лендинги"),
        (r"\bлэндинги\b", "лендинги"),
        (r"\bмодальные\s+окна\b", "modal windows"),
        (r"\bмодальными\s+окнами\b", "modal windows"),
        (r"\bмодального\s+окна\b", "modal window"),
        (r"\bререндеры\b", "re-renders"),
        (r"\bререндеров\b", "re-renders"),
        (r"\bфронтенд\b", "frontend"),
        (r"\bфронт-энд\b", "frontend"),
        (r"\bфронтенде\b", "frontend"),
        (r"\bфронт-энде\b", "frontend"),
        (r"\bбекенд\b", "backend"),
        (r"\bбэкенд\b", "backend"),
        (r"\bbackend\s+api\b", "backend API"),
        (r"\bкорректное\s+состояние\b", "корректный state"),
        (r"\bстарое\s+состояние\b", "старый state"),
        (r"\bбольшое\s+количество\s+состояний\b", "большое количество states"),
        (r"\bбольшим\s+количеством\s+состояний\b", "большим количеством states"),
        (r"\bверстка\b", "layout"),
        (r"\bвёрстка\b", "layout"),
        (r"\bверстки\b", "layout"),
        (r"\bвёрстки\b", "layout"),
    )
    RU_TO_EN_CONTEXTUAL_GLOSSARY_REPLACEMENTS = (
        (
            ("еще один", "ещё один"),
            (r"\bподпроект\b", "пет-проект"),
        ),
    )
    _RU_TO_EN_CONTEXT_SENSITIVE_STARTERS = (
        "как",
        "чтобы",
        "то",
        "который",
        "которая",
        "которое",
        "которые",
        "которым",
        "которую",
    )
    _KNOWN_STANDALONE_STT_HALLUCINATIONS = frozenset({
        "продолжение следует",
    })
    _KNOWN_EN_STANDALONE_STT_HALLUCINATIONS = frozenset({
        "welcome to the american league of legends",
    })
    _TTS_CONTINUATION_LEADERS = frozenset({
        "when",
        "because",
        "since",
        "if",
        "and",
        "but",
    })
    _RU_TO_EN_FINAL_CLEANUP_STOPWORDS = frozenset({
        "я", "ты", "он", "она", "оно", "они", "мы", "вы",
        "это", "эт", "эта", "эти", "как", "что", "в", "во", "на",
        "и", "но", "а", "то", "так", "потому", "если", "когда",
        "мой", "моя", "мои", "его", "ее", "её", "их", "наш", "наша",
        "наши", "ваш", "ваша", "ваши", "для", "через", "у", "из",
        "по", "к", "ко", "с", "со", "до", "или", "же", "ли", "нет",
        "есть", "меня", "мне", "его", "ее", "её", "свой",
    })
    _RU_TO_EN_FINAL_CLEANUP_SUFFIXES = (
        "аются", "яются", "ились", "ались",
        "ывать", "ивать", "овать", "ировать",
        "аться", "яться", "иться", "еться", "нуть",
        "аешь", "яешь", "аете", "яете", "ают", "яют",
        "уешь", "уете", "уют", "ишь", "ите", "им", "ит", "ят", "ют",
        "ался", "ялся", "илась", "илось", "ались", "ялись",
        "ал", "ала", "ало", "али", "ял", "яла", "яло", "яли",
        "аю", "яю", "ую", "ешь", "ете", "ем", "ут", "ют",
        "ить", "ать", "ять", "еть", "оть", "уть", "ти",
        "иями", "ями", "ами", "его", "ого", "ему", "ому", "ыми", "ими",
        "иях", "ах", "ях", "ия", "ья", "ие", "ье", "ий", "ый", "ой",
        "ая", "яя", "ое", "ее", "ую", "ом", "ем", "ам", "ям",
        "ов", "ев", "ей", "иям", "ием",
        "а", "я", "ы", "и", "е", "о", "у", "ю", "ь",
    )

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
        self._recent_translated_compare_norms: list[str] = []
        self._recent_translated_texts: list[str] = []
        self._recent_translated_source_texts: list[str] = []
        self._tts_ready_chunks_count: int = 0
        self._last_stt_activity_at: float = 0.0

        self.final_text_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=32)
        self.low_latency_text_queue: queue.Queue[tuple[str, bool, int]] = queue.Queue(maxsize=32)
        self.tts_text_queue: queue.Queue[TTSRequest] = queue.Queue(maxsize=64)
        self.tts_worker_thread: Optional[threading.Thread] = None
        self._tts_stop_event = threading.Event()
        self._tts_item_counter = 0
        self._tts_item_lock = threading.Lock()
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
        self._ru_to_en_recent_partial_text = ""
        self._ru_to_en_last_emitted_at: float = 0.0
        self._ru_to_en_emitted_phrase_norms: set[str] = set()
        self._ru_to_en_emitted_phrases: list[str] = []
        self._ru_to_en_finalized_phrase_count: int = 0
        self._ru_to_en_pending_phrase_text: str = ""
        self._ru_to_en_pending_phrase_norm: str = ""
        self._ru_to_en_pending_phrase_seen_count: int = 0
        self._ru_to_en_pending_phrase_first_seen_at: float = 0.0
        self._ru_to_en_pending_phrase_last_seen_at: float = 0.0
        self._en_to_ru_last_final_received_at: float = 0.0
        self._en_to_ru_translation_started_at_by_text: dict[str, float] = {}
        self._en_to_ru_tts_queued_at_by_text: dict[str, float] = {}
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
        self._recent_translated_compare_norms = []
        self._recent_translated_texts = []
        self._recent_translated_source_texts = []
        self._tts_ready_chunks_count = 0
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
        self.session.on_playback_started = self._on_playback_started
        self.session.on_playback_finished = lambda text: self._log(f"PLAYBACK finished: {text}")
        self.session.on_playback_skipped = (
            lambda text, reason: self._log(f"PLAYBACK skipped: {text} reason={reason}")
        )
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
        if branch_config.translation_direction == "en_to_ru":
            self._log(
                "EN2RU STT config active | "
                f"backend={self.app_config.stt.backend} "
                f"min_window_sec={self.app_config.stt.silero_min_window_sec:.2f} "
                f"min_silence_ms={self.app_config.stt.silero_min_silence_ms} "
                f"speech_pad_ms={self.app_config.stt.silero_speech_pad_ms} "
                f"final_debounce_sec={self.app_config.stt.final_debounce_sec:.2f} "
                f"partial_emit_enabled={self.app_config.stt.partial_emit_enabled}"
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

        if hasattr(self.translation_service, "stop"):
            try:
                self.translation_service.stop()
            except Exception as error:
                self._handle_error(f"Translation service stop error: {error}")

        if hasattr(self.tts_service, "stop"):
            try:
                self.tts_service.stop()
            except Exception as error:
                self._handle_error(f"TTS service stop error: {error}")

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
        self._recent_translated_compare_norms = []
        self._recent_translated_texts = []
        self._recent_translated_source_texts = []
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
            if self._get_active_branch_config().translation_direction == "ru_to_en":
                self._log(f"FINAL raw: {text}")
            else:
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
                text, source_type = self.final_text_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.running:
                break

            self._process_final_text(text, source_type=source_type)

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
            ready_source_type = "final"

            with self._pending_final_lock:
                if self._pending_final_text:
                    age = time.monotonic() - self._pending_final_updated_at
                    debounce_sec = self.app_config.stt.final_debounce_sec
                    if self._pending_final_is_partial:
                        debounce_sec = self._get_partial_merge_window_sec()
                    if age >= debounce_sec:
                        ready_text = self._pending_final_text
                        ready_source_type = "partial" if self._pending_final_is_partial else "final"
                        self._pending_final_text = ""
                        self._pending_final_updated_at = 0.0
                        self._pending_final_is_partial = False

            if ready_text:
                self._enqueue_final_text(ready_text, source_type=ready_source_type)
                continue

            self._maybe_emit_stable_partial()

            time.sleep(0.05)

    def _process_final_text(self, text: str, *, source_type: str = "final") -> None:
        if not self.running or self._translation_paused:
            return

        if self._process_en_to_ru_final_segments_if_needed(text, source_type=source_type):
            return

        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction == "en_to_ru" and source_type == "final":
            self._mark_en_to_ru_final_received(text)

        if self._stt_backend_outputs_translated_text():
            translated_text = self._normalize_text(text)
            translation_elapsed = 0.0
        else:
            translation_service = self.translation_service
            if translation_service is None:
                self._log("Translation skipped: service is unavailable")
                return

            if branch_config.translation_direction == "en_to_ru" and source_type == "final":
                self._mark_en_to_ru_translation_started(text)
            translation_started_at = time.perf_counter()
            try:
                translated_text = self._translate_ru_to_en_with_optional_context(
                    translation_service,
                    text,
                )
            except Exception as error:
                self._handle_error(f"Translation error: {error}")
                return

            translated_text = self._normalize_text(translated_text)
            translation_elapsed = time.perf_counter() - translation_started_at

        self._log(f"Translation time: {translation_elapsed:.3f} sec")

        if not translated_text:
            self._log("TRANSLATED: <empty>")
            return

        if self._try_replace_pending_translated_duplicate(translated_text):
            self.last_translated_text = translated_text
            self.last_translated_at = time.monotonic()
            self._log("TRANSLATED replaced pending duplicate")
            return

        same_prefix_retry_match = self._find_same_prefix_translated_retry(translated_text)
        if same_prefix_retry_match:
            self.last_translated_text = translated_text
            self.last_translated_at = time.monotonic()
            self._log(f"TRANSLATED skipped same-prefix retry: {translated_text} ~= {same_prefix_retry_match}")
            return

        if self._should_skip_translated_text(translated_text):
            self._log("TRANSLATED skipped: duplicate")
            return

        if self._should_skip_strict_short_translated_fragment(text, translated_text):
            self._log(f"TRANSLATED skipped short low-confidence fragment: {translated_text}")
            return

        self.last_translated_text = translated_text
        self.last_translated_at = time.monotonic()
        self._remember_translated_text_for_compare(translated_text)
        self._remember_translated_source_text_for_compare(text)
        self._log(f"TRANSLATED: {translated_text}")
        self._enqueue_tts_text(translated_text, source_type=source_type)
        if branch_config.translation_direction == "en_to_ru" and source_type == "final":
            self._mark_en_to_ru_tts_queued(translated_text)

    def _process_en_to_ru_final_segments_if_needed(self, text: str, *, source_type: str) -> bool:
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "en_to_ru" or source_type != "final":
            return False
        if self._stt_backend_outputs_translated_text():
            return False

        final_text = self._normalize_text(text)
        if not final_text:
            return False

        segments = self._split_en_to_ru_final_segments(final_text)
        if len(segments) <= 1:
            return False

        translation_service = self.translation_service
        if translation_service is None:
            self._log("Translation skipped: service is unavailable")
            return True

        self._mark_en_to_ru_final_received(final_text)
        self._log(f"EN2RU FINAL received: {final_text}")
        self._log(f"EN2RU FINAL segmented: count={len(segments)}")

        for index, segment in enumerate(segments, start=1):
            if not self.running or self._translation_paused:
                break

            translation_started_at = time.perf_counter()
            self._mark_en_to_ru_translation_started(segment)
            try:
                translated_text = self._normalize_text(translation_service.translate(segment))
            except Exception as error:
                self._handle_error(f"Translation error: {error}")
                continue

            translation_elapsed = time.perf_counter() - translation_started_at
            self._log(
                "EN2RU segment translated "
                f"{index}/{len(segments)} time={translation_elapsed:.3f}s "
                f"source={segment} translation={translated_text or '<empty>'}"
            )

            if not translated_text:
                continue

            if self._should_skip_translated_text(translated_text):
                self._log(f"TRANSLATED skipped: duplicate segment={index}")
                continue

            if self._should_skip_strict_short_translated_fragment(segment, translated_text):
                self._log(
                    "TRANSLATED skipped short low-confidence fragment: "
                    f"{translated_text}"
                )
                continue

            self.last_translated_text = translated_text
            self.last_translated_at = time.monotonic()
            self._remember_translated_text_for_compare(translated_text)
            self._remember_translated_source_text_for_compare(segment)
            self._log(f"TRANSLATED: {translated_text}")
            self._enqueue_tts_text(translated_text, source_type="final")
            self._mark_en_to_ru_tts_queued(translated_text)
            self._log(
                f"EN2RU segment TTS queued {index}/{len(segments)} "
                f"text={translated_text}"
            )

        return True

    def _mark_en_to_ru_final_received(self, text: str) -> None:
        now = time.monotonic()
        self._en_to_ru_last_final_received_at = now
        self._log(
            "EN2RU FINAL received at "
            f"{now:.3f} text={self._normalize_text(text)}"
        )

    def _mark_en_to_ru_translation_started(self, text: str) -> None:
        now = time.monotonic()
        source_norm = self._normalize_compare_text(text)
        if source_norm:
            self._en_to_ru_translation_started_at_by_text[source_norm] = now
            if len(self._en_to_ru_translation_started_at_by_text) > 32:
                self._en_to_ru_translation_started_at_by_text.clear()

        final_delay = (
            now - self._en_to_ru_last_final_received_at
            if self._en_to_ru_last_final_received_at > 0.0
            else 0.0
        )
        self._log(
            "EN2RU translation started at "
            f"{now:.3f} final_to_translation_sec={final_delay:.3f} "
            f"source={self._normalize_text(text)}"
        )

    def _mark_en_to_ru_tts_queued(self, text: str) -> None:
        now = time.monotonic()
        translated_norm = self._normalize_compare_text(text)
        if not translated_norm:
            return

        self._en_to_ru_tts_queued_at_by_text[translated_norm] = now
        if len(self._en_to_ru_tts_queued_at_by_text) > 32:
            self._en_to_ru_tts_queued_at_by_text.clear()

    def _translate_ru_to_en_with_optional_context(
        self,
        translation_service: TranslationService,
        text: str,
    ) -> str:
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "ru_to_en":
            return translation_service.translate(text)

        rewritten_text = self._rewrite_ru_to_en_source_before_translation(text)
        if rewritten_text != self._normalize_text(text):
            self._log(
                "RU2EN source rewrite before translation: "
                f"{self._normalize_text(text)} -> {rewritten_text}"
            )
            text = rewritten_text

        if not self.app_config.translation.ru_to_en_context_aware_translation_enabled:
            return translation_service.translate(text)

        current_text = self._normalize_text(text)
        if not self._should_use_ru_to_en_translation_context(current_text):
            return translation_service.translate(current_text)

        context_text = self._get_ru_to_en_translation_context(current_text)
        if not context_text:
            return translation_service.translate(current_text)

        self._log(
            "RU2EN context translation used:\n"
            f"CONTEXT: {context_text}\n"
            f"CURRENT: {current_text}"
        )

        combined_source = self._normalize_text(f"{context_text} {current_text}")
        combined_translated = self._normalize_text(translation_service.translate(combined_source))
        previous_translated = self._normalize_text(self.last_translated_text)
        contextual_tail = self._normalize_text(
            self._extract_incremental_text(previous_translated, combined_translated)
        )
        if contextual_tail:
            self._log(f"RU2EN context translated output: {contextual_tail}")
            return contextual_tail

        self._log("RU2EN context translation fallback: incremental tail empty")
        return combined_translated

    def _rewrite_ru_to_en_source_before_translation(self, text: str) -> str:
        source_text = self._normalize_text(text)
        normalized = self._normalize_compare_text(source_text)
        if not normalized:
            return source_text

        if normalized.startswith("но упирался то в лимит") or normalized.startswith("но опирался то в лимит"):
            return self._normalize_text(
                re.sub(
                    r"^но\s+(?:у|о)пирался\s+то\s+в\s+лимит\b",
                    "но я сталкивался с лимитом",
                    source_text,
                    count=1,
                    flags=re.IGNORECASE,
                )
            )

        if normalized.startswith("упирался то в лимит") or normalized.startswith("опирался то в лимит"):
            return self._normalize_text(
                re.sub(
                    r"^(?:у|о)пирался\s+то\s+в\s+лимит\b",
                    "я сталкивался с лимитом",
                    source_text,
                    count=1,
                    flags=re.IGNORECASE,
                )
            )

        words = normalized.split()
        if len(words) > 7:
            return source_text
        if "я" in words:
            return source_text
        if not words or words[0] != "которым" or "пользуюсь" not in words:
            return source_text

        raw_words = source_text.split()
        if not raw_words:
            return source_text

        tail_words = raw_words[1:]
        if not tail_words:
            return source_text

        rewritten_words = ["я"]
        inserted_object = False
        for raw_word in tail_words:
            if not inserted_object and raw_word.strip(".,!?;:").lower() == "пользуюсь":
                base_word = raw_word.rstrip(".,!?;:")
                punctuation = raw_word[len(base_word):]
                rewritten_words.append(base_word)
                rewritten_words.append(f"этим{punctuation}")
                inserted_object = True
                continue
            rewritten_words.append(raw_word)

        if not inserted_object:
            return source_text

        return self._normalize_text(" ".join(rewritten_words))

    def _normalize_ru_to_en_source_vocabulary(self, text: str) -> str:
        source_text = self._normalize_text(text)
        if not source_text:
            return source_text

        normalized = self._normalize_compare_text(source_text)
        result = source_text
        for pattern, replacement in self.RU_TO_EN_SOURCE_GLOSSARY_REPLACEMENTS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        for context_markers, replacement in self.RU_TO_EN_CONTEXTUAL_GLOSSARY_REPLACEMENTS:
            pattern, replacement_text = replacement
            if any(marker in normalized for marker in context_markers):
                result = re.sub(pattern, replacement_text, result, flags=re.IGNORECASE)

        return self._normalize_text(result)

    def _should_skip_ru_to_en_known_weak_draft(self, text: str) -> bool:
        return self._normalize_compare_text(text) in {
            "сначала я пользовался бесплатным",
            "это проект для конвертации",
            "modal windows и интеграторами",
        }

    def _find_ru_to_en_partial_rewind_suffix_match(self, candidate: str) -> str:
        candidate_text = self._normalize_text(candidate)
        candidate_norm = self._normalize_compare_text(candidate_text)
        recent_partial = self._normalize_text(self._ru_to_en_recent_partial_text)
        recent_partial_norm = self._normalize_compare_text(recent_partial)
        if not candidate_norm or not recent_partial_norm:
            return ""

        candidate_words = candidate_norm.split()
        if not 3 <= len(candidate_words) <= 5:
            return ""

        standalone_starters = {
            "я",
            "мне",
            "сейчас",
            "также",
            "когда",
            "сначала",
            "это",
            "но",
            "то",
            "который",
            "которая",
            "которое",
            "которые",
            "которым",
            "которую",
        }
        first_word = candidate_words[0]
        if first_word in standalone_starters or candidate_norm.startswith("так как "):
            return ""

        candidate_index = recent_partial_norm.find(candidate_norm)
        if candidate_index < 0:
            return ""

        for emitted_phrase in reversed(self._ru_to_en_emitted_phrases[-5:]):
            emitted_norm = self._normalize_compare_text(emitted_phrase)
            emitted_words = emitted_norm.split()
            if len(emitted_words) < 3:
                continue

            emitted_index = recent_partial_norm.find(emitted_norm)
            if emitted_index < 0 and recent_partial_norm.startswith(emitted_norm):
                emitted_index = 0
            if emitted_index < 0:
                continue

            last_word = emitted_words[-1]
            last_word_start = emitted_norm.rfind(last_word)
            if last_word_start < 0:
                continue

            tail_boundary = emitted_index + last_word_start
            if candidate_index < tail_boundary:
                continue

            if len(first_word) <= 2 or first_word.startswith(last_word):
                return recent_partial

        return ""

    def _should_use_ru_to_en_translation_context(self, text: str) -> bool:
        normalized = self._normalize_compare_text(text)
        if not normalized:
            return False

        words = normalized.split()
        if len(words) > 6:
            return False

        for starter in self._RU_TO_EN_CONTEXT_SENSITIVE_STARTERS:
            if normalized == starter or normalized.startswith(f"{starter} "):
                return True
        return False

    def _get_ru_to_en_translation_context(self, current_text: str) -> str:
        current_norm = self._normalize_compare_text(current_text)
        if not current_norm:
            return ""

        skipped_current = False
        for previous_text in reversed(self._ru_to_en_emitted_phrases):
            previous_norm = self._normalize_compare_text(previous_text)
            if not previous_norm:
                continue
            if not skipped_current and previous_norm == current_norm:
                skipped_current = True
                continue

            context_words = self._normalize_text(previous_text).split()
            if not context_words:
                return ""
            return " ".join(context_words[-12:]).strip()

        return ""

    def _enqueue_tts_text(
        self,
        text: str,
        *,
        source_type: str = "final",
        generation: int = 0,
    ) -> None:
        if not self.running or self._translation_paused:
            return
        text = self._normalize_text(text)
        if not text:
            return
        try:
            self.tts_text_queue.put_nowait(
                TTSRequest(
                    text=text,
                    samplerate=self.current_samplerate,
                    source_type=source_type,
                    generation=generation,
                )
            )
        except queue.Full:
            self._handle_error("TTS text queue is full, dropping chunk")

    def _tts_worker_loop(self) -> None:
        while not self._tts_stop_event.is_set():
            if not self.running:
                break
            try:
                request = self.tts_text_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.running or self._translation_paused:
                continue

            request = self._coerce_tts_request(request)
            request = self._collect_adjacent_tts_request(request)
            if not request.text:
                continue

            tts_service = self.tts_service
            if tts_service is None:
                continue

            text_chunks = self._split_tts_text_chunks(request.text, request.source_type)
            for chunk_index, chunk_text in enumerate(text_chunks):
                if not self.running or self._translation_paused:
                    break

                item_id = self._next_tts_item_id()
                item_created_at = time.monotonic()
                self._log(
                    "TTS item created: "
                    f"id={item_id} sourceType={request.source_type} status=queued "
                    f"textLen={len(chunk_text)} chunk={chunk_index + 1}/{len(text_chunks)} "
                    f"generation={request.generation} text={chunk_text}"
                )

                tts_started_at = time.perf_counter()
                try:
                    tts_audio = tts_service.synthesize(
                        chunk_text,
                        target_samplerate=request.samplerate,
                    )
                except Exception as error:
                    self._handle_error(f"TTS error: {error}")
                    continue

                tts_elapsed = time.perf_counter() - tts_started_at
                self._log(
                    f"TTS time: {tts_elapsed:.3f} sec "
                    f"id={item_id} sourceType={request.source_type} textLen={len(chunk_text)}"
                )

                duration = (
                    tts_audio.shape[0] / float(self.current_samplerate)
                    if tts_audio.size > 0
                    else 0.0
                )
                self._log(
                    f"TTS audio ready: {duration:.2f} sec "
                    f"id={item_id} sourceType={request.source_type} textLen={len(chunk_text)}"
                )
                self._tts_ready_chunks_count += 1
                queued_blocks = self._enqueue_tts_audio(
                    tts_audio,
                    chunk_text,
                    item_id=item_id,
                    source_type=request.source_type,
                    created_at=item_created_at,
                    generation=request.generation,
                )
                queue_size = self.session.output_queue.qsize() if self.session is not None else 0
                item_age = time.monotonic() - item_created_at
                self._log(
                    "TTS item queued: "
                    f"id={item_id} sourceType={request.source_type} status=queued "
                    f"textLen={len(chunk_text)} queueSize={queue_size} "
                    f"itemAge={item_age:.3f}s blocks={queued_blocks} text={chunk_text}"
                )
                self._log(
                    f"PLAYBACK queued: {chunk_text} "
                    f"id={item_id} sourceType={request.source_type} "
                    f"textLen={len(chunk_text)} queueSize={queue_size}"
                )

    def _on_playback_started(self, text: str) -> None:
        self._log(f"PLAYBACK started: {text}")
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "en_to_ru":
            return

        normalized_text = self._normalize_compare_text(text)
        queued_at = self._en_to_ru_tts_queued_at_by_text.pop(normalized_text, 0.0)
        now = time.monotonic()
        if queued_at > 0.0:
            self._log(
                "EN2RU playback started at "
                f"{now:.3f} queued_to_playback_sec={now - queued_at:.3f} "
                f"text={self._normalize_text(text)}"
            )
        else:
            self._log(
                "EN2RU playback started at "
                f"{now:.3f} text={self._normalize_text(text)}"
            )

    def _coerce_tts_request(self, request: TTSRequest | tuple) -> TTSRequest:
        if isinstance(request, TTSRequest):
            return request
        if isinstance(request, tuple) and len(request) >= 2:
            return TTSRequest(
                text=str(request[0]),
                samplerate=int(request[1]),
                source_type="final",
            )
        return TTSRequest(text="", samplerate=self.current_samplerate, source_type="final")

    def _collect_adjacent_tts_request(self, first_request: TTSRequest) -> TTSRequest:
        first_text = self._normalize_text(first_request.text)
        if not first_text:
            return TTSRequest(
                text="",
                samplerate=first_request.samplerate,
                source_type=first_request.source_type,
                generation=first_request.generation,
            )

        if first_request.source_type != "final" or not self._should_wait_for_tts_continuation(first_text):
            return TTSRequest(
                text=first_text,
                samplerate=first_request.samplerate,
                source_type=first_request.source_type,
                generation=first_request.generation,
            )

        merged_parts = [first_text]
        started_at = time.monotonic()
        max_wait_sec = 0.24
        max_parts = 2
        max_chars = 160

        while len(merged_parts) < max_parts and (time.monotonic() - started_at) < max_wait_sec:
            timeout = max(0.0, max_wait_sec - (time.monotonic() - started_at))
            try:
                next_request = self._coerce_tts_request(self.tts_text_queue.get(timeout=timeout))
            except queue.Empty:
                break

            next_text = self._normalize_text(next_request.text)
            if not next_text:
                continue

            if next_request.source_type != first_request.source_type:
                try:
                    self.tts_text_queue.put_nowait(next_request)
                except queue.Full:
                    self._handle_error("TTS text queue is full, dropping deferred chunk")
                break

            if not self._should_merge_tts_continuation(merged_parts[0], next_text):
                try:
                    self.tts_text_queue.put_nowait(next_request)
                except queue.Full:
                    self._handle_error("TTS text queue is full, dropping deferred chunk")
                break

            candidate = " ".join(merged_parts + [next_text]).strip()
            if len(candidate) > max_chars:
                try:
                    self.tts_text_queue.put_nowait(next_request)
                except queue.Full:
                    self._handle_error("TTS text queue is full, dropping deferred chunk")
                break

            merged_parts.append(next_text)

        merged_text = " ".join(part for part in merged_parts if part).strip()
        if len(merged_parts) > 1:
            self._log(f"TTS continuation merged: {len(merged_parts)} chunks")
            self._log(f"PLAYBACK merged: {merged_parts[0]} + {merged_parts[1]}")
        return TTSRequest(
            text=merged_text,
            samplerate=first_request.samplerate,
            source_type=first_request.source_type,
            generation=first_request.generation,
        )

    def _next_tts_item_id(self) -> str:
        with self._tts_item_lock:
            self._tts_item_counter += 1
            return f"tts-{self._tts_item_counter}"

    def _split_tts_text_chunks(self, text: str, source_type: str) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        max_chars = 180 if source_type == "final" else 140
        if len(text) <= max_chars:
            return [text]

        completed_sentences, trailing_fragment = self._split_sentences(text)
        sentence_parts = completed_sentences[:]
        if trailing_fragment:
            sentence_parts.append(trailing_fragment)

        if not sentence_parts:
            sentence_parts = [text]

        chunks: list[str] = []
        current = ""

        for part in sentence_parts:
            part = self._normalize_text(part)
            if not part:
                continue

            if len(part) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_long_tts_part(part, max_chars=max_chars))
                continue

            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks or [text]

    @staticmethod
    def _split_long_tts_part(part: str, *, max_chars: int) -> list[str]:
        part = AudioEngine._normalize_text(part)
        if len(part) <= max_chars:
            return [part] if part else []

        fragments = [
            fragment.strip()
            for fragment in re.split(r"(?<=[,;:])\s+", part)
            if fragment.strip()
        ]
        if len(fragments) <= 1:
            fragments = part.split()

        chunks: list[str] = []
        current = ""
        for fragment in fragments:
            candidate = f"{current} {fragment}".strip() if current else fragment
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(fragment) <= max_chars:
                current = fragment
                continue

            words = fragment.split()
            word_chunk = ""
            for word in words:
                word_candidate = f"{word_chunk} {word}".strip() if word_chunk else word
                if len(word_candidate) <= max_chars:
                    word_chunk = word_candidate
                else:
                    if word_chunk:
                        chunks.append(word_chunk)
                    word_chunk = word
            if word_chunk:
                current = word_chunk

        if current:
            chunks.append(current)

        return chunks

    def _should_wait_for_tts_continuation(self, text: str) -> bool:
        if self._tts_ready_chunks_count == 0:
            return False

        normalized = self._normalize_text(text)
        if not normalized:
            return False

        if len(normalized) > 90 or len(normalized.split()) > 10:
            return False

        if normalized.endswith(","):
            return True

        first_word = normalized.split()[0].lower()
        if first_word in self._TTS_CONTINUATION_LEADERS:
            return True

        return self._looks_like_incomplete_english_thought(normalized)

    def _should_merge_tts_continuation(self, first_text: str, next_text: str) -> bool:
        first = self._normalize_text(first_text)
        second = self._normalize_text(next_text)
        if not first or not second:
            return False

        if self._looks_like_complete_english_sentence(first):
            return False

        if len(second) > 120 or len(second.split()) > 14:
            return False

        second_first_word = second.split()[0].lower()
        if second_first_word in self._TTS_CONTINUATION_LEADERS and not second.startswith("I "):
            return False

        combined = f"{first} {second}".strip()
        if len(combined) > 160:
            return False

        return True

    @classmethod
    def _looks_like_complete_english_sentence(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        if not normalized:
            return False
        return normalized.endswith((".", "!", "?"))

    @classmethod
    def _looks_like_incomplete_english_thought(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        if not normalized or cls._looks_like_complete_english_sentence(normalized):
            return False

        words = normalized.split()
        if not words:
            return False

        first_word = words[0].lower()
        if first_word in cls._TTS_CONTINUATION_LEADERS:
            return True

        if len(words) <= 4:
            return True

        trailing_word = words[-1].lower().strip(",")
        return trailing_word in {"and", "but", "because", "since", "if", "when", "that"}

    def _should_output_processed_audio(self) -> bool:
        if self.processor is None:
            return False

        return self.processor.mode in {ProcessingMode.MUTE, ProcessingMode.TEST_TONE}

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if self.processor is None:
            return chunk.astype(np.float32, copy=True)

        return self.processor.process(chunk)

    def _enqueue_tts_audio(
        self,
        audio: np.ndarray,
        text: str = "",
        *,
        item_id: str,
        source_type: str,
        created_at: float,
        generation: int = 0,
    ) -> int:
        session = self.session
        if session is None or not self.running or session.stopping:
            return 0

        if audio.size == 0:
            return 0

        audio = self._trim_tts_audio_silence(audio)
        if audio.size == 0:
            return 0

        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        self._trim_output_queue_if_needed(session)

        total_frames = audio.shape[0]
        total_blocks = max(1, (total_frames + self.current_blocksize - 1) // self.current_blocksize)
        offset = 0
        block_index = 0
        queued_blocks = 0

        while self.running and offset < total_frames:
            session = self.session
            if session is None or session.stopping:
                return queued_blocks

            piece = audio[offset: offset + self.current_blocksize]

            if piece.shape[0] < self.current_blocksize:
                padding = np.zeros(
                    (self.current_blocksize - piece.shape[0], self.current_channels),
                    dtype=np.float32,
                )
                piece = np.vstack([piece, padding])

            try:
                session.output_queue.put(
                    PlaybackAudioBlock(
                        audio=piece.astype(np.float32, copy=False),
                        item_id=item_id,
                        text=text,
                        source_type=source_type,
                        created_at=created_at,
                        block_index=block_index,
                        total_blocks=total_blocks,
                        generation=generation,
                    ),
                    timeout=0.2,
                )
            except queue.Full:
                if not self.running:
                    return queued_blocks
                continue

            offset += self.current_blocksize
            block_index += 1
            queued_blocks += 1

        return queued_blocks

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

        cleared = session.clear_output_queue(
            reason="stale_tts_audio_queue",
            preserve_final=True,
            preserve_playing=True,
        )
        self._log(
            "Dropped stale TTS audio queue "
            f"({queued_audio_seconds:.2f} sec, clearedBlocks={cleared.cleared_blocks} "
            f"skippedItems={cleared.skipped_items} preservedBlocks={cleared.preserved_blocks} "
            f"preservedItems={cleared.preserved_items} reason=stale_partial_or_overflow)"
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
        self._ru_to_en_recent_partial_text = ""
        self._ru_to_en_last_emitted_at = 0.0
        self._ru_to_en_emitted_phrase_norms = set()
        self._ru_to_en_emitted_phrases = []
        self._ru_to_en_finalized_phrase_count = 0
        self._reset_ru_to_en_pending_phrase()
        with self._utterance_audio_lock:
            self._utterance_audio_chunks = []
            self._utterance_audio_total_frames = 0

    def _reset_output_audio_queue(self) -> None:
        session = self.session
        if session is None:
            return
        session.clear_output_queue(reason="reset_output_queue")

    def _reset_utterance_state(self) -> None:
        self.last_final_text = ""
        self.last_enqueued_final_text = ""
        self.last_emitted_source_text = ""
        self._recent_translated_compare_norms = []
        self._recent_translated_texts = []
        self._recent_translated_source_texts = []
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

    def _enqueue_final_text(
        self,
        text: str,
        source_text: str | None = None,
        *,
        source_type: str = "final",
    ) -> None:
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
            self.final_text_queue.put_nowait((text, source_type))
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
            return False

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
        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction != "ru_to_en":
            return False

        backend = (self.app_config.stt.backend or "nim").strip().lower()
        if backend == "faster_whisper":
            if not self.app_config.stt.partial_emit_enabled:
                return False
            return True
        if backend == "whisper_cpp":
            return True
        return False

    def _handle_low_latency_partial(self, text: str) -> None:
        normalized = self._normalize_text(text)
        if not normalized:
            return

        branch_config = self._get_active_branch_config()
        if branch_config.translation_direction == "ru_to_en":
            now = time.monotonic()
            self._ru_to_en_recent_partial_text = normalized

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
                normalized_candidate = self._normalize_ru_to_en_source_vocabulary(candidate)
                if normalized_candidate != candidate:
                    self._log(f"RU2EN source normalized: {candidate} -> {normalized_candidate}")
                    candidate = normalized_candidate
                if self._should_skip_ru_to_en_known_weak_draft(candidate):
                    self._log(f"LOWLAT skipped known weak draft: {candidate}")
                    self._reset_ru_to_en_pending_phrase()
                    continue
                partial_rewind_match = self._find_ru_to_en_partial_rewind_suffix_match(candidate)
                if partial_rewind_match:
                    self._log(f"LOWLAT skipped partial rewind suffix: {candidate} ~= {partial_rewind_match}")
                    self._reset_ru_to_en_pending_phrase()
                    continue
                if self._should_skip_ru_to_en_partial_refinement(candidate):
                    continue
                if not self._allow_ru_to_en_source_fragment(
                    candidate,
                    source_context="",
                    is_suffix_extracted=False,
                ):
                    continue
                if not self._should_emit_ru_to_en_partial_phrase(candidate):
                    continue
                source_duplicate_match = self._find_ru_to_en_source_contained_duplicate(candidate)
                if source_duplicate_match:
                    self._log(
                        "LOWLAT skipped source contained duplicate: "
                        f"{candidate} ~= {source_duplicate_match}"
                    )
                    self._reset_ru_to_en_pending_phrase()
                    continue
                if not self._mark_ru_to_en_partial_phrase_seen(candidate, now):
                    return

                self._ru_to_en_last_emitted_at = now
                self._remember_ru_to_en_emitted_phrase(candidate)
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
            self._log_ru_to_en_finalized_segments()
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
            for index, sentence in enumerate(completed_sentences):
                previous_sentence = completed_sentences[index - 1] if index > 0 else ""
                normalized_sentence = self._prepare_ru_to_en_phrase_for_queue(sentence)
                if len(normalized_sentence.split()) < phrase_min_words:
                    continue
                if self._should_skip_ru_to_en_known_weak_draft(normalized_sentence):
                    self._log(f"LOWLAT skipped known weak draft: {normalized_sentence}")
                    continue
                phrase_norm = self._normalize_compare_text(normalized_sentence)
                source_duplicate_match = self._find_ru_to_en_source_contained_duplicate(normalized_sentence)
                if (
                    not phrase_norm
                    or phrase_norm in self._ru_to_en_emitted_phrase_norms
                    or source_duplicate_match
                    or self._should_skip_ru_to_en_overlap_phrase(normalized_sentence)
                ):
                    if source_duplicate_match:
                        self._log(
                            "LOWLAT skipped source contained duplicate: "
                            f"{normalized_sentence} ~= {source_duplicate_match}"
                        )
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
                    is_final=is_final,
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
                    is_final=is_final,
                    generation=self._low_latency_generation,
                )
                emitted_any = True

        if is_final:
            trailing_fragment = self._prepare_ru_to_en_phrase_for_queue(trailing_fragment)
            if branch_config.translation_direction == "ru_to_en":
                if trailing_fragment and len(trailing_fragment.split()) >= phrase_min_words:
                    if self._should_skip_ru_to_en_known_weak_draft(trailing_fragment):
                        self._log(f"LOWLAT skipped known weak draft: {trailing_fragment}")
                    else:
                        trailing_norm = self._normalize_compare_text(trailing_fragment)
                        last_queued_norm = self._normalize_compare_text(self._low_latency_last_queued_text)
                        trailing_duplicate_match = self._find_ru_to_en_source_contained_duplicate(trailing_fragment)
                        if (
                            trailing_norm
                            and trailing_norm != last_queued_norm
                            and trailing_norm not in self._ru_to_en_emitted_phrase_norms
                            and not trailing_duplicate_match
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
                        elif trailing_duplicate_match:
                            self._log(
                                "LOWLAT skipped source contained duplicate: "
                                f"{trailing_fragment} ~= {trailing_duplicate_match}"
                            )
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

        self._process_final_text(text, source_type="final" if is_final else "partial")
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
            self._enqueue_final_text(
                normalized_sentence,
                source_text=normalized_sentence,
                source_type="final" if is_final else "partial",
            )
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
                    self._enqueue_final_text(
                        trailing_fragment,
                        source_text=trailing_fragment,
                        source_type="final",
                    )
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

    def _split_en_to_ru_final_segments(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        max_chars = 160
        max_words = 22
        soft_max_chars = 120
        comma_min_words = 8

        primary_segments = self._split_en_to_ru_primary_segments(text)
        if (
            len(primary_segments) <= 1
            and len(text) <= max_chars
            and len(text.split()) <= max_words
        ):
            return [text]

        segments: list[str] = []
        for primary_segment in primary_segments:
            primary_segment = self._normalize_text(primary_segment)
            if not primary_segment:
                continue

            if (
                len(primary_segment) <= max_chars
                and len(primary_segment.split()) <= max_words
            ):
                segments.append(primary_segment)
                continue

            segments.extend(
                self._split_en_to_ru_long_final_segment(
                    primary_segment,
                    max_chars=max_chars,
                    max_words=max_words,
                    soft_max_chars=soft_max_chars,
                    comma_min_words=comma_min_words,
                )
            )

        return [segment for segment in segments if segment]

    @staticmethod
    def _split_en_to_ru_primary_segments(text: str) -> list[str]:
        text = AudioEngine._normalize_text(text)
        if not text:
            return []

        matches = list(re.finditer(r"[^.!?;]+[.!?;]", text))
        segments = [
            match.group(0).strip()
            for match in matches
            if match.group(0).strip()
        ]

        tail_start = matches[-1].end() if matches else 0
        trailing_fragment = text[tail_start:].strip()
        if trailing_fragment:
            segments.append(trailing_fragment)

        return segments or [text]

    def _split_en_to_ru_long_final_segment(
        self,
        text: str,
        *,
        max_chars: int,
        max_words: int,
        soft_max_chars: int,
        comma_min_words: int,
    ) -> list[str]:
        text = self._normalize_text(text)
        if not text:
            return []

        comma_parts = [
            part.strip()
            for part in re.split(r"(?<=,)\s+", text)
            if part.strip()
        ]
        if len(comma_parts) > 1:
            comma_segments: list[str] = []
            current = ""
            for part in comma_parts:
                candidate = f"{current} {part}".strip() if current else part
                candidate_words = len(candidate.split())
                if (
                    current
                    and candidate_words >= comma_min_words
                    and (len(candidate) > soft_max_chars or candidate_words > max_words)
                ):
                    comma_segments.append(current)
                    current = part
                else:
                    current = candidate
            if current:
                comma_segments.append(current)

            if len(comma_segments) > 1:
                result: list[str] = []
                for segment in comma_segments:
                    result.extend(
                        self._split_en_to_ru_word_chunks(
                            segment,
                            max_chars=max_chars,
                            max_words=max_words,
                        )
                    )
                return result

        return self._split_en_to_ru_word_chunks(
            text,
            max_chars=max_chars,
            max_words=max_words,
        )

    def _split_en_to_ru_word_chunks(
        self,
        text: str,
        *,
        max_chars: int,
        max_words: int,
    ) -> list[str]:
        words = self._normalize_text(text).split()
        if not words:
            return []

        chunks: list[str] = []
        current_words: list[str] = []
        for word in words:
            candidate_words = current_words + [word]
            candidate = " ".join(candidate_words)
            if current_words and (
                len(candidate_words) > max_words
                or len(candidate) > max_chars
            ):
                chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

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

    def _should_skip_ru_to_en_partial_refinement(self, current_source: str) -> bool:
        if self._get_active_branch_config().translation_direction != "ru_to_en":
            return False

        current_source = self._normalize_text(current_source)
        previous_source = self._normalize_text(self._low_latency_last_queued_text)
        if not current_source or not previous_source:
            return False

        if current_source.endswith((".", "!", "?")):
            return False

        current_norm = self._normalize_compare_text(current_source).strip(".,!?;: ")
        previous_norm = self._normalize_compare_text(previous_source).strip(".,!?;: ")
        if not current_norm or not previous_norm or current_norm == previous_norm:
            return False

        prefix = f"{previous_norm} "
        if not current_norm.startswith(prefix):
            return False

        suffix = current_norm[len(prefix):].strip()
        if not suffix:
            return False

        delta_words = suffix.split()
        if not 1 <= len(delta_words) <= 2:
            return False

        self._log(f"LOWLAT skipped partial refinement: {previous_source} -> {current_source}")
        return True

    def _should_early_admit_ru_to_en_partial(self, fragment: str) -> bool:
        return False

    def _mark_ru_to_en_partial_phrase_seen(self, phrase: str, now: float) -> bool:
        phrase = self._normalize_text(phrase)
        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return False

        previous_last_seen_at = self._ru_to_en_pending_phrase_last_seen_at
        same_phrase = phrase_norm == self._ru_to_en_pending_phrase_norm
        gap_since_previous = (
            now - previous_last_seen_at
            if same_phrase and previous_last_seen_at > 0.0
            else None
        )

        old_stale_after_sec = 0.55
        stale_after_sec = 0.8
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
        admitted = (
            self._ru_to_en_pending_phrase_seen_count >= 2
            or stable_for >= self._get_ru_to_en_partial_phrase_stability_sec(phrase)
        )
        if (
            admitted
            and self._ru_to_en_pending_phrase_seen_count >= 2
            and gap_since_previous is not None
            and old_stale_after_sec < gap_since_previous <= stale_after_sec
        ):
            self._log(f"LOWLAT phrase_seen relaxed admit: {phrase}")
        return admitted

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

        normalized_phrase = self._normalize_ru_to_en_source_vocabulary(phrase)
        if normalized_phrase != phrase:
            self._log(f"RU2EN source normalized: {phrase} -> {normalized_phrase}")
            phrase = normalized_phrase

        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return ""
        comparable_phrase_norm = self._normalize_ru_to_en_dedup_phrase(phrase)
        phrase_words = phrase.split()
        phrase_norm_words = phrase_norm.split()

        for emitted_phrase in reversed(self._ru_to_en_emitted_phrases[-2:]):
            emitted_norm = self._normalize_compare_text(emitted_phrase)
            if not emitted_norm:
                continue
            comparable_emitted_norm = self._normalize_ru_to_en_dedup_phrase(emitted_phrase)
            emitted_norm_words = emitted_norm.split()

            if self._is_ru_to_en_near_duplicate_phrase(comparable_phrase_norm, comparable_emitted_norm):
                return ""

            common_prefix_len = 0
            for emitted_word, phrase_word in zip(emitted_norm_words, phrase_norm_words):
                if emitted_word != phrase_word:
                    break
                common_prefix_len += 1

            if 0 < common_prefix_len < len(phrase_norm_words):
                remaining_raw_words = phrase_words[common_prefix_len:]
                if remaining_raw_words:
                    remaining_text = self._normalize_text(" ".join(remaining_raw_words))
                    if self._should_skip_ru_to_en_prepare_born_suffix_tail_to(remaining_text):
                        return ""
                    if (
                        self._is_ru_to_en_safe_suffix_emit(remaining_text)
                        and self._allow_ru_to_en_source_fragment(
                            remaining_text,
                            source_context=emitted_phrase,
                            is_suffix_extracted=True,
                        )
                    ):
                        return remaining_text
                    return ""

            incremental = self._normalize_text(
                self._extract_incremental_text(emitted_phrase, phrase)
            )
            incremental_norm = self._normalize_compare_text(incremental)
            if incremental_norm and incremental_norm != phrase_norm:
                if self._should_skip_ru_to_en_prepare_born_suffix_tail_to(incremental):
                    return ""
                if (
                    self._is_ru_to_en_safe_suffix_emit(incremental)
                    and self._allow_ru_to_en_source_fragment(
                        incremental,
                        source_context=emitted_phrase,
                        is_suffix_extracted=True,
                    )
                ):
                    return incremental
                return ""

            if (
                len(phrase_norm_words) > len(emitted_norm_words)
                and len(emitted_norm_words) >= 3
                and phrase_norm_words[-len(emitted_norm_words):] == emitted_norm_words
            ):
                return ""

            overlap_len = self._find_ru_to_en_suffix_prefix_overlap_len(
                emitted_norm_words,
                phrase_norm_words,
            )
            if overlap_len >= 3 and overlap_len == len(emitted_norm_words):
                return ""

            if phrase_norm == emitted_norm:
                return ""

            if len(comparable_phrase_norm.split()) <= 6 and f" {comparable_phrase_norm} " in f" {comparable_emitted_norm} ":
                return ""

        if not self._allow_ru_to_en_source_fragment(
            phrase,
            source_context="",
            is_suffix_extracted=False,
        ):
            return ""

        return phrase

    def _should_skip_ru_to_en_prepare_born_suffix_tail_to(self, fragment: str) -> bool:
        fragment = self._normalize_text(fragment)
        normalized = self._normalize_compare_text(fragment)
        if not normalized:
            return False

        words = normalized.split()
        if len(words) > 5:
            return False

        if words[0] != "то":
            return False

        self._log(f"LOWLAT skipped prepare-born suffix-tail то: {fragment}")
        return True


    def _allow_ru_to_en_source_fragment(
        self,
        fragment: str,
        *,
        source_context: str,
        is_suffix_extracted: bool,
    ) -> bool:
        if self._get_active_branch_config().translation_direction != "ru_to_en":
            return True

        fragment = self._normalize_text(fragment)
        if not fragment:
            return False

        suspicious_starter = self._get_ru_to_en_suspicious_starter(fragment)
        if not suspicious_starter:
            return True

        if not self._RU_TO_EN_DEPENDENT_SUFFIX_ADMISSION_FILTER_ENABLED:
            self._log(f"LOWLAT allowed suspicious fragment: {fragment}")
            return True

        if not is_suffix_extracted:
            self._log(f"LOWLAT allowed suspicious fragment: {fragment}")
            return True

        if not self._is_ru_to_en_weak_dependent_fragment(fragment):
            self._log(f"LOWLAT allowed suspicious fragment: {fragment}")
            return True

        if self._has_ru_to_en_new_semantic_payload(fragment, source_context):
            self._log(f"LOWLAT allowed suspicious fragment: {fragment}")
            return True

        if not self._is_ru_to_en_fragment_covered_by_source_context(fragment, source_context):
            self._log(f"LOWLAT allowed suspicious fragment: {fragment}")
            return True

        self._log(f"LOWLAT blocked dependent suffix fragment: {fragment}")
        return False

    @classmethod
    def _get_ru_to_en_suspicious_starter(cls, fragment: str) -> str:
        normalized = cls._normalize_compare_text(fragment)
        if not normalized:
            return ""

        for starter in (
            "так как",
            "потому что",
            "чтобы",
            "и то",
            "а то",
            "которого",
            "которому",
            "которыми",
            "которыми",
            "которым",
            "которую",
            "которая",
            "которое",
            "которые",
            "которой",
            "который",
            "то",
            "но",
            "и",
            "а",
            "или",
        ):
            if normalized == starter or normalized.startswith(f"{starter} "):
                return starter
        return ""

    def _is_ru_to_en_weak_dependent_fragment(self, fragment: str) -> bool:
        normalized = self._normalize_compare_text(fragment)
        if not normalized:
            return True

        words = normalized.split()
        if len(words) <= 4:
            return True

        keywords = self._extract_ru_to_en_final_cleanup_keywords(fragment)
        return len(words) <= 5 and len(keywords) <= 2

    def _has_ru_to_en_new_semantic_payload(self, fragment: str, source_context: str) -> bool:
        fragment_keywords = self._extract_ru_to_en_final_cleanup_keywords(fragment)
        if not fragment_keywords:
            return False

        if re.search(r"\d", fragment):
            return True

        context_keywords = self._extract_ru_to_en_final_cleanup_keywords(source_context)
        new_keywords = fragment_keywords - context_keywords
        if len(new_keywords) >= 2:
            return True

        meaningful_payload_markers = {
            "активн",
            "быстр",
            "гибк",
            "до сих пор",
            "невероятн",
            "помог",
            "польз",
            "работ",
            "ускор",
        }
        normalized = self._normalize_compare_text(fragment)
        if any(marker in normalized for marker in meaningful_payload_markers):
            return True

        return False

    def _is_ru_to_en_fragment_covered_by_source_context(self, fragment: str, source_context: str) -> bool:
        fragment_norm = self._normalize_ru_to_en_dedup_phrase(fragment)
        context_norm = self._normalize_ru_to_en_dedup_phrase(source_context)
        if not fragment_norm or not context_norm:
            return False

        if f" {fragment_norm} " in f" {context_norm} ":
            return True

        fragment_words = fragment_norm.split()
        context_words = context_norm.split()
        if len(fragment_words) >= 3 and len(context_words) >= len(fragment_words):
            if fragment_words == context_words[-len(fragment_words):]:
                return True

        overlap_len = self._find_ru_to_en_suffix_prefix_overlap_len(
            context_words,
            fragment_words,
        )
        if overlap_len >= max(3, len(fragment_words) - 1):
            return True

        return SequenceMatcher(None, context_norm, fragment_norm).ratio() >= 0.72

    def _remember_ru_to_en_emitted_phrase(self, phrase: str) -> None:
        phrase = self._normalize_text(phrase)
        phrase_norm = self._normalize_compare_text(phrase)
        if not phrase_norm:
            return

        self._ru_to_en_emitted_phrase_norms.add(phrase_norm)
        self._ru_to_en_emitted_phrases.append(phrase)
        if len(self._ru_to_en_emitted_phrases) > 24:
            overflow = len(self._ru_to_en_emitted_phrases) - 24
            self._ru_to_en_emitted_phrases = self._ru_to_en_emitted_phrases[-24:]
            self._ru_to_en_finalized_phrase_count = max(
                0,
                self._ru_to_en_finalized_phrase_count - overflow,
            )

    def _log_ru_to_en_finalized_segments(self) -> None:
        new_phrases = self._ru_to_en_emitted_phrases[self._ru_to_en_finalized_phrase_count:]
        if not new_phrases:
            return

        cleaned_phrases = self._cleanup_ru_to_en_finalized_phrases(new_phrases)
        if not cleaned_phrases:
            cleaned_phrases = new_phrases

        final_text = self._normalize_text(" ".join(cleaned_phrases))
        if not final_text:
            return

        self._ru_to_en_finalized_phrase_count = len(self._ru_to_en_emitted_phrases)
        self._log(f"FINAL: {final_text}")

    def _cleanup_ru_to_en_finalized_phrases(self, phrases: list[str]) -> list[str]:
        cleaned: list[str] = []

        for raw_phrase in phrases:
            phrase = self._normalize_text(raw_phrase)
            if not phrase:
                continue

            cleaned.append(phrase)
            while len(cleaned) >= 2:
                previous = cleaned[-2]
                current = cleaned[-1]
                action, replacement = self._resolve_ru_to_en_final_phrase_pair(previous, current)

                if action == "keep":
                    break

                if action == "replace_previous":
                    self._log(f"FINAL cleanup dropped weak draft: {previous}")
                    cleaned[-2] = replacement or current
                    cleaned.pop()
                    continue

                if action == "skip_current":
                    self._log(f"FINAL cleanup dropped weak return: {current}")
                    cleaned.pop()
                    break

                if action == "merge":
                    if replacement and replacement != previous:
                        self._log(f"FINAL cleanup merged: {previous} + {current} -> {replacement}")
                    cleaned[-2] = replacement or previous
                    cleaned.pop()
                    continue

                break

        return cleaned

    def _resolve_ru_to_en_final_phrase_pair(self, previous: str, current: str) -> tuple[str, str]:
        previous_norm = self._normalize_compare_text(previous)
        current_norm = self._normalize_compare_text(current)
        if not previous_norm or not current_norm:
            return "keep", ""

        if previous_norm == current_norm:
            return "replace_previous", current if len(current) >= len(previous) else previous

        previous_keywords = self._extract_ru_to_en_final_cleanup_keywords(previous)
        current_keywords = self._extract_ru_to_en_final_cleanup_keywords(current)
        shared_keywords = previous_keywords & current_keywords

        overlap_len = self._find_ru_to_en_suffix_prefix_overlap_len(
            previous_norm.split(),
            current_norm.split(),
        )
        merged_text = self._merge_final_texts(previous, current)
        merged_norm = self._normalize_compare_text(merged_text)
        if (
            merged_norm
            and merged_norm not in {previous_norm, current_norm}
            and overlap_len >= 2
        ):
            return "merge", merged_text

        if self._is_ru_to_en_near_duplicate_phrase(previous_norm, current_norm):
            return "replace_previous", current if len(current) >= len(previous) else previous

        if self._is_ru_to_en_weak_final_draft(previous, current, previous_keywords, current_keywords, shared_keywords):
            return "replace_previous", current

        if self._is_ru_to_en_weak_final_draft(current, previous, current_keywords, previous_keywords, shared_keywords):
            return "skip_current", previous

        return "keep", ""

    def _is_ru_to_en_weak_final_draft(
        self,
        draft: str,
        fuller: str,
        draft_keywords: set[str],
        fuller_keywords: set[str],
        shared_keywords: set[str],
    ) -> bool:
        draft_norm = self._normalize_compare_text(draft)
        fuller_norm = self._normalize_compare_text(fuller)
        if not draft_norm or not fuller_norm or draft_norm == fuller_norm:
            return False

        draft_words = draft_norm.split()
        fuller_words = fuller_norm.split()
        if len(fuller_words) < len(draft_words):
            return False

        if self._ru_to_en_phrase_has_unique_marker(draft, fuller):
            return False

        if not draft_keywords:
            return len(draft_words) <= 3 and len(fuller_words) >= len(draft_words) + 2

        if not shared_keywords:
            return False

        overlap_ratio = len(shared_keywords) / max(1, len(draft_keywords))
        if overlap_ratio < 0.74:
            return False

        if len(draft_keywords) == 1:
            return len(fuller_words) >= len(draft_words) + 2

        if fuller_keywords - draft_keywords:
            return True

        return len(fuller_words) >= len(draft_words) + 2

    def _find_ru_to_en_source_contained_duplicate(self, phrase: str) -> str:
        phrase_text = self._normalize_text(phrase)
        phrase_norm = self._normalize_ru_to_en_dedup_phrase(phrase_text)
        phrase_words = phrase_norm.split()
        if len(phrase_words) < 3:
            return ""

        phrase_keywords = self._extract_ru_to_en_final_cleanup_keywords(phrase_text)
        for previous_text in reversed(self._ru_to_en_emitted_phrases[-5:]):
            previous_norm = self._normalize_ru_to_en_dedup_phrase(previous_text)
            previous_words = previous_norm.split()
            if len(previous_words) < len(phrase_words):
                continue

            if (
                len(phrase_words) >= 4
                and f" {phrase_norm} " in f" {previous_norm} "
            ):
                return previous_text

            if (
                len(phrase_words) >= 3
                and phrase_words == previous_words[-len(phrase_words):]
            ):
                previous_joined = "".join(previous_words)
                phrase_joined = "".join(phrase_words)
                if previous_joined and phrase_joined and (len(phrase_joined) / len(previous_joined)) >= 0.5:
                    return previous_text

            previous_keywords = self._extract_ru_to_en_final_cleanup_keywords(previous_text)
            if not phrase_keywords or len(phrase_keywords) < 2 or len(previous_keywords) < len(phrase_keywords):
                continue

            shared_keywords = phrase_keywords & previous_keywords
            if len(shared_keywords) < len(phrase_keywords):
                continue

            if len(phrase_words) + 2 <= len(previous_words):
                return previous_text

            if len(phrase_keywords) >= 3 and len(phrase_words) < len(previous_words):
                return previous_text

        return ""

    @classmethod
    def _ru_to_en_phrase_has_unique_marker(cls, draft: str, fuller: str) -> bool:
        if re.search(r"\d", draft):
            return True

        draft_latin = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", draft)}
        fuller_latin = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", fuller)}
        if draft_latin - fuller_latin:
            return True

        draft_tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", draft)
        fuller_tokens = {token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", fuller)}
        for token in draft_tokens[1:]:
            if token and token[0].isupper() and token.lower() not in fuller_tokens:
                return True

        return False

    @classmethod
    def _extract_ru_to_en_final_cleanup_keywords(cls, text: str) -> set[str]:
        normalized = cls._normalize_compare_text(text)
        if not normalized:
            return set()

        keywords: set[str] = set()
        for word in normalized.split():
            if word in cls._RU_TO_EN_FINAL_CLEANUP_STOPWORDS:
                continue
            normalized_word = cls._normalize_ru_to_en_final_cleanup_word(word)
            if not normalized_word or normalized_word in cls._RU_TO_EN_FINAL_CLEANUP_STOPWORDS:
                continue
            keywords.add(normalized_word)
        return keywords

    @classmethod
    def _normalize_ru_to_en_final_cleanup_word(cls, word: str) -> str:
        word = cls._normalize_compare_text(word)
        if not word:
            return ""

        if word.endswith(("ся", "сь")) and len(word) > 4:
            word = word[:-2]

        if len(word) <= 4:
            return word

        updated = True
        while updated:
            updated = False
            for suffix in cls._RU_TO_EN_FINAL_CLEANUP_SUFFIXES:
                if len(word) - len(suffix) < 4:
                    continue
                if word.endswith(suffix):
                    word = word[:-len(suffix)]
                    updated = True
                    break

        return word

    @staticmethod
    def _is_ru_to_en_safe_suffix_emit(text: str) -> bool:
        normalized = AudioEngine._normalize_compare_text(text)
        if not normalized:
            return False
        return len(normalized.split()) >= 3

    @staticmethod
    def _find_ru_to_en_suffix_prefix_overlap_len(
        emitted_words: list[str],
        phrase_words: list[str],
    ) -> int:
        max_overlap = min(len(emitted_words), len(phrase_words))
        for overlap_len in range(max_overlap, 2, -1):
            if emitted_words[-overlap_len:] == phrase_words[:overlap_len]:
                return overlap_len
        return 0

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
        comparable_phrase_norm = self._normalize_ru_to_en_dedup_phrase(phrase)

        emitted_norm = self._normalize_compare_text(self._low_latency_emitted_text)
        if not emitted_norm:
            return False
        comparable_emitted_norm = self._normalize_ru_to_en_dedup_phrase(self._low_latency_emitted_text)

        if comparable_phrase_norm == comparable_emitted_norm:
            return True

        phrase_words = comparable_phrase_norm.split()
        if len(phrase_words) <= 5 and f" {comparable_phrase_norm} " in f" {comparable_emitted_norm} ":
            return True

        for emitted_phrase in reversed(self._ru_to_en_emitted_phrases[-8:]):
            emitted_phrase_norm = self._normalize_ru_to_en_dedup_phrase(emitted_phrase)
            if (
                emitted_phrase_norm
                and (
                    (len(phrase_words) <= 6 and f" {comparable_phrase_norm} " in f" {emitted_phrase_norm} ")
                    or self._is_ru_to_en_near_duplicate_phrase(comparable_phrase_norm, emitted_phrase_norm)
                )
            ):
                return True

        emitted_words = comparable_emitted_norm.split()
        max_overlap = min(len(phrase_words), len(emitted_words), 6)
        for overlap_len in range(max_overlap, 1, -1):
            if emitted_words[-overlap_len:] == phrase_words[:overlap_len]:
                remaining_words = phrase_words[overlap_len:]
                if not remaining_words or len(remaining_words) <= 2:
                    return True

        return False

    @staticmethod
    def _normalize_ru_to_en_dedup_phrase(text: str) -> str:
        normalized = AudioEngine._normalize_compare_text(text)
        if not normalized:
            return ""

        leading_linkers = {
            "и",
            "а",
            "но",
            "ну",
            "так",
            "тогда",
            "потом",
        }
        words = normalized.split()
        while words and words[0] in leading_linkers:
            words = words[1:]

        return " ".join(words).strip()

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

        previous_text = self._normalize_text(self.last_translated_text)
        previous = self._normalize_compare_text(previous_text)
        current_text = self._normalize_text(text)
        current = self._normalize_compare_text(current_text)
        current_translated_norm = self._normalize_translated_compare_text(text)

        if not current or not current_translated_norm:
            return True

        if not previous:
            return False

        if (time.monotonic() - self.last_translated_at) > 4.0:
            return False

        for previous_translated_norm in reversed(self._recent_translated_compare_norms[-2:]):
            if current_translated_norm == previous_translated_norm:
                return True

        if current == previous:
            return True

        if current in previous:
            return True

        if self._is_translated_suffix_duplicate(previous_text, current_text):
            return True

        return False

    def _try_replace_pending_translated_duplicate(self, text: str) -> bool:
        if self._uses_sentence_partial_streaming() or self._uses_boundary_layer_streaming():
            return False

        current_translated_norm = self._normalize_translated_compare_text(text)
        if not current_translated_norm:
            return False

        with self.tts_text_queue.mutex:
            pending_items = self.tts_text_queue.queue
            if not pending_items:
                return False

            last_request = self._coerce_tts_request(pending_items[-1])
            last_text = last_request.text
            last_translated_norm = self._normalize_translated_compare_text(last_text)
            if current_translated_norm != last_translated_norm:
                return False

            pending_items[-1] = TTSRequest(
                text=text,
                samplerate=last_request.samplerate,
                source_type=last_request.source_type,
                generation=last_request.generation,
            )
            return True

    def _should_skip_strict_short_translated_fragment(
        self,
        source_text: str,
        translated_text: str,
    ) -> bool:
        if not self.app_config.tts.strict_short_translated_fragment_filter:
            return False

        if self._get_active_branch_config().translation_direction != "ru_to_en":
            return False

        normalized_text = self._normalize_text(translated_text)
        normalized_compare = self._normalize_compare_text(normalized_text)
        if not normalized_text or not normalized_compare:
            return False

        if normalized_text.endswith((".", "!", "?")):
            return False

        words = normalized_compare.split()
        if len(words) > 5:
            return False

        if words[0] not in {"and", "but", "because", "when", "if", "that", "which", "so"}:
            return False

        if re.search(r"\d", normalized_text):
            return False

        if self._translated_fragment_has_entity_or_brand_like_token(normalized_text):
            return False

        if self._translated_fragment_has_strong_content_word(words):
            return False

        if self._translated_fragment_starts_new_thought(words):
            return False

        return self._translated_fragment_has_recent_context_coverage(source_text)

    @staticmethod
    def _translated_fragment_has_entity_or_brand_like_token(text: str) -> bool:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
        for index, token in enumerate(tokens):
            if len(token) >= 2 and token.isupper():
                return True
            if any(ch.isupper() for ch in token[1:]):
                return True
            if index > 0 and token[:1].isupper():
                return True
            if re.search(r"[A-Za-z]+\d|\d+[A-Za-z]+", token):
                return True
        return False

    @staticmethod
    def _translated_fragment_has_strong_content_word(words: list[str]) -> bool:
        weak_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "because",
            "been",
            "being",
            "but",
            "by",
            "for",
            "from",
            "if",
            "in",
            "into",
            "is",
            "it",
            "its",
            "me",
            "my",
            "of",
            "on",
            "or",
            "our",
            "so",
            "that",
            "the",
            "their",
            "them",
            "there",
            "these",
            "this",
            "those",
            "to",
            "us",
            "was",
            "we",
            "were",
            "when",
            "which",
            "with",
            "without",
            "you",
            "your",
        }

        for word in words[1:]:
            if len(word) >= 6 and word not in weak_words:
                return True
            if word not in weak_words:
                return True
        return False

    @staticmethod
    def _translated_fragment_starts_new_thought(words: list[str]) -> bool:
        if len(words) < 2:
            return False

        if words[1] in {
            "i",
            "we",
            "he",
            "she",
            "they",
            "you",
            "my",
            "our",
            "his",
            "her",
            "their",
            "this",
            "that",
            "these",
            "those",
        }:
            return True

        return False

    def _translated_fragment_has_recent_context_coverage(self, source_text: str) -> bool:
        current_source = self._normalize_text(source_text)
        if not self._source_fragment_looks_like_weak_continuation(current_source):
            return False

        current_norm = self._normalize_ru_to_en_dedup_phrase(current_source)
        current_words = current_norm.split()
        if len(current_words) < 3:
            return False

        for previous_text in reversed(self._recent_translated_source_texts[-3:]):
            previous_norm = self._normalize_ru_to_en_dedup_phrase(previous_text)
            if not previous_norm or previous_norm == current_norm:
                continue
            previous_words = previous_norm.split()

            if current_words == previous_words[-len(current_words):]:
                return True

            overlap_len = self._find_ru_to_en_suffix_prefix_overlap_len(
                previous_words,
                current_words,
            )
            if overlap_len >= max(3, len(current_words) - 1):
                return True

            if SequenceMatcher(None, previous_norm, current_norm).ratio() >= 0.72:
                return True

        return False

    @classmethod
    def _source_fragment_looks_like_weak_continuation(cls, source_text: str) -> bool:
        normalized = cls._normalize_compare_text(source_text)
        if not normalized:
            return False

        words = normalized.split()
        if not words or len(words) > 6:
            return False

        first_word = words[0]
        if first_word in {"и", "но", "а", "или", "если", "когда", "что", "чтобы", "то"}:
            return True

        if first_word in {
            "который",
            "которая",
            "которое",
            "которые",
            "которого",
            "которой",
            "которому",
            "которым",
            "которую",
        }:
            return True

        return len(words) >= 2 and tuple(words[:2]) in {
            ("потому", "что"),
            ("так", "как"),
            ("так", "что"),
            ("для", "того"),
        }

    def _remember_translated_text_for_compare(self, text: str) -> None:
        normalized_text = self._normalize_text(text)
        normalized = self._normalize_translated_compare_text(text)
        if not normalized or not normalized_text:
            return

        self._recent_translated_compare_norms.append(normalized)
        if len(self._recent_translated_compare_norms) > 8:
            self._recent_translated_compare_norms = self._recent_translated_compare_norms[-8:]

        self._recent_translated_texts.append(normalized_text)
        if len(self._recent_translated_texts) > 8:
            self._recent_translated_texts = self._recent_translated_texts[-8:]

    def _remember_translated_source_text_for_compare(self, text: str) -> None:
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return

        self._recent_translated_source_texts.append(normalized_text)
        if len(self._recent_translated_source_texts) > 8:
            self._recent_translated_source_texts = self._recent_translated_source_texts[-8:]

    def _find_same_prefix_translated_retry(self, text: str) -> str:
        current_text = self._normalize_text(text)
        current_words = self._normalize_compare_text(current_text).split()
        if len(current_words) < 5:
            return ""

        for previous_text in reversed(self._recent_translated_texts[-5:]):
            previous_words = self._normalize_compare_text(previous_text).split()
            if len(previous_words) < 5:
                continue

            if abs(len(previous_words) - len(current_words)) > 2:
                continue

            min_len = min(len(previous_words), len(current_words))
            required_prefix_words = min(6, min_len - 1)
            if required_prefix_words < 4:
                continue

            common_prefix_len = 0
            for left, right in zip(previous_words, current_words):
                if left != right:
                    break
                common_prefix_len += 1

            if common_prefix_len < required_prefix_words:
                continue

            previous_tail = previous_words[common_prefix_len:]
            current_tail = current_words[common_prefix_len:]
            if not previous_tail or not current_tail or previous_tail == current_tail:
                continue

            if len(previous_tail) > 3 or len(current_tail) > 3:
                continue

            return previous_text

        return ""

    @classmethod
    def _is_translated_suffix_duplicate(cls, previous_text: str, current_text: str) -> bool:
        previous_tokens = cls._normalize_compare_text(previous_text).split()
        current_tokens = cls._normalize_compare_text(current_text).split()

        if len(previous_tokens) < 4 or len(current_tokens) < 3:
            return False

        if len(current_tokens) >= len(previous_tokens):
            return False

        if current_tokens != previous_tokens[-len(current_tokens):]:
            return False

        current_joined = "".join(current_tokens)
        previous_joined = "".join(previous_tokens)
        if not current_joined or not previous_joined:
            return False

        return (len(current_joined) / len(previous_joined)) >= 0.5

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
    def _normalize_translated_compare_text(text: str) -> str:
        text = AudioEngine._normalize_text(text).lower()
        return "".join(ch for ch in text if ch.isalnum())

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
