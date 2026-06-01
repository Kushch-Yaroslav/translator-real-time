from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
import time
from typing import Optional, Callable, Any

import numpy as np

from core.audio.audio_utils import create_input_stream, create_output_stream, stop_stream


@dataclass
class AudioSessionConfig:
    input_device_index: int
    output_device_index: int
    samplerate: int = 48000
    channels: int = 1
    blocksize: int = 1024
    input_queue_maxsize: int = 128
    output_queue_maxsize: int = 128


@dataclass(frozen=True)
class PlaybackAudioBlock:
    audio: np.ndarray
    item_id: str
    text: str
    source_type: str
    created_at: float
    block_index: int
    total_blocks: int
    generation: int = 0

    @property
    def is_start(self) -> bool:
        return self.block_index == 0

    @property
    def is_end(self) -> bool:
        return self.block_index >= self.total_blocks - 1


@dataclass(frozen=True)
class ClearOutputQueueResult:
    cleared_blocks: int = 0
    skipped_items: int = 0
    preserved_blocks: int = 0
    preserved_items: int = 0


class AudioSession:
    def __init__(self, config: AudioSessionConfig):
        self.config = config

        self.input_stream = None
        self.output_stream = None

        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.input_queue_maxsize)
        self.output_queue: queue.Queue[Any] = queue.Queue(maxsize=config.output_queue_maxsize)

        self.running = False
        self.stopping = False
        self._lock = threading.Lock()

        self.on_input_overflow: Optional[Callable[[], None]] = None
        self.on_output_underflow: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_playback_started: Optional[Callable[[str], None]] = None
        self.on_playback_finished: Optional[Callable[[str], None]] = None
        self.on_playback_skipped: Optional[Callable[[str, str], None]] = None
        self._playing_item_id: str = ""

    def start(self) -> None:
        with self._lock:
            if self.running:
                return

            self.stopping = False

            try:
                self.input_stream = create_input_stream(
                    device_index=self.config.input_device_index,
                    samplerate=self.config.samplerate,
                    channels=self.config.channels,
                    blocksize=self.config.blocksize,
                    callback=self._input_callback,
                )

                self.output_stream = create_output_stream(
                    device_index=self.config.output_device_index,
                    samplerate=self.config.samplerate,
                    channels=self.config.channels,
                    blocksize=self.config.blocksize,
                    callback=self._output_callback,
                )

                self.input_stream.start()
                self.output_stream.start()
                self.running = True

            except Exception as error:
                self.running = False
                self.stopping = True
                stop_stream(self.input_stream)
                stop_stream(self.output_stream)
                self.input_stream = None
                self.output_stream = None

                if self.on_error:
                    self.on_error(f"AudioSession start error: {error}")
                else:
                    raise

    def stop(self) -> None:
        with self._lock:
            if not self.running and self.stopping:
                return

            self.stopping = True
            self.running = False

            input_stream = self.input_stream
            output_stream = self.output_stream

            self.input_stream = None
            self.output_stream = None

        stop_stream(input_stream)
        stop_stream(output_stream)

        self._clear_queues()

        with self._lock:
            self.stopping = False

    def clear_output_queue(
        self,
        reason: str = "cleared",
        *,
        preserve_final: bool = False,
        preserve_playing: bool = False,
    ) -> ClearOutputQueueResult:
        cleared = 0
        preserved = 0
        preserved_items: set[str] = set()
        skipped_items: dict[str, tuple[str, bool]] = {}
        retained_items: list[Any] = []

        while not self.output_queue.empty():
            try:
                item = self.output_queue.get_nowait()
                if isinstance(item, PlaybackAudioBlock):
                    already_playing = item.item_id == self._playing_item_id
                    should_preserve = (
                        (preserve_playing and already_playing)
                        or (preserve_final and item.source_type == "final")
                    )
                    if should_preserve:
                        retained_items.append(item)
                        preserved += 1
                        preserved_items.add(item.item_id)
                        continue

                    cleared += 1
                    if item.text:
                        skipped_items.setdefault(
                            item.item_id,
                            (
                                self._format_playback_log_text(
                                    item,
                                    already_playing,
                                    status="skipped",
                                ),
                                already_playing,
                            ),
                        )
                    continue

                cleared += 1
                if isinstance(item, tuple) and len(item) == 4:
                    _chunk, text, _is_start, _is_end = item
                    if text:
                        skipped_items.setdefault(text, (text, False))
            except queue.Empty:
                break

        for item in retained_items:
            try:
                self.output_queue.put_nowait(item)
            except queue.Full:
                cleared += 1
                if isinstance(item, PlaybackAudioBlock) and item.text:
                    skipped_items.setdefault(
                        item.item_id,
                        (
                            self._format_playback_log_text(
                                item,
                                item.item_id == self._playing_item_id,
                                status="skipped",
                            ),
                            item.item_id == self._playing_item_id,
                        ),
                    )

        if self.on_playback_skipped:
            for text, already_playing in skipped_items.values():
                detailed_reason = reason
                if already_playing:
                    detailed_reason = f"{reason}:playing"
                self.on_playback_skipped(text, detailed_reason)

        return ClearOutputQueueResult(
            cleared_blocks=cleared,
            skipped_items=len(skipped_items),
            preserved_blocks=preserved,
            preserved_items=len(preserved_items),
        )

    def get_output_queue_duration_seconds(self) -> float:
        queued_blocks = self.output_queue.qsize()
        queued_frames = queued_blocks * self.config.blocksize
        return queued_frames / float(self.config.samplerate)

    def _clear_queues(self) -> None:
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    def _input_callback(self, indata, frames, time_info, status):
        try:
            if self.stopping or not self.running:
                return

            chunk = np.copy(indata)

            try:
                self.input_queue.put_nowait(chunk)
            except queue.Full:
                pass
        except Exception:
            pass

    def _output_callback(self, outdata, frames, time_info, status):
        try:
            if self.stopping or not self.running:
                outdata.fill(0)
                return

            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                return

            text = ""
            is_start = False
            is_end = False
            if isinstance(item, PlaybackAudioBlock):
                chunk = item.audio
                text = item.text
                is_start = item.is_start
                is_end = item.is_end
            elif isinstance(item, tuple) and len(item) == 4:
                chunk, text, is_start, is_end = item
            else:
                chunk = item

            if is_start and text and self.on_playback_started:
                if isinstance(item, PlaybackAudioBlock):
                    self._playing_item_id = item.item_id
                    text = self._format_playback_log_text(item, False, status="playing")
                self.on_playback_started(text)

            if chunk.shape[0] != frames:
                if chunk.shape[0] > frames:
                    chunk = chunk[:frames]
                else:
                    padding = np.zeros((frames - chunk.shape[0], self.config.channels), dtype=np.float32)
                    chunk = np.vstack([chunk, padding])

            outdata[:] = chunk.astype(np.float32, copy=False)
            if is_end and text and self.on_playback_finished:
                if isinstance(item, PlaybackAudioBlock):
                    text = self._format_playback_log_text(item, True, status="finished")
                self.on_playback_finished(text)
                if isinstance(item, PlaybackAudioBlock) and self._playing_item_id == item.item_id:
                    self._playing_item_id = ""
        except Exception:
            outdata.fill(0)

    def _format_playback_log_text(
        self,
        item: PlaybackAudioBlock,
        already_playing: bool,
        *,
        status: str,
    ) -> str:
        age_sec = max(0.0, time.monotonic() - item.created_at)
        return (
            f"{item.text} "
            f"id={item.item_id} sourceType={item.source_type} status={status} "
            f"textLen={len(item.text)} queueSize={self.output_queue.qsize()} "
            f"itemAge={age_sec:.3f}s alreadyPlaying={already_playing} "
            f"generation={item.generation}"
        )
