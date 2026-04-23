from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional, Callable

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


class AudioSession:
    def __init__(self, config: AudioSessionConfig):
        self.config = config

        self.input_stream = None
        self.output_stream = None

        self.input_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.input_queue_maxsize)
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.output_queue_maxsize)

        self.running = False
        self.stopping = False
        self._lock = threading.Lock()

        self.on_input_overflow: Optional[Callable[[], None]] = None
        self.on_output_underflow: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

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

    def clear_output_queue(self) -> int:
        cleared = 0

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break

        return cleared

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
                chunk = self.output_queue.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                return

            if chunk.shape[0] != frames:
                if chunk.shape[0] > frames:
                    chunk = chunk[:frames]
                else:
                    padding = np.zeros((frames - chunk.shape[0], self.config.channels), dtype=np.float32)
                    chunk = np.vstack([chunk, padding])

            outdata[:] = chunk.astype(np.float32, copy=False)
        except Exception:
            outdata.fill(0)
