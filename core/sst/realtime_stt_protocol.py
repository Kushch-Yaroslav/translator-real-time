from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np


class RealtimeSTTService(Protocol):
    def start(
        self,
        partial_callback: Optional[Callable[[str], None]] = None,
        final_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        ...

    def stop(self) -> None:
        ...

    def restart(self) -> None:
        ...

    def send_audio_chunk(self, audio: np.ndarray, samplerate: int) -> None:
        ...

    def commit(self) -> None:
        ...

    def clear(self) -> None:
        ...

    def send_done(self) -> None:
        ...

    def get_last_partial_text(self) -> str:
        ...

    def get_last_final_text(self) -> str:
        ...
