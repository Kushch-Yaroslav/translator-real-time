from __future__ import annotations

from typing import Protocol

import numpy as np


class ConfirmSTTService(Protocol):
    def transcribe(self, audio: np.ndarray, samplerate: int) -> str:
        ...
