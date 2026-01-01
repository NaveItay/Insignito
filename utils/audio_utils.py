from __future__ import annotations
import numpy as np


def normalize_audio(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mx = float(np.max(np.abs(y)) + 1e-12)
    return (peak / mx) * y
