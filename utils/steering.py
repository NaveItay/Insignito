from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DOA:
    azimuth: float
    elevation: float


def doa_to_unit_vector(doa: DOA) -> np.ndarray:
    """
    Convert azimuth/elevation (radians) to unit vector using assignment convention:
        x = cos(-az) * cos(-el)
        y = sin(-az) * cos(-el)
        z = sin(-el)
    """
    az = doa.azimuth
    el = doa.elevation
    u = np.array([
        np.cos(-az) * np.cos(-el),
        np.sin(-az) * np.cos(-el),
        np.sin(-el),
    ], dtype=np.float64)

    n = np.linalg.norm(u) + 1e-12
    return u / n


def compute_delays_seconds(mic_pos: np.ndarray, u: np.ndarray, c: float, sign: float) -> np.ndarray:
    """
    Far-field plane-wave relative delays:
        tau_m = sign * (p_m dot u) / c
    Shift by tau[0] so mic0 is reference.
    """
    tau = sign * (mic_pos @ u) / c
    tau = tau - tau[0]
    return tau


def steering_vectors(freqs_hz: np.ndarray, delays_s: np.ndarray) -> np.ndarray:
    """
    a(f)[m] = exp(-j*2*pi*f*delay[m])
    Returns shape (F, M)
    """
    phase = -2.0 * np.pi * freqs_hz[:, None] * delays_s[None, :]
    return np.exp(1j * phase)
