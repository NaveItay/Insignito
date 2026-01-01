from __future__ import annotations
from typing import Tuple
import numpy as np


def _stft_onesided(
    x: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    pad = n_fft // 2 if center else 0

    x_pad = np.pad(x, (pad, pad), mode="constant") if pad > 0 else x

    if len(x_pad) < n_fft:
        x_pad = np.pad(x_pad, (0, n_fft - len(x_pad)), mode="constant")

    n_frames = int(np.ceil((len(x_pad) - n_fft) / hop)) + 1
    pad_end = (n_frames - 1) * hop + n_fft - len(x_pad)
    if pad_end > 0:
        x_pad = np.pad(x_pad, (0, pad_end), mode="constant")

    F = n_fft // 2 + 1
    Z = np.empty((F, n_frames), dtype=np.complex128)

    for ti in range(n_frames):
        start = ti * hop
        frame = x_pad[start:start + n_fft] * window
        Z[:, ti] = np.fft.rfft(frame, n=n_fft)

    if center:
        times = ((np.arange(n_frames) * hop) - pad) / float(fs)
    else:
        times = (np.arange(n_frames) * hop) / float(fs)

    return Z, times


def _istft_onesided(
    Z: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    length: int,
    center: bool = True
) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.complex128)
    n_frames = Z.shape[1]
    pad = n_fft // 2 if center else 0

    out_len = (n_frames - 1) * hop + n_fft
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)

    win_sq = window.astype(np.float64) ** 2

    for ti in range(n_frames):
        start = ti * hop
        frame = np.fft.irfft(Z[:, ti], n=n_fft).astype(np.float64)
        y[start:start + n_fft] += frame * window
        wsum[start:start + n_fft] += win_sq

    y = y / (wsum + 1e-12)

    if center:
        y = y[pad:pad + length]
    else:
        y = y[:length]

    return y


def stft_multichannel(
    x: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(n_fft).astype(np.float64)

    X_list = []
    times = None

    for m in range(x.shape[1]):
        Z, t = _stft_onesided(x[:, m], fs, n_fft, hop, window, center=True)
        if times is None:
            times = t
        X_list.append(Z)

    X = np.stack(X_list, axis=-1)  # (F,T,M)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return freqs, times, X


def istft_mono(
    Y: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int,
    length: int
) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float64)
    return _istft_onesided(Y, fs, n_fft, hop, window, length=length, center=True)
