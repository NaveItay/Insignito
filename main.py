#!/usr/bin/env python3
"""
Insignito Beamforming Assignment Solution
----------------------------------------
Beamformer: Frequency-domain LCMV (MVDR with linear constraints)
- Output 1: Distortionless response to source #1, null to source #2
- Output 2: Distortionless response to source #2, null to source #1

Inputs:
- Multichannel WAV (N x M)
- array_geometry.yaml (M x 3 mic positions in meters)
- Two source DOAs (azimuth/elevation in radians)

Outputs:
- source1.wav (mono)
- source2.wav (mono)

Dependencies:
- numpy
- scipy
- soundfile
- pyyaml
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import stft, istft


# ------------------------- Configuration -------------------------

SPEED_OF_SOUND = 343.0  # m/s


@dataclass(frozen=True)
class DOA:
    azimuth: float
    elevation: float


# ------------------------- IO -------------------------

def load_geometry_yaml(path: str) -> np.ndarray:
    """Load mic positions from YAML. Expects key: 'array_geometry' -> list of [x,y,z]."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "array_geometry" not in data:
        raise KeyError(f"YAML file does not contain 'array_geometry': {path}")

    geom = np.asarray(data["array_geometry"], dtype=np.float64)
    if geom.ndim != 2 or geom.shape[1] != 3:
        raise ValueError(f"array_geometry must be shape (M,3). Got: {geom.shape}")

    return geom


def load_multichannel_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    Returns:
      x: float64 array shape (N, M), scaled to [-1, 1] if original is int
      fs: sample rate
    """
    x, fs = sf.read(path, always_2d=True)
    x = np.asarray(x, dtype=np.float64)

    # Remove per-channel DC
    x -= np.mean(x, axis=0, keepdims=True)

    return x, fs


# ------------------------- Geometry / Steering -------------------------

def doa_to_unit_vector(doa: DOA) -> np.ndarray:
    """
    Convert azimuth/elevation (radians) to unit vector using the assignment convention:

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

    # Normalize (just in case)
    n = np.linalg.norm(u) + 1e-12
    return u / n


def compute_delays_seconds(mic_pos: np.ndarray, u: np.ndarray, c: float, sign: float) -> np.ndarray:
    """
    Far-field plane-wave relative delays:
        tau_m = sign * (p_m dot u) / c
    We shift by tau[0] so mic0 is reference.
    """
    tau = sign * (mic_pos @ u) / c
    tau = tau - tau[0]
    return tau


def steering_vectors(freqs_hz: np.ndarray, delays_s: np.ndarray) -> np.ndarray:
    """
    a(f)[m] = exp(-j*2*pi*f*delay[m])
    Returns shape (F, M)
    """
    # freqs_hz: (F,)
    # delays_s: (M,)
    phase = -2.0 * np.pi * freqs_hz[:, None] * delays_s[None, :]
    return np.exp(1j * phase)


# ------------------------- Mic Health / Selection -------------------------

def detect_bad_mics(x: np.ndarray, clip_threshold: float = 0.999) -> np.ndarray:
    """
    Simple robust mic selection:
      - remove channels with extremely low RMS (dead)
      - remove channels with extremely high RMS (very noisy)
      - remove channels with frequent clipping (|x| near 1 for float audio)
    Returns:
      good_mask: boolean shape (M,)
    """
    eps = 1e-12
    rms = np.sqrt(np.mean(x**2, axis=0) + eps)
    med = np.median(rms)

    # clipping ratio (works for float audio in [-1,1])
    clip_ratio = np.mean(np.abs(x) >= clip_threshold, axis=0)

    # thresholds (tuned for robustness)
    dead = rms < (0.05 * med)
    noisy = rms > (20.0 * med)
    clipped = clip_ratio > 1e-3  # 0.1% samples clipped is suspicious

    bad = dead | noisy | clipped | ~np.isfinite(rms)
    good = ~bad

    return good


# ------------------------- Beamforming Core -------------------------

def stft_multichannel(
    x: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for each channel.
    Returns:
      freqs, times, X where X has shape (F, T, M)
    """
    nperseg = n_fft
    noverlap = n_fft - hop

    X_list = []
    freqs = times = None

    for m in range(x.shape[1]):
        f, t, Z = stft(
            x[:, m],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
            return_onesided=True
        )
        if freqs is None:
            freqs, times = f, t
        X_list.append(Z)  # (F, T)

    X = np.stack(X_list, axis=-1)  # (F, T, M)
    return freqs, times, X


def istft_mono(
    Y: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int
) -> np.ndarray:
    """Inverse STFT for mono signal. Y shape: (F, T)."""
    nperseg = n_fft
    noverlap = n_fft - hop
    _, y = istft(
        Y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True
    )
    return np.asarray(y, dtype=np.float64)


def choose_delay_sign_by_ds_energy(
    X: np.ndarray,
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    doa: DOA,
    c: float
) -> float:
    """
    The DOA sign convention can flip depending on coordinate definition.
    We test both signs using a simple delay-and-sum score and pick the one
    that yields higher output power.
    """
    u = doa_to_unit_vector(doa)

    def ds_score(sign: float) -> float:
        tau = compute_delays_seconds(mic_pos, u, c, sign)
        a = steering_vectors(freqs, tau)  # (F,M)
        M = a.shape[1]
        # DS weights (distortionless): w ~ a / M
        # y(f,t) = x(f,t,:) @ conj(w)
        # compute quick power score
        power = 0.0
        for fi in range(X.shape[0]):
            w = a[fi, :] / M  # (M,)
            y = X[fi, :, :] @ np.conj(w)  # (T,)
            power += float(np.mean(np.abs(y) ** 2))
        return power

    p_plus = ds_score(+1.0)
    p_minus = ds_score(-1.0)
    return +1.0 if p_plus >= p_minus else -1.0


def lcmv_weights_per_freq(
    R: np.ndarray,
    C: np.ndarray,
    f_vec: np.ndarray
) -> np.ndarray:
    """
    Solve LCMV:
      minimize w^H R w  subject to C^H w = f_vec

    Closed form:
      w = R^{-1} C (C^H R^{-1} C)^{-1} f_vec

    R: (M,M)
    C: (M,K)
    f_vec: (K,)
    Returns: w (M,)
    """
    M = R.shape[0]
    K = C.shape[1]

    # regularization for numerical stability
    trace = np.trace(R).real
    dl = 1e-3 * (trace / M + 1e-12)
    R_reg = R + dl * np.eye(M, dtype=R.dtype)

    # Solve R^{-1} C without explicitly inverting R
    try:
        RinvC = np.linalg.solve(R_reg, C)  # (M,K)
    except np.linalg.LinAlgError:
        RinvC = np.linalg.pinv(R_reg) @ C

    G = C.conj().T @ RinvC  # (K,K)

    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)

    w = RinvC @ (Ginv @ f_vec)  # (M,)
    return w


def beamform_lcmv(
    X: np.ndarray,          # (F,T,M)
    freqs: np.ndarray,      # (F,)
    mic_pos: np.ndarray,    # (M,3)
    doa_target: DOA,
    doa_interf: DOA,
    c: float
) -> np.ndarray:
    """
    LCMV beamforming per-frequency:
      - pass target with gain 1
      - null interferer with gain 0

    Returns:
      Y: (F,T) complex STFT for the target output
    """
    # Decide sign convention automatically using DS energy for the target DOA
    sign = choose_delay_sign_by_ds_energy(X, freqs, mic_pos, doa_target, c)

    u_t = doa_to_unit_vector(doa_target)
    u_i = doa_to_unit_vector(doa_interf)

    tau_t = compute_delays_seconds(mic_pos, u_t, c, sign)
    tau_i = compute_delays_seconds(mic_pos, u_i, c, sign)

    a_t = steering_vectors(freqs, tau_t)  # (F,M)
    a_i = steering_vectors(freqs, tau_i)  # (F,M)

    F, T, M = X.shape
    Y = np.zeros((F, T), dtype=np.complex128)

    f_vec = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)

    for fi in range(F):
        Xf = X[fi, :, :]              # (T,M)
        R = (Xf.conj().T @ Xf) / max(T, 1)  # (M,M)

        C = np.stack([a_t[fi, :], a_i[fi, :]], axis=1)  # (M,2)
        w = lcmv_weights_per_freq(R, C, f_vec)          # (M,)

        # Apply: y(t) = w^H x(t) = x(t,:) @ conj(w)
        Y[fi, :] = Xf @ np.conj(w)

    return Y


# ------------------------- Utilities -------------------------

def normalize_audio(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mx = float(np.max(np.abs(y)) + 1e-12)
    return (peak / mx) * y


# ------------------------- Main -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="50ch Beamforming (LCMV) - Insignito Assignment")
    parser.add_argument("--input_wav", default="recording.wav", help="Path to multichannel wav (50ch)")
    parser.add_argument("--geometry_yaml", default="array_geometry.yaml", help="Path to geometry YAML")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--c", type=float, default=SPEED_OF_SOUND, help="Speed of sound (m/s)")

    # STFT settings
    parser.add_argument("--n_fft", type=int, default=0, help="FFT size (0=auto)")
    parser.add_argument("--hop", type=int, default=0, help="Hop size (0=auto)")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    mic_pos = load_geometry_yaml(args.geometry_yaml)
    x, fs = load_multichannel_wav(args.input_wav)

    if x.shape[1] != mic_pos.shape[0]:
        M = min(x.shape[1], mic_pos.shape[0])
        print(f"[WARN] Channel count mismatch: WAV has {x.shape[1]}ch, geometry has {mic_pos.shape[0]} mics.")
        print(f"[WARN] Using first {M} channels/mics.")
        x = x[:, :M]
        mic_pos = mic_pos[:M, :]

    print(f"[INFO] Loaded audio: {x.shape[0]} samples, {x.shape[1]} channels, fs={fs} Hz")
    print(f"[INFO] Geometry: {mic_pos.shape[0]} mics")

    # Detect bad microphones
    good_mask = detect_bad_mics(x)
    good_idx = np.where(good_mask)[0].tolist()
    bad_idx = np.where(~good_mask)[0].tolist()
    print(f"[INFO] Good mics: {len(good_idx)} / {x.shape[1]}")
    if bad_idx:
        print(f"[INFO] Excluding bad mics: {bad_idx}")

    x = x[:, good_mask]
    mic_pos = mic_pos[good_mask, :]

    # Auto STFT params
    if args.n_fft > 0:
        n_fft = args.n_fft
    else:
        n_fft = 2048 if fs >= 32000 else 1024

    if args.hop > 0:
        hop = args.hop
    else:
        hop = n_fft // 4  # 75% overlap

    print(f"[INFO] STFT: n_fft={n_fft}, hop={hop}")

    # DOAs from assignment
    doa1 = DOA(azimuth=-0.069, elevation=0.0)
    doa2 = DOA(azimuth=1.029, elevation=0.017)

    # STFT
    freqs, times, X = stft_multichannel(x, fs, n_fft, hop)
    print(f"[INFO] STFT shapes: freqs={freqs.shape}, times={times.shape}, X={X.shape}")

    # Beamforming
    print("[INFO] Beamforming source #1 (target=doa1, null=doa2)...")
    Y1 = beamform_lcmv(X, freqs, mic_pos, doa_target=doa1, doa_interf=doa2, c=args.c)

    print("[INFO] Beamforming source #2 (target=doa2, null=doa1)...")
    Y2 = beamform_lcmv(X, freqs, mic_pos, doa_target=doa2, doa_interf=doa1, c=args.c)

    # iSTFT
    y1 = istft_mono(Y1, fs, n_fft, hop)
    y2 = istft_mono(Y2, fs, n_fft, hop)

    # Normalize & save
    y1 = normalize_audio(y1)
    y2 = normalize_audio(y2)

    out1 = os.path.join(args.out_dir, "source1.wav")
    out2 = os.path.join(args.out_dir, "source2.wav")
    sf.write(out1, y1.astype(np.float32), fs)
    sf.write(out2, y2.astype(np.float32), fs)

    print(f"[DONE] Wrote: {out1}")
    print(f"[DONE] Wrote: {out2}")


if __name__ == "__main__":
    main()
