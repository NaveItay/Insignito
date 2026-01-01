#!/usr/bin/env python3
"""
Insignito Beamforming Assignment Solution (NO SciPy)
----------------------------------------------------
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
- soundfile
- pyyaml

How to run
----------
1) Install dependencies:
   pip install numpy soundfile pyyaml

2) From the project root (recommended):
   python .\\main.py --input_wav .\\utils\\recording.wav --geometry_yaml .\\utils\\array_geometry.yaml --out_dir .\\outputs

3) Optional parameters:
   - Choose STFT resolution:
     python .\\main.py --input_wav .\\utils\\recording.wav --geometry_yaml .\\utils\\array_geometry.yaml --out_dir .\\outputs --n_fft 2048 --hop 512

   - Change speed of sound:
     python .\\main.py --input_wav .\\utils\\recording.wav --geometry_yaml .\\utils\\array_geometry.yaml --out_dir .\\outputs --c 343

4) Outputs will be created under:
   outputs\\source1.wav
   outputs\\source2.wav
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import soundfile as sf
import yaml

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
      x: float64 array shape (N, M)
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


# ------------------------- Mic Health / Selection -------------------------

def detect_bad_mics(
    x: np.ndarray,
    clip_threshold: float = 0.999,
    print_report: bool = True,
    print_all: bool = False,
) -> np.ndarray:
    """
    Simple robust mic selection + report.

    Rules:
      - dead:    rms < 0.05 * median_rms
      - noisy:   rms > 20.0 * median_rms
      - clipped: clip_ratio > 1e-3   (more than 0.1% samples saturated)

    Returns:
      good_mask: boolean shape (M,)
    """
    eps = 1e-12

    rms = np.sqrt(np.mean(x ** 2, axis=0) + eps)          # (M,)
    med = float(np.median(rms))

    clip_ratio = np.mean(np.abs(x) >= clip_threshold, axis=0)  # (M,)

    dead = rms < (0.05 * med)
    noisy = rms > (20.0 * med)
    clipped = clip_ratio > 1e-3
    not_finite = ~np.isfinite(rms)

    bad = dead | noisy | clipped | not_finite
    good = ~bad

    if print_report:
        M = x.shape[1]
        print("[MIC CHECK] thresholds:")
        print(f"  median_rms = {med:.6e}")
        print(f"  dead   if rms < {(0.05 * med):.6e}  (0.05 * median)")
        print(f"  noisy  if rms > {(20.0 * med):.6e}  (20 * median)")
        print(f"  clipped if clip_ratio > 1.0e-3  (0.1%) with |x| >= {clip_threshold}")
        print("")

        idxs = list(range(M)) if print_all else np.flatnonzero(bad).tolist()

        if len(idxs) == 0:
            print("[MIC CHECK] No bad microphones detected.")
        else:
            print("[MIC CHECK] Report:")
            for i in idxs:
                reasons = []
                if dead[i]:
                    reasons.append("dead (low RMS)")
                if noisy[i]:
                    reasons.append("noisy (high RMS)")
                if clipped[i]:
                    reasons.append("clipped")
                if not_finite[i]:
                    reasons.append("non-finite RMS")

                ratio = float(rms[i] / (med + eps))
                status = "BAD" if bad[i] else "OK "
                reason_str = ", ".join(reasons) if reasons else "—"

                print(
                    f"  ch {i:02d} | {status} | rms={rms[i]:.6e} (x{ratio:.2f} of med) "
                    f"| clip_ratio={clip_ratio[i]:.6e} | {reason_str}"
                )

        print("")
        bad_list = np.flatnonzero(bad).tolist()
        print(f"[MIC CHECK] bad mics: {bad_list}")
        print(f"[MIC CHECK] good mics: {int(np.sum(good))} / {M}")

    return good


# ------------------------- STFT / ISTFT (NO SciPy) -------------------------

def _stft_onesided(
    x: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal STFT (onesided rFFT) for a 1D signal.
    Returns:
      Z: (F, T) complex
      times: (T,) seconds (frame centers)
    """
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

    # times are not essential for processing; used for debug/shape printouts
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
    """
    Minimal inverse STFT (overlap-add) matching _stft_onesided.
    Z shape: (F, T)
    Returns y trimmed to 'length' samples (original length).
    """
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
    """
    Compute STFT for each channel (NO SciPy).
    Returns:
      freqs: (F,)
      times: (T,)
      X: (F, T, M)
    """
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
    """Inverse STFT for mono signal. Y shape: (F, T)."""
    window = np.hanning(n_fft).astype(np.float64)
    return _istft_onesided(Y, fs, n_fft, hop, window, length=length, center=True)


# ------------------------- Beamforming Core -------------------------

def choose_delay_sign_by_ds_energy(
    X: np.ndarray,
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    doa: DOA,
    c: float
) -> float:
    """
    DOA sign convention can flip depending on coordinate definition.
    We test both signs using a delay-and-sum score and pick the one with higher output power.
    """
    u = doa_to_unit_vector(doa)

    def ds_score(sign: float) -> float:
        tau = compute_delays_seconds(mic_pos, u, c, sign)
        a = steering_vectors(freqs, tau)  # (F,M)
        M = a.shape[1]
        power = 0.0
        for fi in range(X.shape[0]):
            w = a[fi, :] / M
            y = X[fi, :, :] @ np.conj(w)
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
    """
    M = R.shape[0]
    trace = np.trace(R).real
    dl = 1e-3 * (trace / M + 1e-12)
    R_reg = R + dl * np.eye(M, dtype=R.dtype)

    try:
        RinvC = np.linalg.solve(R_reg, C)
    except np.linalg.LinAlgError:
        RinvC = np.linalg.pinv(R_reg) @ C

    G = C.conj().T @ RinvC

    try:
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Ginv = np.linalg.pinv(G)

    w = RinvC @ (Ginv @ f_vec)
    return w


def beamform_lcmv(
    X: np.ndarray,         # (F,T,M)
    freqs: np.ndarray,     # (F,)
    mic_pos: np.ndarray,   # (M,3)
    doa_target: DOA,
    doa_interf: DOA,
    c: float,
    verbose: bool = False
) -> np.ndarray:
    """
    LCMV beamforming per-frequency:
      - pass target with gain 1
      - null interferer with gain 0

    Returns:
      Y: (F,T) complex STFT for the target output
    """
    sign = choose_delay_sign_by_ds_energy(X, freqs, mic_pos, doa_target, c)
    if verbose:
        print(f"        [INFO] Using delay sign = {sign:+.0f} for target DOA")

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
        Xf = X[fi, :, :]  # (T,M)
        R = (Xf.conj().T @ Xf) / max(T, 1)

        C = np.stack([a_t[fi, :], a_i[fi, :]], axis=1)  # (M,2)
        w = lcmv_weights_per_freq(R, C, f_vec)          # (M,)

        Y[fi, :] = Xf @ np.conj(w)

    return Y


# ------------------------- Utilities -------------------------

def normalize_audio(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mx = float(np.max(np.abs(y)) + 1e-12)
    return (peak / mx) * y


# ------------------------- Main -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="50ch Beamforming (LCMV) - Insignito Assignment (NO SciPy)")
    parser.add_argument("--input_wav", default="recording.wav", help="Path to multichannel wav (50ch)")
    parser.add_argument("--geometry_yaml", default="array_geometry.yaml", help="Path to geometry YAML")
    parser.add_argument("--out_dir", default="outputs", help="Output directory")
    parser.add_argument("--c", type=float, default=SPEED_OF_SOUND, help="Speed of sound (m/s)")
    parser.add_argument("--n_fft", type=int, default=0, help="FFT size (0=auto)")
    parser.add_argument("--hop", type=int, default=0, help="Hop size (0=auto)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[STEP 1] Load geometry YAML...")
    mic_pos = load_geometry_yaml(args.geometry_yaml)
    print(f"        Geometry loaded: {mic_pos.shape[0]} mics")

    print("[STEP 2] Load multichannel WAV...")
    x, fs = load_multichannel_wav(args.input_wav)
    N, M_audio = x.shape
    print(f"        Audio loaded: N={N} samples, M={M_audio} channels, fs={fs} Hz")

    print("[STEP 3] Align audio channels with geometry (if mismatch)...")
    if x.shape[1] != mic_pos.shape[0]:
        M = min(x.shape[1], mic_pos.shape[0])
        print(f"        [WARN] Mismatch: WAV has {x.shape[1]}ch, geometry has {mic_pos.shape[0]} mics.")
        print(f"        [WARN] Using first {M} channels/mics.")
        x = x[:, :M]
        mic_pos = mic_pos[:M, :]
    print(f"        Using M={x.shape[1]} channels/mics")

    print("[STEP 4] Detect and exclude bad microphones (dead/noisy/clipped)...")
    good_mask = detect_bad_mics(x, print_report=True, print_all=False)
    good_idx = np.flatnonzero(good_mask).tolist()
    bad_idx = np.flatnonzero(~good_mask).tolist()
    print(f"        Good mics: {len(good_idx)} / {x.shape[1]}")
    if bad_idx:
        print(f"        Excluding bad mics: {bad_idx}")

    x = x[:, good_mask]
    mic_pos = mic_pos[good_mask, :]

    print("[STEP 5] Choose STFT parameters (n_fft / hop)...")
    n_fft = args.n_fft if args.n_fft > 0 else (2048 if fs >= 32000 else 1024)
    hop = args.hop if args.hop > 0 else (n_fft // 4)  # 75% overlap
    print(f"        STFT params: n_fft={n_fft}, hop={hop}")

    print("[STEP 6] Define DOAs (given by assignment)...")
    doa1 = DOA(azimuth=-0.069, elevation=0.0)
    doa2 = DOA(azimuth=1.029, elevation=0.017)
    print(f"        DOA1: az={doa1.azimuth} rad, el={doa1.elevation} rad")
    print(f"        DOA2: az={doa2.azimuth} rad, el={doa2.elevation} rad")

    print("[STEP 7] Compute STFT for all microphones (NO SciPy)...")
    freqs, times, X = stft_multichannel(x, fs, n_fft, hop)
    print(f"        STFT shapes: freqs={freqs.shape}, times={times.shape}, X={X.shape} (F,T,M)")

    print("[STEP 8] Beamform source #1 (target=DOA1, null=DOA2)...")
    Y1 = beamform_lcmv(X, freqs, mic_pos, doa_target=doa1, doa_interf=doa2, c=args.c, verbose=True)

    print("[STEP 9] Beamform source #2 (target=DOA2, null=DOA1)...")
    Y2 = beamform_lcmv(X, freqs, mic_pos, doa_target=doa2, doa_interf=doa1, c=args.c, verbose=True)

    print("[STEP 10] Inverse STFT to time domain (overlap-add)...")
    y1 = istft_mono(Y1, fs, n_fft, hop, length=N)
    y2 = istft_mono(Y2, fs, n_fft, hop, length=N)

    print("[STEP 11] Normalize outputs and save WAV files...")
    y1 = normalize_audio(y1)
    y2 = normalize_audio(y2)

    out1 = os.path.join(args.out_dir, "source1.wav")
    out2 = os.path.join(args.out_dir, "source2.wav")
    sf.write(out1, y1.astype(np.float32), fs)
    sf.write(out2, y2.astype(np.float32), fs)

    print(f"[DONE] Wrote: {out1}")
    print(f"[DONE] Wrote: {out2}\n")


if __name__ == "__main__":
    main()
