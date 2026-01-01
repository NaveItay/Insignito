#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import soundfile as sf

from utils.io_audio import load_geometry_yaml, load_multichannel_wav
from utils.mic_check import detect_bad_mics
from utils.stft import stft_multichannel, istft_mono
from utils.steering import DOA
from utils.beamforming import beamform_lcmv
from utils.audio_utils import normalize_audio

SPEED_OF_SOUND = 343.0


def main() -> None:
    parser = argparse.ArgumentParser(description="50ch Beamforming (LCMV) - Insignito Assignment (NO SciPy)")
    parser.add_argument("--input_wav", default=os.path.join("INPUT", "recording.wav"))
    parser.add_argument("--geometry_yaml", default=os.path.join("INPUT", "array_geometry.yaml"))
    parser.add_argument("--out_dir", default="OUTPUT")
    parser.add_argument("--c", type=float, default=SPEED_OF_SOUND)
    parser.add_argument("--n_fft", type=int, default=0)
    parser.add_argument("--hop", type=int, default=0)
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

    print("[STEP 4] Detect and exclude bad microphones...")
    good_mask = detect_bad_mics(x, print_report=True, print_all=False)
    bad_idx = (good_mask == 0).nonzero()[0].tolist()
    if bad_idx:
        print(f"        Excluding bad mics: {bad_idx}")

    x = x[:, good_mask]
    mic_pos = mic_pos[good_mask, :]

    print("[STEP 5] Choose STFT parameters...")
    n_fft = args.n_fft if args.n_fft > 0 else (2048 if fs >= 32000 else 1024)
    hop = args.hop if args.hop > 0 else (n_fft // 4)
    print(f"        STFT params: n_fft={n_fft}, hop={hop}")

    print("[STEP 6] Define DOAs...")
    doa1 = DOA(azimuth=-0.069, elevation=0.0)
    doa2 = DOA(azimuth=1.029, elevation=0.017)
    print(f"        DOA1: az={doa1.azimuth} rad, el={doa1.elevation} rad")
    print(f"        DOA2: az={doa2.azimuth} rad, el={doa2.elevation} rad")

    print("[STEP 7] Compute STFT...")
    freqs, times, X = stft_multichannel(x, fs, n_fft, hop)
    print(f"        STFT shapes: freqs={freqs.shape}, times={times.shape}, X={X.shape} (F,T,M)")

    print("[STEP 8] Beamform source #1 (target=DOA1, null=DOA2)...")
    Y1 = beamform_lcmv(X, freqs, mic_pos, doa_target=doa1, doa_interf=doa2, c=args.c, verbose=True)

    print("[STEP 9] Beamform source #2 (target=DOA2, null=DOA1)...")
    Y2 = beamform_lcmv(X, freqs, mic_pos, doa_target=doa2, doa_interf=doa1, c=args.c, verbose=True)

    print("[STEP 10] Inverse STFT...")
    y1 = istft_mono(Y1, fs, n_fft, hop, length=N)
    y2 = istft_mono(Y2, fs, n_fft, hop, length=N)

    print("[STEP 11] Normalize + save...")
    y1 = normalize_audio(y1)
    y2 = normalize_audio(y2)

    out1 = os.path.join(args.out_dir, "source1.wav")
    out2 = os.path.join(args.out_dir, "source2.wav")
    sf.write(out1, y1.astype("float32"), fs)
    sf.write(out2, y2.astype("float32"), fs)

    print(f"[DONE] Wrote: {out1}")
    print(f"[DONE] Wrote: {out2}\n")


if __name__ == "__main__":
    main()
