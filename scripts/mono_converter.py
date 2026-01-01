#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert multichannel WAV to mono preview WAV")
    parser.add_argument("--input_wav", required=True, help="Path to input multichannel WAV")
    parser.add_argument("--output_wav", required=True, help="Path to output mono WAV")
    parser.add_argument("--peak", type=float, default=0.99, help="Normalize output peak to this value")
    args = parser.parse_args()

    x, fs = sf.read(args.input_wav, always_2d=True)  # (N, C)
    x = np.asarray(x, dtype=np.float64)

    mono = np.mean(x, axis=1)

    mx = float(np.max(np.abs(mono)) + 1e-12)
    mono = (args.peak / mx) * mono

    sf.write(args.output_wav, mono.astype(np.float32), fs)
    print(f"fs: {fs}, shape: {x.shape}, wrote: {args.output_wav}")


if __name__ == "__main__":
    main()
