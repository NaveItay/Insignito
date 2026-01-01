from __future__ import annotations
from typing import Tuple
import numpy as np
import soundfile as sf
import yaml


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
