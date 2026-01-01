import soundfile as sf
import numpy as np

in_wav = "recording.wav"
x, fs = sf.read(in_wav, always_2d=True)   # shape: (N, C)
print("fs:", fs, "shape:", x.shape, "dtype:", x.dtype)
print("min/max:", x.min(), x.max())

mono = np.mean(x, axis=1)
mono /= (np.max(np.abs(mono)) + 1e-12)   # normalize
sf.write("preview_mono.wav", mono, fs)
print("Wrote preview_mono.wav")
