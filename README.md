# Insignito – Beamforming Assignment (LCMV / MVDR)

![Microphone array geometry (3D)](images/mic_array_3d.png)

This repository contains a complete solution for the **Beamforming** task.

We are given a **50-channel microphone array** recording that contains a mixture of **two acoustic sources** with known directions of arrival (DOA).
The goal is to design a beamformer that enhances each source while suppressing the other (and background noise).

---

## Approach (Short)

This solution uses **frequency-domain LCMV beamforming** (MVDR with linear constraints):

- **Output 1**: distortionless response to **Source #1** + **null** towards **Source #2**
- **Output 2**: distortionless response to **Source #2** + **null** towards **Source #1**

To be robust on real recordings, it performs **basic microphone quality checks** (dead / noisy / clipped channels) and excludes problematic microphones before beamforming.

---

## Repository Structure

```
Insignito/
  main.py
  requirements.txt
  README.md
  images/
    mic_array_3d.png
  INPUT/
    recording.wav
    array_geometry.yaml
    (optional) recording_mono.wav
  OUTPUT/
    source1.wav
    source2.wav
    sound_overlay.png
  tools/
    __init__.py
    io_audio.py
    mic_check.py
    stft.py
    steering.py
    beamforming.py
    audio_utils.py
  scripts/
    mono_converter.py
    plot_geometry.py
    overlay_sources_on_image.py
```

---

## Input Data

Located under `INPUT/`:

- `recording.wav` – multichannel WAV (**50 synchronized channels**)
- `array_geometry.yaml` – microphone positions (`array_geometry`) and (if provided) camera parameters for the overlay script

---

## Outputs

The main script generates (under `OUTPUT/`):

- `OUTPUT/source1.wav` – beamformed mono signal toward **DOA #1**
- `OUTPUT/source2.wav` – beamformed mono signal toward **DOA #2**

Optional script outputs:

- `OUTPUT/sound_overlay.png` – DOA projection on a blank camera canvas

---

## Requirements

This repo intentionally avoids **SciPy** and uses only NumPy-based STFT/ISTFT.

Install dependencies from `requirements.txt`:

```powershell
pip install -r .\requirements.txt
```

If you want to run the overlay script, you may also need:

```powershell
pip install opencv-python
```

---

## One-Command Runs (copy/paste)

> Run all commands from the **project root** (the folder containing `main.py`).

### 1) Run the beamforming pipeline (`main.py`)

```powershell
python .\main.py --input_wav .\INPUT\recording.wav --geometry_yaml .\INPUT\array_geometry.yaml --out_dir .\OUTPUT --n_fft 2048 --hop 512 --c 343
```

### 2) Create a mono preview (`scripts/mono_converter.py`)

```powershell
python .\scripts\mono_converter.py --input_wav .\INPUT\recording.wav --output_wav .\INPUT\recording_mono.wav
```

### 3) Overlay DOAs on a blank camera canvas (`scripts/overlay_sources_on_image.py`)

```powershell
python .\scripts\overlay_sources_on_image.py --yaml .\INPUT\array_geometry.yaml --blank --invert_extrinsics --out .\OUTPUT\sound_overlay.png
```

### 4) Plot microphone array geometry (`scripts/plot_geometry.py`)

```powershell
python .\scripts\plot_geometry.py --yaml .\INPUT\array_geometry.yaml
```

---

## Pipeline Details (Aligned to Script Output)

When you run `main.py`, it prints numbered steps like `[STEP 1] ...` through `[STEP 11] ...`.
Below is what each step does.

### STEP 1) Load geometry YAML
Reads `array_geometry.yaml` and loads **M microphone positions** (shape `(M,3)` in meters).  
This geometry is required to compute **relative propagation delays** between microphones for a given DOA.

### STEP 2) Load multichannel WAV
Loads `recording.wav` as an array of shape `(N, M)` and sample rate `fs`.  
A small cleanup is applied: **DC removal per channel** (subtract mean), because offsets can leak into low-frequency bins and harm covariance estimates.

### STEP 3) Align audio channels with geometry (if mismatch)
If the WAV has a different number of channels than the geometry file has microphones, the script uses the first `min(M_wav, M_geom)` channels/mics so both arrays match.

### STEP 4) Detect and exclude bad microphones (dead/noisy/clipped)
Computes a quick health check per channel:
- **dead**: RMS too low (channel is basically silent / disconnected)
- **noisy**: RMS too high (channel dominated by strong noise)
- **clipped**: too many samples near ±1.0 (ADC saturation)

Bad microphones are excluded from both `x` and `mic_pos`, improving stability and output quality.

### STEP 5) Choose STFT parameters (`n_fft` / `hop`)
Chooses window size and hop:
- `n_fft` controls **frequency resolution**: `Δf = fs / n_fft`
- `hop` controls **time resolution** and overlap (default: hop = n_fft/4 → 75% overlap)

### STEP 6) Define DOAs
Defines the two known DOAs (azimuth/elevation in radians):
- DOA1: az = -0.069, el = 0.0
- DOA2: az =  1.029, el = 0.017

### STEP 7) Compute STFT for all microphones (NO SciPy)
Computes STFT for each microphone channel using Hann window + overlap framing + one-sided rFFT.  
Produces `X` with shape `(F, T, M)`.

### STEP 8) Beamform source #1 (target=DOA1, null=DOA2)
Per frequency bin:
1. Estimate covariance `R(f)` from `X`.
2. Build constraints `C = [a_target, a_interferer]`.
3. Solve LCMV weights `w(f)` s.t. `wᴴ a_target = 1` and `wᴴ a_interferer = 0`.
Then apply weights to get output STFT `Y1`.

### STEP 9) Beamform source #2 (target=DOA2, null=DOA1)
Same as STEP 8, but swap target/interferer constraints to get output STFT `Y2`.

### STEP 10) Inverse STFT to time domain (overlap-add)
IFFT per frame + overlap-add reconstruction back to time domain, trimmed to the original length `N`.

### STEP 11) Normalize outputs and save WAV files
Normalize each output to a fixed peak (0.99) and write:
- `OUTPUT/source1.wav`
- `OUTPUT/source2.wav`

---

## Implementation Notes

- Uses **diagonal loading (regularization)** of the covariance matrix to improve numerical stability in near-singular conditions.
- Automatically chooses the **delay sign convention** by comparing delay-and-sum output energy for both sign options (helps avoid DOA convention mismatches).
- Includes a **microphone quality check** (dead/noisy/clipped) to handle real-world recordings more robustly.
