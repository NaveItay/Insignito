# Insignito – Beamforming Assignment

![Microphone array geometry (3D)](images/mic_array_3d.png)

![Sound sources overlay (blank camera canvas)](outputs/sound_overlay.png)

This repository contains a solution for the **Beamforming** task.

We are given a **50-channel microphone array** recording that contains a mixture of **two acoustic sources** with known directions of arrival (DOA).  
The goal is to design a beamformer that enhances each source while suppressing the other (and noise).

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
  overlay_sources_on_image.py
  README.md
  outputs/
    source1.wav
    source2.wav
    sound_overlay.png
  utils/
    recording.wav
    array_geometry.yaml
    mono_converter.py
    plot_geometry.py
    recording_mono.wav
  images/
    mic_array_3d.png
```

---

## Input Data

Located under `utils/`:

- `recording.wav` – Multichannel WAV (**50 synchronized channels**)
- `array_geometry.yaml` – Microphone positions (`array_geometry`) + camera parameters (for the bonus overlay)
- Optional utilities:
  - `recording_mono.wav` – Mono preview for listening
  - `mono_converter.py` – Mono converter helper
  - `plot_geometry.py` – Microphone array geometry visualization

---

## Outputs

The main script generates (under `outputs/`):

- `outputs/source1.wav` – Beamformed mono signal toward **DOA #1**
- `outputs/source2.wav` – Beamformed mono signal toward **DOA #2**

Bonus script output:

- `outputs/sound_overlay.png` – DOA projection on a blank camera canvas

---

## Requirements

### Main beamforming script (`main.py`) – **NO SciPy**
Python packages:

- `numpy`
- `soundfile`
- `pyyaml`

Install:

```powershell
pip install numpy soundfile pyyaml
```

> Note: SciPy is intentionally not used to avoid NumPy/SciPy binary compatibility issues on some environments.

### Bonus overlay script (`overlay_sources_on_image.py`)
Additional packages:

- `opencv-python`

Install:

```powershell
pip install opencv-python
```

---

## How to Run

From the project root:

```powershell
python .\main.py --input_wav .\utils\recording.wav --geometry_yaml .\utils\array_geometry.yaml --out_dir .\outputs
```

After running, you should have:

- `outputs\source1.wav`
- `outputs\source2.wav`

---

## Optional Arguments

### Control STFT parameters

```powershell
python .\main.py --input_wav .\utils\recording.wav --geometry_yaml .\utils\array_geometry.yaml --out_dir .\outputs --n_fft 2048 --hop 512
```

### Change speed of sound

```powershell
python .\main.py --input_wav .\utils\recording.wav --geometry_yaml .\utils\array_geometry.yaml --out_dir .\outputs --c 343
```

---

## Run from inside `utils/` (Alternative)

If you are inside the `utils` directory:

```powershell
cd .\utils
python ..\main.py --input_wav .\recording.wav --geometry_yaml .\array_geometry.yaml --out_dir ..\outputs
```

---

## Pipeline – Step by Step (Aligned to the Code Output)

When you run the script, it prints numbered steps like:

- `[STEP 1] ...`
- `[STEP 2] ...`
- ...
- `[STEP 11] ...`

Below is what each step means and why it exists.

### STEP 1) Load geometry YAML
Reads `array_geometry.yaml` and loads **M microphone positions** (shape `(M,3)` in meters).  
This geometry is required to compute **relative propagation delays** between microphones for a given direction-of-arrival (DOA).

### STEP 2) Load multichannel WAV
Loads `recording.wav` as an array of shape `(N, M)` and sample rate `fs`.  
A small cleanup is applied: **DC removal per channel** (subtract mean), because offsets can leak into low-frequency bins and harm covariance estimates.

### STEP 3) Align audio channels with geometry (if mismatch)
If the WAV has a different number of channels than the geometry file has microphones, the script uses the first `min(M_wav, M_geom)` channels/mics so both arrays match.

### STEP 4) Detect and exclude bad microphones (dead/noisy/clipped)
Computes a quick health check per channel:
- **dead**: RMS too low
- **noisy**: RMS too high
- **clipped**: too many samples near ±1.0

Bad microphones are excluded from:
- the audio matrix `x`
- the geometry array `mic_pos`

This improves stability and usually improves beamforming quality.

### STEP 5) Choose STFT parameters (`n_fft` / `hop`)
Chooses window size and hop:
- `n_fft` controls **frequency resolution**
- `hop` controls **time resolution** and overlap (typically hop = n_fft/4 → 75% overlap)

You can override both from the command line.

### STEP 6) Define DOAs (given by assignment)
Defines the two known DOAs (azimuth/elevation in radians):
- DOA1: az = -0.069, el = 0.0
- DOA2: az =  1.029, el = 0.017

These are converted to 3D unit direction vectors using the assignment convention.

### STEP 7) Compute STFT for all microphones (**NO SciPy**)
Computes STFT for each microphone channel using:
- Hann window
- overlap-add framing
- FFT per frame

Produces a tensor:
- `X` with shape `(F, T, M)` (frequency bins × time frames × microphones)

### STEP 8) Beamform source #1 (target=DOA1, null=DOA2)
For each frequency bin:
1. Estimate spatial covariance `R(f)` from the multichannel STFT.
2. Build constraint matrix `C = [a_target, a_interferer]`
   where `a(f)` are steering vectors.
3. Solve **LCMV weights** `w(f)` such that:
   - `wᴴ a_target = 1`  (pass target without distortion)
   - `wᴴ a_interferer = 0` (null the other source)

Apply `w(f)` to obtain output STFT `Y1(f,t)`.

### STEP 9) Beamform source #2 (target=DOA2, null=DOA1)
Same as STEP 8, but swap target/interferer constraints to produce `Y2(f,t)`.

### STEP 10) Inverse STFT to time domain (overlap-add)
For each output STFT (`Y1` and `Y2`):
- perform IFFT per frame
- overlap-add frames back into a time-domain mono waveform

### STEP 11) Normalize outputs and save WAV files
Normalizes each output to a fixed peak (e.g., 0.99) and writes:
- `outputs/source1.wav`
- `outputs/source2.wav`

---

## Bonus – DOA overlay on camera canvas (no camera frame needed)

This repository includes a small bonus script that projects the **DOA directions** onto a **blank camera canvas** using the camera intrinsics/extrinsics in `array_geometry.yaml`.

Run:

```powershell
python .\overlay_sources_on_image.py --yaml utils\array_geometry.yaml --blank --invert_extrinsics --out outputs\sound_overlay.png
```

Result image:

![Sound sources overlay (blank camera canvas)](outputs/sound_overlay.png)

---

## Notes

- The solution assumes a **far-field plane-wave** model and uses the assignment-provided **azimuth/elevation → direction** conversion.
- Real recordings often contain degraded channels; excluding **dead/noisy/clipped** mics improves robustness.
- A small **diagonal loading** term is used when solving the beamformer weights for numerical stability.
