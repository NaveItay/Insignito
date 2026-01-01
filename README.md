# Insignito – Beamforming Assignment

![Microphone array geometry (3D)](images/mic_array_3d.png)

![Sound sources overlay (blank camera canvas)](outputs/sound_overlay.png)


This repository contains a solution for the **Beamforming** task.

We are given a **50-channel microphone array** recording that contains a mixture of **two acoustic sources** with known directions of arrival (DOA).  
The goal is to design a beamformer that enhances each source while suppressing the other and additional noise.

---

## Approach (Short)

This solution uses **frequency-domain LCMV beamforming** (a constrained MVDR beamformer):

- **Output 1**: distortionless response to **Source #1**, and a **null** towards **Source #2**
- **Output 2**: distortionless response to **Source #2**, and a **null** towards **Source #1**

It also handles real-world issues by performing **basic microphone quality checks** (dead/noisy/clipped channels) and excluding problematic microphones before beamforming.

---

## Repository Structure

```
Insignito/
  main.py
  README.md
  outputs/
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

- `recording.wav` – Multichannel WAV (50 synchronized channels)
- `array_geometry.yaml` – 3D positions of all microphones (`array_geometry`)
- Optional utilities:
  - `recording_mono.wav` – A mono preview for listening
  - `mono_converter.py` – Mono converter helper
  - `plot_geometry.py` – Microphone array geometry visualization

---

## Outputs

The script generates (under `outputs/`):

- `outputs/source1.wav`
- `outputs/source2.wav`

Each output is a **mono WAV** beamformed toward the corresponding DOA.

---

## Requirements

Python packages:

- `numpy`
- `scipy`
- `soundfile`
- `pyyaml`

Install:

```powershell
pip install numpy scipy soundfile pyyaml
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

## Method – Step by Step (High Level)

### 1) Sanity check: listen to a mono preview
Many audio players cannot play 50-channel WAV files.  
A quick check is to downmix to mono (e.g., mean over channels) and listen to confirm the recording contains the two sources.

### 2) Verify microphone geometry visually
Plot the microphone positions from `array_geometry.yaml` to ensure:
- There are 50 microphones
- Units look like meters
- The shape is physically reasonable  
(In this dataset all x≈0, so the array lies in the YZ plane.)

![Microphone array geometry (3D)](images/mic_array_3d.png)

### 3) Load data & basic cleanup
- Load the 50-channel WAV as `(N, M)`
- Remove DC per channel (subtract mean)

### 4) Detect and exclude bad microphones (real-world robustness)
The code checks each channel for:
- Very low RMS (dead mic)
- Extremely high RMS (very noisy mic)
- Excessive clipping  
Bad channels are removed from both the audio matrix and geometry before beamforming.

### 5) Convert DOA (azimuth/elevation) to a 3D unit vector
Using the assignment convention:
- x = cos(-az) cos(-el)
- y = sin(-az) cos(-el)
- z = sin(-el)

### 6) Compute far-field delays and steering vectors
Assuming a plane wave and speed of sound c:
- delay per mic: tau[m] ∝ (p[m]·u)/c
- steering vector per frequency: a[m](f) = exp(-j2πf tau[m])

### 7) STFT of all microphones
Compute STFT per channel to work in the frequency domain:
- X(f, t, m)

### 8) LCMV beamforming per frequency
For each frequency bin:
- Estimate spatial covariance R(f) from X
- Build constraint matrix C = [a_target, a_interferer]
- Solve LCMV weights so:
  - w^H a_target = 1  (pass target)
  - w^H a_interferer = 0 (null the other source)

### 9) Inverse STFT and save outputs
Apply weights to produce Y(f,t), inverse STFT to time domain, normalize, and write:
- `outputs/source1.wav`
- `outputs/source2.wav`

---

## Bonus – DOA overlay on camera canvas (no camera frame needed)

This repository also includes a small bonus script that projects the **DOA directions** onto a **blank camera canvas** using the camera intrinsics/extrinsics in `array_geometry.yaml`.

Run:

```powershell
python .\overlay_sources_on_image.py --yaml utils\array_geometry.yaml --blank --invert_extrinsics --out outputs\sound_overlay.png
```

Result image:

![Sound sources overlay (blank camera canvas)](outputs/sound_overlay.png)

---

## Notes

- The solution assumes a **far-field plane-wave** model and uses the assignment-provided **azimuth/elevation → direction** conversion.
- Since this is a real recording, some microphones may be degraded. The code detects and excludes obvious bad channels (**dead/noisy/clipped**).
- A small **diagonal loading** term is used for numerical stability when inverting covariance matrices.