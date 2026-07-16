# octsim — a physically-modelled OCT simulation platform

Generate synthetic OCT data — single B-scans or **volumes** of retina (macula,
optic-nerve head, periphery), cornea, or bench targets (mirror, paper) — from a
physical forward model, **per spectrum / per A-scan**. Every sample carries the raw
interferometric spectra, the reconstructed B-scan, and a free per-pixel layer/feature
label map. Built for training deep models in sparse-data regimes: one call yields **N
independent random samples** defined by physics + acquisition parameters.

There is a **programmatic API** and a **PySide6 GUI**.

```python
from octsim import SimConfig, simulate, generate_dataset

cfg = SimConfig(
    anatomy="macula", scan_pattern="volume", n_frames=32, slow_fov_um=6000.0,
    axial_resolution_um=5.0, lateral_fov_um=6000.0, lines_per_frame=512,
    line_rate_hz=100_000, motion_level=1.0, falloff_6db_um=1500.0, noise_level=1.0,
)
sample = simulate(cfg, seed=1)                 # one B-scan or volume
sample.volume_db          # [n_frames, depth, x] display stack (dB)
sample.label_map          # [n_frames, depth, x] layer index (free ground truth)

for s in generate_dataset(cfg, n=200, seed=0): # N random samples, in-memory
    train_on(s.bscan_db, s.labels2d)
```

```bash
octsim                       # launch the GUI (needs the [gui] extra)
octsim --generate 5 --anatomy macula --pattern volume --frames 16   # headless
octsim --selftest            # headless smoke test, exit 0/1
```

## The physical model

Per A-scan: a scene → a complex reflectivity field → the interferometric spectrum
`I(k) = P(k)·|a_ref + Σ r·exp(i·2k·z)|²` → DC-removal + window + IFFT → B-scan. On top
of the ideal fringe the forward model folds in the instrument/acquisition physics:

- **Source & resolution** — Gaussian source sampled linearly in `k`; wavelength +
  bandwidth set the axial resolution.
- **Sensitivity fall-off** — depth-dependent roll-off (`sinc × Gaussian`), one 6 dB-depth knob.
- **Dispersion** — a single intuitive knob: `0` = sharp, magnitude = axial stretch, sign = chirp direction.
- **Acquisition speed** — line/sweep-rate → integration time `T_int`, which drives both the
  collected signal (shot-limited SNR ∝ √T_int) **and** fringe washout — the real OCT trade-off, emergent.
- **Motion & artifacts** — one continuous `Δz(t)` trajectory (drift + cardiac/breathing +
  microsaccades) sampled along the acquisition time axis, so bulk motion, phase drift, and
  washout are **correlated across A-scans and across the frames of a volume**.
- **Detector noise** — shot (∝√I) + RIN (∝I) + read (floor), tunable clean → degraded, with a
  noise-free toggle and a target-SNR (dB) knob.

First-Born / single-scattering: every scatterer contributes once (multiple scattering not modelled).

## Resolvable parameters

Physically-linked quantities may be specified from *either* side; the config back-solves the
rest (a `0` on a linked field means "derive me") and reports both:

| specify… | …or its physical target |
| --- | --- |
| `bandwidth_fwhm_nm` | `axial_resolution_um` |
| `k_span_factor` | `axial_range_um` / `axial_sampling_um` (with `pixels_per_spectrum`) |
| `beam_waist_um` | `lateral_resolution_um` |

Scan geometry (fast axis `lateral_fov_um`/`lines_per_frame`, slow axis `slow_fov_um`/`n_frames`),
speed (`line_rate_hz`, `duty_cycle`, `flyback_time_s`), and the adjustable sample position
(`sample_depth_um`, `sample_lateral_offset_um`) round out the acquisition.

## Anatomy

Retina is modelled in 3D (enface-parameterized) with **reasonable physiological thicknesses**
(total neurosensory retina ≈ 270 µm) and the characteristic OCT layer pattern, so a volume sweeps
a consistent scene:

- **Foveal pit** — inner layers (NFL…OPL) thin toward the centre while the photoreceptor/RPE
  complex stays anchored flat; the ONL bulges centrally.
- **Optic nerve head** — the RPE/photoreceptor bands **terminate at Bruch's-membrane opening**,
  a central cup, RNFL thickening toward the disc, and a neural-canal fill (informed by clinical ONH OCT).
- **Blood vessels** — vertically-oval hyporeflective lumen + hyperreflective wall in the inner
  retina, with a posterior shadow.
- **Choroid, vitreous, curvature.** Optical properties are illustrative and **calibratable** —
  there is no validated per-layer retinal optical-property table; tune against real scans.

`generate_dataset` draws a fresh random anatomy (layer thicknesses, pit/cup, vessels, position) +
speckle + motion per sample, reproducible via `SeedSequence`.

## Package map

```
octsim/
  config.py        SimConfig (resolvable) + resolve() -> ResolvedConfig
  source.py        Gaussian source / k-grid / axial calibration
  optics.py        sensitivity fall-off + beam helpers
  dispersion.py    dispersion phase + compensation
  acquisition.py   ScanGeometry + Timing (speed -> T_int, timestamps)
  motion.py        continuous Δz(t) trajectory; bulk motion / drift / washout
  sample.py        reflectivity core (single-draw speckle, per-column attenuation, features)
  features.py      BloodVessel (+ shadow)
  anatomy/retina.py  RetinaModel (layers, foveal pit, ONH, vessels), random/default builders
  forward.py       I(k) with fall-off + dispersion + signal-scale + motion/washout
  reconstruct.py   IFFT reconstruction (+ optional dispersion compensation)
  noise.py         shot / RIN / read
  generate.py      simulate() (B-scan/volume) + generate_dataset() (N samples)
  presets.py       mirror / paper / cornea / macula (Sample)
  gui/             PySide6 app (launcher, widgets) + CLI (app.py)
```

## Install & test

```bash
pip install -e .            # core (numpy, scipy)
pip install -e .[gui]       # + PySide6, pyqtgraph, matplotlib for the GUI
python tests/test_phase_a.py  # ... phase_b, phase_c, phase_d, test_octsim
QT_QPA_PLATFORM=offscreen python tests/test_gui_smoke.py
```

The physics/anatomy/generation suites (33 checks) run in a couple of seconds and pin the
resolvable config, timing, fall-off, dispersion, speed↔SNR trade, motion correlation, foveal pit,
vessel shadow, ONH termination, and dataset reproducibility.

## Roadmap

- Calibrate retinal layer optical properties against real macular OCT (biggest realism lever).
- Cornea anatomy (endothelial mosaic, iris, lens) on the same feature framework.
- Disk/raw-`uint16`/`.CLB` writers to feed the OCT-Denoiser / Reconstruction / Super-Resolution pipelines.
- Optional GPU/torch forward path for very large corpora.
