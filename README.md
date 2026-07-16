# octsim — a physically-modelled OCT B-scan synthesizer

Generate synthetic OCT B-scans (mirror, paper, cornea, macula, or any layered
sample you define) from a physical forward model, **per spectrum / per A-scan**.
The simulator emits both the raw interferometric spectrum a swept source would
digitise *and* the reconstructed B-scan, plus a free per-pixel layer label map.

It is deliberately small, dependency-light (numpy; matplotlib only for the
gallery), and fast — the whole four-preset gallery renders in a few seconds, and
the physics test suite runs in under a second.

```python
from octsim import LightSource, simulate_bscan, presets, NoiseModel

res = simulate_bscan(presets.macula(), LightSource(),
                     noise=NoiseModel.typical(), seed=1)
res.bscan_db      # [depth, x] display image (dB, 0 dB peak)
res.spectra       # [n_k, x] raw interferometric spectra (pre-reconstruction)
res.label_map     # [depth, x] uint8 layer index (free segmentation ground truth)
```

## The physical model

The signal chain, per A-scan:

```
layered sample  ->  complex reflectivity field r[z, x]  (specular interfaces
                                                          + speckle cells)
     r[z, x]    ->  I(k) = P(k) · | a_ref + Σ r · exp(i·2·k·z) | ²   (forward)
       I(k)     ->  + shot / RIN / read noise
       I(k)     ->  DC high-pass  ->  Hann window  ->  IFFT  ->  |·|  ->  dB
```

* **Source** (`source.py`) — a Gaussian spectrum sampled linearly in wavenumber
  `k` (SS-OCT convention). Two knobs, `center_wavelength_nm` and
  `bandwidth_fwhm_nm`, set the carrier and the axial resolution; the imaging
  depth and depth-pixel size follow from the k-sampling. Defaults model a ~1050 nm /
  100 nm ophthalmic source (~4.9 µm axial resolution, ~1.9 mm optical depth).
* **Sample** (`sample.py`) — an ordered stack of `Layer`s (thickness, refractive
  index, backscatter, scattering coefficient), optionally curved. It becomes a
  complex reflectivity field on the reconstruction grid:
  * **volume speckle** as a circular complex-Gaussian per resolution cell (the
    central-limit limit of many sub-resolution scatterers → fully-developed,
    Rayleigh-magnitude speckle), scaled by backscatter and a Beer–Lambert
    `exp(-μ_t·d)` roll-off;
  * **specular interfaces** as deterministic Fresnel reflectors at each boundary;
  * a **lateral beam convolution** that sets the lateral resolution and the
    speckle grain size.
* **Forward** (`forward.py`) — the interferometric spectrum. Because `k` is
  linear and the depth grid satisfies `2·k·z = 2π·m·p/n_k`, the coherent sum is
  evaluated **exactly** as one FFT per B-scan (verified to machine precision
  against the direct phasor sum). A direct-sum primitive for arbitrary off-grid
  point scatterers is kept for validation.
* **Noise** (`noise.py`) — the three physical components: shot (`∝√I`), relative
  intensity noise (`∝I`), and read (constant), added in the wavenumber domain.
* **Reconstruct** (`reconstruct.py`) — textbook SD/SS-OCT inverse (spectral
  high-pass DC removal → Hann → IFFT → positive-depth half).

This is a **first-Born / single-scattering** model: every scatterer contributes
once, independently. It reproduces speckle, the coherence-gated axial PSF, the
lateral PSF, index-dependent optical path, and a realistic noise floor. It does
**not** model multiple scattering — the deliberate, corpus-scalable fidelity
tier, which bounds fidelity for deep / turbid / edematous tissue and for deep
speckle statistics.

## Presets

| Preset | What it exercises |
| --- | --- |
| `mirror()` | Single specular reflector → axial PSF / resolution validation |
| `paper()` | Dense turbid slab → fully-developed speckle + Beer–Lambert roll-off |
| `cornea()` | Bright anterior/posterior surfaces + weak stroma, curved |
| `macula()` | Layered retina with the hyper/hypo-reflective NFL…RPE…choroid pattern |

Optical parameters in `presets.py` are **physically motivated but illustrative** —
they are the knobs to calibrate against real scans, not measured constants.

## Usage

```bash
# from this directory
python -m pip install numpy matplotlib          # (scipy not required)

python examples/generate_gallery.py             # render all presets -> examples/output/
python examples/generate_gallery.py macula --n-ascans 512
python tests/test_octsim.py                      # physics checks (or: python -m pytest tests/)
```

Build your own sample:

```python
from octsim import Layer, Sample, LightSource, simulate_bscan

sample = Sample("two-layer", top_n=1.0, surface_um=150.0, layers=[
    Layer("film",   thickness_um=60,  n=1.45, backscatter=2e-2, mu_s_per_mm=8),
    Layer("substrate", thickness_um=400, n=1.38, backscatter=1e-2, mu_s_per_mm=3),
])
res = simulate_bscan(sample, LightSource(bandwidth_fwhm_nm=120), seed=0)
```

## What the tests validate

* a mirror reconstructs at its inserted depth; the axial FWHM matches the
  source-bandwidth coherence limit (× the ~1.2 Hann-window factor);
* more bandwidth → sharper axial PSF (monotonic);
* the FFT forward path equals the direct phasor sum to machine precision;
* paper's in-slab speckle is Rayleigh (row-normalised contrast ≈ 0.52);
* detector noise lifts the deep background;
* the macula shows ordered, sub-surface layer structure.

## Roadmap / natural extensions

* **Calibrate the retinal optical properties** (per-layer n, μ_s, g) against real
  macular OCT — the single biggest realism lever.
* **Foveal pit**: x-dependent inner-layer thinning (the current model uses
  fixed-thickness layers + a gentle contour).
* **Raw-format emitters**: write the spectra as the `uint16` `.raw` layouts and
  `.CLB` files consumed by the OCT-Denoiser / OCT-Reconstruction / OCT-Super-
  Resolution pipelines, and cross-check by feeding them through the Line-Field-OCT
  reconstruction.
* **Pathology**: fluid pockets (low-scatter), drusen (RPE deformation), edema
  (thickening) — each carrying free labels.
* **Higher-fidelity tier**: a Monte-Carlo validation subset to bound the
  single-scattering gap.
