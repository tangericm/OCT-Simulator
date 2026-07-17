# octsim — Data-calibrated, spectrum-based realism: PLAN + HANDOFF

> Self-contained handoff for a **fresh session** (esp. a local environment where the real
> Maestro `.raw` data is readable). Read this top-to-bottom, then `docs/CALIBRATION.md`.
> **Status: approved, NOT yet implemented** — this is the plan plus the context to execute it.
> The tissue-optics fit is intentionally deferred until the real Maestro `.raw` frames are readable.

---

## 0. What octsim is (orientation)

`octsim` = a physics-based **synthetic OCT B-scan/volume generator** for ML training data, standalone
repo **`tangericm/OCT-Simulator`, developed on `main`** (user directed: push to `main`). Local path
in the cloud workspace: `/workspace/oct-simulator`.

Generation is a **single-scattering forward model**, and every image is the FFT of a synthesized
detector spectrum (this is the invariant to preserve):
```
source P(k) → layered sample → complex reflectivity r(z,x)
  → I(k) = P(k)·|a_ref + Σ r·e^{i2kz}|²   (octsim/forward.py:synthesize_bscan_spectra)
  → FFT reconstruct (octsim/reconstruct.py:reconstruct_bscan) → magnitude/dB B-scan
```
Outputs carry **exact per-pixel layer labels** for free. Config uses resolvable binding
(`SimConfig.resolve()`). There's a PySide6 GUI (`octsim/gui/`), CLI (`--gui/--selftest/--generate`),
and a test suite (74 checks across 11 modules) + `python -m octsim.gui.app --selftest`.

## 1. Current state (already built + committed on `main`)

- **Platform**: config/resolve, forward model, 3D enface anatomy (`octsim/anatomy/retina.py`
  `RetinaModel`, 11 layers NFL…RPE + choroid), foveal pit + optic disc unified field, blood vessels
  (+shadow, FAZ), acquisition timing (T_int→SNR/washout), continuous motion, tunable noise,
  stochastic B-scan + **contiguous-volume** generation, GUI with settings persistence.
- **Data-calibrated GEOMETRY** (PCA `LayerShapeModel`, bundled `.npz`, raw never committed):
  default = **OCTA-500 volumetric** (`retina_shape_octa500.npz`, deep physiologic foveal pit
  rel-depth 0.63); `jhu` (shallower) and `duke`/`dme` selectable. `SimConfig.shape_model="normal"`.
- **Data-calibrated TEXTURE/appearance** (label-free `TextureProfile`, `octsim/data/unlabeled.py`):
  default healthy pool = **~4,945 NORMAL B-scans across 4 devices** (`texture_pooled.json`:
  Srinivasan+OCTDL+OCTID+OCT-C8). Disease profiles bundled separately (amd/cnv/csr/dme/dr/drusen/mh).
  OCTA-500 texture kept separate (frame-averaged, contrast 0.34). `octa500_macula.json` = geometry ref.
- **Recent realism fixes**: subtler vessel shadows, softer choroid boundary + reduced choroidal-blob
  contrast, **speckle now coherent across volume frames** (AR(1), no boiling), GUI settings persist,
  `Averages` + `Geometry model` knobs exposed. Shared `retina.random_macular_vessels` helper (fixed a
  FAZ-duplication bug between the parametric and shape-model paths).

### The remaining gap (why we're here)
Pieces were calibrated **separately**; the **composite render was never optimized against real
images**, and per-layer **optical properties** (backscatter/attenuation) are still literature-set.
That's the biggest realism lever and the target of this plan.

## 2. Constraints & decisions (locked with the user)

- **HARD**: output must stay **spectrum / forward-model based**. Calibration may ONLY tune *physical
  parameters* that feed `I(k)` → FFT. **No image-space post-filter, no neural refiner.**
- **Wait for the raw frames**: build + validate the calibration loop now, but **defer the definitive
  Maestro tissue-optics fit and the default switch until the real `.raw` frames are available**. Ship
  literature optics as default for now; fitted profile = one-line switch.
- **Calibrate in octsim's own reconstruction**: objective compares octsim's own `.bscan_db` stats.
  The device-RAW round-trip + spectral-domain objective term are **deferred** (need raw frames).

## 3. Data situation (CRITICAL for a fresh session)

- Real **Maestro2 device calibration** arrived via chat upload (session-local dir, NOT the repo):
  - `*_3DOCT1_FUNDUS.CLB` — 16,448 B = 64-byte header + 4096 float32; **first 2048** = a monotone
    [0,1] k-linearization LUT. Confirms **pixels = 2048** per A-line.
  - `*fpn.bin` — 2,097,152 B = **2048×512 uint16** detector fixed-pattern/background (col-major,
    `order="F"`); per-pixel mean ≈ the raw source envelope (peaks mid-array, ~7–8-count tail floor).
  - `*MainInfo.bin` — device header: **3D OCT-1 Maestro2**, serial 000003, 2024-08-27. (SD-OCT, ~840 nm.)
  - (Also six OCTID JPEG zips — Cirrus 8-bit images, already integrated; NOT Maestro spectra.)
- **Raw `.raw` B-scan frames are NOT available in the cloud session.** `OCT-Denoiser/images/` is
  git-ignored/empty; a local path like `C:\Users\etang\...\OCT Data` is **unreachable from the cloud
  container**; chat-upload dirs are session-local (a fresh session won't see them either).
- **To make the raw data readable in a fresh session**: run in the **local environment** (the user's
  plan — then the `OCT Data` folder is directly readable), OR provide a **public download URL**
  (WebFetch/curl can pull it) / **chat attachment** / push a sample into a **repo in scope**. Google
  Drive connector is **unauthorized** in-session, so a Drive path needs a plain public link. Expected
  raw format (from OCT-Denoiser): `uint16`, `2048 pixels × {1024 or 2048} A-lines`, column-major,
  `bscan*.raw`.

## 4. THE PLAN — two tiers (physics-grounded)

Guiding principle: `synthesize_bscan_spectra` stays the only image generator; calibration changes
only the numbers it consumes — `LightSource` envelope `P(k)`, `reference_amp`, `falloff_6db_um`,
`NoiseModel`, and the tissue optics carried by a `CalibrationProfile` applied through the EXISTING
`calibration.retina_from_shape_model(..., optics=...)`.

### Tier 1 — instrument grounding from CLB + FPN (measured, not fit; BUILD NOW; data in hand)
Deterministic one-shot → `octsim/data/instrument_maestro2.json` (builder `scripts/build_instrument.py`,
loader `octsim/instrument.py:load_instrument`).
- **Measured envelope**: `P_raw = FPN.reshape(2048,512,order="F").mean(axis=1)`, first-3-pixel DC fix,
  subtract the ~7–8-count electronic tail → k-linearize with the CLB (reuse
  `OCT-Denoiser/preprocess.read_clb_resampling` + `resample_klinear_cubic_operator`, or `np.interp`
  on the LUT) → `power_envelope[2048]` on octsim's linear-k axis (peak-normalized).
- **`LightSource` change** (`octsim/source.py`, frozen dataclass): add ONE field
  `measured_power: np.ndarray | None = field(default=None, compare=False, repr=False)`;
  `power_spectrum()` returns `np.interp(k, k_grid(), measured_power)` if set, else the Gaussian.
  `None` reproduces today exactly. All derived props keep closed-form defs.
- **Reference/noise/background**: DC-to-floor ratio → seed `reference_amp`; FPN tail + per-pixel
  variance → seed `NoiseModel.read`; store k-linearized FPN mean as `background[2048]`
  (`reconstruct_bscan` already accepts `background=`).
- **Grid/λ/BW**: `n_k=2048`; `center_wavelength_nm=840` (near the 850 nm `tissue.scale_layer` ref);
  Gaussian-fit `power_envelope` → `bandwidth_fwhm_nm` + `k_span_factor` so axial res/depth-pixel match
  the device.
- **Wiring**: `SimConfig.instrument: str = ""`; `resolve()` overlays the JSON. Empty → today's behavior.

### Tier 2 — tissue-optics analysis-by-synthesis loop (BUILD + VALIDATE now; definitive fit DEFERRED)
`octsim/calibrate.py` (objective + loop) + `scripts/calibrate_optics.py` (CLI, `scripts/fit_*.py`
style). One eval `f(θ)`: decode θ → `CalibrationProfile` + global overrides; render N B-scans via
`generate.simulate(cfg, seed=s_i)` over a FIXED seed set (common random numbers),
`cfg.instrument="maestro2"`, `shape_model` matching the target, `averages≈2–3` for frame-averaged
targets, θ injected via `retina_from_shape_model(sm, rng, optics=θ)`; extract ONE synthetic
`TextureProfile` from the `.bscan_db` images via `unlabeled.extract_texture(imgs,
axial_pitch_um=resolved.source.depth_pixel_um, lateral_pitch_um=resolved.derived["fast_pitch_um"])`;
compute distance `D(θ)` to the real target `TextureProfile`.

- **Objective** `D = w_t·MSE(reflectivity_template[120]) + w_h·χ²(intensity_hist[32]) +
  Σ w_s·relerr²(speckle_contrast, heterogeneity, noise_floor_frac) + w_reg·R(θ)`; `w_t≈1.0`,
  `w_h≈0.5`, `w_s≈0.3`; grain/thickness/curvature = low-weight guards only; `R(θ)` (`w≈0.05`) =
  `||log θ_backscatter − log θ_lit||² + ||θ_other − θ_lit||²` toward `literature_profile()`
  (anti-overfit / anti-degeneracy).
- **Stochasticity control**: common random numbers (fixed seeds every eval), N≈8–12 realizations
  averaged, population optimizer, `polish=False`, log-scaled params, per-generation checkpoint.
- **Params (~19)**: 6 per-band `backscatter` `[log]` ({NFL},{inner},{ONL},{photoreceptor},{RPE},
  {choroid}, multipliers preserving intra-band ratios, 0.2–5×); 4 per-band `attenuation` `[log]`
  (0.3–3×); `boundary_sharpness_um` (3–15); `choroid_contrast` (0.1–0.6); `texture_contrast`
  (0.08–0.45); `interface_jitter` (0.1–0.6); `vitreous_backscatter` `[log]` (1e-4–3e-3); Tier-1-seeded
  `reference_amp` (2–12), `falloff_6db_um` (300–1800), `noise_level` `[log]` (0.2–4), `floor_db`
  (−70…−45); optional `directional_strength` (0.3–0.9).
- **Optimizer**: `scipy.optimize.differential_evolution` (scipy already a hard dep — NO new dep):
  `init="sobol"`, `polish=False`, `mutation=(0.4,1.0)`, `recombination=0.7`, `tol=1e-2`, fixed `seed`,
  `workers=`. Optional `try: import cma` (CMA-ES) when installed; `Nelder-Mead` multi-start fallback/polish.
- **CalibrationProfile extension**: add `texture_contrast`, `interface_jitter`, `vitreous_backscatter`,
  `directional_strength`; `retina_from_shape_model` reads them (small per-eye jitter) instead of its
  `rng.uniform(...)` draws. `boundary_sharpness_um`/`choroid_contrast` already threaded.

### Wiring a fitted profile as default (mechanism now; SWITCH later)
Mirror `_BUNDLED_SHAPE`: `generate.py` gets `_BUNDLED_PROFILE = {"literature":None, "parametric":None,
"maestro2":"retina_optics_maestro2.json", "octa500":"retina_optics_octa500.json", ...}` +
`@lru_cache _resolve_profile(name)`; `build_scene` passes `optics=_resolve_profile(config.optics_profile)`
into the existing `retina_from_shape_model(...)` call; `SimConfig.optics_profile: str = "literature"`
(coerced in `normalized()`). **Default stays `"literature"` now** (label_map bit-identical — optics
never touch geometry). Flip default to `"maestro2"` after the raw-frame fit.

## 5. Verification

- **Injected-optics recovery** (`tests/test_calibrate.py`, mirror
  `tests/test_calibration.py:_synthetic_segmented_bscan`): render with a KNOWN θ*, make it the target,
  run the loop (tiny budget) from literature; assert recovered band backscatter ratios +
  `boundary_sharpness_um` match θ* within `rtol≈0.25` and `D(θ̂) < D(θ_lit)`. Proves it works WITHOUT
  real frames.
- **Objective decreases** (monotone incumbent). **Demonstration fit + before/after figure** vs
  `texture_octa500.json` (a script; interim, NOT the shipped default). **Tier-1 checks** (envelope
  length 2048, `n_k==2048`, `center_wavelength_nm==840`, `measured_power=None` ⇒ Gaussian).
  **Label invariance** (`label_map` identical literature vs fitted at same seed). **Regression**: full
  suite + `--selftest` pass; `SimConfig` round-trips new fields.

## 6. Deferred until real `.raw` frames arrive
1. Definitive Maestro tissue fit → `retina_optics_maestro2.json`; flip defaults to it.
2. Spectral-domain objective term `D_spec` (fall-off + noise floor from real spectra).
3. Device-RAW emitter (`octsim/device_raw.py`, inverse-CLB via the same CLB) + OCT-Denoiser
   reconstruction round-trip → real-vs-synthetic in the device's own domain + drop-in labeled data.

## 7. Key code references (verified this session; exact reuse map)

- **Forward/instrument**: `octsim/forward.py:synthesize_bscan_spectra(reflectivity, source, *,
  reference_amp=4.0, falloff_6db_um=0.0, dispersion_coeffs=None, signal_scale=1.0, ...)` → real spectra
  [n_k,n_x]. `octsim/source.py:LightSource` (Gaussian `power_spectrum`, `k_grid` linear in k,
  `depth_pixel_um`; defaults λ0=1050, Δλ=100, n_k=2048). `octsim/reconstruct.py:reconstruct_bscan(...,
  background=None)`. `octsim/noise.py:NoiseModel(shot,rin,read)` `.apply` (shot∝√I, rin∝I, read=const).
- **Free tissue params**: `octsim/anatomy/retina.py` `RETINA_LAYERS` (per-layer `backscatter`,
  `mu_s_per_mm`, `interface_reflectivity`, n/melanin/water); wavelength-scaled in
  `octsim/tissue.py:scale_layer`. Model-level: `boundary_sharpness_um=8`, `choroid_contrast=0.32`,
  `texture_contrast=0.22`, `interface_jitter=0.30`, `vitreous_backscatter=5e-4`, `directional_strength=0.6`.
- **Calibration structure**: `octsim/calibration.py` `CalibrationProfile`/`LayerCal` (LayerCal:
  reflectivity, attenuation_per_um, specular, thickness_um; Profile: layers + boundary_sharpness_um,
  curvature_*, choroid_contrast, `backscatter_gain`, JSON save/load), `literature_profile()`,
  `extract_profile(image, boundaries, depth_axis_um, domain="db"/"amplitude")`,
  `retina_from_shape_model(shape, rng, *, optics=None, ...)` — maps `LayerCal.reflectivity*backscatter_gain
  → backscatter`, `attenuation_per_um*1e3 → mu_s_per_mm`, `specular → interface_reflectivity`.
  **THE GAP**: `generate.build_scene` calls it WITHOUT `optics=`, so optics is always literature.
- **Objective stat machinery**: `octsim/data/unlabeled.py:extract_texture(images, *, axial_pitch_um,
  lateral_pitch_um) → TextureProfile` (fields: reflectivity_template[120], template_axis, speckle_contrast,
  heterogeneity, axial/lateral_grain_um, noise_floor_frac, intensity_hist[32], curvature_concave_frac,
  total_retina_um_mean/std). `_detect_ilm_rpe` label-free surfaces; `_normalize` percentile-stretches
  each image (so synthetic dB [-60,0] ≈ real 8-bit display are comparable). `bundled_texture(name)`
  loads `texture_*.json`. Real targets: `texture_octa500.json` (contrast 0.336), `texture_pooled.json`
  (0.494).
- **Render→stats path**: `generate.simulate(SimConfig(anatomy="macula",...), seed) → SampleResult`
  with `.bscan_db`, `.spectra` (retained frame-0 I(k) [n_k,n_x]), `.k_axis_rad_per_um`,
  `resolved.source.depth_pixel_um`, `resolved.derived["fast_pitch_um"]`.
- **Wiring precedent**: `generate.py:_BUNDLED_SHAPE` + `_resolve_shape_model` + `SimConfig.shape_model`
  — mirror exactly for `_BUNDLED_PROFILE`/`optics_profile`.
- **OCT-Denoiser reconstruction** (`OCT-Denoiser/preprocess.py`, for Tier-1 reuse + the deferred device
  loop): `read_clb_resampling(clb_path, pixels=2048)` (seek 64, float32, first 2048),
  `resample_klinear_cubic_operator`, `BscanProcessor.process_one` (uint16 [2048,alines] order="F" →
  DC-subtract → CLB cubic-spline resample → Gaussian window → IFFT → |.| → crop → log → z-score).
  `FolderSpec` in `configs/default.py` (pixels=2048, alines=1024/2048, crop_depth=(0,1024)).

## 8. New/changed files
New: `octsim/instrument.py`, `octsim/calibrate.py`, `scripts/build_instrument.py`,
`scripts/calibrate_optics.py`, `octsim/data/instrument_maestro2.json`, `tests/test_calibrate.py`.
Edits: `octsim/source.py`, `octsim/calibration.py`, `octsim/generate.py`, `octsim/config.py`. Update
`docs/CALIBRATION.md` when done.

## 9. How to resume in a fresh session
1. Read this doc + `docs/CALIBRATION.md`. Repo: `tangericm/OCT-Simulator`, branch `main`.
2. Sanity: `cd` into the repo; run each `tests/test_*.py` (they self-run via `__main__`) +
   `python -m octsim.gui.app --selftest`. GUI tests need `QT_QPA_PLATFORM=offscreen`.
3. Locate the real data: CLB+FPN and the raw `.raw` frames (local `OCT Data` folder in the local env,
   or a provided download URL).
4. Build **Tier 1** (data in hand), then **Tier 2 loop + injected-recovery test** (no real frames
   needed to validate). Only run the DEFINITIVE Maestro tissue fit + flip defaults once real `.raw`
   frames are readable.
5. Commit conventions (this repo): commit to `main`; do NOT commit raw data/caches — only small fitted
   artifacts (`.json`/`.npz`).
