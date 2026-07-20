# octsim — ANCHOR · FIT · JUDGE: an iterative Maestro-anchored calibration routine

> **BUILD STATUS (2026-07-20).** Harness built + green: `octsim/instrument.py`, `calibrate.py`,
> `experiment.py`; `scripts/build_instrument.py`, `build_real_targets.py`, `calibrate_optics.py`,
> `iterate_loop.py`, `leaderboard.py`. THE GAP is closed (optics wiring + label bit-identity across θ,
> verified) and the injected-recovery proof passes.
> **DATA UNBLOCKED.** `OCT Data` is hydrated locally; the real `instrument_maestro3.json` and the
> full diverse panel of real `texture_maestro3_*` / `texture_maestro2_*` / OCTA-500 / OCT-C8 / OCTID
> targets are built. **`scripts/iterate_loop.py` is the multi-corpus driver** (this doc's §3 loop over
> a panel spanning 2 instruments × 3–12 mm FOV × macula/disc/wide): it fits ONE shared tissue-optics θ
> on a pool (anti-overfit), judges D + gates on all members, and emits a contrast-matched full-depth
> real-vs-sim grid + `FINDINGS.md` per run.
> **First accepted multi-corpus fit** (`runs/20260720T005745Z_loop_e82f054`): pooled mean D
> 12.93 → 3.40, and the JUDGE-only corpora dropped too (OCTA-500 43→1.9, OCTID 45→3.4) — the shared
> fit generalised rather than overfitting M3. Open gaps: retina vertical placement offset,
> disc anatomy renders a macula-like notch (needs true ONH cup), and OCT-C8/OCTID fail the speckle
> gate (8-bit clinical JPEG sources, real speckle ~0.46 vs sim ~0.32). Next: run the loop to
> convergence (below), then a definitive fit before flipping any `optics_profile` default.

> A repeatable experiment loop that drives octsim's rendered B-scans toward the **real
> Maestro3/M3** device data now readable locally, while measuring — every iteration — how well
> the *same* parameters generalize to the other corpora (Maestro2, OCTA-500, OCT-C8, OCTID).
> Physics stays the only image generator (`synthesize_bscan_spectra` → FFT); calibration only
> tunes the numbers it consumes. Builds on `docs/CALIBRATION_PLAN.md` (Tier 1 / Tier 2) and
> `docs/REALISM_PLAN.md`. **The data the plan deferred is now in hand — this routine executes it.**

The loop has one sentence: **ANCHOR** what is measured (instrument + geometry), **FIT** what is
free (tissue optics + appearance) by analysis-by-synthesis, **JUDGE** against real targets and
regression gates, then log a tracked, reproducible experiment and iterate. Match M3 first;
generalization to the other devices is a *tracked metric*, not a separate fit.

---

## 0. What the data gives us (measured this session; the anchor)

`OCT Data/` (local) now contains everything the plan called unreachable:

| Asset | Where | What it grounds (routine stage) |
|---|---|---|
| `bscan*.raw` uint16 `[2048, alines]` order='F' | `Maestro3/*/`, `Maestro2/*/` | real B-scans (reconstruct → real targets). M3 alines 1024/2048; 12-bit (max ~4092), mean ~1200 |
| `*.CLB` (first 2048 float32 ∈[0,1], monotone LUT) | each scan folder | k-linearization → depth calibration; confirms **pixels = 2048** |
| `fpn.bin` uint16 `[2048, 512]` | each scan folder | **measured source envelope P(k)** + background + read-noise floor (Tier 1) |
| **mirror scans** `mirror_6mm_1024Aline*` | `Maestro3/` | **direct sensitivity roll-off `falloff_6db_um` + axial PSF/resolution** (a gift the plan didn't have) |
| `MainInfo.bin` | each folder | device = Topcon 3D OCT-1 Maestro, **~840 nm SD-OCT** (⇒ set `center_wavelength_nm≈840`, near tissue.py's 850 ref) |
| `predictions_tiff/gt_*.tif` + `snr_per_frame*.csv` | some folders | already-reconstructed GT B-scans + **real SNR/CNR** (M3 GT ≈ 94 dB SNR / 54 dB CNR) — a free secondary metric |
| repeated frames (e.g. 64 per M3 6mm line folder) | line-scan folders | single frame = full-speckle target (`averages=1`); GT average = frame-averaged look |
| geometries: macula/disc/wide 3–12 mm | `M3_*`, `Maestro2/*` | per-geometry real targets → cross-geometry generalization |

**Reconstruction bridge** (reuse, don't reinvent): the OCT-Denoiser recipe in
`OCT Denoiser/preprocess.py` — `read_clb_resampling(clb, 2048)` → cubic k-linear resample →
DC/mean-A-line subtract → IFFT → **crop rows `(1024,2048)`** → `log10` → per-frame z-score, with
SNR/CNR in its `engine/metrics.py` (`SNR=20log10(⟨col-max⟩/std_bg)`, `CNR=20log10(mean_sig/std_bg)`,
signal rows 111–600, background bottom 20 rows). Reconstruct real frames **through the identical
normalization octsim's stats use** so `unlabeled.extract_texture` is comparing like with like.

---

## 1. Guardrails (do not violate — they define "valid improvement")

- **Spectrum/forward-model only.** Tune only what feeds `I(k)`: `LightSource` envelope, `reference_amp`,
  `falloff_6db_um`, `NoiseModel`, and the tissue optics in a `CalibrationProfile`. No image-space
  post-filter, no neural refiner.
- **Labels are sacred.** Optics never touch geometry, so `label_map` must stay **bit-identical** at a
  fixed seed across every θ (a gate below). This is what makes the synthetic corpus trainable.
- **Instrument is measured, then frozen.** Tier-1 numbers (P(k), background, read floor, falloff,
  resolution, λ0) come from CLB/FPN/mirror by *measurement*, not by fitting. Freeze them before the
  optics fit so the tissue fit can't absorb an instrument error.
- **Common random numbers (CRN).** Every objective evaluation renders the **same fixed seed set**, so
  D(θ) differences are signal, not speckle. octsim already threads `SeedSequence.spawn` deterministically.
- **Regularize toward literature.** The optics fit is under-determined; anchor it with an
  `R(θ)=‖log θ − log θ_lit‖²` term so it doesn't drift into a degenerate look-alike.

---

## 2. The one-time harness to build (before the loop can run)

Mirrors the plan's file list, plus the routine's tracking + real-data bridge. Small fitted artifacts
(`.json`) are committed; raw data and caches are never committed.

**A. Real-target builder** → `scripts/build_real_targets.py`, `octsim/data/real_maestro.py`
- Reconstruct each real corpus (per device × geometry) from `bscan*.raw` via the OCT-Denoiser recipe
  (import it, or port `read_clb_resampling` + the cubic operator — ~40 lines).
- Cache a corpus of reconstructed dB B-scans, then `unlabeled.extract_texture(imgs,
  axial_pitch_um=…, lateral_pitch_um=…)` → **real `TextureProfile` per corpus**, saved as
  `octsim/data/texture_maestro3_macula.json`, `…_disc.json`, `…_maestro2.json`, etc.
- Also compute and store the real **SNR/CNR distribution** (from `predictions_tiff/*.csv` where present,
  else via the ported metric) and the **mirror-measured falloff + axial FWHM**.

**B. Instrument grounding (Tier 1)** → `octsim/instrument.py`, `scripts/build_instrument.py`,
`octsim/data/instrument_maestro3.json`
- `power_envelope[2048]` = k-linearized `FPN.reshape(2048,512,'F').mean(1)` (DC-fix first 3 px, subtract
  ~7–8-count electronic tail); `background[2048]` = same for `reconstruct_bscan(background=…)`;
  `read` floor from FPN per-pixel variance; `reference_amp` seed from DC/floor ratio.
- **From mirror scans**: fit `falloff_6db_um` (depth-vs-peak roll-off) and the axial PSF FWHM →
  `bandwidth_fwhm_nm` + `k_span_factor`; set `center_wavelength_nm≈840`, `n_k=2048`.
- Wiring: add `LightSource.measured_power` (frozen dataclass, `None` sentinel = today's Gaussian);
  `SimConfig.instrument: str=""`; `resolve()` overlays the JSON. Empty ⇒ behavior unchanged.

**C. Optics wiring — close THE GAP** (`octsim/generate.py`, `config.py`, `calibration.py`)
- `build_scene` currently calls `retina_from_shape_model(...)` **without `optics=`**. Add a
  `_BUNDLED_PROFILE` dict + `@lru_cache _resolve_optics_profile(name)` mirroring `_BUNDLED_SHAPE`, a
  `SimConfig.optics_profile: str="literature"` (coerced in `normalized()`), and pass
  `optics=_resolve_optics_profile(config.optics_profile)` into the call.
- Extend `CalibrationProfile` with the four appearance fields it lacks — `texture_contrast`,
  `interface_jitter`, `vitreous_backscatter`, `directional_strength` — and replace the eight
  `rng.uniform(...)` draws in `retina_from_shape_model`/`retina_from_profile` with profile values
  (small per-eye jitter around them). Keep `"literature"` as default so nothing changes until a fit ships.

**D. Objective + optimizer (Tier 2)** → `octsim/calibrate.py`, `scripts/calibrate_optics.py`
- `f(θ)`: decode θ → `CalibrationProfile` + globals → render N CRN B-scans via `generate.simulate`
  (`instrument="maestro3"`, `shape_model` matching the target, `averages` matching the target) →
  `extract_texture` on the `.bscan_db` → distance `D(θ)` to the real `TextureProfile` (Section 4).
- Optimizer: `scipy.optimize.differential_evolution` (`init="sobol"`, `polish=False`,
  `mutation=(0.4,1.0)`, `recombination=0.7`, fixed `seed`, `workers=-1`); optional CMA-ES if installed.
- **Params (~19), log-scaled where noted**: 6 per-band `backscatter` [log], 4 per-band `attenuation`
  [log], `boundary_sharpness_um`, `choroid_contrast`, `texture_contrast`, `interface_jitter`,
  `vitreous_backscatter` [log], `directional_strength`, + Tier-1-seeded `reference_amp`,
  `falloff_6db_um`, `noise_level` [log], `floor_db`.

**E. Experiment tracker** → `octsim/experiment.py` (Section 5). Small, dependency-free.

---

## 3. THE ROUTINE — the iterate loop (run this repeatedly)

Each turn of the crank is one tracked **experiment**. Steps:

**① ANCHOR.** Pick a *target corpus* (start: `maestro3_macula_6x6`). Load its frozen instrument profile
and real `TextureProfile`. Fix: `shape_model` (macula→`octa500`, disc→ONH-parametric), `averages`
(1 for single-frame raw targets; 2–3 for the frame-averaged GT), the CRN seed set (N≈8–12), and
`center_wavelength_nm≈840`. Nothing measured is free this turn.

**② RENDER.** Generate the N CRN B-scans at the current incumbent θ (first turn: `literature`).

**③ MEASURE.** Extract the synthetic `TextureProfile` through the identical normalization; compute the
primary objective **D(θ)** and the full secondary metric panel (Section 4). One render → all metrics.

**④ FIT.** Run the optimizer for a bounded budget (e.g. a few DE generations, or a targeted 1–3 param
sweep when diagnosing a specific mismatch). Record the incumbent trajectory (must be monotone-improving).

**⑤ JUDGE + GATE.** Accept the new θ only if it (a) lowers D(θ), (b) passes the **regression gates**
(label-map bit-identical at fixed seed; `--selftest` + full test suite green; `SimConfig` round-trips),
and (c) stays inside the **physics sanity envelope** (Section 4, "Sanity"). A θ that lowers D but breaks
a gate is rejected and logged as a failure — that is signal about the objective.

**⑥ GENERALIZE.** With the accepted θ **fixed**, evaluate D against the *other* corpora (M2, OCTA-500,
OCT-C8, OCTID) and record the **generalization gap** `D_other − D_target`. This is how the routine keeps
the model honest to "should also produce similar results to the other images": a fit that overfits M3
shows a widening gap and is walked back (raise the regularization weight `w_reg`, or fit shared params
on the pooled corpus and only device-specific params per device).

**⑦ TRACK.** Append one experiment row (Section 5), save the before/after figure and the θ-delta, commit
the fitted `.json` if accepted. Then return to ①.

**Milestones (exit criteria for the phase, not the loop):**
1. *Instrument locked* — Tier-1 checks pass; rendered falloff/axial-FWHM/SNR match mirror+CSV within tol.
2. *Loop proven without over-reliance on eyeballing* — **injected-optics recovery test**: render with a
   known θ\*, make it the target, fit from literature, assert recovered band ratios + `boundary_sharpness_um`
   within `rtol≈0.25` and `D(θ̂) < D(θ_lit)`. Proves the machinery works.
3. *Definitive M3 fit* — `retina_optics_maestro3.json`; flip `SimConfig.optics_profile` default to
   `"maestro3"` (one line) once gates + generalization hold. Before/after figure vs real M3 in the run log.

---

## 4. Metrics — the justification panel (logged every experiment)

Split into the **objective** (what the optimizer minimizes), **diagnostics** (interpretability — *why*
D moved), **sanity** (hard physics gates), and **generalization** (cross-corpus honesty). Only the
domain-invariant `TextureProfile` stats enter D, because real B-scans are 8-bit log-display and
synthetic are dB — per-image percentile normalization destroys absolute brightness, keeping only
*relative* structure comparable.

### 4a. Primary objective D(θ) — minimized
```
D(θ) = w_t·MSE(reflectivity_template[120])          # relative layer-brightness fingerprint  (w_t≈1.0)
     + w_h·χ²(intensity_hist[32])                   # tissue-intensity distribution           (w_h≈0.5)
     + Σ w_s·relerr²( speckle_contrast,             # fully-developed-speckle ratio           (w_s≈0.3)
                      axial_grain_um, lateral_grain_um,  # speckle cell size ↔ resolution
                      heterogeneity,                # intra-layer mottle
                      total_retina_um_mean )        # geometry sanity (low weight)
     + w_reg·R(θ)                                   # ‖log θ_bs − log θ_lit‖² + ‖θ_other−θ_lit‖²  (w_reg≈0.05)
```
Stochasticity-controlled: N≈8–12 CRN realizations averaged per eval. **Do NOT use SSIM/PSNR** to judge
speckle. `intensity_hist`/`noise_floor_frac` are display-domain — only valid because real and synthetic
pass through the *same* normalization; keep them low-weight.

### 4b. Diagnostics — logged, not optimized (tell you *what* is still wrong)
- **Per-layer dB ladder**: mean dB per layer (labels are free). Expect RPE > EZ ≳ NFL/IZ > plexiform >
  nuclear. Compare against the real template peaks.
- **Speckle autocorrelation FWHM** vs beam waist (should equal the resolved lateral/axial resolution).
- **Foveal morphometry** vs normative: pit depth ~120 µm, CFT ~270 µm, FAZ radius ~320 µm.
- **SNR / CNR** (OCT-Denoiser metric) of synthetic vs the real `snr_per_frame*.csv`: M3 GT ≈ 94 dB /
  54 dB. Tunes `reference_amp` / `noise_level` / `floor_db` independently of the texture fit.
- **Falloff & axial FWHM** vs the mirror-measured curve.
- **Vessel placement**: fraction of vessels inside NFL/GCL (target ~100%), none in vitreous/FAZ,
  sub-vessel shadow depth 30–60%.
- **Directional checks**: flip `pupil_entry_offset`/`henle_sign` → Henle wedge flips; sweep tilt →
  EZ/IZ dim while RPE stays stable.

### 4c. Sanity envelope — hard gates (reject θ if violated)
- `label_map` bit-identical at fixed seed across θ (label invariance).
- Speckle contrast ∈ [0.45, 0.53] for `averages=1` (Rayleigh regime); lower and consistent with √N_avg
  for averaged targets (0.52/0.34 ≈ √2.3 cross-checks the averaging model).
- Curvature predominantly concave-up (bowl). Retinal thickness physiologic.
- `--selftest` + full suite green; `SimConfig` round-trips new fields.

### 4d. Generalization — cross-corpus (the "other images" requirement)
- `D(θ_target)` on M3 **and** on each of {M2, OCTA-500, OCT-C8, OCTID}; report `gap = D_other − D_target`
  and the trend across iterations. A healthy fit lowers M3's D without the gaps blowing up.
- Optional distribution-level **FID** between synthetic and real patch sets per corpus — a coarse
  end-to-end cross-check (report, don't optimize; it conflates many effects).
- **Ultimate downstream metric** (when a segmenter is available): Dice of a model trained on the
  synthetic corpus, evaluated on held-out real B-scans. This is the metric the whole simulator exists for.

---

## 5. Experiment tracking — schema & discipline

Every experiment writes an immutable run folder + one manifest row. Dependency-free (JSON/CSV); no MLflow
required, though the schema maps cleanly onto it if you want a UI later.

```
runs/<UTC-timestamp>_<target>_<shortsha>/
  config.json         # full SimConfig snapshot + instrument name + CRN seed set + N
  theta.json          # the CalibrationProfile / θ vector evaluated (+ parent run id)
  metrics.json        # every number in Section 4: D and its weighted terms, diagnostics,
                      #   sanity flags (pass/fail), generalization {corpus: D, gap}
  incumbent.csv       # per-DE-generation: best D so far (monotone), wall-clock, evals
  figures/            # before/after real-vs-synthetic B-scan grid; template overlay;
                      #   per-layer dB ladder bar; falloff curve; SNR/CNR vs real
  env.txt             # git SHA (octsim + this doc), package versions, seed, host
```
Append one row per run to `runs/manifest.jsonl`:
```
{ run_id, ts, git_sha, target_corpus, N, seeds_key,
  D_total, D_template, D_hist, D_speckle, D_reg,
  speckle_contrast, axial_grain_um, thickness_um, concave_frac,
  snr_db, cnr_db, falloff_6db_um, axial_fwhm_um,
  gates_pass, gen_gap_maestro2, gen_gap_octa500, gen_gap_octc8, gen_gap_octid,
  accepted, parent_run_id, note }
```
Discipline that makes the tracking mean something:
- **Reproducibility first**: run id includes the git SHA; `env.txt` + `seeds_key` make any row re-runnable
  bit-for-bit. A metric with no reproducible run behind it is not evidence.
- **Leaderboard**: `scripts/leaderboard.py` sorts `manifest.jsonl` by `D_total` among `gates_pass=true`
  rows, printing the θ-delta from the current default — this is your "did it actually improve" view and
  the justification artifact for flipping the default.
- **One variable per experiment when diagnosing**: for attribution, change one param group (e.g. only RPE
  backscatter) and read the diagnostic panel; use full DE only for the joint fit.
- **Always log the failures**: a θ that lowered D but broke a gate, or widened the generalization gap, is
  recorded with `accepted=false` and a `note`. The failure trail is how you learn the objective's blind spots.
- **Guard against objective-gaming**: because D is a texture distance, a monotone D drop is *necessary not
  sufficient* — a run is only "an improvement" when D drops **and** the diagnostic panel + sanity gates hold
  **and** the generalization gap doesn't worsen. Report all three together, never D alone.

---

## 6. Suggested first three experiments (concrete starting point)

1. **E0 — Anchor & baseline.** Build Tier-1 `instrument_maestro3.json` + real targets; render literature
   optics through it; log the full panel. This is the *baseline row* every later run is judged against.
   Expected immediate wins from the instrument alone: correct falloff, SNR/CNR, and DC/background.
2. **E1 — Injected-recovery proof.** Milestone-2 test. No real data needed; proves the loop before you
   trust it on M3.
3. **E2 — First M3 optics fit.** DE over the ~19 params against `texture_maestro3_macula`, CRN N=10,
   `w_reg=0.05`. Accept only on gates + non-worsening generalization; save `retina_optics_maestro3.json`;
   produce the before/after figure. Do **not** flip the default yet — let a few iterations of E2-style
   runs converge and stabilize the leaderboard first.

---

## 7. Deferred / future (not blocking)
- Spectral-domain objective term `D_spec` (falloff + noise floor from the real spectra directly).
- Device-RAW emitter (inverse-CLB) + OCT-Denoiser round-trip → compare in the device's own domain and
  emit drop-in labeled `.raw` for the denoiser/super-res pipelines.
- `.fds/.fda` native reader (no parser exists yet in either repo) to pull Topcon's own processed volumes
  as an additional real reference.
- Hybrid macular geometry (OCTA-500 pit inside ±1.5 mm + JHU finer layers outside).
