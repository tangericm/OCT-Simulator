# octsim realism — state after round 2, and the round-3 plan

Pickup document. Round 1 is recorded in `REALISM_PLAN.md`; this supersedes its priority
list. Everything below is merged to `main` as of `85d1f63` (PR #1), CI green on py3.11 and
py3.13, 89 tests.

Read the "traps" section before measuring anything. Most of the wrong turns in round 2 were
bad *measurements*, not bad models, and they are easy to repeat.

---

## 0. Session baseline, 2026-07-26 — re-measured, plus three corrections

None of §3 has been started; `main` is still at the doc commit and the tree is clean. The
suite and the validation command were re-run to confirm the numbers below are live, not
remembered: selftest exit 0, 81/81 fast tests pass (`--ignore=tests/test_calibrate.py`),
and `validate_realism.py` reports

```
SUMMARY  training 9/9   maestro2 9/9   octa500 2/2   octid 0/0
```

Three things that section 1–3 get wrong or omit, in value order.

### 0a. The lateral speckle grain was defined in A-lines, so it was not a physical length

**Status: fixed on `realism-round3-r0`.** The clearest generalization defect in the model,
and it was invisible because it passed both gates.

| corpus | A-line pitch dx | real grain | sim before | delta | sim after | delta |
|---|---|---|---|---|---|---|
| Maestro3 6 mm/1024 | 5.86 µm | 5.56 µm | 7.94 µm | **+2.38** | 5.79 µm | **+0.22** |
| Maestro2 6 mm/2048 | 2.93 µm | 5.15 µm | 2.60 µm | **−2.55** | 4.80 µm | **−0.35** |

The delta flipped sign across the two corpora and only stayed inside the ±3 µm gate by
luck; halving the pitch again would have failed it.

**The real grain is one physical length, and the evidence is now seven corpora, not two.**
Maestro3 ships several FOVs at fixed A-line counts, which sweeps `dx` on one instrument with
one optical train:

| dx µm | 2.93 | 5.86 | 5.86 | 8.79 | 11.72 | 17.58 | 23.44 |
|---|---|---|---|---|---|---|---|
| real grain µm | 5.15 | 5.56 | 6.16 | 5.79 | 7.85 | 11.01 | 14.01 |
| grain / dx | 1.76 | 0.95 | 1.05 | 0.66 | 0.67 | 0.63 | 0.60 |

The apparent growth at coarse pitch is **`_acf_fwhm` bottoming out at its own 0.5 px floor**
(measured: pure white noise reads 0.499 px), not a wider grain — the ratio marches toward
0.5 and never below. Forward-modelling the metric confirms it: one correlation length of
**5.0 µm** reproduces all seven (RMS 1.4 µm, and ≤0.7 µm on the four line scans, which are
the only ones that resolve it). The per-A-line model — a grain of exactly `dx`, which is
what the code drew — fits almost twice as badly, RMS 2.6 µm.

There were **two** defects in `octsim/sample.py:372-378`, not one:

1. **Wrong units.** The `speckle_decorrelation = 0.12` component was drawn as
   `rng.standard_normal((n_depth, n_x))` — white *per column index*, so its correlation
   length was one A-line and in micrometres it was `dx`. The comment above it reasoned that
   real decorrelation comes from per-A-line effects (galvo jitter, motion between A-lines),
   which would indeed be A-line-quantised. The seven-corpus measurement says otherwise, so
   that premise was wrong.
2. **No variance matching, which is the bigger effect.** `_lateral_beam_filter` is
   sum-normalised: it preserves smooth structure but shrinks *speckle* variance by Σk²,
   which depends on `beam_waist_um / dx_um` — i.e. on the sampling pitch. Mixing an
   unfiltered component against it with plain √(1−f)/√f weights meant f was never a power
   fraction. From one nominal f = 0.12 the effective decorrelated fraction ran:

   | dx µm | 2.93 | 5.86 | 8.79 | 11.72 | 23.44 |
   |---|---|---|---|---|---|
   | Σk² | 0.078 | 0.156 | 0.234 | 0.312 | 0.736 |
   | white/beam amplitude ratio | 1.17 | 0.83 | 0.68 | 0.59 | 0.38 |

   At 2.93 µm the "12%" component *dominated* the speckle; at 23.44 µm it was a third of
   the beam term. That is what produced the 3× grain swing, and it moved frame brightness
   with A-line density too.

The fix is both: `speckle_corr_lateral_um` (a physical Gaussian σ, on `SimConfig` so it is
an instrument property rather than anatomy) and scaling the injected component by `√Σk²` so
`f` is a true power fraction at every pitch. `_beam_variance_keep` computes Σk² in closed
form from the same kernel the filter uses, and a test pins the two against each other.

`tests/test_speckle_pitch.py` is the regression gate, in the genre of `test_wavelength.py`:
render the same scene at two A-line densities and require the grain to agree. It fails on
the old code by 3.85 µm (6.62 vs 2.77) and passes on the new by 0.91 µm (5.62 vs 4.71).

#### What the fix revealed: the model cannot have both grain and lamination

Once `f` means what it says, it has to be re-fitted, and the fit is **much** tighter than
expected. Sweeping it at the training geometry (L = 4 µm, deltas against real):

| f | grain (tol 3.0) | inner_swing (tol 2.5) | inner_gradient | contrast_inner_vitreous |
|---|---|---|---|---|
| 0.00 | **+13.56** | +0.09 | +0.67 | −0.63 |
| 0.60 | **+3.92** | −1.41 | +0.20 | −1.15 |
| 0.75 | +2.44 | −1.98 | −0.07 | −1.27 |
| 0.85 | +1.01 | −2.20 | −0.21 | −1.38 |
| 0.90 | +0.40 | **−2.50** | −0.36 | −1.42 |
| 1.00 | −0.40 | **−2.74** | −0.51 | −1.54 |

The two gated metrics move in opposite directions and monotonically: at f = 0 the
inner-retina lamination is nearly perfect (+0.09 dB) and the grain is 13.6 µm too coarse;
at f = 1 the grain is right to 0.4 µm and the swing is 2.7 dB low. **The window where both
gates hold is only 0.75–0.85 wide.** `speckle_decorrelation` ships at **0.75**, the end of
that window with the best worst-case margin — full 8-seed validation gives grain +2.03
(1.7σ of the seed spread from its tolerance) and swing −1.98 (1.2σ), against 3.2σ / 0.7σ at
f = 0.80. Both settings return 9/9 on every corpus; 0.75 is the one that is not one noisy
seed set away from failing.

That narrowness is a finding, not a tuning detail. The old code reconciled the two by
accident — the pitch-dependent effective fraction happened to sit at a different operating
point on each corpus. With that artifact gone, matching real speckle grain *costs* real
lamination contrast, which says the inner-retina backscatter/texture parameters are
carrying error that the speckle model was masking. That is R1's territory, and R1 should
treat `f` as jointly fitted with the inner-layer parameters rather than fixed.

Two side effects worth recording, both on the **held-out** Maestro2 corpus:

- **`shadow_count` was almost entirely a speckle artifact.** Real 42.6 vs sim 14.9 before
  (delta −27.8), sim **42.4** after (delta −0.25). R6 lists this as needing a
  "vessel-conditioned statistic"; it needed a correct speckle model instead.
- **The speckle *amplitude* statistics got worse**, and are still ungated:
  `speckle_intensity_contrast` 2.20 → 1.70 against a real 9.73, and
  `speckle_amp_mu_over_sigma` 1.23 → 1.43 against a real 0.85. Note the real
  `speckle_intensity_contrast` is 4.27 on Maestro3 and 9.73 on Maestro2 — a 2.3× spread on
  the *real* measurement across two similar devices, which suggests the metric is unstable
  before concluding anything about the model. Worth resolving in R6 before gating it.

### 0b. The Maestro2 device numbers were not reachable from the public API

**Status: fixed on `realism-round3-r0` (R0b).** Kept below because the reasoning — and one
wrong turn — is what the fix had to be shaped around.

§1 says `pivot_distance_um` and `scan_distortion_correction` are "already on the instrument
side". They are on `SimConfig` (`octsim/config.py:102,111`), which is the acquisition side,
not the instrument side — and neither `instrument_maestro2.json` nor `instrument_maestro3.json`
carries either field, nor an SNR.

The two numbers that took Maestro2 from 4/7 to 9/9 live only in the validation harness's
holdout table (`octsim/data/display_domain.py:137`: `distortion=0.43, snr_db=45.33`), applied
by `scripts/validate_realism.py:173-175`. So `simulate(SimConfig(instrument="maestro2"))`
still renders Maestro3 curvature at Maestro3 noise. The headline result of round 2 is
reproducible only from the validation script, not from the simulator.

`Instrument.read_floor` already differs per device (0.0402 vs 0.0893) and is deliberately
*not* adopted by `resolve()` — it is written as an 8-bit FPN scale (`read_std / 255`,
`scripts/build_instrument.py:301`) that is not commensurate with the reference-DC units, and
adopting it raises the floor ~11× (`octsim/config.py:347`).

**Correction (measured after this section was first written).** `target_snr_db` is *not*
the right channel, though it looks like one. It back-solves the `read` term alone
(`config.py:365`), while the noise floor is partly shot- and RIN-dominated — the comment at
`config.py:134` says so explicitly. Measured on a Maestro2 macula render, deep floor as the
median of the last 80 rows:

| channel | floor | shift |
|---|---|---|
| default | −57.92 dB | — |
| `noise_level = 2.165` | −51.26 dB | **+6.65 dB** |
| `target_snr_db = 45.33` | −54.92 dB | +3.00 dB |

The device gap is 52.04 − 45.33 = 6.71 dB, so only the `noise_level` route reproduces the
round-2 Maestro2 result; `target_snr_db` delivers under half of it. R0b must overlay a
`noise_level` derived through `display_domain.noise_level_for_snr`, not a `target_snr_db`.

There is also **no field-level instrument overlay to extend**: `resolve()` uses the loaded
`Instrument` only to replace the `LightSource` wholesale (`config.py:274-289`), and
`reference_amp`, `read_floor`, `floor_db` and `falloff_6db_um` are all loaded and ignored.
`falloff_6db_um` is in fact a third field with this same reachability bug — SimConfig
defaults it to 0 (= off) while both JSONs carry ~1960 µm, and `validate_realism.py:91`
works around it by hard-coding the Maestro3 value for every corpus, Maestro2 included.

Two traps for whoever writes R0b. **Neither field has a spare sentinel**:
`scan_distortion_correction = 0.0` already means "fully flattened" and `noise_level <= 0`
already means "noise off", and `SimConfig` is `slots=True`, so there is no way to ask
whether a field was explicitly passed. Comparing against the class default is not safe
either — `octsim/calibrate.py:316` and `scripts/iterate_loop.py:183` pass a *fitted*
`noise_level` that can legitimately land on exactly 1.0 while selecting
`instrument="maestro2"`. An explicit `None`-typed optional is the only clean option.
And **`generate.py:92-93` reads `config.scan_distortion_correction` directly**, not the
resolved value, so an overlay that only populates `ResolvedConfig` would make the R0b test
pass while leaving the rendered image unchanged.

### 0c. "9/9" gates 9 metrics out of 36, and one holdout gates nothing

36 statistics are computed and printed; `GATES` (`octsim/realism_metrics.py:615-628`) has
tolerances for 9 of them. The other 27 are reported and ignored, so a regression in any of
them is silent. The largest ungated deltas right now:

| metric | corpus | real | sim | note |
|---|---|---|---|---|
| `shadow_count` | maestro2 | 42.6 | 14.9 | known (R6); worse cross-device than on training (25.5 vs 15.9) |
| `speckle_intensity_contrast` | maestro2 | 9.73 | 2.20 | real value is 4.27 on M3 vs 9.73 on M2 — the *real* measurement is not device-stable, so suspect the metric before the model |
| `band_above_floor_db.onl` | training | 10.73 | 14.87 | +4.1 dB; the largest band error, and consistent with §6's "inner retina too dim relative to outer" |
| `inner_extrema` | octa500 | 4.88 | 2.38 | sim under-laminated in the display domain, though it matches 3.0 vs 3.0 on maestro2 |
| `speckle_grain_axial_um` | octa500 | 3.13 | 5.48 | axial grain has no gate at all — only the lateral one does |
| `choroid_dark_frac_4db` | training | 0.004 | 0.049 | 12× too much dark choroid; matches §6's "rounder lumens" |
| `ilm_row` / `rpe_row` | training | 217.8 / 338.1 | 199.6 / 323.4 | sim retina sits ~60 µm shallow — a fixed offset, both landmarks move together |

Two structural problems behind the number:

- **`octid` gates 0/0 and still counts as a passing holdout.** At `tier=compressed` only 4
  metrics survive, and none of the 4 has a `GATES` entry, so the corpus can never fail. It
  is currently decoration. Either give the compressed tier a gated statistic or stop
  reporting octid as a holdout result. *(Fixed in R0c: `rpe_peak_separation_um` is gated at
  18 µm and octid is now 0/1, recorded as an expected failure rather than as a pass.)*
- **Some tolerances cannot discriminate.** `sag_optical_um` has tol 150 µm, but real sag is
  503 µm on Maestro3 and 254 µm on Maestro2 — the gate is wider than the entire device
  effect it is supposed to be sensitive to. *(Fixed in R0c: tightened to 60 µm.)*

This does not invalidate round 2 (the 9 gated metrics did move), but "9/9" should be read as
"9 gated metrics pass", not "the sim matches".

---

## 1. What round 2 established

### The overfitting was in the device parameters, not the tissue model

This is the central result and it reframes the whole problem.

Round 1 matched one Topcon Maestro3 corpus to 7/7 gates. The worry was that the layer
table had absorbed that device's tone curve. Partly true — but the dominant effect was
that **device properties had been baked into `RetinaModel`**: `pivot_distance_um` lived on
the anatomy, and the noise coefficients were global constants tuned to one unit.

Held-out Maestro2 went **4/7 → 9/9 by supplying only its own two device numbers**, with
zero tissue refitting:

| parameter | Maestro3 | Maestro2 |
|---|---|---|
| SNR | 52.04 dB | 45.33 dB (already in the repo, unused) |
| `scan_distortion_correction` | 0.86 | 0.43 |

The scan-distortion number is not a fudge. Raw geometric sag over a nominal 6 mm scan is
~585 µm; Maestro3 shows 504 µm and Maestro2 shows 249 µm — **more than 2× apart on the same
anatomy**, because vendors flatten their B-scans differently. Curvature is therefore a
device property as much as an anatomical one, and any future device needs its own value.

**Implication for round 3:** before fitting anything, ask whether the quantity belongs to
the tissue or the instrument. The remaining device-conditioned parameters are listed in §4.

### Metrics have to be tiered by domain

8-bit display-domain corpora (OCTA-500, OCTID) have passed through an unknown monotone tone
curve. dB-valued and floor-referenced statistics do not survive it; geometry and spatial
correlations do. `realism_metrics.TIER_COMPRESSED` / `TIER_DISPLAY` / `TIER_FULL` encode
this, and `restrict(agg, tier)` applies it.

Neither corpus ships voxel metadata. Assuming a pitch gave a **449 µm retina**, which is
anatomically impossible. `display_domain.calibrate_axial_pitch()` solves the pitch against
a normative 300 µm retina instead — and `thickness_tissue_um` is then explicitly dropped,
because it is pinned by construction and is no longer an independent measurement.

### Wavelength behaviour is right, and it is the one thing fitting cannot fake

`tests/test_wavelength.py`. Choroid brightness relative to the **inner retina** rises
monotonically with wavelength:

| λ | choroid − inner |
|---|---|
| 840 nm | −7.55 dB |
| 1050 nm | −6.53 dB |
| 1310 nm | −5.78 dB |

matching the model's own predicted 1.84 dB cumulative difference. No 840 nm data
constrains this, so it is a genuine test of the tissue model rather than of the fit. Keep
it green; if a future change to the layer table breaks it, the change is wrong.

### Out-of-range wrap needed no special-case code

The physics was already correct — only the grid prevented it, because the scene was
rasterised onto exactly `n_k//2` rows. Rendering onto the full `n_k` grid produces
fold-over exactly as the device does, and gives conjugate/mirror artifacts for free.
Verified inert in range (0.74 dB speckle-averaged residual) and active out of range (46 dB
at 3300 µm against `z_max` 3963). Cost 1.17×.

---

## 2. Negative result: do not redo O3 as specified

**`octsim/attenuation.py` implements the Vermeer 2014 depth-resolved estimator, and it
cannot supply per-layer µ for the retina.** This was measured, not assumed, and the module
docstring carries the numbers.

The estimator itself is correct — on synthetic A-lines it recovers 1.5 → 1.51 and
3.0 → 3.04 /mm. The problem is its stated assumption: constant backscatter-to-attenuation
ratio with depth. Tested directly:

> Synthetic A-line, **uniform true µ = 4 /mm**, with a 30× brighter band inserted:
> **24.8 /mm inside the band**, 3.3 /mm above it, 4.0 /mm below (where the assumption is
> undisturbed).

It converts backscatter contrast into apparent attenuation. The retina is the canonical
violation — the RPE is ~30× brighter than the inner retina — so on real Maestro3 data it
reports ~27 /mm at the RPE and 0.1–0.5 /mm through the inner retina, against published
RNFL attenuation of ~4–5 /mm. The inner retina reads an order of magnitude low precisely
because the bright RPE beneath it inflates the denominator.

**The layer table was deliberately left unchanged.** Updating it from these numbers would
have swapped tone-matched values for differently-wrong ones while looking more principled.

What the module *is* good for: the shape of the µ(depth) profile, and regions of roughly
uniform reflectivity (choroid, deep tail). See §3 for the successor task.

---

## 3. Round-3 plan

Ordered by value. Effort in {S, M, L}. **R0 was added by the 2026-07-26 baseline and comes
first**: it is cheap, and it fixes the measurement apparatus that every later item is scored
by. Fitting anything (R1) against a 9-metric gate set with a pitch-dependent speckle model
would bake those defects into the fitted numbers.

### R0. Make the harness able to fail · **S, do first**

Three sub-tasks, all from §0. Each is independently shippable and each has a test.

- **R0a — physical lateral speckle correlation length. ✅ done** (branch
  `realism-round3-r0`). `speckle_corr_lateral_um = 4.0` on `SimConfig`, variance-matched
  mixing via `_beam_variance_keep`, `speckle_decorrelation` re-fitted 0.12 → 0.75, and
  `tests/test_speckle_pitch.py` added. Grain delta +2.38/−2.55 → +2.03/+0.83, and
  `shadow_count` on held-out Maestro2 −27.8 → −0.25. Validation stays 9/9 · 9/9 · 2/2 · 0/0,
  exit 0. See §0a for the numbers and for the grain-vs-lamination trade-off it exposed.

  One regression it caused and fixed: raising `f` made the decorrelated draw the dominant
  speckle term, and that draw was fresh per frame, so `test_volume_speckle_is_coherent_across_frames`
  started failing — a densely-sampled volume boiled. The decorrelation is *lateral*, so the
  draw now carries its own slow-axis AR(1) chain alongside the base speckle field. The
  single-B-scan path is untouched (both fields are `None` there, and the builder still draws
  from `rng` in the same order), so the validation numbers above are unaffected by that fix.
- **R0b — move the device numbers into the instrument JSONs. ✅ done** (branch
  `realism-round3-r0`). `Instrument` gained `scan_distortion_correction` and `snr_db`, both
  JSONs carry them, and `SimConfig(instrument="maestro2")` now renders at 0.43 with the
  Maestro2 noise level. `validate_realism.py` reads them from the instrument rather than
  from its own table, so the harness and the public API cannot disagree. Validation stays
  9/9 · 9/9 · 2/2 · 0/0, exit 0, and the Maestro2 holdout now renders as
  `instrument=maestro2` rather than as Maestro3 with two numbers patched in.

  Four decisions worth knowing, all forced by what §0b found:

  - **Precedence is an explicit `None`, not a `0` sentinel.** `scan_distortion_correction`
    is now `float | None`, resolved config → instrument → 0.86 by
    `SimConfig.effective_scan_distortion()`. `0.0` could not be reused because it already
    means "fully flattened", and `slots=True` rules out detecting an explicit set.
  - **The resolver is a method on `SimConfig`, not an overlay inside `resolve()`**, because
    `generate.build_scene` stamps this onto the retina from the *config*. An overlay into
    `ResolvedConfig` alone would have left the rendered image unchanged while making
    `resolved.scan_distortion_correction` report the device value — a test that passes on a
    bug. `tests/test_instrument_device_numbers.py` asserts both paths agree.
  - **SNR goes through `noise_level`, via `Instrument.noise_scale`, not `target_snr_db`.**
    It multiplies all three coefficients because the floor is shot-dominated. `noise_level`
    keeps its meaning as a user multiplier applied on top, so no sentinel was needed and a
    fitted `noise_level` of 1.0 still means 1.0× *that device's* level.
  - **`REFERENCE_SNR_DB` is full precision** (52.04221179196247, not 52.04) so
    `noise_scale` is exactly 1.0 for the calibration device and overlaying an instrument
    cannot nudge the corpus it was calibrated on. `display_domain` now re-exports it rather
    than keeping a second copy.

  `scripts/build_instrument.py` carries the two values across on a rebuild, since it cannot
  measure either and would otherwise reset them to the dataclass defaults.

  **Still open, deliberately:** `falloff_6db_um` is the third field with this same
  reachability bug — `SimConfig` defaults it to 0 (= off) while both JSONs carry ~1960 µm,
  and `validate_realism.py:91` still hard-codes the Maestro3 value for every corpus. It was
  left alone because `calibrate.py` treats it as a frozen fitted parameter
  (`ParamSpec("falloff_6db_um", ..., frozen=True)`) and adopting it from the instrument
  would change what that fit means. Decide it with R1, not here.
- **R0c — close the gate holes. ✅ done** (branch `realism-round3-r0`). Gates went
  **9 → 18** of the 36 statistics, `sag_optical_um` tightened 150 → 60 µm, and an
  expected-failing mechanism was built because none existed. Validation now reads
  `training 16/17 · maestro2 17/17 · octa500 3/5 · octid 0/1` with four recorded waivers —
  and octid finally carries a gate at all, instead of the 0/0 that could never fail.

  **A blanket "k × real SD" rule does not work**, which is why each tolerance is set from
  its own evidence. The shipped gates already ranged k = 6–46 (median 10), and the
  candidates break the rule in both directions: `inner_extrema` has a real SD of *exactly
  zero* (always 3.0 on the raw corpora), so 8×SD is a tolerance of 0; `shadow_count`'s real
  value nearly doubles between devices (25.5 vs 42.6), so 8×SD is 32 — wider than the
  metric's whole range. Tolerances added:

  | metric | tol | why that number |
  |---|---|---|
  | `sag_optical_um` | 150 → **60** | real sag is 503 / 254 / 41 µm across the three corpora; 60 passes all (+7, +21, +14) and still catches a device mix-up (~250 µm) |
  | `speckle_grain_axial_um` | **1.0** | real 5.24 ± 0.12 and 4.87 ± 0.06 — the most stable statistic in the vector, and the axial partner the lateral-grain gate never had |
  | `shadow_count` | **15** | loose by necessity, but still catches the −27.8 the pre-R0a speckle model produced |
  | `band_above_floor_db.onl` | **4.0** | current +2.6 / −1.9; would have caught the +4.1 of §0c |
  | `column_energy_sd_db` | **0.8** | real 3.34 ± 0.08 / 2.41 ± 0.07 |
  | `median_tissue_snr_db` | **3.0** | guards the R0b per-device noise wiring: if an instrument's SNR stops reaching the renderer, this is where it shows |
  | `choroid_dark_frac_4db` | **0.010** | gated *and waived* — see below |
  | `inner_extrema` | **1.0** | lamination must be right to within one extremum; guards the round-1 "featureless slab" failure |
  | `rpe_peak_separation_um` | **18** | the only gate the compressed tier can carry |

  **The waiver mechanism.** `GATES` values stayed bare floats; the new
  `realism_metrics.EXPECTED_FAIL` maps `(corpus, metric) → reason`, and `gate_outcome()`
  returns `PASS` / `FAIL` / `XFAIL` / `XPASS`. An `XFAIL` is reported in the table and the
  summary but does not set exit 1; an `XPASS` is reported as a prompt to delete the waiver.
  The alternative — widening a tolerance until it passes — is what turns a known gap back
  into a silent one, which is the whole failure R0c exists to remove. Four waivers, each
  carrying its reason in the source:

  - `training / choroid_dark_frac_4db` — sim holds ~7× too many locally-dark choroidal
    voxels (0.004 → 0.031). §6's "rounder lumens", now a number.
  - `octa500 / inner_extrema` — sim resolves 2.4 extrema against a real 4.9 in the display
    domain while matching 3.0 vs 3.0 on Maestro2. Under-lamination; R1.
  - `octid / rpe_peak_separation_um` — real 13.3 vs sim 40.6. Sim always resolves exactly 2
    RPE-complex peaks at a fixed spacing; heavily JPEG-compressed real data merges them.
  - `octa500 / speckle_grain_axial_um` — **found by the new gate on its first run.** Real
    3.13 vs sim 5.40, where the two raw corpora agree to 0.2 µm. Axial grain is set by
    source bandwidth and OCTA-500 has no instrument model, so it renders with Maestro3's
    source: this compares two devices' axial PSFs, and its pitch is solved rather than
    measured. The fix is an instrument model for that corpus (the R0b pattern), not a wider
    gate.

  `tests/test_gates.py` pins the machinery rather than the numbers: every gate names a real
  metric (a typo'd key is silently a dead gate), every waiver points at a gated metric and
  carries a reason, **every tier has at least one gated metric**, and the compressed-tier
  gate survives the harness's own drop set — the specific way octid became decoration.

  Four things that shaped it, worth keeping:

  - **Only two metrics can gate octid at all.** The compressed tier holds four, but
    `validate_realism.py:161-169` then drops `thickness_tissue_um` (pitch-calibrated) and
    `sag_optical_um` (`fov_known=False`), leaving `rpe_peak_separation_um` and
    `shadow_median_width_um`. Tightening `sag` therefore does nothing for octid.
  - **`shadow_median_width_um` is a trap.** OCTA-500's real value is 0.0 — the dataclass
    default, meaning no shadow runs were detected, not a measurement — against a sim 66.6 µm.
    Any tolerance derived from the training spread fails octa500 on a non-measurement. Its
    scale is also `columns × dx_um`, so the same `fov_known=False` argument that gets `sag`
    dropped for octid applies to it and is not applied.
  - **`rpe_peak_separation_um` is the defensible octid gate, and it is expected to fail
    there**: real 13.30 vs sim 40.64. A ~2-SD tolerance (2 × 8.285 ≈ 17 µm from the training
    real spread) passes training (+5.2), maestro2 (+9.0) and octa500 (+12.8) and fails octid
    alone. That is exactly the case the expected-failing marker is for.
  - **There is no xfail mechanism to reuse.** `GATES` values are bare floats and both gate
    counters filter on `r[7] is not None`; a waiver needs a parallel set plus a change to the
    `n_pass != n_gated` comparison. Also note `compare()` silently skips a `GATES` key that
    is missing from either side, so a typo'd key is a dead entry with no error —
    `band_above_floor_db` has no `vitreous`, for instance.

Note the ordering constraint: R0c will turn several currently-silent deltas into failures.
That is intended, but it means R0c should land with the gate list explicitly marked
"expected-failing" for the ones no fix exists for yet, rather than with tolerances widened
to make them pass.

### R1. Joint (µ, backscatter) fit — the real successor to O3 · **L, highest value**

Separating attenuation from backscatter needs both solved together, not µ alone. The
simulator can do this where the Vermeer estimator cannot, because it can *render*: propose
(µ, backscatter) per layer, render, compare against the real A-line through the existing
metric code, iterate. `octsim/calibrate.py` already has the ANCHOR·FIT·JUDGE machinery and
`tests/test_calibrate.py` proves it recovers injected parameters.

Check identifiability per dimension before fitting it — µ and backscatter trade off within
a thin layer, and the round-1 experience is that a 4-parameter toy fit already conflates
adjacent outer bands at 512 px. ~~Start with the layers that are thick and well-separated
(NFL, ONL, choroid) and hold the outer complex fixed.~~

#### R1a — identifiability, measured 2026-07-26. Two corrections to this plan.

Method: finite-difference the realism metric vector against each of the 9 band parameters
under common random numbers, normalise every metric by the **real** corpus's frame-to-frame
SD — so a direction counts to the extent the gates can actually see it — and take pairwise
cosines of the Jacobian columns. `runs/probe_identifiability.py`.

Sensitivity, in real-SD units per decade. This is how visible each parameter is *to the
judge*:

| | bs_NFL | bs_inner | bs_ONL | bs_outer | bs_choroid | at_inner | at_nuclear | at_outer | at_choroid |
|---|---|---|---|---|---|---|---|---|---|
| ‖∂m/∂θ‖ | 71 | 182 | 96 | 179 | **4.0** | 96 | 34 | 25 | **1.6** |

Degenerate pairs, |cos| ≥ 0.9:

| pair | cos |
|---|---|
| `bs_outer` vs `at_outer` | **−0.991** |
| `bs_outer` vs `at_nuclear` | **−0.985** |
| `at_nuclear` vs `at_outer` | **+0.973** |
| `bs_choroid` vs `at_choroid` | **−0.898** |

**Correction 1: the choroid is the worst starting point, not one of the best.** This
paragraph proposed NFL, ONL and choroid as the thick, well-separated layers to fit first.
NFL and ONL hold up — `bs_NFL` peaks at |cos| 0.36 against anything else, `bs_ONL` at 0.73.
The choroid does not: its two parameters are 90 % degenerate with each other **and** they
are the two least visible dimensions in the set, 45× and 114× weaker than `bs_inner`. A
choroid fit converges to a confident number no gate can contradict.

**Correction 2: `theta_space`'s freeze list is backwards.** It freezes `at_inner` and
`at_nuclear` as "only weakly identifiable" while leaving `at_outer` and `at_choroid` free.
Measured, `at_inner` is the *most* visible attenuation dimension (96) — more than the two
that are actually fitted (25 and 1.6) combined — while `at_outer` is 99 % degenerate with
`bs_outer` and `at_choroid` is nearly invisible. The fit spends its attenuation freedom
almost entirely on directions the judge cannot see.

The 9-parameter problem has condition number **389** (singular values 207 … 0.53), so two
or three combinations are effectively unrecoverable. Fitting per-*layer* µ, which is what
R1's headline asks for, would add 13 dimensions to a problem that cannot support 9.

**The flat direction is almost pure choroid.** The smallest right-singular vector of the
Jacobian is `at_choroid` −0.95, `bs_choroid` −0.29, everything else below 0.13. Walking it
and comparing against the best-seen direction at the same step (`runs/probe_null2.py`):

| step (decades) | ‖Δm‖ along the null direction | along the best direction | ratio |
|---|---|---|---|
| 0.10 | 0.07 | 21.4 | **304×** |
| 0.20 | 0.27 | 41.5 | **151×** |
| 0.40 | 0.32 | 78.0 | **245×** |

Choroidal attenuation can be multiplied by **2.5×** and the whole gated realism vector moves
by 0.32 SD units *in total, across 23 metrics* — less than a third of one metric's
frame-to-frame noise. Whatever a fit reports for the choroid, no gate in the harness can
contradict it.

**Caveat, stated because it bounds the claim.** This Jacobian is taken against the realism
metric vector — the *judge* — not against `calibrate.D`, the objective the fit minimises,
which carries a `D_falloff` term built specifically to identify deep attenuation. So the
honest statement is not "the fit cannot move `at_outer`/`at_choroid`" but "**nothing in the
gate set can tell whether it moved them correctly**", which is the overfitting shape this
whole round is about. An attempt to walk the same direction through `calibrate.evaluate`
returned `D_total ≈ 222` where the accepted multi-corpus fit records ≈ 3.4, i.e. the
RenderSpec/target pairing was wrong — that probe is not reported, and reproducing the
fitting session's exact anchor is a prerequisite for the objective-side half of this.

**What R1 should do**, superseding the struck-through text: fit `bs_NFL`, `bs_inner`,
`bs_ONL` and `at_inner` — the four dimensions that are both visible and mutually
distinguishable — and hold the outer complex and the choroid at their literature values
instead of fitting them. Quote per-layer µ only for bands whose Jacobian column is
resolvable, and say so plainly for the rest rather than reporting a fitted number. Note
also that `speckle_decorrelation` has to join the free set (§0a): it trades directly
against `inner_swing_db`, which is one of the metrics this fit moves.

Success criterion: per-layer µ in 1/mm that is (a) within range of published values and
(b) **improves the held-out score**, not just the training score.

### R2. Finish the device/tissue split + provenance tags · **S, high value**

Was O1; still not done, and §1 shows why it matters most. **R0b is the first half of this**
— once the device numbers live in the instrument JSONs, the `fitted:<device>` tag has
somewhere to go. `speckle_decorrelation` in the table below is superseded by R0a: it is
instrument-side, and it also has the wrong units.

Tag every parameter `measured | literature | fitted:<device>` so device-conditioned numbers
are visible rather than indistinguishable from physical ones. Still device-conditioned and
currently unlabelled:

| parameter | fitted against | belongs to |
|---|---|---|
| all 13 layer `backscatter` values | Maestro3 mean A-line | tissue (should be physical) |
| `speckle_decorrelation = 0.12` | Maestro2 2048-A-line grain | instrument |
| `ms_fraction`, `ms_length_um` | Maestro3 deep tail | tissue, and λ-dependent |
| `ONH_CUP_DEPTH_RANGE` | 2 disc scans, probably 1 eye | population statistic |

`pivot_distance_um` and `scan_distortion_correction` are already on the instrument side —
follow that pattern.

### R3. ONH interior · **M**

Still too uniform, and it is the most visible remaining anatomy gap. In payoff order:
prelaminar **texture** (currently a smooth exponential fade — needs its own mottling field
and a laterally varying lamina insertion depth), nasal/temporal **asymmetry** (the model is
radially symmetric in `r_disc`), **BMO overhang lip**, and optional peripapillary atrophy.

### R4. `octsim/device_raw.py` — the OCT Denoiser bridge · **M**

Deferred twice. Inverse-CLB warp → uint16 `bscan*.raw` + `.CLB`, so the simulator can feed
the denoiser in its native format. The payoff is unique: **noise-free training targets**,
which no real corpus can provide. `octsim/data/real_maestro.py` has the forward pipeline to
invert.

### R5. Pathology primitives · **M–L**

Drusen, IRF/SRF, PED, GA, ERM, CNV. **Do GA first**: hypertransmission below the atrophy
should fall out of the attenuation physics rather than being painted. If it does, that
validates the attenuation model; if it has to be painted, the model is wrong.

### R6. Remaining metric gaps · **S**

- ~~`shadow_count` is speckle-dominated (13.8 vs 24.3 while `shadow_max_depth` and
  `column_energy_sd` both match). Replace with a vessel-conditioned statistic.~~
  **Largely resolved by R0a**: it was speckle-dominated, and the speckle model was the bug.
  Held-out Maestro2 went from real 42.6 / sim 14.9 to sim 42.4. The training corpus still
  reads 25.5 vs 13.4, so a residual gap remains — but re-measure before redesigning the
  statistic.
- No never-fitted held-out metric subset *within* a device. Every statistic is currently
  both objective and score, so single-device overfitting is still undetectable. Reserve
  e.g. histogram shape and the axial gradient spectrum.

### R7. Housekeeping · **S**

`actions/checkout@v4` and `actions/setup-python@v5` target Node 20, currently force-run on
Node 24. Bump when convenient.

---

## 4. Traps — read before measuring

Round 2 produced several confident, wrong conclusions. All were measurement errors.

1. **Measure contrast through a window wide enough to contain the peak.** A clipped window
   gave "EZ is 5.8 dB brighter than RPE". Measured properly, **RPE is ~3 dB brighter than
   EZ**.
2. **Check which band you actually cropped.** "Sim inner retina is featureless vs real
   laminated" was the **ONL**, not the inner retina. At the correct ILM→OPL band both are
   textured (3 extrema each).
3. **Choroid − RPE is not a penetration probe.** Those bands are ~14 µm of tissue apart, so
   the ratio barely responds to attenuation and reports ~0.1 dB where the effect is >1 dB.
   Reference the **inner retina**.
4. **Measure lamination per column, then reduce.** Averaging A-lines first smears real
   lamination away, and a metric built that way cannot distinguish *flat* from
   *featureless* — which is how round-1 fitting kept erasing layer structure.
5. **Report which optics profile you measured.** The round-1 headline "2.2× too wide
   contrast" was measured with `optics_profile="maestro3"` (a fitted artifact), not the
   shipped default `"literature"`. Literature gives RPE−ONL 16.2 against real 13.8;
   maestro3 gives 31.0.
6. **Flattening must be sub-pixel.** Integer `np.roll` is circular *and* quantised into a
   staircase; it stamped notches into every flattened image and contaminated the choroid and
   speckle statistics measured on them. Fixed, but re-baseline anything measured before it.
7. **Landmark detection is the fragile part.** Failure modes hit in sequence: argmax locking
   onto the DC band, then the ILM specular; "deepest strong peak" resolving to the RPE on
   one image and the EZ on another (the RPE sits at 0.51 of the EZ, right on a 0.5
   threshold); a global row cut failing across 550 µm of sag. Current approach is an FFT
   cross-correlation template with a deepest-strong-peak anchor.

### Bugs found by automated review, worth remembering as classes

- **Circular correlation aliases when the lag window is too wide for the frame.** With
  `2·max_shift+1 > n` the lag slices overlap, `argmax` returns the negative alias, and a
  +125 px sag reads as −260 px. Measured: 375 px of error on a 500-row frame at 2 µm/px,
  5 px after clamping to `(n-1)//2`. Deep frames never reach the bound, so this was
  **display-domain only** — i.e. exactly the held-out corpora.
- **A `None` collection that used to self-generate now means "nothing".** `ChoroidTexture`
  generated its own field when unset; `ChoroidVasculature` takes a list, so unset means a
  flat choroid. `default_retina`, `retina_from_profile` and direct construction all silently
  lost their choroid. Fixed in `RetinaModel.__post_init__` so hand-built models are covered.
- **A validation command that ignores its own held-out result.** `validate_realism.py`
  exited 0 on a cross-device regression. Exit codes are now 0 pass / 1 gate failed /
  2 a requested holdout could not be evaluated.

---

## 5. Running it

```powershell
.venv\Scripts\python.exe -m octsim.gui.app --selftest        # 1 s, exit 0/1
.venv\Scripts\python.exe -m pytest -q                        # 89 tests, ~11 min
.venv\Scripts\python.exe -m pytest -q --ignore=tests/test_calibrate.py   # ~3 min
.venv\Scripts\python.exe tests\test_wavelength.py            # physics gate alone
```

`test_calibrate` is 4 differential-evolution fits — 670 s of the 11 min. Skip it while
iterating; CI runs the full suite on py3.11 and py3.13.

Sim-vs-real, needs the local corpus (never committed):

```powershell
$env:OCT_DATA_ROOT = "...\OCT Data"
.venv\Scripts\python.exe scripts\validate_realism.py `
    --folder "$env:OCT_DATA_ROOT\Maestro3\6mm_1024Aline" `
    --holdout maestro2 --holdout octa500 --holdout octid
```

Training score first, then each held-out corpus. **A change that improves training while
worsening generalization is overfitting** — that comparison is the whole point of the
command, so run it with holdouts or not at all.

---

## 6. Known remaining gaps (visual, unresolved)

- Real inner retina is brighter relative to the outer complex than sim, by ~4 dB in the
  A-line overlay. Now has a number: `band_above_floor_db.onl` is +4.1 dB in sim (§0c).
- Real choroid has more vertical striation than sim's rounder lumens. Now has a number:
  `choroid_dark_frac_4db` 0.004 real vs 0.049 sim (§0c).
- ONH interior too uniform (→ R3).
- **Sim retina sits ~60 µm shallower than real** on the training corpus — `ilm_row` −18.1
  and `rpe_row` −14.8 rows at 3.87 µm/px, moving together, so it is a placement offset
  rather than a thickness error (`thickness_tissue_um` is +20 µm, a separate and smaller
  discrepancy). `SAMPLE_DEPTH_UM = 430.0` in `validate_realism.py:74` is the knob; it was set
  by eye. Cheap to re-derive, and worth doing before R1 so the fit is not compensating for a
  window offset. This is the "retina vertical placement offset" left open in
  `CALIBRATION_ROUTINE.md`.
