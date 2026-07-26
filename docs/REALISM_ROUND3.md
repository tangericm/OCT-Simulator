# octsim realism — state after round 2, and the round-3 plan

Pickup document. Round 1 is recorded in `REALISM_PLAN.md`; this supersedes its priority
list. Everything below is merged to `main` as of `85d1f63` (PR #1), CI green on py3.11 and
py3.13, 89 tests.

Read the "traps" section before measuring anything. Most of the wrong turns in round 2 were
bad *measurements*, not bad models, and they are easy to repeat.

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

Ordered by value. Effort in {S, M, L}.

### R1. Joint (µ, backscatter) fit — the real successor to O3 · **L, highest value**

Separating attenuation from backscatter needs both solved together, not µ alone. The
simulator can do this where the Vermeer estimator cannot, because it can *render*: propose
(µ, backscatter) per layer, render, compare against the real A-line through the existing
metric code, iterate. `octsim/calibrate.py` already has the ANCHOR·FIT·JUDGE machinery and
`tests/test_calibrate.py` proves it recovers injected parameters.

Check identifiability per dimension before fitting it — µ and backscatter trade off within
a thin layer, and the round-1 experience is that a 4-parameter toy fit already conflates
adjacent outer bands at 512 px. Start with the layers that are thick and well-separated
(NFL, ONL, choroid) and hold the outer complex fixed.

Success criterion: per-layer µ in 1/mm that is (a) within range of published values and
(b) **improves the held-out score**, not just the training score.

### R2. Finish the device/tissue split + provenance tags · **S, high value**

Was O1; still not done, and §1 shows why it matters most.

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

- `shadow_count` is speckle-dominated (13.8 vs 24.3 while `shadow_max_depth` and
  `column_energy_sd` both match). Replace with a vessel-conditioned statistic.
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
  A-line overlay.
- Real choroid has more vertical striation than sim's rounder lumens.
- ONH interior too uniform (→ R3).
