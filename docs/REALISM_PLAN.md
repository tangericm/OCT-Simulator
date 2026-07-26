# octsim — audit findings & realism plan

> **Superseded for planning purposes by [`REALISM_ROUND3.md`](REALISM_ROUND3.md).** This
> file remains the record of the round-1 audit and the reasoning behind it; the current
> priority list, the measured results of round 2, and the measurement traps to avoid are
> in the round-3 document.

Living record of the 2026-07 repo audit (40 adversarially-verified findings) and the
retinal-OCT realism strategy (grounded in clinical-appearance research). Drives the
fix batches. Update as items land.

## Why the current output looks fake (root causes)

1. **Vessels float above the layers.** `BloodVessel` is placed at an *absolute* optical
   depth (`depth_opt_um`), not anchored to the local ILM/NFL. Where the ILM dips (foveal
   pit) or curves, the vessel detaches into the vitreous, and its bright wall ring deposits
   speckle into empty space. (features.py — the #1 user complaint, numerically reproduced.)
2. **Rigidly parallel, perfectly smooth boundaries.** Every interface is an exact vertical
   offset of one parabola; no low-frequency undulation. The classic rigid-sim tell.
3. **Uniform per-layer reflectivity.** Each layer = one scalar × attenuation × speckle —
   no intra-layer mottling, no fiber striping, no directional reflectance.
4. **No directional reflectance.** Missing RNFL angular dependence, Henle-layer angular
   sign, photoreceptor Stiles–Crawford (EZ/IZ), and the contrasting angle-stable RPE.
5. **Little per-eye/spatial variation.** Symmetric pit, one curvature, no tilt/asymmetry —
   a corpus looks like one template + speckle.
6. **Device cues absent.** No default sensitivity roll-off, no DC/zero-delay band, no
   stationary fixed-pattern lines; deep structure doesn't fade into a flat floor.

## Verified bugs / physics

- **random_retina drops `melanin` + `water_fraction`** (positional dataclass rebuild) → the
  whole randomized corpus loses RPE/choroid NIR absorption. **Fix:** `dataclasses.replace`.
- **backscatter scaled by `s` not `√s`** (tissue.py): amplitude should scale as `√μs` so
  intensity ∝ μs. **Fix:** `backscatter_ref*np.sqrt(s)`.
- **read noise scales with LO/reference power** → inverts read-limited SNR vs source power.
- **`_seed_key` identical for every sample** in a dataset (spawn index dropped).
- **vessel shadow starts at the lumen center**, not the posterior wall; OD unrelated to caliber. ✅
- ✅ beam PSF used the **intensity** profile `exp(-2t²/w²)` on the complex field, imaging points
  √2 too sharp vs the reported lateral resolution. Now the **amplitude** profile `exp(-t²/w²)`,
  so a point images at intensity FWHM = `waist_to_lateral_fwhm_um(w)` (pinned by a test). The
  fixed-physical-spot sampling itself was already correct (waist in px = waist_um/dx).
- ✅ dead `compensation_phase` removed; `sample_depth_um` now positions paper + cornea too.

## Realism strategy (prioritized)

| # | area | change | effort/impact |
|---|------|--------|------|
| 1 | vessels | ✅ anchor depth to local ILM/NFL (`ctx.opt_b`); clamp masks ≥ ILM; shadow from posterior wall, caliber-scaled | med / **high** |
| 2 | geometry | ✅ per-boundary low-freq undulation (RMS few µm, corr len 350–600 µm), partly shared across neighbours (`BoundaryUndulation`) | low / **high** |
| 3 | texture | ✅ per-layer low-freq multiplicative mottling under the speckle (`texture_contrast`) + specular-interface break-up (`interface_jitter`) | low / med |
| 4 | directionality | ✅ EZ/IZ Stiles–Crawford `floor+(1-floor)·exp(-(θ/θ0)²)` on local tilt, Henle angular-sign wedge on the ONL (`henle_sign`), stable RPE (`_directional_gain`) | med / med |
| 5 | variation | ✅ asymmetric pit (skewed walls), global tilt, contour on anatomic centre, per-eye texture/henle/curvature | low / med |
| 5b | FOV-curvature | ✅ **spherical-cap posterior-pole curvature** `sag = R_eye − √(R_eye²−(FOV/2)²)` (`R_eye_um` fittable) replaces the saturating exp that under-curved wide scans. FOV now drives curvature (3 mm flat → 12 mm bowed); fit against `D_curve` (FOV-normalized RPE curvature coef). Single global R_eye is a known compromise (real curvature *saturates* aspherically flatter than a globe cap). See `docs/CURVATURE_FALLOFF.md` | low / **high** |
| 6 | device | ✅ **depth-signal fall-off now an objective**: `D_falloff` matches the peak-normalized sub-ILM→choroid intensity envelope; fit via unfrozen deep-tissue attenuation (`at_choroid`/`at_outer`) + capped `floor_db` (roll-off stays a frozen measured guardrail — it is near-flat across the shallow retina). Deep background now fades dark like real. DC band / fixed-pattern lines still TODO | low / med |
| 7 | vitreous | ✅ faint vitreous backscatter floor (`vitreous_backscatter`); posterior-hyaloid membrane still TODO | low / low |
| 8 | vessels | multiple plexus depths ✅ (caliber↔depth), shape jitter; FAZ respected ✅ | low / low |
| 9 | speckle | ✅ non-Rayleigh (K-distribution) volume speckle: Gamma-modulate the circular-Gaussian variance (`speckle_shape`=M, mean-preserving, grain-correlated so it survives the PSF), fit in θ. Breaks the D_hist calibration plateau. See `docs/SPECKLE_STATISTICS.md` | med / **high** |
| — | hybrid | optional SDEdit/CycleGAN refinement keeping the sim's exact labels | high / med |

## Performance

Done (output-safe; reproducibility tests still exact):
- ✅ multithread the forward/reconstruct FFTs via `octsim/_fft.py` (`scipy.fft` with
  `workers=-1`, bit-identical to numpy pocketfft; numpy fallback). 0.125→0.034 s on 4
  cores, scales with more.
- ✅ bound the per-layer attenuation `exp` to each layer's occupied depth rows —
  `build_reflectivity` 0.25→0.11 s, every masked value identical.
- ✅ bound the ChoroidTexture + intra-layer-mottling Gaussian filters to their bands
  (full-height noise still drawn, so the RNG stream is stable) — ndimage 0.59→0.20 s.
- ✅ vessel wall draws only the ring samples (earlier batch).

Net: macula 2048×512 ~430→260 ms/B-scan on a 4-core box; larger machines gain more
from the FFT threading. Remaining (lower value; default `averages=1` is raw):
- split averaging into a deterministic base (bounded exps + interfaces) + per-draw
  speckle/texture tail — saves the exps ×(navg−1) on averaged clean-target runs;
- cache k/P/carrier/Hann across a volume's frames; vectorize `_rasterize_labels`.

## Validation (how we'll know it improved)

speckle contrast ≈0.52 (dense) / >0.52 (nuclear); patch autocorrelation FWHM = beam waist;
per-layer dB ladder RPE>EZ≳RNFL/IZ>plexiform>nuclear; foveal morphometry vs normative
(pit ~120 µm, CFT ~270 µm); vessels centred inside NFL/GCL, none in vitreous/FAZ, ~30–60%
sub-vessel shadow; directional checks (flip `pupil_entry_offset` → Henle wedge flips; sweep
tilt → EZ dims, RPE stable). Do NOT use SSIM/PSNR to judge speckle. Ultimately: FID drop and
downstream Dice on real data.

_Note: the audit's `correctness` dimension aborted on a transient 529, so a dedicated
correctness sweep is still owed._
