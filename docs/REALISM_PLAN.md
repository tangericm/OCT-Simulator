# octsim — audit findings & realism plan

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
- **vessel shadow starts at the lumen center**, not the posterior wall; OD unrelated to caliber.
- beam PSF uses the **intensity** profile on the complex field (should be amplitude → √2 in
  effective lateral resolution). (User-flagged; otherwise the fixed-spot model is correct.)

## Realism strategy (prioritized)

| # | area | change | effort/impact |
|---|------|--------|------|
| 1 | vessels | ✅ anchor depth to local ILM/NFL (`ctx.opt_b`); clamp masks ≥ ILM; shadow from posterior wall, caliber-scaled | med / **high** |
| 2 | geometry | ✅ per-boundary low-freq undulation (RMS few µm, corr len 350–600 µm), partly shared across neighbours (`BoundaryUndulation`) | low / **high** |
| 3 | texture | ✅ per-layer low-freq multiplicative mottling under the speckle (`texture_contrast`) + specular-interface break-up (`interface_jitter`) | low / med |
| 4 | directionality | ✅ EZ/IZ Stiles–Crawford `floor+(1-floor)·exp(-(θ/θ0)²)` on local tilt, Henle angular-sign wedge on the ONL (`henle_sign`), stable RPE (`_directional_gain`) | med / med |
| 5 | variation | ✅ asymmetric pit (skewed walls), global tilt, contour on anatomic centre, per-eye texture/henle/curvature | low / med |
| 6 | device | default sensitivity roll-off; DC band; stationary fixed-pattern lines; flat noise floor | low / med |
| 7 | vitreous | ✅ faint vitreous backscatter floor (`vitreous_backscatter`); posterior-hyaloid membrane still TODO | low / low |
| 8 | vessels | multiple plexus depths ✅ (caliber↔depth), shape jitter; FAZ respected ✅ | low / low |
| 9 | speckle | heavier-tailed (K/gamma) statistics in sparse nuclear layers (INL/ONL) | med / low |
| — | hybrid | optional SDEdit/CycleGAN refinement keeping the sim's exact labels | high / med |

## Performance (11 findings)

- multithread + complex64 the forward/reconstruct FFTs (largest single cost);
- bound the per-layer attenuation `exp` to each layer's depth rows (~L× → ~1×);
- draw only the needed samples for the vessel wall + bound ChoroidTexture to its band;
- split averaging into a deterministic precompute + per-draw speckle tail;
- cache k/P/carrier/Hann; vectorize `_rasterize_labels`; resolve once per dataset.

## Validation (how we'll know it improved)

speckle contrast ≈0.52 (dense) / >0.52 (nuclear); patch autocorrelation FWHM = beam waist;
per-layer dB ladder RPE>EZ≳RNFL/IZ>plexiform>nuclear; foveal morphometry vs normative
(pit ~120 µm, CFT ~270 µm); vessels centred inside NFL/GCL, none in vitreous/FAZ, ~30–60%
sub-vessel shadow; directional checks (flip `pupil_entry_offset` → Henle wedge flips; sweep
tilt → EZ dims, RPE stable). Do NOT use SSIM/PSNR to judge speckle. Ultimately: FID drop and
downstream Dice on real data.

_Note: the audit's `correctness` dimension aborted on a transient 529, so a dedicated
correctness sweep is still owed._
