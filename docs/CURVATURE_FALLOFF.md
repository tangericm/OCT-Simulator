# FOV-dependent curvature, depth fall-off, and cross-dataset heterogeneity

Closing the "everything looks too uniform" gap: the sim rendered near-identical curvature at every
FOV and a uniformly bright depth column, so a corpus looked like one template + speckle. Fixed by
making curvature FOV-driven, the depth fade an objective term, and letting datasets diverge.

## Measured gaps (real vs sim, before)

| gap | real | sim (before) | verdict |
|---|---|---|---|
| curvature sag vs FOV | 269 µm (3 mm) → 840 µm (9 mm) → 876 µm (12 mm) | ~half, ~flat everywhere | too flat at wide FOV |
| depth fall-off (deep-slab / peak) | 0.05–0.17 (fades) | ~0.27 uniformly | deep signal too bright |
| cross-dataset heterogeneity | corpora differ | near-identical | one shared profile + saturating bowl |

## The three changes

**1. Spherical-cap curvature (`retina.py::_contour`).** Replaced the saturating exp (capped ~520 µm,
which erased the FOV signal) with the true cap `sag = R_eye − √(R_eye²−h²)`, `h ≈ FOV/2`. Sag grows
~`h²/2R`, so FOV drives curvature for free: a 3 mm scan stays flat, a 12 mm scan bows. `R_eye_um` is a
fittable θ dim, identified by the **geometric** `D_curve` term (FOV-normalized RPE curvature coef,
intensity-blind). Default R = 12 mm (globe radius); the fit lands flatter (~30 mm) because real device
B-scans are **aspherically flatter** than a globe cap (posterior-pole flattening + device display) —
the incumbent R=12 mm visibly *over*-curves 9/12 mm, the fit matches real's moderate bow.

**2. Depth fall-off as an objective (`unlabeled.py` + `calibrate.py`).** The old objective ran on
percentile-normalized ILM→RPE texture, blind to absolute depth-brightness. Added a peak-normalized
**depth-intensity envelope** (8 slabs from the ILM down into the choroid) to `TextureProfile`, and a
`D_falloff` MSE term. The honest lever is **tissue attenuation, not instrument roll-off**: the measured
`falloff_6db_um` (~1960 µm) is near-flat across the shallow retina (top ~24 % of the window), so it
stays frozen; the fade is `exp(−μ_t·d)` driven by the now-unfrozen deep bands (`at_choroid`,
`at_outer`, upper bound widened to 6×), plus a `floor_db` upper bound tightened to −54 dB so a bright
noise floor can't fake-fill the deep region. Result: deep-slab/peak now 0.12–0.15 sim vs 0.12–0.15
real on the wide corpora.

**3. Heterogeneity is emergent — no explicit term.** Once curvature is FOV-driven and each corpus fits
its own deep attenuation against its own target, corpora render visibly distinct (flat 3 mm macula,
bowed 12 mm wide, cupped disc, dark-faded deep). The uniformity was an artifact of one shared profile +
a saturating bowl; both are gone. Guardrail: do **not** pool the depth-envelope / curvature fields
across FOVs (`pool_textures` leaves them empty — FOV-averaging would erase exactly the heterogeneity).

## Orthogonality (why the added dims are safe)

`D_curve` is geometric (RPE boundary position) and `D_falloff` is photometric on the sub-RPE envelope;
both are **outside** the ILM→RPE band that carries the K-speckle / histogram / brightness calibration,
so the new levers don't perturb the earlier fit. `R_eye` moves geometry *and* labels together and
consistently — determinism (same θ+seed → same labels) holds, verified by
`test_optics_never_touch_labels`. New tests: `test_curvature_cap_grows_with_fov_and_stays_in_window`,
`test_r_eye_recovery`, `test_deep_falloff_recovery`.

## Deep-research update — conic curvature + the fan-beam finding (2026-07-20)

A single spherical cap over-curved wide-FOV scans (12 mm sim sag ~920 µm vs real ~520 µm). Deep
research (16 sources, adversarially verified) reframed the model:

- **B-scan curvature is largely a fan-beam SCAN-GEOMETRY artifact, not eye anatomy.** OCT A-scans fan
  out from a pivot (~nodal point / axial length) but display on a rectangular raster, curving the
  periphery down; uncorrected OCT radius differs from MRI by **24 ± 46 mm**, distortion-corrected by
  only **−0.23 ± 2.47 mm**. Devices apply distortion/flattening correction, so the *displayed* wide
  B-scan is **flatter** than a raw sphere — which is exactly why a pure cap over-curves.
- **True posterior pole ≈ 12.7 mm radius but ASPHERIC/oblate**; the central macula is ~spherical and
  asphericity matters only at wide FOV (so narrow scans need no conic, wide scans do).
- **Retina sits near the TOP of the axial window** (zero-delay just anterior to the ILM) — confirms
  the sim's shallow placement is correct; do **not** move `sample_depth_um`.
- **Deeper structures appear more curved** (fan-beam arc widens with depth) — the sim applies rigid
  curvature to all layers; a depth-scaled curvature is a future lever.

**Implemented:** the spherical cap became a **conic sag** `z = (h²/R)/(1+√(1−(1+K)(h/R)²))` with a
fittable conic constant `curvature_conic` K (K=0 sphere = the old cap, exactly, label-invariant;
K<−1 flattens the periphery). Now DE fits the near-fovea curvature via apex `R` **and** the wide-FOV
flattening via `K` at once — matching both instead of compromising. The curvature metric
`rpe_curvature_coef` was also made **signed** so `D_curve` penalises a dome/over-curve mismatch, not
just magnitude (targets rebuilt).

## Next realism levers (from the research, not yet built)
- **Multiple-scattering diffuse tail** below dense structures (RPE/choroid) — single-scattering
  degrades with depth × scattering density; adds a depth-growing haze the first-Born model omits.
- **Depth-dependent (fan-beam) curvature** — deeper layers more curved than the ILM.
- **Fan-beam lateral compression** with depth; posterior-pole tilt ~11° + nasal decentration ~0.5 mm.

## Honest limitations
- octc8/octid deep envelopes differ (8-bit clinical-JPEG domain); their curve/falloff terms are
  judge-only and excluded from the pool fit.

## Macula smile / sag-metric fix (2026-07-31)

The fan-beam default and 0.86/0.43 distortion pair were calibrated to **peak-to-peak** RPE sag.
On Device A that p2p (~500–600 µm) is mostly tilt/edge noise; the tilt-free bowl is ~310 µm.
Fan-beam is now off by default; distortion is 0.82/0.67 vs tilt-free sag; `sag_optical_um` is
the quadratic \|a2\|·dz bowl depth.
