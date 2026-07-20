"""Injected-optics recovery — prove the ANCHOR·FIT·JUDGE loop WITHOUT real data.

Render a corpus at a KNOWN θ*, make it the target, then fit from the literature θ with a tiny
DE budget and assert the machinery recovers θ* (band backscatter ratios + boundary sharpness
within a loose rtol) and lowers D below the literature point. This is milestone 2 of
``docs/CALIBRATION_ROUTINE.md``: it gates trusting the loop on Maestro data.

Runs headless and fast (small source, few CRN seeds, a masked 4-D fit). No PySide6, no OCT Data.
"""

from __future__ import annotations

import numpy as np

from octsim import calibrate as C
from octsim import generate


# fast, pure-synthetic anchor (no real instrument needed for a self-consistency proof). Only
# texture-identifiable, well-separated dims are injected/fit: the RPE peak and the inner-retinal
# brightness are distinct template features; boundary sharpness and texture contrast drive the
# transition width and speckle contrast. (Adjacent outer bands like EZ/IZ conflate with RPE at
# this axial resolution, so they need a larger budget — exercised in the real fit, not this proof.)
# 1024 px so the outer bright complex (ELM/EZ/IZ/RPE) is axially resolvable and its backscatter is
# identifiable — at 512 px it blurs together and the band is unrecoverable (the real fits use 2048).
_SPEC = C.RenderSpec(instrument="", shape_model="octa500", pixels_per_spectrum=1024,
                     lines_per_frame=96, lateral_fov_um=6000.0, averages=1,
                     seeds=(11, 22, 33, 44))
# inject/fit orthogonal, independently-recoverable dims: the outer-band brightness (a distinct
# template peak), the boundary transition width, and the mottle contrast. (Inner-vs-outer
# brightness trade off in a 4-param toy fit, so the inner band is left at literature here.)
_FREE = ["bs_outer", "boundary_sharpness_um", "texture_contrast"]
_INJECT = {"bs_outer": np.log10(2.2), "boundary_sharpness_um": 5.0, "texture_contrast": 0.34}


def _theta_star():
    th = C.theta_literature()
    for k, v in _INJECT.items():
        th[C._index(k)] = v
    return th


def _build_target(th_star):
    prof, glob = C.decode(th_star)
    imgs, resolved = C.render_crn(prof, glob, _SPEC)
    return C.synth_texture(imgs, resolved)


def test_injected_optics_recovery():
    th_star = _theta_star()
    target = _build_target(th_star)

    th_lit = C.theta_literature()
    d_lit = C.evaluate(th_lit, target, _SPEC)["D_total"]

    th_hat, d_hat, incumbent, res = C.run_de(
        target, _SPEC, free_names=_FREE, maxiter=12, popsize=8, seed=0, workers=1)

    # 1) the fit closed most of the gap to the θ* floor (the loop actually optimises)
    assert d_hat < 0.5 * d_lit, f"D barely improved: D_hat={d_hat:.4f} vs D_lit={d_lit:.4f}"

    # 2) the incumbent trajectory is monotone-non-increasing
    best = [row["best_D"] for row in incumbent]
    assert all(b2 <= b1 + 1e-9 for b1, b2 in zip(best, best[1:])), f"non-monotone: {best}"

    # 3) recovered band ratios + boundary sharpness within a loose rtol of θ*. The merged outer
    # band (ELM/EZ/IZ/RPE) is a flatter lever than a single layer, so its recovery is looser at
    # this tiny budget — the proof is the right magnitude/direction, not precision.
    def mult(th, name):
        return 10.0 ** th[C._index(name)]
    assert np.isclose(mult(th_hat, "bs_outer"), mult(th_star, "bs_outer"), rtol=0.45), \
        f"outer backscatter not recovered: {mult(th_hat,'bs_outer'):.2f} vs {mult(th_star,'bs_outer'):.2f}"
    bh = th_hat[C._index("boundary_sharpness_um")]
    assert np.isclose(bh, _INJECT["boundary_sharpness_um"], rtol=0.30), \
        f"boundary sharpness not recovered: {bh:.2f} vs {_INJECT['boundary_sharpness_um']:.2f}"


def test_optics_never_touch_labels():
    """Guardrail: switching the optics profile leaves the label map bit-identical."""
    from octsim.config import SimConfig
    from octsim.calibration import literature_profile

    th = _theta_star()
    prof, _ = C.decode(th)
    cfg = SimConfig(anatomy="macula", shape_model="octa500", pixels_per_spectrum=512,
                    lines_per_frame=64, lateral_fov_um=6000.0)
    a = generate.simulate(cfg, seed=7)
    generate.set_optics_override(prof)
    try:
        b = generate.simulate(cfg, seed=7)
    finally:
        generate.set_optics_override(None)
    assert np.array_equal(a.label_map, b.label_map), "label map changed across optics profile"
    assert float(np.mean(np.abs(a.bscan_db - b.bscan_db))) > 0.05, "optics change had no effect"


def test_decode_encode_roundtrip():
    th = _theta_star()
    prof, glob = C.decode(th)
    th2 = C.encode(prof, glob)
    assert np.allclose(th, th2, atol=1e-6), f"round-trip drift: {np.max(np.abs(th-th2)):.4g}"


# --- non-Rayleigh (K-distribution) speckle: identifiability + recovery -------------------------
_SPEC_K = C.RenderSpec(instrument="", shape_model="octa500", pixels_per_spectrum=1024,
                       lines_per_frame=128, lateral_fov_um=6000.0, averages=1,
                       seeds=tuple(11 + 11 * i for i in range(8)))


def test_speckle_shape_decode_sentinel():
    """The top of the M range decodes to the None sentinel (exact Rayleigh, no draw); a value
    below it decodes to M and encode() inverts both directions."""
    i = C._index("speckle_shape")
    th = C.theta_literature()                       # lit = log10(50) => None (Rayleigh)
    prof0, glob0 = C.decode(th)
    assert prof0.speckle_shape is None
    assert np.isclose(C.encode(prof0, glob0)[i], th[i], atol=1e-9)   # None -> hi round-trips
    th[i] = np.log10(4.0)                            # inject M=4 (super-Rayleigh)
    prof, glob = C.decode(th)
    assert prof.speckle_shape is not None and abs(prof.speckle_shape - 4.0) < 1e-9
    assert np.isclose(C.encode(prof, glob)[i], np.log10(4.0), atol=1e-9)


def test_speckle_shape_recovery():
    """Inject a known K-shape M*, make it the target, and prove the objective (a) responds to M
    with a unique minimum at M* (identifiability, not washed out by the PSF/beam) and (b) is
    recovered by DE from the Rayleigh start — while speckle contrast stays in a sane band."""
    i = C._index("speckle_shape")
    th_star = C.theta_literature()
    th_star[i] = np.log10(3.0)                       # M* = 3
    prof_star, glob_star = C.decode(th_star)
    imgs, resolved = C.render_crn(prof_star, glob_star, _SPEC_K)
    target = C.synth_texture(imgs, resolved)

    # (a) identifiability: D over an M grid has its unique minimum at M*=3, and is NOT flat
    grid = [2.0, 3.0, 4.0, 8.0, 25.0]
    losses = []
    for m in grid:
        th = th_star.copy(); th[i] = np.log10(m)
        losses.append(C.evaluate(th, target, _SPEC_K)["D_total"])
    assert grid[int(np.argmin(losses))] == 3.0, f"min not at M*: {dict(zip(grid, np.round(losses,4)))}"
    assert losses[grid.index(25.0)] > 2.0 * losses[grid.index(3.0)], \
        f"objective too flat in M (param near-inert): {dict(zip(grid, np.round(losses,4)))}"

    # (b) DE from the Rayleigh literature start recovers M* within a loose factor
    d_lit = C.evaluate(C.theta_literature(), target, _SPEC_K)["D_total"]
    th_hat, d_hat, _, _ = C.run_de(target, _SPEC_K, free_names=["speckle_shape"],
                                   maxiter=12, popsize=10, seed=0, workers=1)
    m_hat = 10.0 ** th_hat[i]
    assert d_hat < d_lit, f"fit did not improve: D_hat={d_hat:.4f} vs D_lit={d_lit:.4f}"
    assert 2.0 <= m_hat <= 5.0, f"M not recovered: M_hat={m_hat:.2f} (M*=3)"
    # speckle contrast at the recovered M stays in the absolute sanity envelope
    prof_hat, glob_hat = C.decode(th_hat)
    imgs_h, res_h = C.render_crn(prof_hat, glob_hat, _SPEC_K)
    sc = C.synth_texture(imgs_h, res_h).speckle_contrast
    assert 0.25 <= sc <= 0.60, f"speckle contrast out of gate at M_hat: {sc:.3f}"


# --- FOV-dependent spherical-cap curvature + depth-falloff recovery ---------------------------
def test_curvature_cap_grows_with_fov_and_stays_in_window():
    """The spherical cap gives ~0 sag at 3 mm and grows monotonically with FOV, clamped in-window."""
    from octsim.anatomy.retina import RetinaModel
    sags = {}
    for fov in (3000.0, 9000.0, 12000.0):
        m = RetinaModel(location="macula", vessels=[], curvature_radius_um=12000.0,
                        curvature_sag_um=1800.0, tilt_um_per_mm=0.0, center_x_um=fov / 2)
        x = np.linspace(0.0, fov, 300)
        d = m._contour(x)
        sags[fov] = float(d.max() - d.min())
        assert abs(int(np.argmax(d)) - len(x) // 2) <= 1, "not concave-up (center should be deepest)"
        assert d.max() <= 1800.0 + 1e-6, "curvature left the depth window (clamp failed)"
    assert sags[3000.0] < sags[9000.0] < sags[12000.0], f"sag not FOV-monotone: {sags}"
    assert sags[12000.0] > 1000.0, f"12 mm cap too flat: {sags[12000.0]:.0f} µm (real ~1600)"


def test_r_eye_recovery():
    """Inject a known eye radius, make it the target on a WIDE-FOV spec (where the global bowl
    dominates), and recover R_eye from the geometric D_curve term."""
    spec = C.RenderSpec(instrument="", shape_model="octa500", pixels_per_spectrum=1024,
                        lines_per_frame=96, lateral_fov_um=12000.0, averages=1,
                        seeds=(11, 22, 33, 44))
    i = C._index("R_eye_um")
    th_star = C.theta_literature(); th_star[i] = 16000.0
    prof, glob = C.decode(th_star)
    imgs, res = C.render_crn(prof, glob, spec)
    target = C.synth_texture(imgs, res)
    w = {**C.DEFAULT_WEIGHTS, "curve": 1.5}
    th_hat, d_hat, _, _ = C.run_de(target, spec, free_names=["R_eye_um"], weights=w,
                                   maxiter=12, popsize=8, seed=0, workers=1)
    assert np.isclose(th_hat[i], 16000.0, rtol=0.30), f"R_eye not recovered: {th_hat[i]:.0f} (*=16000)"


def test_deep_falloff_recovery():
    """Inject deeper choroid attenuation, and recover at_choroid from the photometric D_falloff term
    (depth-envelope). Proves the depth-fade lever is identifiable and independent of the ILM→RPE fit.
    Needs a FULL-DEPTH window (2048 spectrum → ~3963 µm) so the 720 µm sub-ILM envelope fits — the
    real Maestro geometry the loop uses; a shallow 1024-spectrum window truncates it."""
    spec = C.RenderSpec(instrument="maestro3", shape_model="octa500", pixels_per_spectrum=2048,
                        lines_per_frame=96, lateral_fov_um=9000.0, averages=1,
                        seeds=(11, 22, 33, 44))
    i = C._index("at_choroid")
    th_star = C.theta_literature(); th_star[i] = np.log10(3.5)
    prof, glob = C.decode(th_star)
    imgs, res = C.render_crn(prof, glob, spec)
    target = C.synth_texture(imgs, res)
    w = {**C.DEFAULT_WEIGHTS, "falloff": 1.5}
    th_hat, d_hat, _, _ = C.run_de(target, spec, free_names=["at_choroid"], weights=w,
                                   maxiter=12, popsize=8, seed=0, workers=1)
    at_hat = 10.0 ** th_hat[i]
    assert 2.0 <= at_hat <= 6.0, f"at_choroid not recovered toward 3.5: {at_hat:.2f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} calibrate checks passed")
