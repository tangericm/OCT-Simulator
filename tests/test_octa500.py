"""OCTA-500 integration: bundled artifacts are sane, and the real macular geometry is a
validation reference for the simulator (surfacing the foveal-pit-depth gap).

OCTA-500 is the first *volumetric* dataset here (dense foveal-centred 3 mm cubes with a 3D
layer segmentation), so it pins the real macular thickness profile and FAZ far better than
the sparse 49-B-scan JHU volumes that drive the default shape model.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim.data import octa500 as o5          # noqa: E402
from octsim.data import unlabeled as ul         # noqa: E402


def test_bundled_macula_reference_is_physiological():
    m = o5.bundled_macula()
    # real normal fovea: ~210 um central rising to a ~335 um perifoveal plateau
    assert 195.0 < m.central_thickness_um < 235.0
    assert 315.0 < m.perifoveal_plateau_um < 355.0
    assert 0.58 < m.pit_relative_depth < 0.68          # a deep, physiologic foveal pit
    assert m.central_thickness_um < m.perifoveal_plateau_um
    # radial profile rises monotonically from the fovea outward (smoothed)
    prof = np.array(m.total_retina_thickness_um)
    sm = np.convolve(prof, np.ones(3) / 3, "same")
    assert (np.diff(sm[:len(sm) - 1]) >= -0.5).mean() > 0.9
    assert m.curvature_concave_frac > 0.9              # posterior pole is a concave-up bowl
    assert 250.0 < m.faz_radius_um_median < 420.0      # normal FAZ equivalent radius
    assert m.n_eyes_normal >= 80 and m.n_eyes_disease >= 5   # geometric filter split


def test_octa500_texture_is_frame_averaged_and_separate():
    p = ul.bundled_texture("octa500")
    # RTVue-XR structural B-scans are frame-averaged -> speckle contrast BELOW single-frame
    assert 0.28 < p.speckle_contrast < 0.42
    assert p.curvature_concave_frac > 0.9
    assert 250.0 < p.total_retina_um_mean < 360.0
    assert p.n_images > 500
    # it must NOT be pooled into the single-frame healthy default (different acquisition regime)
    default = ul.bundled_texture()
    assert p.speckle_contrast < default.speckle_contrast - 0.08
    assert default.n_images == 4945                    # OCTA-500 deliberately excluded
    assert default.speckle_contrast > 0.45             # single-frame Rayleigh regime preserved


def test_faz_avascular_zone_matches_octa500():
    from octsim.anatomy import retina as R
    m = o5.bundled_macula()
    # the simulator's avascular-zone radius tracks the measured OCTA-500 median
    assert abs(R.FAZ_RADIUS_UM - m.faz_radius_um_median) < 60.0
    # a randomized macula rings the fovea: no vessel core inside the avascular zone
    rng = np.random.default_rng(0)
    fov = 6000.0
    mdl = R.random_retina(rng, location="macula", lateral_fov_um=fov)
    cx = mdl.center_x_um
    for v in mdl.vessels:
        assert abs(v.x0_um - cx) >= R.FAZ_RADIUS_UM - 1.0


def test_default_shape_model_is_octa500_volumetric():
    from octsim.generate import _resolve_shape_model
    for name in ("normal", "default", "octa500"):
        sm = _resolve_shape_model(name)
        assert "OCTA-500" in sm.source
    # the model's own mean fovea reproduces the real deep pit
    sm = _resolve_shape_model("normal")
    e = np.abs(sm.ecc_grid_um); tot = sm.mean_thickness.sum(axis=0)
    rel = tot[e < 75].mean() / tot[(e >= 1000) & (e <= 1300)].mean()
    assert abs(rel - o5.bundled_macula().pit_relative_depth) < 0.05


def test_default_fovea_matches_octa500_depth():
    """The default (now OCTA-500 volumetric) fovea is deep, matching the real pit — not the
    shallow, parafovea-diluted JHU pit it replaced."""
    from octsim.generate import _resolve_shape_model
    from octsim.calibration import retina_from_shape_model, literature_profile
    from octsim.anatomy import retina as R
    sm = _resolve_shape_model("normal")
    x = np.linspace(0.0, 3000.0, 304)
    bm = [l.name for l in R.RETINA_LAYERS].index("RPE") + 1
    ratios = []
    for seed in range(12):
        mdl = retina_from_shape_model(sm, np.random.default_rng(seed), optics=literature_profile(),
                                      location="macula", lateral_fov_um=3000.0)
        th = mdl.cross_section(0.0, x).opt_b[bm] - mdl.cross_section(0.0, x).opt_b[0]
        sm_th = np.convolve(th, np.ones(9) / 9, "same")
        fi = int(np.argmin(sm_th[40:264])) + 40
        ecc = np.abs(x - x[fi])
        plat = th[(ecc >= 1000) & (ecc <= 1300)].mean()
        if np.isfinite(plat) and plat > 0:
            ratios.append(th[ecc < 75].mean() / plat)
    sim_rel = float(np.mean(ratios))
    real_rel = o5.bundled_macula().pit_relative_depth
    assert abs(sim_rel - real_rel) < 0.12          # deep, physiologic pit
    assert sim_rel < 0.80                          # clearly deeper than the old JHU ~0.86


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} OCTA-500 checks passed")
