"""Statistical shape model: PCA fit/sample/IO, and data-driven RetinaModel geometry."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import calibration as cal  # noqa: E402
from octsim.config import SimConfig  # noqa: E402
from octsim.generate import simulate  # noqa: E402
from octsim.shape_model import LayerShapeModel, align_and_resample, locate_fovea_um  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "octsim", "data")
BUNDLED = os.path.join(_DATA, "retina_shape_jhu_hc.npz")   # the default normal-eye model


def _toy_stack(n=40, n_layer=7, n_ecc=61, seed=0):
    """A stack of thickness profiles with a foveal dip in the inner layers."""
    rng = np.random.default_rng(seed)
    ecc = np.linspace(-1500, 1500, n_ecc)
    pit = np.exp(-(ecc / 500.0) ** 2)              # 1 at fovea -> 0 outside
    stack = np.zeros((n, n_layer, n_ecc))
    for i in range(n):
        depth = rng.uniform(0.5, 1.0)              # per-eye pit depth
        for j in range(n_layer):
            base = 30.0 + 10 * j
            thin = base * (1 - depth * pit) if j < 4 else base * (1 + 0.3 * depth * pit)
            stack[i, j] = thin + rng.normal(0, 1.5, n_ecc)
    return np.clip(stack, 1.0, None), ecc


def test_fit_sample_and_explained_variance():
    stack, ecc = _toy_stack()
    sm = LayerShapeModel.fit(stack, ecc, [f"L{j}" for j in range(7)], n_modes=6)
    assert sm.mean_thickness.shape == (7, ecc.size)
    assert sm.modes.shape[0] == 6
    # the dominant pit-depth axis should carry most variance
    assert sm.explained_variance_ratio[0] > 0.5
    s = sm.sample(np.random.default_rng(0))
    assert s.shape == (7, ecc.size) and np.isfinite(s).all() and (s > 0).all()


def test_shape_model_round_trips(tmp_path=None):
    import tempfile
    stack, ecc = _toy_stack()
    sm = LayerShapeModel.fit(stack, ecc, [f"L{j}" for j in range(7)])
    p = os.path.join(tempfile.mkdtemp(), "sm.npz")
    sm.save(p)
    q = LayerShapeModel.load(p)
    assert np.allclose(q.mean_thickness, sm.mean_thickness)
    assert np.allclose(q.modes, sm.modes) and q.layer_names == sm.layer_names


def test_locate_fovea_finds_the_dip():
    cols = np.linspace(0, 4000, 200)
    inner = 150 - 90 * np.exp(-((cols - 2200) / 400) ** 2)   # dip at 2200 µm
    f = locate_fovea_um(inner, cols)
    assert abs(f - 2200) < 150


def test_align_and_resample_edge_clamps():
    cols = np.linspace(0, 3000, 120)
    th = np.vstack([np.full(120, 40.0), np.full(120, 60.0)])
    ecc = np.linspace(-2000, 2000, 51)
    out = align_and_resample(th, cols, 1500.0, ecc)
    assert out.shape == (2, 51)
    assert np.isfinite(out).all()                # extrapolated tails are clamped, not NaN


def test_retina_from_shape_model_has_data_pit():
    if not os.path.exists(BUNDLED):
        return
    sm = LayerShapeModel.load(BUNDLED)
    m = cal.retina_from_shape_model(sm, np.random.default_rng(3), lateral_fov_um=6000.0)
    assert m.data_thickness is not None and m.data_thickness.shape[0] == len(m.layers)
    x = np.linspace(0, 6000, 300)
    cross = m.cross_section(0.0, x, 840.0)
    assert np.isfinite(cross.opt_b).all()
    # inner retina (NFL..OPL, boundaries 0..5) thinner at the fovea than at the edge
    c = int(np.argmin(np.abs(x - m.center_x_um)))
    inner_c = cross.opt_b[5, c] - cross.opt_b[0, c]
    inner_e = cross.opt_b[5, 0] - cross.opt_b[0, 0]
    assert inner_c < 0.7 * inner_e, (inner_c, inner_e)


def test_config_default_normal_shape_model():
    # the default shape_model="normal" (JHU healthy-eye) yields a data-driven foveal pit
    cfg = SimConfig(anatomy="macula", lateral_fov_um=6000.0,
                    pixels_per_spectrum=512, lines_per_frame=120)
    assert cfg.shape_model == "normal"
    res = simulate(cfg, seed=2)
    lab = res.labels2d
    prof = np.isin(lab, [1, 2, 3, 4, 5]).sum(axis=0)     # inner-retina pixel thickness
    center = prof[lab.shape[1] // 2]
    edge = np.median(prof[:30])
    assert center < 0.75 * edge, (center, edge)          # foveal thinning present


def test_both_bundled_models_load():
    from octsim.generate import _resolve_shape_model
    for name in ("normal", "jhu", "duke", "dme"):
        sm = _resolve_shape_model(name)
        assert sm.mean_thickness.shape[0] in (7, 8)      # Duke=7, JHU=8 layers


def test_parametric_opt_out():
    cfg = SimConfig(anatomy="macula", shape_model="parametric", lateral_fov_um=4000.0,
                    pixels_per_spectrum=512, lines_per_frame=96)
    res = simulate(cfg, seed=1)                          # must not raise; uses the pit formula
    assert res.bscan.shape[0] == 256


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} shape-model checks passed")
