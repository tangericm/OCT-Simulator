"""Tests for layer photometry + device_raw Denoiser bridge."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import device_raw  # noqa: E402
from octsim.data import layer_photometry as lp  # noqa: E402


def test_measure_frame_on_synthetic_labels():
    rng = np.random.default_rng(0)
    db = rng.normal(-20.0, 5.0, size=(200, 64))
    lab = np.zeros((200, 64), dtype=int)
    lab[40:60] = 1
    lab[60:100] = 2
    means, slopes = lp.measure_frame(db, lab, ["A", "B"])
    assert np.isfinite(means["A"]) and np.isfinite(means["B"])


def test_from_simulation_returns_layers():
    photo = lp.from_simulation(n_frames=1, seed=1, instrument="")
    assert photo.n_frames == 1
    assert "RPE" in photo.mean_db
    ratios = lp.backscatter_targets_from_photometry(photo, ref_layer="ONL")
    assert ratios["ONL"] == 1.0
    assert ratios["RPE"] > 0


def test_parafoveal_inner_on_synthetic_labels():
    db = np.full((120, 64), -40.0)
    lab = np.zeros((120, 64), dtype=int)
    # Bright NFL / dim GCL / mid IPL; ILM pit at centre (deeper row)
    for c in range(64):
        ilm = 30 + int(8 * (1.0 - abs(c - 32) / 32.0))  # deepest at col 32
        db[ilm:ilm + 8, c] = -20.0
        lab[ilm:ilm + 8, c] = 1
        db[ilm + 8:ilm + 20, c] = -30.0
        lab[ilm + 8:ilm + 20, c] = 2
        db[ilm + 20:ilm + 30, c] = -25.0
        lab[ilm + 20:ilm + 30, c] = 3
        db[ilm + 55:ilm + 58, c] = -5.0   # fake RPE peak for flatten helpers
    m = lp.measure_parafoveal_inner(db, lab, ["NFL", "GCL", "IPL"],
                                    dz_um=3.87, dx_um=6000.0 / 64,
                                    r_inner_mm=0.5, r_outer_mm=2.5, fovea_col=32)
    assert m["n_cols"] > 0
    assert m["nfl_gcl_contrast_db"] > 5.0
    assert m["nfl_ipl_contrast_db"] > 0.0
    assert m["nfl_thick_px"] == 8.0


def test_parafoveal_column_mask_annulus():
    # 6 mm / 1024 → ~5.86 µm/px; 0.75–1.5 mm → ~128–256 px from centre
    dx = 6000.0 / 1024.0
    fovea = 512
    m = lp.parafoveal_column_mask(1024, fovea, dx, r_inner_mm=0.75, r_outer_mm=1.5)
    assert m.dtype == bool and m.shape == (1024,)
    assert not m[fovea]                                 # pit itself excluded
    assert m[fovea + int(round(1.0e3 / dx))]            # ~1 mm in annulus
    assert not m[fovea + int(round(2.0e3 / dx))]        # 2 mm outside
    assert int(m.sum()) > 100


def test_layer_band_means_from_labels_pools_inner_stack():
    db = np.full((80, 40), -40.0)
    lab = np.zeros((80, 40), dtype=int)
    # names: NFL=1, GCL=2, IPL=3, INL=4, OPL=5, ONL=6, RPE=7
    names = ["NFL", "GCL", "IPL", "INL", "OPL", "ONL", "RPE"]
    db[10:15] = -10.0; lab[10:15] = 1          # NFL bright
    db[15:20] = -20.0; lab[15:20] = 2          # GCL
    db[20:25] = -22.0; lab[20:25] = 3
    db[25:30] = -24.0; lab[25:30] = 4
    db[30:35] = -18.0; lab[30:35] = 5
    db[40:50] = -30.0; lab[40:50] = 6          # ONL
    db[55:60] = -5.0;  lab[55:60] = 7          # RPE
    means = lp.layer_band_means_from_labels(db, lab, names)
    assert abs(means["NFL"] - (-10.0)) < 1e-6
    assert abs(means["RPE"] - (-5.0)) < 1e-6
    assert abs(means["ONL"] - (-30.0)) < 1e-6
    # inner_stack = mean of GCL..OPL pixels
    assert -25.0 < means["inner_stack"] < -15.0


def test_parafoveal_mean_aline_uses_mask_only():
    # Bright strip only in the annulus columns; fovea/far periphery dark.
    H, W = 100, 200
    dz, dx = 4.0, 30.0          # 6 mm FOV
    db = np.full((H, W), -50.0)
    # Fake RPE ridge + ILM pit at centre so find_fovea / flatten work
    for c in range(W):
        rpe = 70 + int(0.0004 * (c - W // 2) ** 2)
        ilm = rpe - 40 + int(12 * np.exp(-((c - W // 2) / 25.0) ** 2))
        db[rpe - 2:rpe + 2, c] = -5.0
        db[ilm:ilm + 3, c] = -12.0
        db[ilm + 3:rpe - 2, c] = -25.0
    fovea = lp.find_fovea_column(db, dz, dx)
    mask = lp.parafoveal_column_mask(W, fovea, dx)
    # Boost only parafoveal tissue so the masked mean sits above full-FOV mean
    db[:, mask] += 8.0
    z, prof, meta = lp.parafoveal_mean_aline(db, dz, dx, fovea_col=fovea, flatten=False)
    assert meta["n_cols"] == int(mask.sum()) > 0
    assert np.nanmean(prof) > np.nanmean(db.mean(axis=1))


def test_device_raw_roundtrip_uint16_shape():
    with tempfile.TemporaryDirectory() as td:
        paths = device_raw.export_simulation(
            td, n_frames=1, seed=0, instrument="", noise_free=True, lines=64)
        assert len(paths) == 1
        raw = np.fromfile(paths[0], dtype=np.uint16)
        assert raw.size == 2048 * 64
        assert os.path.isfile(os.path.join(td, "sim.CLB"))


def test_inverse_clb_identity():
    lut = np.linspace(0.0, 1.0, 128)
    spec = np.linspace(1.0, 2.0, 128)
    back = device_raw.inverse_clb_warp(spec, lut)
    assert np.allclose(back, spec, atol=1e-6)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
