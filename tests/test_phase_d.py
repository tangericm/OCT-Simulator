"""Phase D checks: config-driven B-scan / volume generation and stochastic datasets."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim.config import SimConfig  # noqa: E402
from octsim.generate import generate_dataset, simulate  # noqa: E402


def _cfg(**kw):
    base = dict(pixels_per_spectrum=1024, lines_per_frame=96, lateral_fov_um=1200.0,
                sample_depth_um=180.0)
    base.update(kw)
    return SimConfig(**base)


def test_single_bscan_shapes_and_labels():
    res = simulate(_cfg(anatomy="macula", scan_pattern="bscan"), seed=1)
    assert not res.is_volume
    assert res.volume.shape[0] == 1
    assert res.bscan.shape == (512, 96)
    assert res.labels2d.max() >= 8            # several retinal layers labelled
    assert res.bscan_db.max() <= 0.0 + 1e-6


def test_volume_has_contiguous_frames_and_slow_axis():
    cfg = _cfg(anatomy="macula", scan_pattern="volume", n_frames=8, slow_fov_um=700.0)
    res = simulate(cfg, seed=2)
    assert res.is_volume and res.volume.shape[0] == 8
    assert res.y_axis_um.size == 8
    # known slow-axis spacing
    pitch = np.diff(res.y_axis_um)
    assert np.allclose(pitch, 700.0 / 7, rtol=1e-6)


def test_volume_motion_is_correlated_across_frames():
    cfg = _cfg(anatomy="macula", scan_pattern="volume", n_frames=10, slow_fov_um=600.0,
               motion_level=1.0, axial_motion_um=3.0, microsaccade=False)
    res = simulate(cfg, seed=5)
    dz = res.motion_dz_um
    assert dz is not None
    lines = cfg.lines_per_frame
    diffs = np.abs(np.diff(dz))
    boundary_idx = [f * lines - 1 for f in range(1, 10)]
    boundary = diffs[boundary_idx]
    within = np.delete(diffs, boundary_idx)
    assert boundary.max() < 2.0 * within.max()      # continuous across frame boundaries


def test_generate_dataset_yields_distinct_reproducible_samples():
    cfg = _cfg(anatomy="macula", scan_pattern="bscan", n_frames=1)
    a = list(generate_dataset(cfg, 4, seed=42))
    b = list(generate_dataset(cfg, 4, seed=42))
    c = list(generate_dataset(cfg, 4, seed=43))
    assert len(a) == 4
    # same master seed -> identical dataset
    assert np.allclose(a[0].bscan, b[0].bscan) and np.allclose(a[3].bscan, b[3].bscan)
    # samples within a dataset are distinct anatomy/speckle realisations
    assert not np.allclose(a[0].bscan, a[1].bscan)
    # different master seed -> different dataset
    assert not np.allclose(a[0].bscan, c[0].bscan)


def test_faster_acquisition_lowers_snr():
    # shot-limited SNR ~ sqrt(T_int): faster line rate -> shorter T_int -> lower SNR
    def deep_bg_ratio(rate):
        cfg = _cfg(anatomy="mirror", scan_pattern="bscan", line_rate_hz=rate,
                   noise_level=1.0, sample_depth_um=400.0)
        res = simulate(cfg, seed=0)
        mag = np.abs(res.bscan)
        return mag.max() / (mag[-60:, :].mean() + 1e-12)   # signal / deep background
    slow = deep_bg_ratio(50_000.0)     # longer T_int
    fast = deep_bg_ratio(400_000.0)    # shorter T_int
    assert fast < slow, (fast, slow)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} Phase D checks passed")
