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


def test_volume_anatomy_recedes_across_frames():
    # the foveal pit is deepest near the central frame and recedes toward the slow-axis edges
    cfg = _cfg(anatomy="macula", scan_pattern="volume", n_frames=9, slow_fov_um=2000.0,
               lateral_fov_um=6000.0)
    res = simulate(cfg, seed=6)
    lab = res.label_map                                   # [F, nz, nx]
    cen = lab.shape[2] // 2
    inner = np.isin(lab, [1, 2, 3, 4, 5]).sum(axis=1)[:, cen]   # inner-retina px at centre/frame
    assert inner.argmin() in range(3, 6)                 # thinnest (pit) in the middle frames
    assert inner[0] > inner.min() + 5                    # thicker (parafoveal) at both edges
    assert inner[-1] > inner.min() + 5


def test_coherent_texture_field_is_smooth_in_y():
    # the volume-coherent texture field decorrelates with slow-axis distance (no flicker)
    from octsim.sample import CoherentField
    f = CoherentField(np.random.default_rng(0), corr_y_um=420.0, contrast=0.3)
    z = np.linspace(0.0, 300.0, 80); x = np.linspace(0.0, 6000.0, 200)
    g0, g_near, g_far = f.sample(z, x, 0.0), f.sample(z, x, 60.0), f.sample(z, x, 300.0)
    cc = lambda a, b: float((lambda u, v: (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))(
        a.ravel() - a.mean(), b.ravel() - b.mean()))
    assert cc(g0, g_near) > 0.75                          # adjacent frames stay correlated (no flicker)
    assert cc(g0, g_near) > cc(g0, g_far)                 # decorrelates with slow-axis distance


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


def test_spectra_are_retained_and_reconstruct_the_bscan():
    """The image is FFT-grounded: the retained detector spectra I(k,x) reconstruct
    (to float32) to exactly the displayed A-scans — provably the source of the B-scan."""
    from octsim.reconstruct import reconstruct_bscan
    res = simulate(_cfg(anatomy="macula", scan_pattern="bscan"), seed=1)
    assert res.spectra is not None
    n_k = res.resolved.source.n_k
    assert res.spectra.shape == (n_k, 96)             # [pixels/spectrum, lines]
    assert res.k_axis_rad_per_um.shape == (n_k,)
    peak = np.abs(res.volume).max()
    for col in (10, 48, 80):
        recon = np.abs(reconstruct_bscan(res.spectra[:, col]))
        recon_db = np.clip(20 * np.log10(np.maximum(recon.ravel() / peak, 1e-12)), -60, 0)
        assert np.max(np.abs(recon_db - res.ascan_db(col))) < 1e-3


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


def test_every_anatomy_simulates():
    """All scene types render a well-formed B-scan (labels, dB range, retained spectra)."""
    for anatomy in ("macula", "onh", "periphery", "mirror", "paper", "cornea"):
        res = simulate(_cfg(anatomy=anatomy, scan_pattern="bscan"), seed=1)
        n_depth = res.resolved.source.n_k // 2
        assert res.bscan.shape == (n_depth, 96), anatomy
        assert res.bscan_db.max() <= 0.0 + 1e-6, anatomy
        assert res.spectra is not None and res.spectra.shape[1] == 96, anatomy
        assert np.isfinite(res.bscan).all(), anatomy


def test_sample_depth_positions_paper_and_mirror():
    """The adjustable sample position places the surface near sample_depth_um."""
    from octsim.reconstruct import reconstruct_bscan
    for anatomy in ("mirror", "paper"):
        for depth in (250.0, 500.0):
            res = simulate(_cfg(anatomy=anatomy, sample_depth_um=depth, noise_free=True), seed=0)
            aline = np.abs(reconstruct_bscan(res.spectra[:, 0])).ravel()
            surf_um = int(np.argmax(aline)) * res.resolved.source.depth_pixel_um
            assert abs(surf_um - depth) < 60.0, (anatomy, depth, surf_um)


def test_config_serialization_round_trips():
    from octsim.config import config_from_mapping
    cfg = SimConfig(anatomy="onh", scan_pattern="volume", n_frames=6, dispersion=1.5,
                    bandwidth_fwhm_nm=80.0, line_rate_hz=200_000.0, noise_level=0.5)
    back = config_from_mapping(cfg.to_dict())
    assert back.to_dict() == cfg.normalized().to_dict()
    # a resolvable field left at 0 stays derivable after the round-trip
    assert config_from_mapping({"axial_resolution_um": 5.0, "bandwidth_fwhm_nm": 0.0}
                               ).resolve().derived["bandwidth_fwhm_nm"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} Phase D checks passed")
