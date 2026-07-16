"""Phase A checks: resolvable config, acquisition timing, fall-off, dispersion,
signal scaling, and the vectorized lateral filter."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import LightSource, reconstruct_bscan, axial_fwhm_um  # noqa: E402
from octsim.config import SimConfig  # noqa: E402
from octsim.forward import synthesize_bscan_spectra  # noqa: E402
from octsim.optics import sensitivity_falloff  # noqa: E402
from octsim.sample import _lateral_beam_filter  # noqa: E402


def _mirror_reflectivity(source, depth_um, n_x=1):
    n_depth = source.n_k // 2
    r = np.zeros((n_depth, n_x), dtype=complex)
    p = int(round(depth_um / source.depth_pixel_um))
    r[p, :] = 0.9
    return r


def test_config_resolves_bandwidth_from_axial_resolution():
    cfg = SimConfig(bandwidth_fwhm_nm=0.0, axial_resolution_um=5.0)
    rc = cfg.resolve()
    assert abs(rc.source.axial_resolution_um - 5.0) < 0.05
    assert rc.derived["bandwidth_fwhm_nm"] > 0


def test_config_resolves_axial_range_to_k_window():
    cfg = SimConfig(axial_range_um=1500.0, pixels_per_spectrum=2048)
    rc = cfg.resolve()
    assert abs(rc.source.max_optical_depth_um - 1500.0) / 1500.0 < 0.02
    # sampling = 2*range/n_k
    assert abs(rc.source.depth_pixel_um - 2 * 1500.0 / 2048) / rc.source.depth_pixel_um < 0.02


def test_config_over_specification_raises():
    # bandwidth and axial_resolution both set but inconsistent
    bad = SimConfig(bandwidth_fwhm_nm=100.0, axial_resolution_um=2.0)
    try:
        bad.normalized()
    except ValueError:
        return
    raise AssertionError("expected ValueError on over-specified axial")


def test_timing_integration_and_frame_rate():
    cfg = SimConfig(scan_pattern="volume", line_rate_hz=100_000.0, lines_per_frame=500,
                    n_frames=10, duty_cycle=0.8)
    rc = cfg.resolve()
    t = rc.timing
    assert abs(t.integration_time_s - 0.8e-5) < 1e-12
    # halving the line rate doubles the integration time
    cfg2 = SimConfig(line_rate_hz=50_000.0, duty_cycle=0.8, lines_per_frame=500)
    assert abs(cfg2.resolve().timing.integration_time_s - 1.6e-5) < 1e-12
    assert t.frame_rate_hz > 0 and t.volume_time_s > 0


def test_falloff_is_minus_6db_at_configured_depth():
    z = np.array([0.0, 500.0])
    h = sensitivity_falloff(z, falloff_6db_um=500.0, max_depth_um=1e9,
                            include_sampling_sinc=False)
    intensity_db = 20 * np.log10(h[1] / h[0])
    assert abs(intensity_db - (-6.02)) < 0.2, intensity_db


def test_forward_falloff_attenuates_deep_signal():
    src = LightSource(n_k=1024)
    shallow = reconstruct_bscan(
        synthesize_bscan_spectra(_mirror_reflectivity(src, 150.0), src, falloff_6db_um=400.0))
    deep = reconstruct_bscan(
        synthesize_bscan_spectra(_mirror_reflectivity(src, 800.0), src, falloff_6db_um=400.0))
    assert np.abs(deep).max() < np.abs(shallow).max()


def test_dispersion_broadens_and_compensation_restores():
    src = LightSource(n_k=2048)
    r = _mirror_reflectivity(src, 500.0)
    coeffs = [20.0]  # a2 in (rad/µm)^-2; ~several rad of quadratic phase across the band
    clean = reconstruct_bscan(synthesize_bscan_spectra(r, src))
    disp = reconstruct_bscan(synthesize_bscan_spectra(r, src, dispersion_coeffs=coeffs))
    comp = reconstruct_bscan(synthesize_bscan_spectra(r, src, dispersion_coeffs=coeffs),
                             source=src, dispersion_coeffs=coeffs)
    f_clean = axial_fwhm_um(clean[:, 0], src)
    f_disp = axial_fwhm_um(disp[:, 0], src)
    f_comp = axial_fwhm_um(comp[:, 0], src)
    assert f_disp > 1.3 * f_clean, (f_clean, f_disp)
    assert abs(f_comp - f_clean) < 0.25 * f_clean, (f_clean, f_comp)


def test_signal_scale_reduces_signal():
    src = LightSource(n_k=1024)
    r = _mirror_reflectivity(src, 400.0)
    full = np.abs(reconstruct_bscan(synthesize_bscan_spectra(r, src, signal_scale=1.0))).max()
    half = np.abs(reconstruct_bscan(synthesize_bscan_spectra(r, src, signal_scale=0.5))).max()
    assert abs(half / full - 0.5) < 0.05, half / full


def test_vectorized_lateral_filter_matches_reference_loop():
    rng = np.random.default_rng(0)
    r = (rng.standard_normal((64, 200)) + 1j * rng.standard_normal((64, 200)))
    waist_px = 6.0
    fast = _lateral_beam_filter(r, waist_px)
    # explicit reference: np.pad reflect + per-row np.convolve valid
    half = int(np.ceil(3.0 * waist_px))
    t = np.arange(-half, half + 1)
    k = np.exp(-2.0 * (t / waist_px) ** 2)
    k = k / k.sum()
    pad = np.pad(r, ((0, 0), (half, half)), mode="reflect")
    ref = np.stack([np.convolve(pad[row], k, mode="valid") for row in range(r.shape[0])])
    assert np.max(np.abs(fast - ref)) < 1e-9, np.max(np.abs(fast - ref))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} Phase A checks passed")
