"""Phase B checks: the continuous motion / phase-drift / washout trajectory."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import LightSource, reconstruct_bscan  # noqa: E402
from octsim.acquisition import Timing  # noqa: E402
from octsim.forward import synthesize_bscan_spectra  # noqa: E402
from octsim.motion import build_motion  # noqa: E402


def _timing(lines=200, frames=1, rate=100_000.0):
    return Timing(line_rate_hz=rate, lines_per_frame=lines, n_frames=frames)


def _motion(timing, level=1.0, cardiac=True, micro=False, seed=0):
    return build_motion(timing, motion_level=level, axial_motion_um=2.0,
                        phase_drift_rad=0.5, cardiac=cardiac, microsaccade=micro,
                        rng=np.random.default_rng(seed))


def test_zero_motion_is_static():
    m = _motion(_timing(), level=0.0)
    assert m.is_static
    assert not np.any(m.dz_um) and not np.any(m.velocity_um_s)


def test_trajectory_is_correlated_across_frames():
    # a volume; the trajectory is ONE continuous process, so a jump across a frame
    # boundary is no larger than the largest jump *within* a frame. Independent
    # per-frame motion would instead jump by ~the full amplitude at each boundary.
    timing = _timing(lines=100, frames=10)
    m = _motion(timing, micro=False)
    amp = float(np.std(m.dz_um))
    lines = timing.lines_per_frame
    diffs = np.abs(np.diff(m.dz_um))
    boundary_idx = [f * lines - 1 for f in range(1, timing.n_frames)]
    boundary = diffs[boundary_idx]
    within = np.delete(diffs, boundary_idx)
    assert boundary.max() < 2.0 * within.max(), (boundary.max(), within.max())
    assert boundary.max() < 0.25 * amp, (boundary.max(), amp)   # no ~amp discontinuity


def test_velocity_matches_gradient():
    timing = _timing(lines=300)
    m = _motion(timing, micro=False)
    assert np.allclose(m.velocity_um_s, np.gradient(m.dz_um, m.times_s), rtol=1e-6, atol=1e-6)


def test_washout_gain_decreases_with_velocity():
    timing = _timing(lines=300)
    m = _motion(timing, level=3.0, micro=True, seed=2)
    src = LightSource(n_k=512)
    g = m.washout_gain(src.k0, timing.integration_time_s)
    assert g.min() >= 0.0 and g.max() <= 1.0 + 1e-9
    # the fastest-velocity A-scan is washed out more than the slowest
    order = np.argsort(np.abs(m.velocity_um_s))
    assert g[order[-1]] < g[order[0]] + 1e-9


def test_bulk_motion_imprints_per_column_phase_not_magnitude():
    src = LightSource(n_k=1024)
    n_depth = src.n_k // 2
    n_x = 64
    r = np.zeros((n_depth, n_x), dtype=complex)
    p = int(round(400.0 / src.depth_pixel_um))
    r[p, :] = 0.9
    dz = np.linspace(0.0, 0.3, n_x)  # µm of bulk motion across the columns
    b0 = reconstruct_bscan(synthesize_bscan_spectra(r, src))
    b1 = reconstruct_bscan(synthesize_bscan_spectra(r, src, motion_dz_um=dz))
    ph0 = np.angle(b0[p, :])
    ph1 = np.angle(b1[p, :])
    # magnitude preserved; phase varies across columns with the motion
    assert abs(np.abs(b1[p, :]).mean() / np.abs(b0[p, :]).mean() - 1.0) < 0.02
    assert np.std(np.unwrap(ph1)) > 5.0 * np.std(np.unwrap(ph0)) + 0.1


def test_washout_reduces_peak_signal():
    src = LightSource(n_k=1024)
    n_depth = src.n_k // 2
    n_x = 32
    r = np.zeros((n_depth, n_x), dtype=complex)
    p = int(round(400.0 / src.depth_pixel_um))
    r[p, :] = 0.9
    gain = np.full(n_x, 1.0)
    gain[n_x // 2:] = 0.3  # half the columns heavily washed out
    b = reconstruct_bscan(synthesize_bscan_spectra(r, src, washout_gain=gain))
    left = np.abs(b[p, :n_x // 2]).mean()
    right = np.abs(b[p, n_x // 2:]).mean()
    assert right < 0.5 * left, (left, right)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} Phase B checks passed")
