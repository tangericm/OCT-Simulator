"""Lateral speckle grain is a physical length, not a number of A-lines.

This is the same genre of check as ``test_wavelength.py``: it asserts a property the model
must have for a reason outside the corpus it was fitted to, so it cannot be satisfied by
tuning against one device.

Measured across seven Maestro corpora spanning 2.93-23.44 um of A-line pitch, the real
lateral speckle grain is one physical length of ~5 um. The apparent growth at coarse pitch
is ``realism_metrics._acf_fwhm`` bottoming out at its 0.5 px floor, not a wider grain:

    dx (um)   2.93   5.86   5.86   8.79   11.72   17.58   23.44
    grain     5.15   5.56   6.16   5.79    7.85   11.01   14.01
    grain/dx  1.76   0.95   1.05   0.66    0.67    0.63    0.60   <- approaching 0.5

So a scan that samples the same tissue twice as finely must not report a grain twice as
fine. Before the fix it did: the decorrelated speckle component was drawn white *per
column*, giving it a physical width of exactly ``dx``, and the mixture was not
variance-matched, so the effective decorrelated fraction moved with the pitch as well.
The sim measured 8.97 um at dx=5.86 and 2.66 um at dx=2.93 -- a 5.3 um swing across a
2x change in sampling, and the two errors had opposite signs, which is why both stayed
inside the +/-3 um gate while the model underneath was wrong.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import SimConfig, simulate  # noqa: E402
from octsim import realism_metrics as rmet  # noqa: E402
from octsim.sample import (  # noqa: E402
    _beam_variance_keep, _lateral_beam_filter, _lateral_corr_filter,
)

FOV_UM = 3000.0


def _grain(alines: int, seed: int = 7) -> float:
    """Lateral speckle grain (um) of one macular render at the given A-line count."""
    cfg = SimConfig(anatomy="macula", location="macula", shape_model="normal",
                    lateral_fov_um=FOV_UM, lines_per_frame=alines,
                    sample_depth_um=430.0, seed=seed, randomize=True)
    res = simulate(cfg)
    dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
    return rmet.measure(res.volume_db[0], dz, FOV_UM / alines).speckle_grain_lateral_um


def test_lateral_grain_does_not_track_the_a_line_pitch():
    """Doubling the A-line density must not halve the reported speckle grain.

    Measured at these two settings: 6.62 um vs 2.77 um before the fix (a 3.85 um swing,
    failing this bound), 5.62 um vs 4.71 um after it (0.91 um).
    """
    coarse, fine = _grain(384), _grain(768)          # dx 7.81 um vs 3.91 um
    assert abs(coarse - fine) < 2.0, (coarse, fine)


def test_corr_filter_preserves_variance():
    """The correlation filter shapes the speckle without rescaling its power.

    It is normalised to sum(k^2)=1, not sum(k)=1, because its input is zero-mean noise
    whose variance must survive rather than a brightness envelope whose level must.
    """
    rng = np.random.default_rng(0)
    g = (rng.standard_normal((256, 512)) + 1j * rng.standard_normal((256, 512))) / np.sqrt(2.0)
    for sigma_px in (0.5, 1.0, 2.0, 4.0):
        out = _lateral_corr_filter(g, sigma_px)
        ratio = float(np.mean(np.abs(out) ** 2) / np.mean(np.abs(g) ** 2))
        assert 0.9 < ratio < 1.1, (sigma_px, ratio)


def test_corr_filter_sets_the_grain_it_is_asked_for():
    """A larger correlation length must produce a proportionately coarser grain."""
    rng = np.random.default_rng(1)
    g = (rng.standard_normal((512, 2048)) + 1j * rng.standard_normal((512, 2048))) / np.sqrt(2.0)
    grains = []
    for sigma_px in (1.0, 2.0, 4.0):
        inten = np.abs(_lateral_corr_filter(g, sigma_px)) ** 2
        grains.append(rmet._acf_fwhm(inten - inten.mean(), 1))
    assert grains[0] < grains[1] < grains[2], grains
    # roughly linear in sigma, and never below the metric's own 0.5 px floor
    assert 0.9 < (grains[2] / grains[0]) / 4.0 < 1.3, grains
    assert grains[0] > 0.5, grains


def test_beam_variance_keep_matches_the_beam_filter():
    """``_beam_variance_keep`` must equal the variance the beam filter actually removes.

    The mixture weight is only a power fraction if this holds; the beam filter is
    sum-normalised, so it preserves smooth structure but averages uncorrelated speckle
    down by a factor that depends on ``beam_waist_um / dx_um`` -- i.e. on the A-line pitch.
    """
    rng = np.random.default_rng(2)
    g = (rng.standard_normal((512, 2048)) + 1j * rng.standard_normal((512, 2048))) / np.sqrt(2.0)
    p0 = float(np.mean(np.abs(g) ** 2))
    for waist_px in (1.0, 2.0, 3.0, 5.0):
        measured = float(np.mean(np.abs(_lateral_beam_filter(g, waist_px)) ** 2)) / p0
        analytic = _beam_variance_keep(waist_px)
        assert abs(measured - analytic) < 0.02 * max(analytic, 0.05), (
            waist_px, measured, analytic)


def test_beam_variance_keep_is_inert_where_the_filter_is():
    """Below the filter's own 0.5 px guard it is a no-op, so nothing is removed."""
    assert _beam_variance_keep(0.4) == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} speckle-pitch checks passed")
