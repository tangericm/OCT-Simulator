"""Wavelength physics: the simulator must behave correctly at wavelengths it was not tuned at.

Every photometric number in the layer table was obtained by matching a commercial SD-OCT device
B-scan at 840 nm. Those numbers can be right for the wrong reasons — a value that has
absorbed the device's gain and tone curve will still reproduce that device. The check that
cannot be passed by fitting is whether the model reproduces the *known* consequences of
changing the source wavelength, since no 840 nm data constrains them.

The physics, from :mod:`octsim.tissue`:

* scattering falls as ``lambda^-1.1``           -> less attenuation at longer lambda
* melanin absorption falls as ``lambda^-1.5``   -> the RPE blocks less at longer lambda
* water absorption is small until ~1200 nm and then climbs steeply

Together these are why 1050 nm swept-source systems image the choroid far better than
840 nm spectral-domain systems, and why 1310 nm does not keep improving indefinitely.
The rendered images must show that, and the assertions below are deliberately about
ORDERING and SIGN rather than absolute values, because only the 840 nm case is calibrated.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import realism_metrics as rmet, tissue  # noqa: E402
from octsim.config import SimConfig  # noqa: E402
from octsim.generate import simulate  # noqa: E402

#: bandwidths chosen so the transform-limited axial resolution matches at every wavelength
#: (dz ~ lambda^2 / dlambda), isolating the TISSUE response from the source's resolution
_AXIAL_RES_UM = 6.0


def _render(lambda_nm: float, seed: int = 7):
    cfg = SimConfig(
        center_wavelength_nm=lambda_nm,
        bandwidth_fwhm_nm=0.0,              # derive from the axial resolution
        axial_resolution_um=_AXIAL_RES_UM,
        pixels_per_spectrum=2048,
        instrument="",                      # no measured instrument: lambda must stay free
        optics_profile="literature",
        lateral_fov_um=6000.0, lines_per_frame=384,
        sample_depth_um=430.0, seed=seed,
    )
    return simulate(cfg)


def test_tissue_scaling_has_the_right_signs():
    assert tissue.scattering_scale(1050.0) < 1.0 < tissue.scattering_scale(700.0)
    assert tissue.melanin_absorption_scale(1050.0) < 1.0
    assert tissue.melanin_absorption_scale(1310.0) < tissue.melanin_absorption_scale(1050.0)
    # water is a minor term in the NIR window but climbs steeply past ~1200 nm
    assert (tissue.water_absorption_per_um(1310.0)
            > 5.0 * tissue.water_absorption_per_um(1050.0))


def test_axial_resolution_follows_lambda_squared_over_bandwidth():
    a = _render(840.0).resolved
    b = _render(1050.0).resolved
    # both were asked for the same axial resolution, so the DERIVED bandwidth must grow
    # as lambda^2 for that to hold
    ratio = b.source.bandwidth_fwhm_nm / a.source.bandwidth_fwhm_nm
    assert abs(ratio - (1050.0 / 840.0) ** 2) < 0.05, ratio
    assert abs(a.source.axial_resolution_um - b.source.axial_resolution_um) < 0.2


def _choroid_minus_inner_db(res):
    """Choroid level relative to the INNER retina.

    Deliberately not relative to the RPE complex: those two bands are separated by only
    ~14 µm of tissue, so their ratio barely responds to a change in attenuation and reports
    ~0.1 dB where the real effect is >1 dB. Referencing the inner retina integrates the
    attenuation through the entire outer retina and RPE, which is what "penetration" means.
    """
    db = res.volume_db[0]
    dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
    m = rmet.measure(db, dz, 6000.0 / db.shape[1])
    return m.band_db["choroid"] - m.band_db["inner"]


def test_longer_wavelength_penetrates_the_choroid_better():
    """Choroid brightness relative to the inner retina must rise monotonically with lambda.

    Less scattering on the way down and less melanin absorption in the RPE both put more
    light into the choroid. This is the most reported wavelength effect in OCT and nothing
    in the 840 nm calibration constrains it, so it is a genuine test of the tissue model
    rather than of the fit.
    """
    vals = [np.mean([_choroid_minus_inner_db(_render(lam, s)) for s in (7, 19)])
            for lam in (840.0, 1050.0, 1310.0)]
    assert vals[1] > vals[0] + 0.5, vals
    assert vals[2] > vals[1], vals


def test_deep_tail_decays_more_slowly_at_longer_wavelength():
    """The sub-RPE signal must fall off more slowly at 1050 nm than at 840 nm."""
    def decay(lam):
        res = _render(lam)
        db = res.volume_db[0]
        dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
        flat, rows, ref = rmet.flatten_to_rpe(db, dz_um=dz)
        p = flat.mean(axis=1)
        a, b = ref + int(80 / dz), ref + int(360 / dz)
        return (p[a] - p[b]) / ((b - a) * dz)      # dB per µm, positive = decaying
    assert decay(1050.0) < decay(840.0), (decay(840.0), decay(1050.0))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} wavelength checks passed")
