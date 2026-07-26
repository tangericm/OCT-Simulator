"""Data-driven calibration: extraction round-trips, profile I/O, and model application."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import calibration as cal  # noqa: E402
from octsim.anatomy.retina import RetinaModel  # noqa: E402


def test_literature_profile_is_well_formed():
    p = cal.literature_profile()
    assert len(p.layers) == 13          # NFL..choroid, incl. the dark OS band and BM+CC
    assert p.curvature_concave is True
    assert p.boundary_sharpness_um > 0
    names = [ly.name for ly in p.layers]
    assert names[0] == "NFL" and names[-1] == "choroid"


def test_profile_round_trips_through_json(tmp_path=None):
    import tempfile
    p = cal.literature_profile()
    d = tempfile.mkdtemp()
    path = os.path.join(d, "profile.json")
    p.save(path)
    q = cal.CalibrationProfile.load(path)
    assert [l.name for l in q.layers] == [l.name for l in p.layers]
    assert abs(q.curvature_radius_um - p.curvature_radius_um) < 1e-6
    assert q.layers[0].reflectivity == p.layers[0].reflectivity


def _synthetic_segmented_bscan(refl, thick_um, atten, dz=2.0, n_x=120):
    """A labelled AMPLITUDE B-scan with KNOWN per-layer optics so extraction checks out."""
    L = len(refl)
    n_depth = 400
    depth = np.arange(n_depth) * dz
    boundaries = np.zeros((L + 1, n_x))
    boundaries[0] = 100.0
    for j in range(L):
        boundaries[j + 1] = boundaries[j] + thick_um[j]
    rng = np.random.default_rng(0)
    img = np.zeros((n_depth, n_x))
    for j in range(L):
        for c in range(n_x):
            lo = int(boundaries[j, c] / dz); hi = int(boundaries[j + 1, c] / dz)
            z = (np.arange(hi - lo)) * dz
            amp = refl[j] * np.exp(-atten[j] * z)                   # amplitude, attenuated
            rayleigh = rng.rayleigh(1.0, hi - lo)                   # fully-developed speckle
            img[lo:hi, c] = amp * rayleigh                          # magnitude image
    return img, boundaries, depth


def test_extract_recovers_injected_reflectivity_ratios():
    names = ["A", "B", "C"]
    refl = [0.10, 0.02, 0.06]          # bright / dark / medium (amplitude)
    thick = [80.0, 120.0, 100.0]
    atten = [2e-3, 1e-3, 2e-3]
    img, bnd, depth = _synthetic_segmented_bscan(refl, thick, atten)
    prof = cal.extract_profile(img, bnd, depth, layer_names=names)
    r = np.array([ly.reflectivity for ly in prof.layers])
    inj = np.array(refl) / np.median(refl)
    # recovered amplitude RATIOS match the injected ratios (normalised)
    assert np.allclose(r, inj, rtol=0.25), (r, inj)
    # thickness recovered
    assert abs(prof.layers[0].thickness_um - 80.0) < 3.0
    # attenuation ordering preserved (A,C attenuate faster than B)
    a = [ly.attenuation_per_um for ly in prof.layers]
    assert a[0] > a[1] and a[2] > a[1]


def test_extract_measures_concave_curvature():
    # RPE boundary as a shallow bowl: centre deeper than the edges -> concave
    n_x = 200
    x = np.linspace(-1, 1, n_x)
    names = ["A", "B"]
    img = np.ones((300, n_x))
    bnd = np.zeros((3, n_x))
    bnd[0] = 100.0; bnd[1] = 180.0
    bnd[2] = 300.0 + 60.0 * (1.0 - x ** 2)      # centre 60 um deeper than edges (bowl)
    depth = np.arange(300) * 2.0
    prof = cal.extract_profile(img, bnd, depth, layer_names=names)
    assert prof.curvature_concave is True


def test_retina_from_profile_applies_optics():
    p = cal.literature_profile()
    # bump one layer's reflectivity and confirm it propagates into the model
    p.layers[0].reflectivity *= 3.0
    m = cal.retina_from_profile(p, np.random.default_rng(0), lateral_fov_um=2000.0)
    assert isinstance(m, RetinaModel)
    assert m.boundary_sharpness_um == p.boundary_sharpness_um
    assert m.curvature_concave == p.curvature_concave
    # the NFL backscatter reflects the bumped reflectivity (× gain, within per-eye jitter)
    nfl = next(ly for ly in m.layers if ly.name == "NFL")
    assert nfl.backscatter > 1.2 * p.layers[0].reflectivity * p.backscatter_gain / 1.5


def test_average_profiles_pools():
    a = cal.literature_profile()
    b = cal.literature_profile()
    b.boundary_sharpness_um = 9.0
    a.boundary_sharpness_um = 5.0
    avg = cal.average_profiles([a, b])
    assert abs(avg.boundary_sharpness_um - 7.0) < 1e-6
    assert avg.n_scans == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} calibration checks passed")
