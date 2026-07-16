"""Checks for the realism pass: wavelength-dependent optics + optional averaging."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim.config import SimConfig  # noqa: E402
from octsim.generate import simulate  # noqa: E402
from octsim.tissue import scale_layer, scattering_scale  # noqa: E402


def test_scattering_decreases_with_wavelength():
    assert scattering_scale(1310) < scattering_scale(1050) < scattering_scale(850)
    # longer wavelength -> lower total attenuation for a scattering-dominated layer
    mu_850, _ = scale_layer(4.0, 3e-2, 850, water_fraction=0.6)
    mu_1050, _ = scale_layer(4.0, 3e-2, 1050, water_fraction=0.6)
    assert mu_1050 < mu_850


def test_wavelength_changes_penetration():
    # a uniform scattering slab's reconstructed A-line decays as exp(-2·mu_t·z);
    # longer wavelength -> lower scattering -> slower decay -> deeper penetration.
    from octsim import LightSource, reconstruct_bscan
    from octsim.forward import synthesize_bscan_spectra
    from octsim.sample import build_reflectivity_from_boundaries

    def decay_per_mm(wl):
        src = LightSource(n_k=2048, center_wavelength_nm=wl, bandwidth_fwhm_nm=100)
        depth = src.depth_axis_um()
        n_x = 160
        x = np.linspace(0.0, 1000.0, n_x)
        mu_t, bs = scale_layer(4.0, 3e-2, wl, water_fraction=0.6)
        opt_b = np.array([np.full(n_x, 100.0), np.full(n_x, 1200.0)])
        phys_b = np.array([np.zeros(n_x), np.full(n_x, 1200.0 / 1.38)])
        acc = np.zeros(depth.size)
        for a in range(6):
            r = build_reflectivity_from_boundaries(
                opt_b, phys_b, layer_n=np.array([1.38]), layer_backscatter=np.array([bs]),
                layer_mu_t=np.array([mu_t]), layer_interface=[None], top_n=1.0,
                depth_axis_um=depth, x_axis_um=x, beam_waist_um=20.0, lateral_extent_um=1000.0,
                rng=np.random.default_rng(a))
            acc += np.abs(reconstruct_bscan(synthesize_bscan_spectra(r, src)).mean(axis=1))
        prof = acc / 6
        m = (depth > 200.0) & (depth < 900.0)
        return -np.polyfit(depth[m], np.log(prof[m] + 1e-9), 1)[0]

    assert decay_per_mm(1050) < decay_per_mm(850)
    assert decay_per_mm(1310) < decay_per_mm(1050)


def test_averaging_reduces_speckle_contrast():
    def contrast(navg):
        cfg = SimConfig(anatomy="macula", pixels_per_spectrum=1024, lines_per_frame=96,
                        lateral_fov_um=2000.0, sample_depth_um=180.0, averages=navg)
        res = simulate(cfg, seed=1)
        mag = np.abs(res.bscan)
        band = mag[(res.label_map[0] == 1)]        # within the NFL
        return band.std() / (band.mean() + 1e-12)
    raw = contrast(1)
    avg = contrast(16)
    assert avg < 0.6 * raw, (raw, avg)             # averaging suppresses speckle


def test_averages_default_is_raw():
    assert SimConfig().averages == 1


def test_directional_reflectance_fades_photoreceptors_but_not_rpe():
    """Stiles-Crawford: the EZ/IZ band is brightest where flat (fovea) and fades where
    the layer tilts (periphery); the melanin-rich RPE stays angle-stable."""
    from octsim.anatomy.retina import RetinaModel, RETINA_LAYERS

    m = RetinaModel(location="macula", center_x_um=3000.0, directional_strength=0.7,
                    curvature_radius_um=18000.0, vessels=[])
    x = np.linspace(0.0, 6000.0, 400)
    cross = m.cross_section(0.0, x, 1050.0)
    gain = cross.layer_directional
    assert gain is not None
    centre = int(np.argmin(np.abs(x - 3000.0)))
    edge = 0
    groups = [ly.group for ly in RETINA_LAYERS]
    ez = groups.index("photoreceptor")               # first photoreceptor band (ELM/EZ/IZ)
    rpe = groups.index("rpe")
    # photoreceptor gain drops from the flat fovea to the tilted periphery
    assert gain[ez][centre] > 1.25 * gain[ez][edge], (gain[ez][centre], gain[ez][edge])
    # RPE stays angle-stable across the whole width
    assert np.ptp(gain[rpe]) < 1e-6, np.ptp(gain[rpe])
    # disabling the effect removes the field entirely
    m0 = RetinaModel(location="macula", center_x_um=3000.0, directional_strength=0.0,
                     vessels=[])
    assert m0.cross_section(0.0, x, 1050.0).layer_directional is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} realism checks passed")
