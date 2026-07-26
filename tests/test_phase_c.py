"""Phase C checks: 3D retinal anatomy (foveal pit, layers, vessel shadow, ONH)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import LightSource, reconstruct_bscan  # noqa: E402
from octsim.anatomy import default_retina, random_retina  # noqa: E402
from octsim.anatomy.retina import RetinaModel  # noqa: E402
from octsim.forward import synthesize_bscan_spectra  # noqa: E402
from octsim.sample import build_reflectivity_from_boundaries  # noqa: E402

FOV = 1000.0
INNER = (1, 2, 3, 4)  # GCL, IPL, INL, OPL


def _render(model, *, n_x=256, y=0.0, seed=0, beam=12.0, n_k=1024):
    src = LightSource(n_k=n_k)
    depth = src.depth_axis_um()
    x = np.linspace(0.0, FOV, n_x)
    cross = model.cross_section(y, x)
    r = build_reflectivity_from_boundaries(
        cross.opt_b, cross.phys_b, layer_n=cross.layer_n,
        layer_backscatter=cross.layer_backscatter, layer_mu_t=cross.layer_mu_t,
        layer_interface=cross.layer_interface, top_n=cross.top_n,
        depth_axis_um=depth, x_axis_um=x, beam_waist_um=beam, lateral_extent_um=FOV,
        rng=np.random.default_rng(seed), features=cross.features, y_um=y)
    return np.abs(reconstruct_bscan(synthesize_bscan_spectra(r, src))), depth, x, cross


def test_foveal_pit_thins_inner_layers_at_centre():
    m = RetinaModel(location="macula", center_x_um=FOV / 2, vessels=[])
    x = np.linspace(0.0, FOV, 256)
    cross = m.cross_section(0.0, x)
    centre = int(np.argmin(np.abs(x - FOV / 2)))
    edge = 0
    inner_centre = sum(cross.opt_b[i + 1, centre] - cross.opt_b[i, centre] for i in INNER)
    inner_edge = sum(cross.opt_b[i + 1, edge] - cross.opt_b[i, edge] for i in INNER)
    assert inner_centre < 0.1 * inner_edge, (inner_centre, inner_edge)
    # ONL is preserved / thicker at the centre (index 5)
    onl_centre = cross.opt_b[6, centre] - cross.opt_b[5, centre]
    onl_edge = cross.opt_b[6, edge] - cross.opt_b[5, edge]
    assert onl_centre > onl_edge


def test_photoreceptor_rpe_complex_stays_flat_under_the_pit():
    # the RPE/choroid boundary (posterior anchor) must not move across the pit.
    # Flatten the global contour to isolate the foveal-thinning effect: that now means
    # zeroing BOTH sag terms — the anatomic radius and the fan-beam scan pivot.
    m = RetinaModel(location="macula", center_x_um=FOV / 2, vessels=[],
                    curvature_radius_um=1e12, pivot_distance_um=0.0)
    x = np.linspace(0.0, FOV, 256)
    cross = m.cross_section(0.0, x)
    rpe_choroid = cross.opt_b[10]      # boundary below RPE
    assert np.ptp(rpe_choroid) < 2.0, np.ptp(rpe_choroid)


def test_macula_reconstructs_layered_structure():
    img, depth, x, cross = _render(default_retina(center_x_um=FOV / 2))
    # brightest band is the outer retina (EZ/RPE), deeper than the ILM
    row_energy = img.mean(axis=1)
    surface_row = int(np.argmax(cross.opt_b[0] < depth[-1]) )  # crude; just ensure structure
    peak_row = int(np.argmax(row_energy))
    assert peak_row * (depth[1] - depth[0]) > 300.0     # outer retina is deep
    assert (row_energy > 0.2 * row_energy.max()).sum() > 20   # several bright bands


def test_blood_vessel_casts_a_shadow():
    from octsim.features import BloodVessel
    src_depth = 200.0
    vessel = BloodVessel(x0_um=FOV / 2, y0_um=0.0, caliber_um=90.0)
    m = RetinaModel(location="macula", center_x_um=FOV * 0.15,  # fovea off to the side
                    surface_opt_um=src_depth, vessels=[vessel])
    img, depth, x, cross = _render(m, seed=1)
    col_v = int(np.argmin(np.abs(x - FOV / 2)))
    col_ref = int(np.argmin(np.abs(x - FOV * 0.85)))
    below = depth > (src_depth + 90.0)
    shadowed = img[below, col_v].mean()
    normal = img[below, col_ref].mean()
    assert shadowed < 0.8 * normal, (shadowed, normal)


def test_vessel_never_floats_above_ilm():
    from octsim.features import BloodVessel
    # vessel over the foveal pit, where the ILM dips ~100 µm: it must stay anchored
    # to the local surface and deposit NOTHING in the vitreous above the ILM.
    m = RetinaModel(location="macula", center_x_um=FOV / 2, surface_opt_um=180.0,
                    vessels=[BloodVessel(x0_um=FOV / 2, y0_um=0.0, caliber_um=100.0)])
    x = np.linspace(0.0, FOV, 256)
    cross = m.cross_section(0.0, x)
    depth = LightSource(n_k=1024).depth_axis_um()
    r = build_reflectivity_from_boundaries(
        cross.opt_b, cross.phys_b, layer_n=cross.layer_n,
        layer_backscatter=cross.layer_backscatter, layer_mu_t=cross.layer_mu_t,
        layer_interface=cross.layer_interface, top_n=cross.top_n,
        depth_axis_um=depth, x_axis_um=x, beam_waist_um=12.0, lateral_extent_um=FOV,
        rng=np.random.default_rng(0), features=cross.features, y_um=0.0)
    col = int(np.argmin(np.abs(x - FOV / 2)))
    ilm = cross.opt_b[0, col]
    above = depth < ilm - 2.0
    assert np.max(np.abs(r[above, col])) < 1e-9, "vessel deposited signal in the vitreous"


def test_wide_scan_shows_both_fovea_and_disc():
    # a wide macula scan reaches the optic disc: the unified field renders both, with the
    # disc's canal fill present, a cupped ILM, and the outer complex terminating at the BMO.
    m = RetinaModel(location="macula", center_x_um=3000.0, disc_dx_um=4760.0,
                    onh_radius_um=850.0, onh_cup_depth_um=320.0, vessels=[])
    x = np.linspace(0.0, 9000.0, 700)
    cross = m.cross_section(0.0, x)
    assert "canal" in cross.label_names                    # disc in view -> unified field
    disc_x = 3000.0 + 4760.0
    disc = int(np.argmin(np.abs(x - disc_x)))
    ilm = cross.opt_b[0]
    # The disc is LOCALLY cupped (a central excavation). Check the cup against the average of the two
    # peripapillary rims — the symmetric average cancels the global posterior-pole curvature (which now,
    # realistically, lifts the mid-periphery ~0.9 mm at 9 mm FOV), isolating the true local dip.
    rim_l = int(np.argmin(np.abs(x - (disc_x - 950.0))))
    rim_r = int(np.argmin(np.abs(x - (disc_x + 950.0))))
    rim_baseline = 0.5 * (ilm[rim_l] + ilm[rim_r])
    assert ilm[disc] > rim_baseline + 100.0                # central excavation deeper than the rim
    para = int(np.argmin(np.abs(x - 4500.0)))              # parafovea (RPE intact) for the BMO check
    rpe_i = cross.label_names.index("RPE")
    rpe_thick = cross.opt_b[rpe_i + 1] - cross.opt_b[rpe_i]
    assert rpe_thick[disc] < 0.3 * rpe_thick[para]         # RPE terminates at the BMO


def test_narrow_macula_has_no_disc():
    # a 6 mm macula scan does not reach the disc -> fast macular path, no canal label
    m = RetinaModel(location="macula", center_x_um=3000.0, disc_dx_um=4760.0, vessels=[])
    cross = m.cross_section(0.0, np.linspace(0.0, 6000.0, 300))
    assert "canal" not in cross.label_names
    assert len(cross.label_names) == 13


def test_onh_shows_central_excavation():
    m = RetinaModel(location="onh", center_x_um=FOV / 2, vessels=[])
    x = np.linspace(0.0, FOV, 256)
    cross = m.cross_section(0.0, x)
    centre = int(np.argmin(np.abs(x - FOV / 2)))
    edge = 0
    # the surface is excavated (deeper) at the disc centre
    assert cross.opt_b[0, centre] > cross.opt_b[0, edge] + 100.0


def test_random_retina_is_reproducible_and_varies():
    a1, *_ = _render(random_retina(np.random.default_rng(7), lateral_fov_um=FOV), seed=3)
    a2, *_ = _render(random_retina(np.random.default_rng(7), lateral_fov_um=FOV), seed=3)
    a3, *_ = _render(random_retina(np.random.default_rng(8), lateral_fov_um=FOV), seed=3)
    assert np.allclose(a1, a2)          # same rng seed -> identical anatomy
    assert not np.allclose(a1, a3)      # different seed -> different anatomy


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} Phase C checks passed")
