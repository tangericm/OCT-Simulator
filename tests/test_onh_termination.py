"""A layer terminated at the Bruch's-membrane opening must stop drawing.

Reported from a real-vs-sim review as "the RPE just disappears into the nerve pit". It is a
geometry bug, not a texture one. `opt_b` is cumulative from `opt_b[0]`, and `opt_b[0]`
carries the cup excavation, so a layer whose THICKNESS correctly goes to zero at the BMO
still has its BOUNDARY deposited -- at the cup floor rather than at its own depth. Measured
on one `anatomy="onh"` cross-section, the RPE boundary spans 1763 um across the frame: ~644
um at the edge, ~1230 um at the disc margin, 2407 um inside the disc.

Every terminated layer then laid a Fresnel line into that same collapsed depth bin, and ~10
of them summed coherently into a bright wedge tracing the cup floor -- the band appearing to
dive into the pit instead of ending at the opening.

These tests pin the half that is fixed: interface deposition is now weighted by whether the
layer is thick enough to present a resolvable interface. The positional half (the outer
complex should not be dragged down by the cup at all) is open, and
test_rpe_boundary_is_dragged_into_the_cup records it as a known gap so it cannot be
quietly forgotten -- it will fail, loudly, the day someone fixes it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import SimConfig  # noqa: E402
from octsim.generate import build_scene  # noqa: E402
from octsim.sample import build_reflectivity_from_boundaries  # noqa: E402

FOV, ALINES = 6000.0, 256


def _onh_cross_section():
    cfg = SimConfig(anatomy="onh", location="onh", lateral_fov_um=FOV,
                    lines_per_frame=ALINES, shape_model="normal", seed=7, randomize=True)
    scene = build_scene(cfg, np.random.default_rng(7))
    x = np.linspace(0.0, FOV, ALINES)
    return scene.cross_section(0.0, x, 840.0), x


def test_terminated_layers_have_zero_thickness_inside_the_disc():
    """The premise: thickness termination at the BMO already works."""
    cs, x = _onh_cross_section()
    i = cs.label_names.index("RPE")
    thick = cs.phys_b[i + 1] - cs.phys_b[i]
    absent = np.where(thick < 1e-6)[0]
    assert absent.size > 4, "expected a contiguous zero-thickness band inside the disc"
    # A-lines well outside the absent band must still carry RPE.
    left_out = slice(0, max(1, int(absent[0] * 0.5)))
    right_out = slice(min(ALINES - 1, absent[-1] + (ALINES - absent[-1]) // 2), ALINES)
    assert thick[absent].max() < 1e-6
    assert thick[left_out].min() > 1.0 or thick[right_out].min() > 1.0


def test_collapsed_layers_do_not_stack_their_interfaces():
    """The fix: coincident boundaries must not sum their Fresnel lines.

    Two interfaces of amplitude 0.5. Separated, each lands in its own depth bin and the
    frame peaks at 0.5. Collapsed onto the same bin they used to sum to 1.0, because the
    zero-thickness layer still deposited its line -- and in the optic disc ~10 layers
    collapse together, which is the bright wedge. With the fix the collapsed case peaks at
    0.5 too: only the boundary the surviving neighbours actually share is drawn.
    """
    n_x, n_z = 48, 400
    depth = np.arange(n_z) * 3.87
    x_axis = np.linspace(0.0, 1000.0, n_x)

    def render(thick_um):
        opt_b = np.zeros((3, n_x))
        opt_b[0] = 300.0
        opt_b[1] = opt_b[0] + thick_um
        opt_b[2] = opt_b[1] + 100.0
        phys = opt_b.copy() - 300.0
        return build_reflectivity_from_boundaries(
            opt_b, phys,
            layer_n=np.array([1.38, 1.38]),
            layer_backscatter=np.array([0.0, 0.0]),     # interfaces only, no volume signal
            layer_mu_t=np.array([0.0, 0.0]),
            layer_interface=[0.5, 0.5], top_n=1.336,
            depth_axis_um=depth, x_axis_um=x_axis, beam_waist_um=15.0,
            lateral_extent_um=1000.0, rng=np.random.default_rng(0))

    separated = float(np.abs(render(40.0)).max())
    collapsed = float(np.abs(render(0.0)).max())
    assert separated > 1e-6
    # pre-fix this was ~2x separated (0.5 + 0.5 in one bin); it must now be ~1x
    assert collapsed < 1.2 * separated, (collapsed, separated)


def test_rpe_boundary_stays_at_bmo_plane():
    """R3 positional fix: RPE does not follow the cup excavation.

    Absolute center-vs-edge RPE depth still differs by globe contour sag (~FOV bowl);
    the bug was RPE tracking the *cup* (~500–900 µm extra). Assert RPE's center-edge
    delta is far smaller than the ILM's (which carries the cup).
    """
    cs, x = _onh_cross_section()
    i_rpe = cs.label_names.index("RPE")
    # Cup may be off FOV centre; locate excavation from the ILM itself.
    cup_i = int(np.argmax(cs.opt_b[0]))
    half = max(8, ALINES // 20)
    mid = slice(max(0, cup_i - half), min(ALINES, cup_i + half + 1))
    edge = slice(0, int(0.15 * ALINES))
    ilm_dive = float(cs.opt_b[0][mid].mean() - cs.opt_b[0][edge].mean())
    rpe_dive = float(cs.opt_b[i_rpe][mid].mean() - cs.opt_b[i_rpe][edge].mean())
    assert ilm_dive > 400.0, f"cup not excavating ILM ({ilm_dive:.0f} um)"
    # Contour sag alone is ~0.4–0.5 of the ILM dive on a 6 mm ONH frame; require RPE to
    # shed the cup component (pre-fix RPE dive ≈ ILM dive).
    assert rpe_dive < 0.70 * ilm_dive, (
        f"RPE still tracks the cup: RPE dive {rpe_dive:.0f} um vs ILM dive {ilm_dive:.0f} um")
    canal_i = cs.label_names.index("canal")
    assert float(cs.opt_b[i_rpe][mid].mean()) < float(cs.opt_b[canal_i + 1][mid].mean()) - 100.0, (
        "RPE should sit above the canal floor, not at it")


def test_onh_disc_is_off_center_like_device_a():
    """Real Device A disc frames sit ~1.1 mm off FOV centre; sim must not pin the disc.

    Pre-fix |disc−FOV/2| was ≤80 µm on every seed. Require a clear majority of
    randomized ONH scenes to land ≥400 µm off centre (still keeping the BMO in-frame).
    """
    offs = []
    for seed in range(40):
        cfg = SimConfig(anatomy="onh", location="onh", lateral_fov_um=FOV,
                        lines_per_frame=64, shape_model="normal", seed=seed, randomize=True)
        scene = build_scene(cfg, np.random.default_rng(seed))
        _, disc_x = scene._landmarks()
        offs.append(abs(float(disc_x) - 0.5 * FOV))
    offs = np.asarray(offs, float)
    assert (offs >= 400.0).mean() >= 0.85, offs
    assert offs.min() >= 400.0, offs.min()  # ONH draw is Uniform(500,1500) signed
    assert offs.max() <= 0.5 * FOV - 200.0, offs.max()


def test_cup_is_narrower_than_bmo_and_biased_temporally():
    """Cup FWHM < BMO diameter; cup centre sits temporally of the disc axis."""
    cfg = SimConfig(anatomy="onh", location="onh", lateral_fov_um=FOV,
                    lines_per_frame=ALINES, shape_model="normal", seed=7, randomize=True)
    scene = build_scene(cfg, np.random.default_rng(7))
    x = np.linspace(0.0, FOV, ALINES)
    fovea_x, disc_x = scene._landmarks()
    cs = scene.cross_section(0.0, x, 840.0)
    ilm = cs.opt_b[0]
    # ILM half-max width of the excavation
    edge = np.percentile(np.r_[ilm[:int(0.12 * ALINES)], ilm[int(0.88 * ALINES):]], 20)
    floor = float(ilm.max())
    half = edge + 0.5 * (floor - edge)
    inside = np.where(ilm >= half)[0]
    assert inside.size > 4
    cup_fwhm = float(x[inside[-1]] - x[inside[0]])
    bmo_diam = 2.0 * float(scene.onh_radius_um)
    assert cup_fwhm < 0.90 * bmo_diam, (cup_fwhm, bmo_diam)
    # Temporal = toward fovea
    cup_x = float(x[int(np.argmax(ilm))])
    toward_fovea = np.sign(fovea_x - disc_x) or -1.0
    assert toward_fovea * (cup_x - disc_x) > 0.0, (cup_x, disc_x, fovea_x)
    # BMO termination still centred on disc_x, not the cup
    i_rpe = cs.label_names.index("RPE")
    thick = cs.phys_b[i_rpe + 1] - cs.phys_b[i_rpe]
    absent = np.where(thick < 1.0)[0]
    assert absent.size > 4
    bmo_c = 0.5 * (x[absent[0]] + x[absent[-1]])
    assert abs(bmo_c - disc_x) < 0.25 * scene.onh_radius_um, (bmo_c, disc_x)


def test_terminated_layers_follow_off_center_disc():
    """Zero-thickness RPE band must track the (possibly off-centre) disc, not FOV mid."""
    cfg = SimConfig(anatomy="onh", location="onh", lateral_fov_um=FOV,
                    lines_per_frame=ALINES, shape_model="normal", seed=3, randomize=True)
    scene = build_scene(cfg, np.random.default_rng(3))
    x = np.linspace(0.0, FOV, ALINES)
    _, disc_x = scene._landmarks()
    cs = scene.cross_section(0.0, x, 840.0)
    i = cs.label_names.index("RPE")
    thick = cs.phys_b[i + 1] - cs.phys_b[i]
    absent = np.where(thick < 1.0)[0]
    assert absent.size > 4
    bmo_c = 0.5 * (x[absent[0]] + x[absent[-1]])
    assert abs(bmo_c - disc_x) < 200.0, (bmo_c, disc_x)
    assert abs(disc_x - 0.5 * FOV) > 400.0, disc_x


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} ONH termination checks passed")
