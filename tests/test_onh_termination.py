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
    mid = slice(int(0.45 * ALINES), int(0.55 * ALINES))
    assert thick[mid].max() < 1e-6, thick[mid].max()
    assert thick[:int(0.15 * ALINES)].min() > 1.0     # present outside the disc


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


def test_rpe_boundary_is_dragged_into_the_cup():
    """KNOWN GAP, recorded deliberately: the positional half of the bug is NOT fixed.

    The cup excavates `opt_b[0]`, and every boundary is cumulative from it, so the RPE
    boundary follows the cup down instead of staying at Bruch's level and ending. Real
    anatomy is the opposite. This asserts the CURRENT broken behaviour so the gap is
    visible in the suite; fixing R3's positional half will fail this test, which is the
    point -- invert it then.
    """
    cs, x = _onh_cross_section()
    i = cs.label_names.index("RPE")
    top = cs.opt_b[i]
    inside = top[int(0.45 * ALINES):int(0.55 * ALINES)].mean()
    outside = top[:int(0.15 * ALINES)].mean()
    assert inside - outside > 500.0, (
        f"RPE boundary now only dives {inside - outside:.0f} um into the cup -- if the "
        f"positional fix has landed, invert this test to assert it stays flat")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} ONH termination checks passed")
