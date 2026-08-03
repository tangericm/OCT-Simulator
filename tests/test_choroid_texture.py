"""Choroid vasculature texture: overlap composition and dark-frac levers."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim.anatomy.retina import CHOROID_SUBLAYERS, random_choroid_lumens  # noqa: E402
from octsim.features import ChoroidLumen, ChoroidVasculature  # noqa: E402


class _Ctx:
    def __init__(self, depth, x, y_um=0.0):
        self.depth_axis_um = depth
        self.x_axis_um = x
        self.y_um = y_um
        self.dz = float(depth[1] - depth[0])
        self.rng = np.random.default_rng(0)


def test_overlapping_lumens_do_not_compound_darkness():
    """Two co-located voids take min(gain), not gain*gain.

    Multiplicative stacking punched past the −4 dB local-dark gate used by
    ``choroid_dark_frac_4db`` even when each lumen alone stayed above it.
    """
    depth = np.arange(0.0, 200.0, 2.0)
    x = np.linspace(0.0, 400.0, 80)
    top = np.full(x.size, 40.0)
    bot = np.full(x.size, 160.0)
    r = np.ones((depth.size, x.size))
    lu = ChoroidLumen(x0_um=200.0, y0_um=0.0, z_rel=0.5,
                      rx_um=30.0, ry_um=80.0, rz_um=40.0, gain=0.55)
    ChoroidVasculature(top_opt=top, bot_opt=bot, lumens=[lu, lu]).apply(r, _Ctx(depth, x))
    assert float(r.min()) >= 0.54, r.min()


def test_choroid_lumen_gains_stay_moderately_hypo():
    """SD-OCT choroidal blood is moderately hypo; near-black gains fail dark-frac."""
    lumens = random_choroid_lumens(np.random.default_rng(0), lateral_fov_um=6000.0)
    gains = np.array([lu.gain for lu in lumens])
    assert gains.min() >= 0.56 and gains.max() <= 0.74
    aspects = np.array([lu.rz_um / max(lu.rx_um, 1e-6) for lu in lumens])
    assert float(np.median(aspects)) >= 2.0
    # Resolvable calibres (structure-SD uses a 3 px smooth) + packed densities.
    rx = np.array([lu.rx_um for lu in lumens])
    assert float(np.median(rx)) >= 12.0
    dens = {row[0]: row[4] for row in CHOROID_SUBLAYERS}
    assert dens["sattler"] >= 16.0 and dens["haller"] >= 6.0
    assert all(row[4] > 0 for row in CHOROID_SUBLAYERS)


def test_choroid_edge_softness_keeps_visible_walls():
    """Walls sharper than soft 12 µm dark-frac fix; floor held by gain + min-compose."""
    from octsim.features import ChoroidVasculature
    assert 6.0 <= ChoroidVasculature.__dataclass_fields__["edge_um"].default <= 10.0


def test_stroma_mottle_does_not_deepen_lumen_floor():
    """Bright stroma + lumen uses absolute gain floor, not stroma×gain."""
    depth = np.arange(0.0, 200.0, 2.0)
    x = np.linspace(0.0, 400.0, 80)
    top = np.full(x.size, 40.0)
    bot = np.full(x.size, 160.0)
    r = np.ones((depth.size, x.size))

    class _BrightStroma:
        def sample(self, z, x_um, y_um):
            return np.full((z.size, x_um.size), 1.30)

    lu = ChoroidLumen(x0_um=200.0, y0_um=0.0, z_rel=0.5,
                      rx_um=40.0, ry_um=80.0, rz_um=50.0, gain=0.55)
    ChoroidVasculature(
        top_opt=top, bot_opt=bot, lumens=[lu],
        stroma_field=_BrightStroma(), stroma_mix=1.0,
    ).apply(r, _Ctx(depth, x))
    assert float(r.min()) >= 0.54, r.min()
    assert float(r.max()) >= 1.25, r.max()


if __name__ == "__main__":
    test_overlapping_lumens_do_not_compound_darkness()
    test_choroid_lumen_gains_stay_moderately_hypo()
    print("ok")
