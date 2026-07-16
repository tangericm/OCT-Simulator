"""Composable anatomical features that edit the reflectivity field.

A feature receives the complex reflectivity ``r[z, x]`` for one rendered
cross-section and a :class:`~octsim.sample.SceneContext` (grids + geometry at the
frame's slow position ``y``), and mutates ``r`` in place. Geometry that changes
*layer thickness* (the foveal pit, the ONH cup) is handled upstream in the anatomy
model's boundary fields; features here are localized inclusions.

The retinal :class:`BloodVessel` is the main one: a vessel runs through the enface
plane at a plexus depth; where it crosses this B-scan it appears as a dark round
lumen and, crucially, **casts a shadow** — everything below it in that column is
attenuated (blood strongly attenuates the beam).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BloodVessel:
    """A retinal vessel crossing the enface plane.

    Its enface path is a line ``x(y) = x0 + slope·(y - y0)`` over ``|y - y0| <=
    y_half_extent``; it sits at optical depth ``depth_opt_um`` (a plexus) with
    diameter ``caliber_um`` and casts a shadow of optical density ``shadow_od``.
    """

    x0_um: float
    y0_um: float
    depth_opt_um: float
    caliber_um: float
    slope: float = 0.0
    y_half_extent_um: float = 1e9
    shadow_od: float = 1.2
    lumen_gain: float = 0.15

    def apply(self, r: np.ndarray, ctx) -> None:
        y = ctx.y_um
        if abs(y - self.y0_um) > self.y_half_extent_um:
            return
        x = ctx.x_axis_um
        if x.size < 2:
            return
        dx = x[1] - x[0]
        dz = ctx.dz
        xv = self.x0_um + self.slope * (y - self.y0_um)
        n_depth, n_x = r.shape

        col = (x - xv) / max(0.5 * self.caliber_um, 1e-6)          # [n_x], normalized
        zc = self.depth_opt_um / dz
        row = (np.arange(n_depth) - zc) / max(0.5 * self.caliber_um / dz, 1e-6)

        # dark lumen (elliptical disk at the vessel)
        disk = (row[:, None] ** 2 + col[None, :] ** 2) <= 1.0
        r[disk] *= self.lumen_gain

        # shadow: attenuate everything below the vessel within its lateral footprint
        under_col = np.abs(col) <= 1.0                            # [n_x]
        below = np.arange(n_depth)[:, None] > zc                  # [n_depth, 1]
        shadow_mask = below & under_col[None, :]
        r *= np.where(shadow_mask, np.exp(-self.shadow_od), 1.0)
