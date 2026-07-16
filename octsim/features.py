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

    #: brightness of the hyperreflective vessel wall (relative reflectivity) — subtle
    wall_amp: float = 1.6e-2

    def apply(self, r: np.ndarray, ctx) -> None:
        y = ctx.y_um
        if abs(y - self.y0_um) > self.y_half_extent_um:
            return
        depth = ctx.depth_axis_um
        x = ctx.x_axis_um
        if x.size < 2:
            return
        xv = self.x0_um + self.slope * (y - self.y0_um)
        cal = max(self.caliber_um, 1e-6)

        # slightly vertically-oval footprint (real vessels read a touch taller than wide)
        X = (x[None, :] - xv) / (0.5 * cal)                        # [1, n_x]
        Z = (depth[:, None] - self.depth_opt_um) / (0.62 * cal)    # [n_depth, 1]
        rr = X ** 2 + Z ** 2

        lumen = rr <= 1.0
        wall = (rr > 1.0) & (rr <= 1.4)                            # thin hyperreflective wall ring
        r[lumen] *= self.lumen_gain
        if self.wall_amp > 0 and np.any(wall):
            g = (ctx.rng.standard_normal(r.shape)
                 + 1j * ctx.rng.standard_normal(r.shape)) / np.sqrt(2.0)
            r[wall] += self.wall_amp * g[wall]

        # graded posterior shadow: full under the lumen centre, tapering laterally,
        # only below the vessel (smooth, no hard-edged band).
        lateral = np.exp(-((x - xv) / (0.6 * cal)) ** 2)[None, :]  # [1, n_x]
        below = (depth[:, None] > self.depth_opt_um).astype(float)
        r *= 1.0 - (1.0 - np.exp(-self.shadow_od)) * lateral * below


@dataclass
class ChoroidTexture:
    """Large-scale vascular lucencies of the choroid (Sattler/Haller vessels).

    Modulates the reflectivity within the choroid band by a smooth low-spatial-
    frequency random field, producing the characteristic mottled, spongy choroid with
    darker roundish vessel lumina rather than a flat speckle field.
    """

    top_opt: np.ndarray          # [n_x] optical depth of the choroid top (per column)
    bot_opt: np.ndarray          # [n_x] optical depth of the choroid bottom
    blob_um: float = 180.0       # characteristic vessel size
    contrast: float = 0.55       # depth of the lucencies (0..1)

    def apply(self, r: np.ndarray, ctx) -> None:
        depth = ctx.depth_axis_um
        x = ctx.x_axis_um
        n_depth, n_x = r.shape
        top = np.interp(np.arange(n_x), np.arange(n_x), self.top_opt)
        band = (depth[:, None] >= top[None, :]) & (depth[:, None] < self.bot_opt[None, :])
        if not np.any(band):
            return
        dz = ctx.dz
        dx = (x[1] - x[0]) if n_x > 1 else 1.0
        # white noise smoothed to ~blob size -> low-frequency vascular field
        field = ctx.rng.standard_normal((n_depth, n_x))
        try:
            from scipy.ndimage import gaussian_filter
            field = gaussian_filter(field, (max(self.blob_um / dz, 1.0),
                                            max(self.blob_um / dx, 1.0)))
        except Exception:
            pass
        field = (field - field.mean()) / (field.std() + 1e-9)
        gain = 1.0 + self.contrast * np.clip(field, -1.5, 1.5)
        r[band] *= gain[band]
