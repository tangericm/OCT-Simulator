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
    """A retinal vessel embedded in the inner retina, crossing the enface plane.

    Its enface path is a line ``x(y) = x0 + slope·(y - y0)`` over ``|y - y0| <=
    y_half_extent``. Crucially the vessel is anchored to the LOCAL inner-retinal
    surface, not an absolute depth: its centre sits ``plexus_offset_um`` below the
    local ILM (``ctx.opt_b[0]``) so it rides the foveal pit and the dome and never
    floats in the vitreous. Rendered as a vertically-oval hyporeflective lumen with a
    thin hyperreflective wall, plus a posterior shadow that starts at the vessel's
    back wall and deepens with caliber.
    """

    x0_um: float
    y0_um: float
    caliber_um: float
    plexus_offset_um: float = 0.0    # EXTRA depth below "top tucked under the ILM"
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
        z_half = 0.62 * cal
        x_half = 0.5 * cal

        # anchor to the LOCAL ILM depth at this column; place the vessel so its top
        # tucks just under the surface (never in the vitreous), deeper for deep plexi.
        ilm = float(np.interp(xv, x, ctx.opt_b[0])) if ctx.opt_b is not None else depth[0]
        center_z = ilm + z_half + 2.0 * ctx.dz + self.plexus_offset_um

        X = (x[None, :] - xv) / x_half                             # [1, n_x]
        Z = (depth[:, None] - center_z) / z_half                   # [n_depth, 1]
        rr = X ** 2 + Z ** 2
        not_vitreous = depth[:, None] >= ilm                       # hard guard: nothing above ILM

        lumen = (rr <= 1.0) & not_vitreous
        wall = (rr > 1.0) & (rr <= 1.4) & not_vitreous             # thin hyperreflective wall ring
        r[lumen] *= self.lumen_gain
        if self.wall_amp > 0:
            n = int(np.count_nonzero(wall))
            if n:
                g = (ctx.rng.standard_normal(n) + 1j * ctx.rng.standard_normal(n)) / np.sqrt(2.0)
                r[wall] += self.wall_amp * g

        # posterior shadow: begins at the vessel's BACK wall, tapers laterally, and
        # deepens with caliber (longer blood path -> more attenuation). The vertical onset
        # ramps in gradually over ~one caliber (a hard step reads as an unphysical blob edge
        # that is not seen in frame-averaged clinical B-scans).
        od = self.shadow_od * (0.6 + 0.5 * min(cal / 120.0, 1.5))
        lateral = np.exp(-((x - xv) / (0.6 * cal)) ** 2)[None, :]  # [1, n_x]
        onset = max(0.5 * cal, 4.0 * ctx.dz)
        below = np.clip((depth[:, None] - (center_z + z_half)) / onset, 0.0, 1.0)
        r *= 1.0 - (1.0 - np.exp(-od)) * lateral * below


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
    field: object = None         # optional CoherentField -> volume-coherent choroid pattern

    def apply(self, r: np.ndarray, ctx) -> None:
        depth = ctx.depth_axis_um
        x = ctx.x_axis_um
        n_depth, n_x = r.shape
        band = ((depth[:, None] >= self.top_opt[None, :])
                & (depth[:, None] < self.bot_opt[None, :]))
        if not np.any(band):
            return
        dz = ctx.dz
        dx = (x[1] - x[0]) if n_x > 1 else 1.0
        r0 = max(0, int(np.floor(self.top_opt.min() / dz)))
        r1 = min(n_depth, int(np.ceil(self.bot_opt.max() / dz)) + 1)
        gain = np.ones((n_depth, n_x))
        if self.field is not None:
            # volume-coherent: the choroidal vascular pattern is a fixed 3-D structure sampled
            # at this frame's slow position, so it evolves smoothly across a volume.
            gain[r0:r1] = self.field.sample(depth[r0:r1], x, ctx.y_um)
        else:
            # confine the (expensive) smoothing to the choroid band; full-height noise is still
            # drawn so the per-frame RNG stream is stable.
            sub = ctx.rng.standard_normal((n_depth, n_x))[r0:r1]
            try:
                from scipy.ndimage import gaussian_filter
                sub = gaussian_filter(sub, (max(self.blob_um / dz, 1.0),
                                            max(self.blob_um / dx, 1.0)))
            except Exception:
                pass
            sub = (sub - sub.mean()) / (sub.std() + 1e-9)
            gain[r0:r1] = 1.0 + self.contrast * np.clip(sub, -1.5, 1.5)
        r[band] *= gain[band]
