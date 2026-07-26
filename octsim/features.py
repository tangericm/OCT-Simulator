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
    #: Whole-blood attenuation coefficient (1/µm, amplitude) at ~840 nm. Scattering-
    #: dominated. 0.016/µm = 16/mm sits in the published range for whole blood in the NIR
    #: and reproduces the deepest shadows measured on the Maestro3 corpus (~-14 dB under a
    #: ~100 µm vessel: 20·log10(exp(-2·0.016·100)) = -13.9 dB).
    mu_blood_per_um: float = 0.016
    #: lumen reflectivity relative to the surrounding tissue — blood scatters MORE, not less
    lumen_scatter: float = 1.35
    #: free multiplier on the blood attenuation, for calibration only (1.0 = physical)
    shadow_scale: float = 1.0
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

        # Blood is a STRONG SCATTERER, which is the reason the shadow exists at all. The
        # lumen is therefore made *more* reflective than the surrounding neural tissue, not
        # less; the dark appearance of a vessel on a B-scan is the attenuation accumulating
        # through the blood column, not an absence of scatterers. Rendering the lumen as a
        # low-reflectivity hole (the previous model) is self-contradictory: it removes the
        # scatterers and then imposes a shadow by hand.
        #
        # Both the intra-lumen decay and the shadow below now come from one coefficient.
        # Vertical chord through the ellipse at each lateral position:
        #     half_chord(x) = z_half · sqrt(1 - ((x-xv)/x_half)²)
        # so the lateral shadow profile falls out of the geometry instead of being a
        # hand-tuned Gaussian, and the caliber dependence is automatic.
        inside_x = np.clip(1.0 - ((x - xv) / x_half) ** 2, 0.0, None)
        half_chord = z_half * np.sqrt(inside_x)                    # [n_x]
        mu = self.mu_blood_per_um * self.shadow_scale

        # attenuation accumulated from the lumen's front wall down to each depth
        front = center_z - half_chord[None, :]
        path = np.clip(depth[:, None] - front, 0.0, 2.0 * half_chord[None, :])
        atten = np.exp(-mu * path)                                 # amplitude, one-way·2 below
        r *= np.where(not_vitreous, atten, 1.0)

        r[lumen] *= self.lumen_scatter
        if self.wall_amp > 0:
            n = int(np.count_nonzero(wall))
            if n:
                g = (ctx.rng.standard_normal(n) + 1j * ctx.rng.standard_normal(n)) / np.sqrt(2.0)
                r[wall] += self.wall_amp * g


@dataclass
class LaminaCribrosa:
    """The laminar / prelaminar block filling the neural canal.

    Real disc B-scans do not show a uniform grey plug inside the canal. They show
    prelaminar tissue that fades with depth, then the lamina cribrosa as a distinctly
    brighter, laterally-striated slab (the collagenous beams between the pores), below
    which the signal terminates. A single flat ``disc_fill_backscatter`` reproduces none
    of that and is the main reason a simulated disc reads as a grey notch.

    Applied as a multiplicative field inside the canal radius, so it composes with the
    canal layer's own backscatter and attenuation.
    """

    disc_x_um: float
    disc_y_um: float
    radius_um: float
    top_opt: np.ndarray            # [n_x] optical depth of the canal top (the cup floor)
    depth_um: float = 260.0        # optical depth below the cup at which the lamina sits
    thickness_um: float = 170.0    # laminar slab thickness (optical µm)
    gain: float = 2.6              # laminar brightness relative to prelaminar tissue
    prelaminar_fade: float = 520.0 # e-folding depth of the prelaminar fade (optical µm)
    pore_um: float = 55.0          # lateral period of the laminar pore striation

    def apply(self, r: np.ndarray, ctx) -> None:
        x = ctx.x_axis_um
        depth = ctx.depth_axis_um
        dy = abs(ctx.y_um - self.disc_y_um)
        if dy >= self.radius_um:
            return
        # in-plane canal radius at this slow position
        rad = self.radius_um * np.sqrt(max(1.0 - (dy / self.radius_um) ** 2, 0.0))
        if rad < 1.0:
            return
        inside = np.clip(1.0 - _smooth01((np.abs(x - self.disc_x_um) - rad * 0.75)
                                         / max(rad * 0.35, 1.0)), 0.0, 1.0)[None, :]
        if not np.any(inside > 0):
            return
        below = depth[:, None] - self.top_opt[None, :]

        # prelaminar tissue: fades with depth below the cup floor
        gain = np.where(below > 0.0, np.exp(-np.clip(below, 0.0, None) / self.prelaminar_fade), 1.0)

        # laminar slab: brighter, with a lateral pore striation
        lam = _smooth01((below - self.depth_um) / 40.0) * (
            1.0 - _smooth01((below - self.depth_um - self.thickness_um) / 40.0))
        stri = 1.0 + 0.45 * np.cos(2.0 * np.pi * x / max(self.pore_um, 1.0))[None, :]
        gain = gain * (1.0 + (self.gain * stri - 1.0) * lam)

        r *= 1.0 + (gain - 1.0) * inside


def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class ChoroidLumen:
    """One choroidal vessel lumen, as a 3-D ellipsoid.

    Stored in the anatomy model and cross-sectioned per frame, so the choroidal vascular
    pattern is a fixed 3-D structure that evolves smoothly through a volume rather than a
    fresh random field per B-scan.

    ``z_rel`` is the position within the choroid band (0 = Bruch's membrane, 1 = the
    choroid-sclera interface), so lumens ride the RPE contour instead of sitting at an
    absolute depth.
    """

    x0_um: float
    y0_um: float
    z_rel: float
    rx_um: float
    ry_um: float
    rz_um: float
    gain: float = 0.25           # residual reflectivity inside the lumen


@dataclass
class ChoroidVasculature:
    """Stratified choroid: choriocapillaris band + Sattler/Haller vessel lumens.

    Replaces a Gaussian-smoothed random gain field. That field could reproduce roughly the
    right *amount* of choroidal variation but never the right *kind*: its structure was
    isotropic and smooth-edged, measuring a ~50 µm lateral correlation length where real
    choroid is sharp-edged, and it produced almost none of the deep dark lumina that
    dominate a real choroid's appearance.

    Lumens are rasterised with a hard edge (a smoothstep one resolution cell wide), which
    is the whole point — a Gaussian blob and a sharp-walled vessel have very different
    intensity histograms even at identical variance.
    """

    top_opt: np.ndarray          # [n_x] optical depth of the choroid top (per column)
    bot_opt: np.ndarray          # [n_x] optical depth of the choroid bottom
    lumens: list                 # list[ChoroidLumen]
    #: 0 disables the legacy in-choroid CC gain; BM+CC is now its own layer (see
    #: octsim.anatomy.retina.RETINA_LAYERS), which is where it belongs.
    cc_thickness_um: float = 0.0
    cc_gain: float = 1.0
    edge_um: float = 6.0              # lumen wall softness

    def apply(self, r: np.ndarray, ctx) -> None:
        depth = ctx.depth_axis_um
        x = ctx.x_axis_um
        n_depth, n_x = r.shape
        top = self.top_opt[None, :]
        bot = self.bot_opt[None, :]
        d = depth[:, None]
        band = (d >= top) & (d < bot)
        if not np.any(band):
            return
        dz = ctx.dz
        r0 = max(0, int(np.floor(self.top_opt.min() / dz)))
        r1 = min(n_depth, int(np.ceil(self.bot_opt.max() / dz)) + 1)
        if r1 <= r0:
            return

        sub = slice(r0, r1)
        thick = np.maximum(bot - top, 1e-6)
        z_rel = (d[sub] - top) / thick                      # 0 at BM, 1 at the sclera
        gain = np.ones((r1 - r0, n_x))

        # choriocapillaris: a thin bright band hugging Bruch's membrane
        cc = (d[sub] - top) < self.cc_thickness_um
        gain = np.where(cc & band[sub], self.cc_gain, gain)

        y = ctx.y_um
        e = max(self.edge_um, 1e-6)
        x_lo, x_hi = (float(x[0]), float(x[-1])) if n_x else (0.0, 0.0)
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        for lu in self.lumens:
            dy = abs(y - lu.y0_um)
            if dy >= lu.ry_um:
                continue                                     # this B-scan misses the vessel
            # Laterally off-frame lumens are EXACTLY a no-op, not an approximation: outside
            # the wall the smoothstep saturates at 1 and the gain factor becomes
            # lu.gain + (1 - lu.gain)*1 == 1. Skipping them is free correctness. It matters
            # because the vessel bed is drawn over a field wide enough for any scan width,
            # so on a 6 mm frame about half the y-surviving lumens contribute nothing and
            # were still costing a full [n_band, n_x] multiply each.
            if lu.x0_um + lu.rx_um <= x_lo or lu.x0_um - lu.rx_um >= x_hi:
                continue
            # elliptical foreshortening: the in-plane cross-section of an ellipsoid
            shrink = np.sqrt(max(1.0 - (dy / lu.ry_um) ** 2, 0.0))
            rx = lu.rx_um * shrink
            rz = lu.rz_um * shrink
            if rx < 1.0 or rz < 1.0:
                continue
            zc = lu.z_rel * thick + top                      # follows the RPE contour
            # normalised radius; distance is measured in µm so the edge width is physical
            rr = np.sqrt(((x[None, :] - lu.x0_um) / rx) ** 2
                         + ((d[sub] - zc) / rz) ** 2)
            # hard-edged: smoothstep across one wall width, not a Gaussian falloff
            t = np.clip((rr - 1.0) * (min(rx, rz) / e) + 1.0, 0.0, 1.0)
            wall = t * t * (3.0 - 2.0 * t)                   # 0 inside, 1 outside
            gain *= lu.gain + (1.0 - lu.gain) * wall

        # r[sub] is a basic slice (a view), so the masked in-place multiply writes through
        r[sub][band[sub]] *= gain[band[sub]]


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
