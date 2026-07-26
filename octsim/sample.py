"""Layered sample model and complex-reflectivity-field generation.

A sample is an ordered stack of :class:`Layer` objects beneath a top medium
(air for cornea/paper/mirror, vitreous for retina). The physical model turns
that stack into a **complex reflectivity field** ``r[z, x]`` sampled on the
reconstruction depth grid — the object the OCT actually measures at its own
resolution. Two contributions:

* **Volume speckle** — within a layer, each resolution cell holds a *circular
  complex-Gaussian* reflectivity whose amplitude scales with the layer's
  ``backscatter`` and a round-trip Beer–Lambert attenuation ``exp(-μ_t·d)``.
  This is the central-limit-theorem limit of summing many random sub-resolution
  scatterers per cell, and it is exactly the fully-developed-speckle model
  (magnitude is Rayleigh, intensity contrast 1). Convolving the field laterally
  by the beam gives speckle its correct grain size and imposes the lateral PSF.
* **Specular interfaces** — every layer boundary adds a deterministic-phase
  reflector at the boundary depth with a Fresnel amplitude
  ``r = (n1 - n2)/(n1 + n2)`` (or an explicit override), rendering as a bright,
  laterally-coherent line (mirrors, crisp cornea/retina boundaries).

Depths are tracked in two units: *physical* depth (µm; drives attenuation and
geometry) and *optical* depth (``Σ n·dz``; drives interferometric phase and so
where a feature reconstructs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

#: Correlation length of the K-distribution Gamma modulator, as a MULTIPLE of the resolution cell.
#: The modulator must vary SLOWLY relative to the reconstruction PSF / lateral beam so it factors
#: out of the convolution as a slow envelope: intensity ≈ tex·|PSF⊛g|² = tex·(Rayleigh) → a genuine
#: compound-K distribution. Grain ≈ 1 cell gets averaged back to Rayleigh (CLT) and makes M inert.
_K_AXIAL_GRAIN_PX = 6.0        # axial: several recon pixels (a few axial resolution cells)
_K_LATERAL_GRAIN_BEAMS = 2.5   # lateral: multiples of the beam waist (slow vs the beam filter)


@dataclass
class Layer:
    """One homogeneous slab.

    Attributes
    ----------
    name : label (also used for the segmentation mask).
    thickness_um : physical thickness.
    n : group refractive index (sets optical path / apparent depth).
    backscatter : volume backscatter amplitude scale (relative units; sets how
        bright the layer's speckle is). 0 disables volume scatter (e.g. a mirror).
    mu_s_per_mm : scattering/attenuation coefficient driving the Beer–Lambert
        depth roll-off (amplitude ``exp(-μ_t·d_physical)``, μ_t in 1/µm).
    density_per_um2 : retained as a physical descriptor; with the complex-Gaussian
        cell model it only gates whether a layer scatters (``> 0``). The speckle
        realisation is set by ``backscatter`` and the random field.
    interface_reflectivity : explicit field reflectivity of this layer's TOP
        boundary. ``None`` -> computed from the Fresnel index mismatch.
    """

    name: str
    thickness_um: float
    n: float = 1.38
    backscatter: float = 1.0e-2
    mu_s_per_mm: float = 3.0
    density_per_um2: float = 0.25
    interface_reflectivity: float | None = None

    @property
    def mu_t_per_um(self) -> float:
        return self.mu_s_per_mm * 1e-3


@dataclass
class Sample:
    """A stack of layers plus geometry.

    ``top_n`` is the medium above the first layer (1.0 air, ~1.336 vitreous).
    ``surface_um`` sets the physical depth of the first interface so the
    structure sits inside the imaging window. ``curvature`` is an optional
    ``x -> Δz`` callable (µm) added to every boundary — cornea's arc, the
    macular contour.
    """

    name: str
    layers: list[Layer]
    top_n: float = 1.0
    surface_um: float = 120.0
    curvature: Callable[[np.ndarray], np.ndarray] | None = None
    _meta: dict = field(default_factory=dict)

    def boundary_depths_um(self, x: np.ndarray) -> np.ndarray:
        """Physical depth of every interface at each ``x`` — shape ``[L+1, len(x)]``."""
        x = np.atleast_1d(x)
        base = np.array([0.0] + [ly.thickness_um for ly in self.layers]).cumsum()
        depths = self.surface_um + base[:, None] + np.zeros((1, x.size))
        if self.curvature is not None:
            depths = depths + self.curvature(x)[None, :]
        return depths

    def optical_boundary_depths_um(self, x: np.ndarray) -> np.ndarray:
        """Optical-path depth of every interface (``Σ n·dz`` from the top)."""
        x = np.atleast_1d(x)
        thick_opt = np.array([0.0] + [ly.n * ly.thickness_um for ly in self.layers])
        base = thick_opt.cumsum()
        opt = self.top_n * self.surface_um + base[:, None] + np.zeros((1, x.size))
        if self.curvature is not None:
            opt = opt + (self.top_n * self.curvature(x))[None, :]
        return opt


@dataclass
class SceneContext:
    """Grids and geometry passed to :class:`~octsim.features.Feature` objects."""

    depth_axis_um: np.ndarray     # [n_depth]
    x_axis_um: np.ndarray         # [n_x]
    y_um: float                   # slow-axis position of this frame
    opt_b: np.ndarray             # [L+1, n_x] optical boundary depths
    dz: float                     # depth pixel (µm)
    rng: np.random.Generator

    def depth_bin(self, opt_depth_um) -> np.ndarray:
        return np.clip(np.round(np.asarray(opt_depth_um) / self.dz).astype(np.intp),
                       0, self.depth_axis_um.size - 1)


def build_reflectivity_from_boundaries(
    opt_b: np.ndarray,
    phys_b: np.ndarray,
    *,
    layer_n: np.ndarray,
    layer_backscatter: np.ndarray,
    layer_mu_t: np.ndarray,
    layer_interface: list,
    top_n: float,
    depth_axis_um: np.ndarray,
    x_axis_um: np.ndarray,
    beam_waist_um: float,
    lateral_extent_um: float,
    rng: np.random.Generator,
    features=None,
    y_um: float = 0.0,
    texture_contrast: float = 0.0,
    interface_jitter: float = 0.0,
    vitreous_backscatter: float = 0.0,
    layer_directional: np.ndarray | None = None,
    boundary_sharpness_um: float = 0.0,
    layer_sharpness_um: np.ndarray | None = None,
    ms_fraction: float = 0.0,
    ms_length_um: float = 120.0,
    speckle_decorrelation: float = 0.0,
    speckle_corr_lateral_um: float = 0.0,
    texture_field: "CoherentField | None" = None,
    speckle: np.ndarray | None = None,
    speckle_decorr: np.ndarray | None = None,
    k_shape: float | None = None,
) -> np.ndarray:
    """Core reflectivity builder from per-column layer boundaries.

    ``opt_b`` / ``phys_b`` are ``[L+1, n_x]`` optical / physical boundary depths (so
    layers may vary in thickness with lateral position — foveal pit, cup, curvature).
    Speckle is a **single** circular-complex-Gaussian draw for the whole frame scaled
    by a per-cell backscatter×attenuation map; attenuation accumulates **per column**
    (so shadows and thinned layers attenuate correctly); optional ``features`` then
    edit the field (vessels + shadows) before the lateral beam convolution.
    """
    depth_axis_um = np.asarray(depth_axis_um, float)
    x_axis_um = np.asarray(x_axis_um, float)
    n_depth = depth_axis_um.size
    n_x = x_axis_um.size
    dz = float(depth_axis_um[1] - depth_axis_um[0]) if n_depth > 1 else 1.0
    dx_um = lateral_extent_um / (n_x - 1) if n_x > 1 else 1.0
    d = depth_axis_um[:, None]
    cols = np.arange(n_x)

    r = np.zeros((n_depth, n_x), dtype=np.complex128)
    backscatter_map = np.zeros((n_depth, n_x))
    cum_mu = np.zeros(n_x)                                  # per-column attenuation above layer i
    L = len(layer_n)
    # per-pixel transition width, filled from each layer's own sharpness as we rasterise it
    sharp_map = (np.full((n_depth, n_x), float(boundary_sharpness_um))
                 if layer_sharpness_um is not None else None)
    for i in range(L):
        top_opt = opt_b[i][None, :]
        bot_opt = opt_b[i + 1][None, :]
        # bound the attenuation exp to the depth rows this layer actually occupies —
        # every masked value is identical, but the exp shrinks from full-frame to a band.
        if layer_backscatter[i] > 0:
            r0 = max(0, int(np.floor(opt_b[i].min() / dz)))
            r1 = min(n_depth, int(np.ceil(opt_b[i + 1].max() / dz)) + 1)
            if r1 > r0:
                ds = d[r0:r1]
                msk = (ds >= top_opt) & (ds < bot_opt)
                d_in_phys = np.clip(ds - top_opt, 0.0, None) / layer_n[i]
                bmap_i = layer_backscatter[i] * np.exp(
                    -(cum_mu[None, :] + layer_mu_t[i] * d_in_phys))
                if layer_directional is not None:   # per-column directional reflectance
                    bmap_i = bmap_i * layer_directional[i][None, :]
                backscatter_map[r0:r1][msk] = bmap_i[msk]
                if sharp_map is not None:
                    sub = sharp_map[r0:r1]
                    sub[msk] = float(layer_sharpness_um[i])
                    sharp_map[r0:r1] = sub

        r_if = layer_interface[i]
        if r_if is None:
            n_above = top_n if i == 0 else layer_n[i - 1]
            r_if = (n_above - layer_n[i]) / (n_above + layer_n[i])
        if abs(r_if) > 1e-6:
            bins = np.clip(np.round(opt_b[i] / dz).astype(np.intp), 0, n_depth - 1)
            # A layer thinner than the depth pixel presents no RESOLVABLE interface, and a
            # layer of zero thickness presents none at all — its neighbours are in contact,
            # so the physical boundary is theirs, not one each.
            #
            # Without this, every layer terminated at the optic disc still deposits its
            # Fresnel line. Inside the disc their boundaries have all collapsed to the same
            # depth (the stack is cumulative from the cup-excavated ILM), so ~10 lines land
            # in one depth bin and sum coherently into a bright wedge tracing the cup floor:
            # the RPE appears to dive into the nerve pit instead of terminating at Bruch's
            # membrane opening, which is what a real disc B-scan shows. Dimming the
            # interfaces was the earlier mitigation; it did not remove the wedge.
            thick_opt = np.abs(opt_b[i + 1] - opt_b[i])
            resolvable = np.clip(thick_opt / max(dz, 1e-9), 0.0, 1.0)
            amp = r_if * np.exp(-cum_mu) * resolvable
            if interface_jitter > 0:  # break the too-perfect lateral coherence
                amp = amp * (1.0 + interface_jitter * _smooth1d(n_x, rng, 90.0 / dx_um))
                amp = amp * np.exp(1j * (0.4 * interface_jitter) * _smooth1d(n_x, rng, 60.0 / dx_um))
            r[bins, cols] += amp

        cum_mu += layer_mu_t[i] * (phys_b[i + 1] - phys_b[i])

    # faint vitreous backscatter above the ILM (nearly transparent, not pure black)
    if vitreous_backscatter > 0:
        above = d < opt_b[0][None, :]
        backscatter_map = np.where(above, vitreous_backscatter, backscatter_map)

    # slow intra-layer mottling under the fine speckle (fiber striping / gradients)
    if texture_contrast > 0:
        tr0 = max(0, int(np.floor(opt_b[0].min() / dz)))
        tr1 = min(n_depth, int(np.ceil(opt_b[-1].max() / dz)) + 1)
        if texture_field is not None:
            # volume-coherent: a fixed 3-D field sampled at this frame's slow position, so the
            # mottling evolves smoothly across frames rather than flickering.
            gain = texture_field.sample(depth_axis_um[tr0:tr1], x_axis_um, y_um)
            backscatter_map[tr0:tr1] = backscatter_map[tr0:tr1] * gain
        else:
            gain = _smooth_gain(n_depth, n_x, rng, texture_contrast,
                                max(18.0 / dz, 1.0), max(70.0 / dx_um, 1.0), band=(tr0, tr1))
            backscatter_map = backscatter_map * gain

    # soften the layer edges: real inter-layer reflectivity transitions are gradual
    # (a biological gradient beyond the axial PSF), not step functions. Smoothing the
    # deterministic amplitude MAP (not the speckle field) blurs the too-crisp band edges
    # while leaving speckle fully developed. Driven by a measured boundary width.
    # A single global width is wrong on its face: inner-retinal reflectivity transitions
    # really are gradual (8-12 µm), but the photoreceptor/RPE transitions are physically
    # sub-micron. Blurring the whole map at the inner-retinal width washes the 11 µm dark
    # outer-segment band into its bright neighbours, which is what collapses the EZ/RPE
    # bimodality into one spike. ``layer_sharpness_um`` gives each layer its own width;
    # the blur is then a per-pixel select between the (few) distinct widths in use.
    if layer_sharpness_um is not None and sharp_map is not None:
        try:
            from scipy.ndimage import gaussian_filter1d
        except Exception as exc:
            raise RuntimeError("per-layer boundary sharpness requires SciPy") from exc
        widths = sorted({float(w) for w in np.asarray(layer_sharpness_um, float)}
                        | {float(boundary_sharpness_um)})
        out = backscatter_map
        for w in widths:
            sig = w / dz
            if sig <= 0.3:
                blurred = backscatter_map
            else:
                blurred = gaussian_filter1d(backscatter_map, sig, axis=0, mode="nearest")
            sel = sharp_map == w
            if sel.any():
                out = np.where(sel, blurred, out)
        backscatter_map = out
    elif boundary_sharpness_um > 0:
        sigma_px = boundary_sharpness_um / dz
        if sigma_px > 0.3:
            try:
                from scipy.ndimage import gaussian_filter1d
                backscatter_map = gaussian_filter1d(backscatter_map, sigma_px, axis=0,
                                                    mode="nearest")
            except Exception:
                pass

    # --- multiple-scattering diffuse pedestal -----------------------------------------
    # The forward model is first-Born single scattering (see octsim.forward), which cannot
    # produce the diffuse tail real OCT shows beneath every strongly-scattering block: the
    # measured mean A-line falls ~7 dB too fast through the choroid, and the deficit is
    # largest directly under the dense inner-plexiform stack and under the RPE — the
    # signature of missing multiple scattering rather than of a wrong backscatter value.
    #
    # Added in POWER, not amplitude, and before the speckle draw. That is the physically
    # correct superposition: a diffuse, phase-randomised contribution is an INDEPENDENT
    # circular-Gaussian, so its variance adds to the ballistic one. E[I] therefore adds
    # while the field stays fully developed, and the K-modulator's E[tex]=1 invariant is
    # untouched.
    #
    # The kernel is a causal exponential tail of length ``ms_length_um``, evaluated as a
    # one-pass IIR recursion (O(n_depth·n_x)) rather than a convolution, plus a lateral
    # blur for the transverse spread of a multiply-scattered path.
    if ms_fraction > 0:
        inten = backscatter_map ** 2
        decay = float(np.exp(-dz / max(ms_length_um, 1e-6)))
        acc = np.zeros(n_x)
        ped = np.empty_like(inten)
        for zi in range(n_depth):
            acc = acc * decay + inten[zi] * (1.0 - decay)
            ped[zi] = acc
        try:
            from scipy.ndimage import gaussian_filter1d
            ped = gaussian_filter1d(ped, max(2.0 * beam_waist_um / dx_um, 1.0),
                                    axis=1, mode="nearest")
        except Exception:
            pass
        backscatter_map = np.sqrt(inten + ms_fraction * ped)

    # unit circular-complex-Gaussian speckle. For a volume the caller passes a field that is
    # CORRELATED across the slow axis (``speckle``), so adjacent frames share speckle instead of
    # boiling; a fresh draw (default) is the independent single-frame case.
    if speckle is not None:
        g = speckle
    else:
        g = (rng.standard_normal((n_depth, n_x))
             + 1j * rng.standard_normal((n_depth, n_x))) / np.sqrt(2.0)
    # --- K-distributed speckle: Gamma-modulate the local circular-Gaussian variance. ---------
    # amplitude = |g|·sqrt(tex), tex ~ Gamma(M, 1/M) with E[tex]=1, so E[|field|²] is UNCHANGED
    # (backscatter map, layer-brightness template, interface reflectors, Beer-Lambert roll-off
    # all carried by the mean → preserved). Intensity contrast rises to sqrt(1+2/M): M→∞ recovers
    # fully-developed Rayleigh (today's look); small M gives the super-Rayleigh heavy tail real
    # OCT shows in sparse layers (ONL/INL/vitreous). ``None`` skips it → byte-identical to before.
    # The modulator is smoothed to the resolution-cell grain so the lateral beam filter / axial PSF
    # don't average it away (that grain-matching is what makes M identifiable), then renormalized
    # to exact per-frame E[tex]=1. Uses a spawned child stream so ``features`` below draw identically
    # whether K-mode is on or off (clean A/B; label map is upstream and unaffected regardless).
    k_tex = None
    if k_shape is not None and np.isfinite(k_shape):
        try:
            kc = rng.spawn(1)[0]
        except Exception:
            kc = rng
        M = float(k_shape)
        tex = kc.gamma(M, 1.0 / M, size=g.shape)          # mean 1, target intensity var 1/M
        try:
            from scipy.ndimage import gaussian_filter
            lat_grain = max(_K_LATERAL_GRAIN_BEAMS * beam_waist_um / dx_um, 1.0)
            tex = gaussian_filter(tex, (_K_AXIAL_GRAIN_PX, lat_grain), mode="nearest")
        except Exception:
            pass
        # Smoothing gave the modulator its slow (super-PSF) correlation but shrank its variance;
        # restore the marginal variance to 1/M so M actually controls the compound-K tail, keeping
        # mean 1 (brightness preserved). Clip keeps sqrt real; final renorm pins E[tex]=1 exactly.
        c = tex - float(np.mean(tex))
        cur = float(np.std(c))
        if cur > 1e-9:
            tex = 1.0 + c * (np.sqrt(1.0 / M) / cur)
        tex = np.clip(tex, 0.02, None)
        tex *= 1.0 / (float(np.mean(tex)) + 1e-12)
        g = g * np.sqrt(tex)
        k_tex = tex
    r += g * backscatter_map

    if features:
        ctx = SceneContext(depth_axis_um, x_axis_um, y_um, opt_b, dz, rng)
        for feat in features:
            feat.apply(r, ctx)

    out = _lateral_beam_filter(r, beam_waist_um / dx_um)

    # Sub-resolution lateral speckle decorrelation.
    #
    # Real lateral speckle grain is finer than the optical spot: a 15 µm waist predicts
    # ~12 µm and the Maestro corpora measure ~5 µm. The fix is NOT to shrink
    # beam_waist_um — that would falsify the instrument's real lateral resolution for
    # actual structure (vessels, layer boundaries) in order to fix a statistic of the
    # noise. Instead a fraction of the speckle power is replaced by a separately
    # decorrelated realisation riding the same smooth brightness envelope, which leaves
    # the imaged resolution of deterministic structure untouched.
    #
    # That decorrelation length is PHYSICAL, not per-A-line. Measured across seven Maestro
    # corpora spanning 2.93–23.44 µm of A-line pitch, one correlation length of ~5 µm
    # reproduces every one of them (RMS 1.4 µm, and ≤0.7 µm on the four line scans, which
    # are the corpora fine enough to resolve it at all). The apparent growth of the
    # measured grain at coarse pitch is `realism_metrics._acf_fwhm` bottoming out at its
    # 0.5 px floor, not a wider grain. The rival "decorrelates in 1–2 A-lines" model — a
    # laterally *white* realisation, which is what this code used to draw — is a grain of
    # exactly `dx` and fits those seven corpora almost twice as badly (RMS 2.6 µm).
    #
    # The mixture is variance-matched. `_lateral_beam_filter` is sum-normalised, so it
    # preserves smooth structure but shrinks the speckle variance by Σk², which depends on
    # `beam_waist_um / dx_um` and therefore on the sampling pitch. Weighting an unfiltered
    # component against it by √f alone made the *effective* decorrelated fraction 0.54 at
    # 2.93 µm and 0.13 at 23.44 µm from one nominal f = 0.12 — so both the speckle grain
    # and the frame brightness moved with the A-line density. Scaling the injected
    # component by the same √Σk² makes f a true power fraction at every pitch.
    #
    # The K-distribution modulator rides the main speckle path, so the decorrelated draw
    # carries it too: at a large ``f`` most of the speckle comes from this draw, and
    # without the modulator a profile that fits ``speckle_shape`` would silently get
    # Rayleigh speckle back. It is divided out of the envelope before being re-applied so
    # it is not counted twice — ``|r|`` already carries it. Inert in the shipped
    # ``literature`` profile, where ``speckle_shape`` is None.
    f = float(np.clip(speckle_decorrelation, 0.0, 1.0))
    if f > 0:
        mag = np.abs(r)
        if k_tex is not None:
            mag = mag / np.sqrt(k_tex)
        env = _lateral_beam_filter(mag.astype(np.complex128),
                                   beam_waist_um / dx_um).real
        # Like ``speckle``, this draw is slow-axis-correlated for a volume so adjacent
        # frames share it instead of boiling. It has to be: at the fitted f most of the
        # speckle comes from here, so a fresh per-frame draw would decorrelate the volume
        # however densely it was sampled.
        if speckle_decorr is not None:
            g2 = speckle_decorr
        else:
            g2 = (rng.standard_normal((n_depth, n_x))
                  + 1j * rng.standard_normal((n_depth, n_x))) / np.sqrt(2.0)
        g2 = _lateral_corr_filter(g2, speckle_corr_lateral_um / dx_um)
        if k_tex is not None:
            g2 = g2 * np.sqrt(k_tex)
        keep = _beam_variance_keep(beam_waist_um / dx_um)
        out = np.sqrt(1.0 - f) * out + np.sqrt(f * keep) * env * g2
    return out


class CoherentField:
    """A smooth 3-D random gain field ``g(z, x, y) ≈ 1 ± contrast``, coherent in all axes.

    A sum of ``k`` random-frequency cosines; sampling at a slow position ``y`` gives a 2-D
    ``[n_z, n_x]`` gain that evolves **smoothly with y** (correlation length ~ ``corr_y_um``).
    Stored once per volume on the anatomy model and sampled per frame, it makes the intra-layer
    mottling and choroidal vascular pattern change contiguously across a volume instead of
    flickering — they are fixed 3-D structures, not independent per-frame noise.
    """

    def __init__(self, rng: np.random.Generator, *, k: int = 8, corr_z_um: float = 30.0,
                 corr_x_um: float = 80.0, corr_y_um: float = 350.0, contrast: float = 0.2):
        u = lambda: rng.uniform(0.6, 2.0, k)
        self.fz = (2.0 * np.pi) / (corr_z_um * u())
        # RANDOMISE the lateral wavevector SIGN per component. Without this every cosine tilts the
        # SAME (+z,+x) diagonal, so the k components reinforce into a coherent DIAGONAL GRATING — an
        # obviously non-physical banding in the choroid/mottle. Random signs make the waves point in
        # different directions, giving an isotropic (within the intended z/x anisotropy) mottle.
        self.fx = (2.0 * np.pi) / (corr_x_um * u()) * rng.choice([-1.0, 1.0], k)
        # y-frequencies are kept LOW (periods >= corr_y) so the field varies slowly across the
        # slow axis — the correlation length in y is set by the shortest period, so a broad
        # high-frequency spread here would decorrelate adjacent frames. ``corr_y_um`` should be
        # several frame spacings for a smooth (non-flickering) volume.
        self.fy = (2.0 * np.pi) / (corr_y_um * rng.uniform(1.0, 3.0, k))
        self.ph = rng.uniform(0.0, 2.0 * np.pi, k)
        self.w = rng.uniform(0.5, 1.0, k)
        self.contrast = float(contrast)
        self._norm = float(np.sqrt((self.w ** 2).sum())) or 1.0

    def sample(self, z_um: np.ndarray, x_um: np.ndarray, y_um: float) -> np.ndarray:
        z = np.asarray(z_um, float); x = np.asarray(x_um, float)
        pz = self.fz[:, None] * z[None, :]                          # [k, nz]
        px = self.fx[:, None] * x[None, :]                          # [k, nx]
        ph = (self.fy * float(y_um) + self.ph)[:, None, None]       # [k, 1, 1]
        field = (self.w[:, None, None] * np.cos(pz[:, :, None] + px[:, None, :] + ph)).sum(0)
        return 1.0 + self.contrast * np.clip(field / self._norm, -2.5, 2.5)


def _smooth1d(n: int, rng: np.random.Generator, sigma_px: float) -> np.ndarray:
    """Zero-mean unit-std smooth 1-D field (low-frequency lateral modulation)."""
    v = rng.standard_normal(n)
    try:
        from scipy.ndimage import gaussian_filter1d
        v = gaussian_filter1d(v, max(sigma_px, 1.0), mode="reflect")
    except Exception:
        pass
    return (v - v.mean()) / (v.std() + 1e-9)


def _smooth_gain(nz: int, nx: int, rng: np.random.Generator, contrast: float,
                 sigma_z: float, sigma_x: float,
                 band: tuple[int, int] | None = None) -> np.ndarray:
    """Positive multiplicative low-frequency 2-D gain field ~ 1 ± contrast.

    The full-height noise is always drawn (keeping the RNG stream stable) but the
    expensive Gaussian smoothing is confined to ``band`` rows ``[r0, r1)`` — the gain
    is unity outside, where the backscatter map is empty anyway.
    """
    f = rng.standard_normal((nz, nx))
    r0, r1 = (0, nz) if band is None else band
    sub = f[r0:r1]
    try:
        from scipy.ndimage import gaussian_filter
        sub = gaussian_filter(sub, (sigma_z, sigma_x), mode="reflect")
    except Exception:
        pass
    sub = (sub - sub.mean()) / (sub.std() + 1e-9)
    gain = np.ones((nz, nx))
    gain[r0:r1] = 1.0 + contrast * np.clip(sub, -2.5, 2.5)
    return gain


def build_reflectivity(
    sample: Sample,
    depth_axis_um: np.ndarray,
    x_axis_um: np.ndarray,
    *,
    beam_waist_um: float,
    lateral_extent_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reflectivity field for a flat :class:`Sample` (back-compat wrapper).

    Computes per-column boundaries and layer property arrays from the ``Sample`` and
    delegates to :func:`build_reflectivity_from_boundaries`.
    """
    x_axis_um = np.asarray(x_axis_um, float)
    opt_b = sample.optical_boundary_depths_um(x_axis_um)
    phys_b = sample.boundary_depths_um(x_axis_um)
    layers = sample.layers
    return build_reflectivity_from_boundaries(
        opt_b, phys_b,
        layer_n=np.array([ly.n for ly in layers]),
        layer_backscatter=np.array([ly.backscatter if ly.density_per_um2 > 0 else 0.0
                                    for ly in layers]),
        layer_mu_t=np.array([ly.mu_t_per_um for ly in layers]),
        layer_interface=[ly.interface_reflectivity for ly in layers],
        top_n=sample.top_n, depth_axis_um=depth_axis_um, x_axis_um=x_axis_um,
        beam_waist_um=beam_waist_um, lateral_extent_um=lateral_extent_um, rng=rng)


def _beam_kernel(waist_px: float) -> np.ndarray:
    """Sum-normalised Gaussian beam field-amplitude kernel ``exp(-t²/w²)``, in pixels."""
    half = max(1, int(np.ceil(3.0 * waist_px)))
    t = np.arange(-half, half + 1)
    k = np.exp(-((t / waist_px) ** 2))
    return k / k.sum()


def _beam_variance_keep(waist_px: float) -> float:
    """Fraction of a white field's variance surviving :func:`_lateral_beam_filter`.

    Σk² for the sum-normalised beam kernel. The filter preserves smooth structure (Σk = 1)
    but averages an uncorrelated field down by this factor, which falls as the beam covers
    more pixels — i.e. it depends on the A-line pitch. Anything mixed against beam-filtered
    speckle has to be scaled by ``√keep`` or the mixing weight is not a power fraction.
    """
    if waist_px <= 0.5:
        return 1.0
    k = _beam_kernel(waist_px)
    return float((k ** 2).sum())


def _lateral_corr_filter(g: np.ndarray, sigma_px: float) -> np.ndarray:
    """Give a white complex field a lateral correlation length, preserving its variance.

    Unlike :func:`_lateral_beam_filter` the kernel is normalised to **Σk² = 1** rather than
    Σk = 1, because the input is zero-mean noise whose variance must survive rather than a
    brightness envelope whose level must survive. Filtering a circular-complex-Gaussian
    field leaves it circular-complex-Gaussian, so the speckle stays fully developed.
    """
    if sigma_px <= 0.05 or g.shape[1] < 3:
        return g
    half = max(1, int(np.ceil(4.0 * sigma_px)))
    t = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (t / sigma_px) ** 2)
    k = k / np.sqrt((k ** 2).sum())
    try:
        from scipy.ndimage import convolve1d
        return convolve1d(g, k, axis=1, mode="mirror")
    except Exception:
        pad = np.pad(g, ((0, 0), (half, half)), mode="reflect")
        out = np.empty_like(g)
        for row in range(g.shape[0]):
            out[row] = np.convolve(pad[row], k, mode="valid")
        return out


def _lateral_beam_filter(r: np.ndarray, waist_px: float) -> np.ndarray:
    """Convolve each depth row by the Gaussian beam **amplitude** profile (along x).

    ``r`` is the complex reflectivity *field*, so the kernel is the beam field-amplitude
    profile ``exp(-t² / w²)`` with ``w`` the ``1/e²`` intensity waist in pixels — NOT the
    intensity profile ``exp(-2t²/w²)``. This makes a point reflector image with intensity
    FWHM ``√(2 ln2)·w`` = the reported :func:`~octsim.optics.waist_to_lateral_fwhm_um`; the
    intensity kernel would image points √2 too sharp. Sum-normalised so smooth structure
    keeps its amplitude; reflect-padded. Filtering a circular-complex-Gaussian field leaves
    it circular-complex-Gaussian, so speckle stays fully developed but gains a grain size
    ~ the beam width.
    """
    if waist_px <= 0.5 or r.shape[1] < 3:
        return r
    k = _beam_kernel(waist_px)
    half = (k.size - 1) // 2
    # Vectorized separable convolution along x via scipy (falls back to numpy):
    # convolve1d over axis=1 in one call instead of a per-depth-row Python loop.
    try:
        # scipy 'mirror' == numpy pad 'reflect' (edge sample not repeated).
        from scipy.ndimage import convolve1d
        return convolve1d(r, k, axis=1, mode="mirror")
    except Exception:
        pad = np.pad(r, ((0, 0), (half, half)), mode="reflect")
        out = np.empty_like(r)
        for row in range(r.shape[0]):
            out[row] = np.convolve(pad[row], k, mode="valid")
        return out
