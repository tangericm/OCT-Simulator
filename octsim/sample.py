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
    for i in range(L):
        top_opt = opt_b[i][None, :]
        bot_opt = opt_b[i + 1][None, :]
        mask = (d >= top_opt) & (d < bot_opt)
        d_in_phys = np.clip(d - top_opt, 0.0, None) / layer_n[i]
        atten = np.exp(-(cum_mu[None, :] + layer_mu_t[i] * d_in_phys))
        if layer_backscatter[i] > 0:
            bmap_i = layer_backscatter[i] * atten
            if layer_directional is not None:   # per-column directional reflectance
                bmap_i = bmap_i * layer_directional[i][None, :]
            backscatter_map[mask] = bmap_i[mask]

        r_if = layer_interface[i]
        if r_if is None:
            n_above = top_n if i == 0 else layer_n[i - 1]
            r_if = (n_above - layer_n[i]) / (n_above + layer_n[i])
        if abs(r_if) > 1e-6:
            bins = np.clip(np.round(opt_b[i] / dz).astype(np.intp), 0, n_depth - 1)
            amp = r_if * np.exp(-cum_mu)
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
        gain = _smooth_gain(n_depth, n_x, rng, texture_contrast,
                            max(18.0 / dz, 1.0), max(70.0 / dx_um, 1.0))
        backscatter_map = backscatter_map * gain

    g = (rng.standard_normal((n_depth, n_x))
         + 1j * rng.standard_normal((n_depth, n_x))) / np.sqrt(2.0)
    r += g * backscatter_map

    if features:
        ctx = SceneContext(depth_axis_um, x_axis_um, y_um, opt_b, dz, rng)
        for feat in features:
            feat.apply(r, ctx)

    return _lateral_beam_filter(r, beam_waist_um / dx_um)


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
                 sigma_z: float, sigma_x: float) -> np.ndarray:
    """Positive multiplicative low-frequency 2-D gain field ~ 1 ± contrast."""
    f = rng.standard_normal((nz, nx))
    try:
        from scipy.ndimage import gaussian_filter
        f = gaussian_filter(f, (sigma_z, sigma_x), mode="reflect")
    except Exception:
        pass
    f = (f - f.mean()) / (f.std() + 1e-9)
    return 1.0 + contrast * np.clip(f, -2.5, 2.5)


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


def _lateral_beam_filter(r: np.ndarray, waist_px: float) -> np.ndarray:
    """Convolve each depth row by the Gaussian beam amplitude profile (along x).

    Kernel ``exp(-2 t² / w²)`` (``1/e²`` waist ``w`` in pixels), sum-normalised so
    smooth structure keeps its amplitude; reflect-padded at the edges. Filtering a
    circular-complex-Gaussian field leaves it circular-complex-Gaussian, so speckle
    stays fully developed but gains a grain size ~ the beam width.
    """
    if waist_px <= 0.5 or r.shape[1] < 3:
        return r
    half = max(1, int(np.ceil(3.0 * waist_px)))
    t = np.arange(-half, half + 1)
    k = np.exp(-2.0 * (t / waist_px) ** 2)
    k = k / k.sum()
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
