"""Broadband OCT light source.

The source is sampled **linearly in wavenumber** ``k`` (the swept-source / SS-OCT
convention, and the natural grid for a spectral-domain reconstruction). A Gaussian
power spectrum ``P(k)`` centred at ``k0`` is fully described by two physical knobs:

* ``center_wavelength_nm``  -> ``k0``            (carrier; sets the phase-per-depth)
* ``bandwidth_fwhm_nm``     -> ``Δk``            (sets the axial resolution)

Everything downstream (axial point-spread width, imaging depth, depth pixel size)
is *derived* from these plus the number of detector samples ``n_k`` and the width
of the sampled ``k`` window.

Depth calibration (used by :mod:`octsim.reconstruct`): a scatterer at one-way
optical depth ``z`` contributes ``exp(i·2·k·z)``. Sampling ``k`` at spacing ``δk``
places its reconstructed peak at DFT bin ``p`` where ``2·δk·z = 2π·p/n_k``, i.e.

    dz_opt = π / (n_k · δk)          # optical-path depth per pixel
    z_max  = π / (2 · δk)            # one-way optical depth at the last positive bin

Divide an optical depth by a layer's group index ``n`` to get physical depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class LightSource:
    """A Gaussian broadband source sampled linearly in ``k``.

    Defaults describe a typical ophthalmic swept source (~1050 nm, ~100 nm
    bandwidth), giving ~5 µm axial resolution and ~1.9 mm one-way optical depth
    at ``n_k = 2048``.
    """

    center_wavelength_nm: float = 1050.0
    bandwidth_fwhm_nm: float = 100.0
    n_k: int = 2048
    #: Half-width of the sampled k-window, in units of the spectral FWHM. ~1.5
    #: captures the Gaussian to ~3.5σ (>99.9% of the energy) while keeping the
    #: imaging depth generous; larger values trade depth range for tail fidelity.
    k_span_factor: float = 1.5

    # -- carrier / bandwidth in k-space (rad / µm) --------------------------
    @property
    def k0(self) -> float:
        """Central wavenumber (rad/µm)."""
        return TWO_PI / (self.center_wavelength_nm * 1e-3)

    @property
    def fwhm_k(self) -> float:
        """Spectral FWHM expressed in k (rad/µm): ``Δk = 2π·Δλ / λ0²``."""
        lam0_um = self.center_wavelength_nm * 1e-3
        dlam_um = self.bandwidth_fwhm_nm * 1e-3
        return TWO_PI * dlam_um / (lam0_um ** 2)

    @property
    def sigma_k(self) -> float:
        return self.fwhm_k * _FWHM_TO_SIGMA

    # -- sampling grid ------------------------------------------------------
    def k_grid(self) -> np.ndarray:
        """Detector wavenumber axis (rad/µm), linearly spaced, length ``n_k``."""
        half = self.k_span_factor * self.fwhm_k
        return np.linspace(self.k0 - half, self.k0 + half, self.n_k)

    @property
    def dk(self) -> float:
        """Wavenumber sample spacing (rad/µm)."""
        return (2.0 * self.k_span_factor * self.fwhm_k) / (self.n_k - 1)

    def power_spectrum(self, k: np.ndarray | None = None) -> np.ndarray:
        """Gaussian source power ``P(k)`` (peak-normalised to 1)."""
        if k is None:
            k = self.k_grid()
        return np.exp(-0.5 * ((k - self.k0) / self.sigma_k) ** 2)

    # -- derived resolution / depth calibration -----------------------------
    @property
    def axial_resolution_um(self) -> float:
        """Coherence-limited round-trip axial resolution in air (one-way optical
        path), ``δz = (2 ln2 / π) · λ0² / Δλ``. A Hann reconstruction window
        broadens the realised PSF by ~1.5x."""
        lam0_um = self.center_wavelength_nm * 1e-3
        dlam_um = self.bandwidth_fwhm_nm * 1e-3
        return (2.0 * np.log(2.0) / np.pi) * lam0_um ** 2 / dlam_um

    @property
    def depth_pixel_um(self) -> float:
        """Optical-path depth per reconstructed pixel, ``π / (n_k · δk)``."""
        return np.pi / (self.n_k * self.dk)

    @property
    def max_optical_depth_um(self) -> float:
        """One-way optical depth at the last positive-frequency bin, ``π/(2 δk)``."""
        return np.pi / (2.0 * self.dk)

    def depth_axis_um(self) -> np.ndarray:
        """Optical-path depth axis for the positive-depth half (length ``n_k//2``)."""
        return np.arange(self.n_k // 2) * self.depth_pixel_um

    def describe(self) -> str:
        return (
            f"LightSource(λ0={self.center_wavelength_nm:.0f} nm, "
            f"Δλ={self.bandwidth_fwhm_nm:.0f} nm, n_k={self.n_k})\n"
            f"  axial resolution : {self.axial_resolution_um:.2f} µm (coherence limit)\n"
            f"  depth pixel      : {self.depth_pixel_um:.3f} µm/px (optical path)\n"
            f"  max optical depth: {self.max_optical_depth_um:.0f} µm "
            f"(~{self.max_optical_depth_um / 1.38:.0f} µm tissue @ n=1.38)"
        )
