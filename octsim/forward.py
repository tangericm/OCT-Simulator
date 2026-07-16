"""Forward model: reflectivity field / scatterers -> interferometric spectrum.

The physical heart of the simulator. The detected spectral interferogram for one
beam position is

    I(k) = P(k) · | a_ref + E_s(k) | ²,   E_s(k) = Σ_j a_j · exp(i · 2 · k · z_j)

where ``P(k)`` is the source power spectrum, ``a_ref`` the reference-arm field
(a bright zero-delay term whose self-beat is DC), and each scatterer contributes
its complex field amplitude ``a_j`` at one-way **optical** depth ``z_j``. This is
a first-Born / single-scattering model: every scatterer contributes once,
independently; multiple scattering is not modelled.

Two entry points:

* :func:`synthesize_bscan_spectra` — the corpus-scale generator. It takes a
  complex reflectivity field already sampled on the reconstruction depth grid
  (:func:`octsim.sample.build_reflectivity`) and evaluates ``E_s`` **exactly** as
  a single FFT per B-scan: because ``k`` is linear and ``z_p = p·dz`` satisfies
  ``2·k_m·z_p = 2π·m·p/n_k`` up to the carrier ``exp(i·2·k_min·z_p)``,
  ``E_s = n_k · IFFT(carrier · r)``. No gridding approximation — the field *is*
  defined on the grid.
* :func:`synthesize_ascan_spectrum` — the exact direct phasor sum for an
  arbitrary set of off-grid point scatterers (used to validate the PSF).
"""

from __future__ import annotations

import numpy as np

from .source import LightSource


def synthesize_bscan_spectra(
    reflectivity: np.ndarray,
    source: LightSource,
    *,
    reference_amp: float = 4.0,
) -> np.ndarray:
    """Raw spectral interferogram ``[n_k, n_x]`` from a reflectivity field.

    ``reflectivity`` is complex ``[n_depth, n_x]`` on the reconstruction depth
    grid (``n_depth == source.n_k // 2``). Returns the real detector spectra a
    swept source would digitise, one column per A-scan.
    """
    r = np.asarray(reflectivity)
    if r.ndim == 1:
        r = r[:, None]
    n_depth, n_x = r.shape
    k = source.k_grid().astype(np.float64)
    P = source.power_spectrum(k)
    n_k = k.size

    depth = np.arange(n_depth) * source.depth_pixel_um
    carrier = np.exp(1j * 2.0 * k[0] * depth)[:, None]     # exp(i 2 k_min z_p)

    g = np.zeros((n_k, n_x), dtype=np.complex128)
    g[:n_depth] = r * carrier
    e_s = n_k * np.fft.ifft(g, axis=0)                     # = Σ_p r_p exp(i 2 k z_p)
    e_total = reference_amp + e_s
    return P[:, None] * (e_total.real ** 2 + e_total.imag ** 2)


def synthesize_ascan_spectrum(
    z_opt_um: np.ndarray,
    amp: np.ndarray,
    source: LightSource,
    *,
    reference_amp: float = 4.0,
) -> np.ndarray:
    """Exact single-A-scan spectrum for arbitrary point scatterers (direct sum).

    ``z_opt_um`` and ``amp`` are the optical depths and complex field amplitudes
    of the scatterers seen by one beam. Returns the real spectrum ``I(k)`` of
    length ``source.n_k``. This is the reference model — it makes no grid
    assumption, so it is used to validate the FFT path and the axial PSF.
    """
    k = source.k_grid().astype(np.float64)
    P = source.power_spectrum(k)
    if np.size(z_opt_um) == 0:
        return P * reference_amp ** 2
    phase = np.exp(1j * 2.0 * np.outer(k, np.asarray(z_opt_um, float)))
    e_s = phase @ np.asarray(amp, np.complex128)
    e_total = reference_amp + e_s
    return P * (e_total.real ** 2 + e_total.imag ** 2)
