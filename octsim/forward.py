"""Forward model: reflectivity field / scatterers -> interferometric spectrum.

The physical heart of the simulator. The detected spectral interferogram for one
beam position is

    I(k) = P(k) · | a_ref + E_s(k) | ²,   E_s(k) = Σ_j a_j · exp(i · 2 · k · z_j)

where ``P(k)`` is the source power spectrum, ``a_ref`` the reference-arm field
(a bright zero-delay term whose self-beat is DC), and each scatterer contributes
its complex field amplitude ``a_j`` at one-way **optical** depth ``z_j``. First-Born /
single-scattering: every scatterer contributes once; multiple scattering is not modelled.

On top of the ideal fringe, the forward model folds in the depth/detection physics that
depend on the instrument and the acquisition:

* **sensitivity fall-off** ``H(z)`` — an amplitude pre-multiply on the depth reflectivity
  (:mod:`octsim.optics`);
* **dispersion** — a wavenumber phase on the sample field (:mod:`octsim.dispersion`);
* **signal scale** — ``√(T_int/T_ref)`` from the acquisition speed, so faster scans collect
  less sample signal (shot-limited SNR ∝ √T_int);
* **motion / washout** — an optional per-A-scan depth phase ``exp(i·2·k·Δz_x)`` and a
  per-A-scan contrast gain (fringe washout), supplied by :mod:`octsim.motion`.

Two entry points: :func:`synthesize_bscan_spectra` (the corpus-scale generator, exact FFT
per B-scan) and :func:`synthesize_ascan_spectrum` (the direct phasor-sum reference).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .dispersion import apply_dispersion
from .optics import sensitivity_falloff
from .source import LightSource


def synthesize_bscan_spectra(
    reflectivity: np.ndarray,
    source: LightSource,
    *,
    reference_amp: float = 4.0,
    falloff_6db_um: float = 0.0,
    dispersion_coeffs: Sequence[float] | None = None,
    signal_scale: float = 1.0,
    motion_dz_um: np.ndarray | None = None,
    washout_gain: np.ndarray | None = None,
    column_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Raw spectral interferogram ``[n_k, n_x]`` from a reflectivity field.

    ``reflectivity`` is complex ``[n_depth, n_x]`` on the reconstruction depth grid
    (``n_depth == source.n_k // 2``). Optional per-column (length ``n_x``) arrays:
    ``motion_dz_um`` (bulk axial motion → k-dependent depth phase), ``washout_gain``
    (fringe-washout contrast loss), and ``column_phase`` (k-independent phase drift).
    Returns the real detector spectra a swept source would digitise, one per A-scan.
    """
    r = np.asarray(reflectivity)
    if r.ndim == 1:
        r = r[:, None]
    n_depth, n_x = r.shape
    k = source.k_grid().astype(np.float64)
    P = source.power_spectrum(k)
    n_k = k.size
    depth = np.arange(n_depth) * source.depth_pixel_um

    # sensitivity fall-off: amplitude weight on the depth reflectivity
    if falloff_6db_um and falloff_6db_um > 0:
        h = sensitivity_falloff(depth, falloff_6db_um, max_depth_um=source.max_optical_depth_um)
        r = r * h[:, None]

    carrier = np.exp(1j * 2.0 * k[0] * depth)[:, None]     # exp(i 2 k_min z_p)
    g = np.zeros((n_k, n_x), dtype=np.complex128)
    g[:n_depth] = r * carrier
    e_s = n_k * np.fft.ifft(g, axis=0)                     # = Σ_p r_p exp(i 2 k z_p)

    if dispersion_coeffs:
        e_s = apply_dispersion(e_s, k, source.k0, dispersion_coeffs)
    if signal_scale != 1.0:
        e_s = e_s * float(signal_scale)
    if washout_gain is not None:
        e_s = e_s * np.asarray(washout_gain, dtype=np.float64)[None, :]
    if column_phase is not None:
        e_s = e_s * np.exp(1j * np.asarray(column_phase, dtype=np.float64))[None, :]
    if motion_dz_um is not None:
        dz = np.asarray(motion_dz_um, dtype=np.float64)
        e_s = e_s * np.exp(1j * 2.0 * np.outer(k, dz))     # per-column depth phase

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

    ``z_opt_um`` and ``amp`` are the optical depths and complex field amplitudes of the
    scatterers seen by one beam. Returns the real spectrum ``I(k)`` of length
    ``source.n_k``. The grid-free reference model, used to validate the FFT path.
    """
    k = source.k_grid().astype(np.float64)
    P = source.power_spectrum(k)
    if np.size(z_opt_um) == 0:
        return P * reference_amp ** 2
    phase = np.exp(1j * 2.0 * np.outer(k, np.asarray(z_opt_um, float)))
    e_s = phase @ np.asarray(amp, np.complex128)
    e_total = reference_amp + e_s
    return P * (e_total.real ** 2 + e_total.imag ** 2)
