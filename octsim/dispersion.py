"""Dispersion mismatch and numerical compensation.

A mismatch between the dispersion of the sample and reference arms imprints a
wavenumber-dependent phase on the sample field,

    phi(k) = Σ_{m>=2} a_m · (k - k0)^m

with the quadratic term ``a2`` (group-velocity dispersion) dominating. Uncompensated,
this **chirps** the fringe and broadens every reconstructed peak with depth. It is
applied as a phase multiply on the complex sample field ``E_s(k)`` in the forward
model (:mod:`octsim.forward`); numerical compensation multiplies by the conjugate
phase before the reconstruction FFT.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dispersion_phase(k: np.ndarray, k0: float, coeffs: Sequence[float] | None) -> np.ndarray:
    """Complex phase ``exp(i·phi(k))`` for the given coefficients.

    ``coeffs[0]`` is the quadratic term ``a2`` (order 2), ``coeffs[1]`` the cubic
    ``a3``, and so on. Empty/``None`` returns unity (no dispersion).
    """
    k = np.asarray(k, dtype=np.float64)
    if not coeffs:
        return np.ones_like(k, dtype=np.complex128)
    dk = k - float(k0)
    phi = np.zeros_like(k)
    for order, a in enumerate(coeffs, start=2):
        if a:
            phi = phi + float(a) * dk ** order
    return np.exp(1j * phi)


def apply_dispersion(e_s: np.ndarray, k: np.ndarray, k0: float,
                     coeffs: Sequence[float] | None) -> np.ndarray:
    """Multiply a sample field ``E_s(k)`` (``[n_k]`` or ``[n_k, n_x]``) by the phase."""
    if not coeffs:
        return e_s
    ph = dispersion_phase(k, k0, coeffs)
    if e_s.ndim == 2:
        ph = ph[:, None]
    return e_s * ph


def compensation_phase(k: np.ndarray, k0: float, coeffs: Sequence[float] | None) -> np.ndarray:
    """Conjugate phase ``exp(-i·phi(k))`` for numerical dispersion compensation."""
    return np.conj(dispersion_phase(k, k0, coeffs))
