"""Wavelength-dependent tissue optical properties.

OCT operates in the near-infrared, and both scattering and absorption vary strongly
with the source wavelength — which is why 1050/1310 nm systems penetrate the choroid
far better than 830-880 nm systems:

* **Scattering** falls with wavelength, ``mu_s(λ) = mu_s_ref · (λ/λ_ref)^(-b)`` with a
  scattering power ``b ≈ 1.1`` for retinal tissue (a Mie/Rayleigh mix). Less scattering
  at longer λ ⇒ deeper penetration and less superficial backscatter.
* **Absorption** is dominated by water at these wavelengths, with a coarse spectrum: low
  in the 800-880 nm "therapeutic window", a bump near 980 nm, a relative window at
  1050-1100 nm, and rising strongly past ~1200 nm. Melanin (RPE/choroid) absorption
  decreases with wavelength.

Layer optical properties are stored at a reference wavelength (850 nm) and scaled to the
source's centre wavelength by :func:`scale_layer`.
"""

from __future__ import annotations

import numpy as np

REF_WAVELENGTH_NM = 850.0
SCATTER_POWER = 1.1

# Coarse pure-water absorption spectrum in µm^-1 (= cm^-1 / 1e4) at NIR OCT
# wavelengths (Hale & Querry), interpolated between these anchors. Note it is tiny
# vs scattering (~1e-3 µm^-1) until it climbs past ~1300 nm — so scattering, not
# absorption, sets NIR penetration, and longer wavelengths reach the choroid better.
_WATER_NM = np.array([800, 850, 900, 950, 980, 1000, 1050, 1100, 1200, 1310, 1400])
_WATER_MU_A = np.array([2.0e-6, 4.3e-6, 6.8e-6, 3.9e-5, 5.0e-5, 3.6e-5, 1.1e-5,
                        1.3e-5, 1.0e-4, 2.0e-4, 1.2e-3])


def scattering_scale(wavelength_nm: float, power: float = SCATTER_POWER) -> float:
    """Multiplicative factor on ``mu_s`` relative to the 850 nm reference."""
    return (float(wavelength_nm) / REF_WAVELENGTH_NM) ** (-power)


def water_absorption_per_um(wavelength_nm: float) -> float:
    """Water absorption coefficient (µm^-1) at ``wavelength_nm`` (interpolated)."""
    return float(np.interp(wavelength_nm, _WATER_NM, _WATER_MU_A))


def melanin_absorption_scale(wavelength_nm: float) -> float:
    """Melanin absorption relative to 850 nm (decreases with wavelength)."""
    return (float(wavelength_nm) / REF_WAVELENGTH_NM) ** (-1.5)


def scale_layer(mu_s_ref_per_mm: float, backscatter_ref: float, wavelength_nm: float,
                *, water_fraction: float = 0.7, melanin: float = 0.0) -> tuple[float, float]:
    """Effective ``(mu_t_per_um, backscatter)`` at the source wavelength.

    ``mu_t`` = scaled scattering + water absorption + melanin absorption. Backscatter
    scales with scattering (more scattering ⇒ brighter superficial signal).
    """
    s = scattering_scale(wavelength_nm)
    mu_s = mu_s_ref_per_mm * 1e-3 * s
    # melanin: ~3/mm at 850 nm for the fully-pigmented RPE (melanin=1), scaling down
    # with wavelength; water absorption from the tabulated spectrum.
    mu_a = (water_fraction * water_absorption_per_um(wavelength_nm)
            + melanin * 3.0e-3 * melanin_absorption_scale(wavelength_nm))
    # backscatter is an AMPLITUDE, so it scales as sqrt(mu_s) (intensity ~ mu_s).
    return mu_s + mu_a, backscatter_ref * np.sqrt(s)
