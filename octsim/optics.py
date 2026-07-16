"""Optical transfer effects that shape the reconstructed A-scan.

Two things live here:

* **Sensitivity fall-off** — the depth-dependent roll-off of OCT sensitivity. In a
  Fourier-domain system a reflector deeper in the imaging window returns a weaker,
  more washed-out fringe because (a) each detector pixel / swept-source sample
  integrates a finite range of ``k`` (a ``sinc`` envelope with its first null at the
  Nyquist depth ``z_max``) and (b) the source has a finite instantaneous linewidth
  (a Gaussian envelope). We model the amplitude fall-off as

      H(z) = sinc(z / (2 z_max)) · exp(-ln2 · (z / z_6db)²)

  so the **intensity** ``|H|²`` drops to −6 dB (¼) at the user-facing ``falloff_6db_um``
  from the Gaussian term. ``H`` is applied as an amplitude pre-multiply on the depth
  reflectivity before the forward FFT (:mod:`octsim.forward`).

* **Beam / lateral resolution** — small helpers relating the Gaussian beam ``1/e²``
  waist to the FWHM lateral resolution, used by the resolvable config.
"""

from __future__ import annotations

import numpy as np

_LN2 = np.log(2.0)
# FWHM = sqrt(2 ln2) * (1/e^2 waist)  for a Gaussian intensity profile exp(-2 r^2/w^2)
_WAIST_TO_FWHM = np.sqrt(2.0 * _LN2)


def sensitivity_falloff(
    depth_um: np.ndarray,
    falloff_6db_um: float,
    *,
    max_depth_um: float | None = None,
    include_sampling_sinc: bool = True,
) -> np.ndarray:
    """Amplitude fall-off envelope ``H(z)`` over ``depth_um`` (length ``n_depth``).

    ``falloff_6db_um <= 0`` disables the roll-off (returns ones). ``max_depth_um``
    (the positive-depth Nyquist range) drives the sampling ``sinc``; pass ``None``
    to use the last depth sample.
    """
    z = np.asarray(depth_um, dtype=np.float64)
    if falloff_6db_um is None or falloff_6db_um <= 0:
        h = np.ones_like(z)
    else:
        h = np.exp(-_LN2 * (z / float(falloff_6db_um)) ** 2)
    if include_sampling_sinc:
        zmax = float(max_depth_um) if max_depth_um else float(z[-1] if z.size else 1.0)
        if zmax > 0:
            h = h * np.sinc(z / (2.0 * zmax))  # np.sinc(x) = sin(pi x)/(pi x)
    return h


def waist_to_lateral_fwhm_um(beam_waist_um: float) -> float:
    """FWHM lateral resolution from a Gaussian ``1/e²`` intensity waist."""
    return _WAIST_TO_FWHM * float(beam_waist_um)


def lateral_fwhm_to_waist_um(lateral_resolution_um: float) -> float:
    """Gaussian ``1/e²`` waist from a target FWHM lateral resolution."""
    return float(lateral_resolution_um) / _WAIST_TO_FWHM
