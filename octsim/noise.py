"""Detector noise model.

Three physically-motivated components are added to the spectral interferogram
``I(k)`` *before* reconstruction, matching the standard OCT noise decomposition:

* **shot noise**  — amplitude ``∝ sqrt(I)``  (photon statistics)
* **RIN**         — amplitude ``∝ I``        (relative intensity noise of the source)
* **read noise**  — constant                 (non-optical / electronic floor)

All three are drawn white in ``k`` so they spread across the whole depth axis
after the FFT, producing a realistic, roughly depth-flat noise floor whose
level averages down as ``1/sqrt(N)`` over incoherent frame averaging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseModel:
    """Relative strengths of the three noise components (0 disables a term).

    ``shot`` and ``rin`` are dimensionless multipliers on the spectrum (``sqrt(I)`` and
    ``I`` respectively). ``read`` is an **absolute** per-pixel electronic standard
    deviation in the same units as the spectrum.

    Read noise used to be scaled by ``mean(|spectrum|)``, which is dominated by the
    reference term ``P(k)·a_ref²``. That made the "non-optical electronic floor" grow with
    the reference-arm power, so raising ``reference_amp`` raised the noise in step and SNR
    was independent of reference power — the opposite of a real SD-OCT system, where
    increasing reference power improves SNR up to the shot-noise limit. It also meant the
    floor could not be grounded in the measured ``Instrument.read_floor``.
    """

    shot: float = 0.0
    rin: float = 0.0
    read: float = 0.0            # absolute per-pixel std (not a fraction of the spectrum)

    @classmethod
    def typical(cls) -> "NoiseModel":
        """A mild, visually realistic default (shot-dominated floor)."""
        return cls(shot=0.03, rin=0.005, read=0.01)

    @classmethod
    def none(cls) -> "NoiseModel":
        """A noise-free model (all components zero) for clean data."""
        return cls(shot=0.0, rin=0.0, read=0.0)

    @property
    def is_noise_free(self) -> bool:
        return not (self.shot or self.rin or self.read)

    def apply(self, spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return ``spectrum`` with additive noise (does not mutate the input)."""
        if not (self.shot or self.rin or self.read):
            return spectrum
        out = spectrum.astype(np.float64, copy=True)
        if self.shot:
            out += self.shot * np.sqrt(np.abs(spectrum)) * rng.standard_normal(spectrum.shape)
        if self.rin:
            out += self.rin * np.abs(spectrum) * rng.standard_normal(spectrum.shape)
        if self.read:
            out += self.read * rng.standard_normal(spectrum.shape)
        return out
