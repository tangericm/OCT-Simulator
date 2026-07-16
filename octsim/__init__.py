"""octsim — a physically-modelled OCT B-scan synthesizer.

Signal chain (per A-scan): a layered :class:`~octsim.sample.Sample` becomes a
cloud of discrete scatterers (specular interfaces + random volume scatterers);
:func:`~octsim.forward.synthesize_bscan_spectra` sums them coherently under a
Gaussian source and beam into the raw interferometric spectrum ``I(k)``;
:mod:`~octsim.noise` adds a detector floor; and :mod:`~octsim.reconstruct`
inverts it to a B-scan.

Quick start::

    from octsim import LightSource, simulate_bscan, presets, NoiseModel

    result = simulate_bscan(presets.macula(), LightSource(),
                            noise=NoiseModel.typical(), seed=1)
    img = result.bscan_db          # [depth, x] display image, 0 dB peak
"""

from __future__ import annotations

from . import presets
from .forward import synthesize_ascan_spectrum, synthesize_bscan_spectra
from .noise import NoiseModel
from .reconstruct import axial_fwhm_um, reconstruct_bscan, to_db
from .sample import Layer, Sample, build_reflectivity
from .scan import BscanResult, simulate_bscan
from .source import LightSource

__all__ = [
    "LightSource",
    "Layer",
    "Sample",
    "build_reflectivity",
    "NoiseModel",
    "synthesize_ascan_spectrum",
    "synthesize_bscan_spectra",
    "reconstruct_bscan",
    "to_db",
    "axial_fwhm_um",
    "BscanResult",
    "simulate_bscan",
    "presets",
]

__version__ = "0.1.0"
