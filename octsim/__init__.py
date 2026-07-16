"""octsim — a physically-modelled OCT simulation platform.

Signal chain (per A-scan): a scene (a physiological :mod:`~octsim.anatomy` model or a
:class:`~octsim.sample.Sample` preset) becomes a complex reflectivity field;
:func:`~octsim.forward.synthesize_bscan_spectra` turns it into the raw interferometric
spectrum ``I(k)`` under a Gaussian source/beam, with sensitivity fall-off, dispersion,
acquisition-speed signal scaling, and motion/washout; :mod:`~octsim.noise` adds a
detector floor; and :mod:`~octsim.reconstruct` inverts it to a B-scan.

High-level API (config-driven, stochastic)::

    from octsim import SimConfig, simulate, generate_dataset

    cfg = SimConfig(anatomy="macula", scan_pattern="volume", n_frames=32,
                    axial_resolution_um=5.0, lateral_fov_um=6000.0, motion_level=1.0)
    sample = simulate(cfg, seed=1)             # one B-scan or volume
    for s in generate_dataset(cfg, n=100, seed=0):   # N random samples (in-memory)
        img = s.bscan_db

Low-level API (build a scene by hand)::

    from octsim import LightSource, simulate_bscan, presets, NoiseModel
    result = simulate_bscan(presets.macula(), LightSource(), noise=NoiseModel.typical())
"""

from __future__ import annotations

from . import presets
from .config import ResolvedConfig, SimConfig, config_from_mapping
from .forward import synthesize_ascan_spectrum, synthesize_bscan_spectra
from .generate import SampleResult, generate_dataset, simulate
from .noise import NoiseModel
from .reconstruct import axial_fwhm_um, reconstruct_bscan, to_db
from .sample import Layer, Sample, build_reflectivity, build_reflectivity_from_boundaries
from .scan import BscanResult, simulate_bscan
from .source import LightSource

__all__ = [
    # high-level
    "SimConfig",
    "ResolvedConfig",
    "config_from_mapping",
    "simulate",
    "generate_dataset",
    "SampleResult",
    # physics / low-level
    "LightSource",
    "Layer",
    "Sample",
    "build_reflectivity",
    "build_reflectivity_from_boundaries",
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

__version__ = "0.2.0"
