"""Top-level B-scan simulation API.

``simulate_bscan(sample, source)`` is the one call most users need: it builds the
scatterer cloud, synthesizes every A-scan's raw spectrum, adds detector noise,
reconstructs, and returns everything (raw spectra, complex B-scan, dB image,
axes, and a free per-pixel layer label map) bundled in :class:`BscanResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward import synthesize_bscan_spectra
from .noise import NoiseModel
from .reconstruct import reconstruct_bscan, to_db
from .sample import Sample, build_reflectivity
from .source import LightSource


@dataclass
class BscanResult:
    """Everything a single simulated B-scan produces."""

    spectra: np.ndarray          # raw interferogram [n_k, n_x] (pre-reconstruction)
    bscan: np.ndarray            # complex A-lines [n_depth, n_x]
    bscan_db: np.ndarray         # display image [n_depth, n_x], 0 dB peak
    label_map: np.ndarray        # uint8 layer index per pixel [n_depth, n_x]
    depth_axis_um: np.ndarray    # optical-path depth per row
    x_axis_um: np.ndarray        # lateral position per column
    source: LightSource
    sample: Sample

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.bscan)

    def to_uint16(self) -> np.ndarray:
        """Quantise the dB image to uint16 (mimics a 16-bit export)."""
        db = self.bscan_db
        norm = (db - db.min()) / (db.max() - db.min() + 1e-12)
        return (norm * 65535.0).astype(np.uint16)


def simulate_bscan(
    sample: Sample,
    source: LightSource | None = None,
    *,
    n_ascans: int = 512,
    lateral_extent_um: float = 1000.0,
    beam_waist_um: float = 15.0,
    reference_amp: float = 4.0,
    noise: NoiseModel | None = None,
    seed: int = 0,
) -> BscanResult:
    """Simulate one physically-modelled OCT B-scan.

    Parameters mirror a real acquisition: ``n_ascans`` lateral samples across
    ``lateral_extent_um`` with a Gaussian beam of ``1/e²`` waist ``beam_waist_um``.
    Pass a :class:`~octsim.noise.NoiseModel` (or ``NoiseModel.typical()``) to add
    a realistic floor; ``seed`` makes the speckle realisation reproducible.
    """
    source = source or LightSource()
    rng = np.random.default_rng(seed)

    x_axis = np.linspace(0.0, lateral_extent_um, n_ascans)
    depth_axis = source.depth_axis_um()
    reflectivity = build_reflectivity(
        sample, depth_axis, x_axis,
        beam_waist_um=beam_waist_um,
        lateral_extent_um=lateral_extent_um,
        rng=rng,
    )
    spectra = synthesize_bscan_spectra(
        reflectivity, source, reference_amp=reference_amp,
    )
    if noise is not None:
        spectra = noise.apply(spectra, rng)

    bscan = reconstruct_bscan(spectra)
    bscan_db = to_db(bscan)
    label_map = _rasterize_labels(sample, x_axis, depth_axis)

    return BscanResult(
        spectra=spectra,
        bscan=bscan,
        bscan_db=bscan_db,
        label_map=label_map,
        depth_axis_um=depth_axis,
        x_axis_um=x_axis,
        source=source,
        sample=sample,
    )


def _rasterize_labels(
    sample: Sample, x_axis: np.ndarray, depth_axis_um: np.ndarray
) -> np.ndarray:
    """Per-pixel layer index on the reconstruction grid (0 = above/below tissue).

    Uses the *optical* boundary depths, since that is the axis the reconstruction
    lives on — so a label lines up with the structure it describes.
    """
    opt_b = sample.optical_boundary_depths_um(x_axis)   # [L+1, Nx]
    n_depth = depth_axis_um.size
    labels = np.zeros((n_depth, x_axis.size), dtype=np.uint8)
    for i in range(len(sample.layers)):
        top = opt_b[i][None, :]
        bot = opt_b[i + 1][None, :]
        d = depth_axis_um[:, None]
        mask = (d >= top) & (d < bot)
        labels[mask] = i + 1
    return labels
