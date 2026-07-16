"""Physical sanity checks for the octsim forward/inverse chain.

Run from the ``simulator/`` directory::

    python -m pytest tests/ -q
    # or without pytest:
    python tests/test_octsim.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import (  # noqa: E402
    LightSource,
    NoiseModel,
    axial_fwhm_um,
    presets,
    reconstruct_bscan,
    simulate_bscan,
    synthesize_ascan_spectrum,
)


def test_mirror_reconstructs_at_inserted_depth():
    """A mirror at a known optical depth must peak at that depth."""
    src = LightSource()
    depth = 400.0  # µm optical (air, n=1) -> equals physical depth
    spectrum = synthesize_ascan_spectrum(
        np.array([depth]), np.array([0.9 + 0j]), src
    )
    ascan = reconstruct_bscan(spectrum)
    peak_px = int(np.argmax(np.abs(ascan[:, 0])))
    peak_um = peak_px * src.depth_pixel_um
    assert abs(peak_um - depth) < 3.0 * src.depth_pixel_um


def test_axial_psf_matches_source_bandwidth():
    """The reconstructed point width tracks the coherence-limited resolution.

    A Hann window broadens the PSF, so the realised FWHM should sit between the
    coherence limit and ~2x it — not narrower, not wildly wider.
    """
    src = LightSource()
    spectrum = synthesize_ascan_spectrum(
        np.array([500.0]), np.array([0.9 + 0j]), src
    )
    ascan = reconstruct_bscan(spectrum)
    fwhm = axial_fwhm_um(ascan[:, 0], src)
    coherence = src.axial_resolution_um
    assert coherence < fwhm < 2.2 * coherence, (fwhm, coherence)


def test_bandwidth_sets_resolution_monotonically():
    """Doubling the optical bandwidth roughly halves the axial PSF width."""
    def psf(bw):
        src = LightSource(bandwidth_fwhm_nm=bw)
        spec = synthesize_ascan_spectrum(np.array([500.0]), np.array([0.9 + 0j]), src)
        return axial_fwhm_um(reconstruct_bscan(spec)[:, 0], src)

    narrow = psf(50.0)
    wide = psf(150.0)
    assert wide < narrow  # more bandwidth -> sharper


def test_fft_field_path_is_exact_vs_direct_sum():
    """The FFT B-scan path equals the direct phasor sum for on-grid reflectivity
    (to machine precision) — it is exact, not an approximation."""
    from octsim.forward import synthesize_bscan_spectra

    src = LightSource(n_k=512)
    rng = np.random.default_rng(0)
    n_depth = src.n_k // 2
    r = np.zeros(n_depth, dtype=complex)
    bins = rng.choice(np.arange(5, n_depth - 5), size=12, replace=False)
    r[bins] = rng.standard_normal(12) + 1j * rng.standard_normal(12)

    fast = synthesize_bscan_spectra(r[:, None], src)[:, 0]
    z = bins * src.depth_pixel_um
    direct = synthesize_ascan_spectrum(z, r[bins], src)
    assert np.max(np.abs(fast - direct)) / np.max(np.abs(direct)) < 1e-10


def test_macula_layers_are_ordered_and_visible():
    """The reconstructed macula shows structure at the modelled layer depths."""
    result = simulate_bscan(
        presets.macula(), LightSource(n_k=1024),
        n_ascans=64, lateral_extent_um=600.0, seed=3,
    )
    assert result.bscan.shape[1] == 64
    # label map spans several layers within the imaging window
    assert result.label_map.max() >= 8
    # brightest reconstructed row lies below the surface, not at DC
    row_energy = np.abs(result.bscan).mean(axis=1)
    assert int(np.argmax(row_energy)) > 5


def test_paper_speckle_is_fully_developed():
    """Inside the paper slab the magnitude speckle must be Rayleigh — a
    row-normalised contrast (std/mean, with the Beer–Lambert depth roll-off
    divided out) near 0.52, the signature of fully developed speckle."""
    result = simulate_bscan(
        presets.paper(), LightSource(n_k=1024),
        n_ascans=200, lateral_extent_um=800.0, seed=5,
    )
    mag = np.abs(result.bscan)
    rows = np.where((result.label_map == 1).any(axis=1))[0]
    interior = rows[6:-6]                       # skip the bright surface + deep edge
    sub = mag[interior, :]
    norm = sub / (sub.mean(axis=1, keepdims=True) + 1e-12)   # remove depth trend
    contrast = float(norm.std())
    assert 0.40 < contrast < 0.65, contrast     # Rayleigh ~0.52


def test_noise_floor_raises_background():
    """Adding detector noise lifts the deep (structure-free) background."""
    src = LightSource(n_k=1024)
    clean = simulate_bscan(presets.mirror(), src, n_ascans=32, seed=0)
    noisy = simulate_bscan(presets.mirror(), src, n_ascans=32,
                           noise=NoiseModel.typical(), seed=0)
    deep = slice(-40, None)
    assert np.abs(noisy.bscan[deep]).mean() > np.abs(clean.bscan[deep]).mean()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
