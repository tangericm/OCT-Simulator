"""Spectral-domain reconstruction: raw spectra -> B-scan.

Deliberately the textbook SD/SS-OCT inverse, so a synthetic spectrum round-trips
to the structure that produced it:

    DC removal  ->  Hann window  ->  IFFT along k  ->  keep positive-depth half
                ->  magnitude    ->  (optional) log compression

Because the source is sampled linearly in ``k`` (see :mod:`octsim.source`) no
k-linearisation resampling is needed here. Feeding these spectra through the
Line-Field-OCT ``lfoct_pipeline.reconstruction`` path (which adds the resample +
dispersion stages) is the intended external cross-check.
"""

from __future__ import annotations

import numpy as np

from .source import LightSource


def reconstruct_bscan(
    spectra: np.ndarray,
    *,
    remove_dc: bool = True,
    window: bool = True,
    dc_highpass_frac: float = 1.0 / 16.0,
    background: np.ndarray | None = None,
    source=None,
    dispersion_coeffs=None,
) -> np.ndarray:
    """Reconstruct a complex B-scan from raw spectra ``[n_k, n_x]``.

    DC / source-envelope handling (the strong ``P(k)``-shaped reference term must
    be removed or it swamps the image near zero delay):

    * ``background`` given -> subtract it column-wise (the physical
      reference-spectrum subtraction; e.g. ``P(k)·a_ref²``).
    * otherwise ``remove_dc`` applies a **spectral high-pass**: subtract a smooth
      moving-average envelope of width ``dc_highpass_frac · n_k``. This removes
      the slowly-varying source/autocorrelation background while preserving the
      interferometric fringes, and is self-contained (works on a lone A-scan).

    Numerical **dispersion compensation** is applied when both ``source`` and
    ``dispersion_coeffs`` are given: the DC-removed spectrum is multiplied by
    ``exp(-i·phi(k))`` (the conjugate of the forward dispersion phase) before the
    IFFT, sharpening the positive-depth peaks.

    Returns the positive-depth complex A-lines ``[n_k//2, n_x]``. Take ``np.abs``
    for magnitude or :func:`to_db` for a display image.
    """
    s = np.asarray(spectra, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if background is not None:
        s = s - np.asarray(background, dtype=np.float64).reshape(-1, 1)
    elif remove_dc:
        width = max(3, int(dc_highpass_frac * s.shape[0]))
        s = s - _moving_average(s, width)
    if window:
        s = s * np.hanning(s.shape[0])[:, None]
    sc = s.astype(np.complex128, copy=False)
    if source is not None and dispersion_coeffs:
        # The positive-depth half (full[:n_depth]) selects the exp(-i·2kz) fringe
        # component, which carries -phi under the forward's exp(+i·phi); compensating
        # it therefore multiplies by exp(+i·phi) = the forward dispersion phase.
        from .dispersion import dispersion_phase
        ph = dispersion_phase(source.k_grid(), source.k0, dispersion_coeffs)
        sc = sc * ph[:, None]
    full = np.fft.ifft(sc, axis=0)
    n_depth = s.shape[0] // 2
    return full[:n_depth, :]


def _moving_average(s: np.ndarray, width: int) -> np.ndarray:
    """Centered box moving-average along axis 0 (reflect-padded), numpy only.

    Used as the smooth-envelope estimate for the spectral high-pass. ``width`` is
    forced odd so the window is symmetric about each sample.
    """
    width = int(width) | 1
    pad = width // 2
    sp = np.pad(s, ((pad, pad), (0, 0)), mode="reflect")
    csum = np.concatenate([np.zeros((1, s.shape[1])), np.cumsum(sp, axis=0)], axis=0)
    return (csum[width:] - csum[:-width]) / width


def to_db(bscan_complex: np.ndarray, *, floor_db: float = -60.0) -> np.ndarray:
    """Magnitude -> dB, peak-normalised to 0 dB and clipped at ``floor_db``."""
    mag = np.abs(bscan_complex)
    peak = mag.max() or 1.0
    db = 20.0 * np.log10(np.maximum(mag / peak, 1e-12))
    return np.clip(db, floor_db, 0.0)


def axial_fwhm_um(ascan_complex: np.ndarray, source: LightSource) -> float:
    """Full-width-at-half-maximum (µm, optical path) of the brightest peak in a
    single reconstructed A-line — used to validate the axial PSF against
    :attr:`LightSource.axial_resolution_um`."""
    mag = np.abs(np.asarray(ascan_complex).ravel())
    p = int(np.argmax(mag))
    half = mag[p] / 2.0

    # left crossing: last index l < p with mag[l] <= half < mag[l+1]
    left = p
    while left > 0 and mag[left - 1] > half:
        left -= 1
    if left == 0:
        x_left = 0.0
    else:
        lo, hi = mag[left - 1], mag[left]
        x_left = (left - 1) + (half - lo) / (hi - lo)

    # right crossing: first index r > p with mag[r] <= half < mag[r-1]
    right = p
    while right < mag.size - 1 and mag[right + 1] > half:
        right += 1
    if right == mag.size - 1:
        x_right = float(right)
    else:
        hi, lo = mag[right], mag[right + 1]
        x_right = right + (hi - half) / (hi - lo)

    return (x_right - x_left) * source.depth_pixel_um
