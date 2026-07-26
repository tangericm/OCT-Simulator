"""Depth-resolved optical attenuation coefficients from OCT A-lines.

Implements the single-scattering, energy-conservation estimator of Vermeer et al.,
"Depth-resolved model-based reconstruction of attenuation coefficients in optical
coherence tomography", *Biomed Opt Express* 2014;5(1):322-37
(https://doi.org/10.1364/BOE.5.000322):

    mu[i] ~= I[i] / (2 * dz * sum_{j>i} I[j])

The idea is that the light remaining below pixel ``i`` is proportional to everything the
detector still records below it, so the local attenuation follows from the ratio of the
signal at a depth to the integral beneath it — no model fitting and no per-layer slope
regression.

Why this matters here: it returns **mu in physical units (1/µm)**, a property of the
tissue, whereas matching a layer's brightness to a particular B-scan yields a number that
has silently absorbed that device's gain, tone curve and reference power. The simulator's
layer table was built the second way and is therefore only valid at 840 nm on a Topcon
Maestro; this module is the route to values that transfer.

**Known limitation, measured here — read before using the output as per-layer mu.**
The derivation assumes the backscatter-to-attenuation ratio is CONSTANT with depth. The
retina violates that badly: the RPE is roughly 30x more reflective than the inner retina.
On a synthetic A-line with uniform true mu = 4 /mm and a 30x bright band inserted, this
estimator returns **24.8 /mm inside the band** and 3.3 /mm above it (4.0 /mm below, where
the assumption is undisturbed). It converts backscatter contrast into apparent attenuation.

Measured on the Maestro3 corpus that shows up as ~27 /mm at the RPE and 0.1-0.5 /mm through
the inner retina, against published RNFL attenuation of ~4-5 /mm — the inner retina biased
low by an order of magnitude because most of the light it passes is returned by the bright
RPE beneath it, inflating the denominator.

So the output is trustworthy for the *shape* of the depth profile and for regions of
roughly uniform reflectivity (choroid, deep tail), and is NOT a drop-in source of per-layer
mu for the layer table. Separating mu from backscatter needs either segmentation plus a
forward model, or a joint fit of both against the measured A-line.

Two hard requirements, both easy to violate:

* **Linear intensity, not dB and not amplitude.** A log-domain or 8-bit display image
  gives a meaningless answer, which rules out every public 8-bit corpus and leaves the
  Maestro ``.raw`` reconstruction as the only usable source here.
* **The noise floor must be removed.** The estimator divides by the tail sum; if the tail
  is noise rather than signal, the denominator is dominated by the noise pedestal and mu
  is driven toward zero exactly where the signal is weakest (deep choroid).
"""

from __future__ import annotations

import numpy as np


def depth_resolved_mu(intensity: np.ndarray, dz_um: float, *,
                      noise_floor: float | None = None,
                      tail_floor: float = 1e-9) -> np.ndarray:
    """Per-pixel attenuation coefficient (1/µm) for ``intensity`` ``[n_depth, n_x]``.

    ``intensity`` must be LINEAR intensity with depth on axis 0. ``noise_floor`` is
    subtracted first; pass ``None`` to estimate it from the shallowest 5 % of rows.
    """
    I = np.asarray(intensity, float)
    if I.ndim == 1:
        I = I[:, None]
    n_depth = I.shape[0]

    if noise_floor is None:
        noise_floor = float(np.median(I[: max(n_depth // 20, 1)]))
    I = np.clip(I - float(noise_floor), 0.0, None)

    # sum strictly BELOW each pixel: reverse-cumsum shifted by one
    tail = np.cumsum(I[::-1], axis=0)[::-1]
    below = np.empty_like(tail)
    below[:-1] = tail[1:]
    below[-1] = 0.0

    return I / (2.0 * float(dz_um) * np.maximum(below, tail_floor))


def mu_profile(frames, dz_um: float, *, anchor_rows, ref_row: int,
               noise_floor=None, lo_frac: float = 0.1, hi_frac: float = 0.9):
    """Mean mu(depth) profile aligned to a per-frame anchor row.

    ``frames`` is a list of linear-intensity B-scans, ``anchor_rows`` the matching list of
    per-column anchor rows (e.g. from :func:`octsim.realism_metrics.find_rpe`). Returns
    ``mu[n_depth]`` on a depth axis whose index ``ref_row`` is the anchor.
    """
    acc = None
    n = 0
    for img, rows in zip(frames, anchor_rows):
        mu = depth_resolved_mu(img, dz_um, noise_floor=noise_floor)
        n_depth, n_x = mu.shape
        lo, hi = int(lo_frac * n_x), int(hi_frac * n_x)
        idx = np.arange(n_depth)
        for c in range(lo, hi):
            shift = float(rows[c]) - ref_row
            col = np.interp(idx + shift, idx, mu[:, c], left=np.nan, right=np.nan)
            if acc is None:
                acc = np.zeros(n_depth)
                cnt = np.zeros(n_depth)
            ok = np.isfinite(col)
            acc[ok] += col[ok]
            cnt[ok] += 1.0
            n += 1
    if acc is None:
        raise ValueError("no frames")
    return acc / np.maximum(cnt, 1.0)


def per_layer_mu(mu_prof: np.ndarray, dz_um: float, ref_row: int,
                 layer_bounds_um) -> dict:
    """Median mu per layer, given each layer's ``(top, bottom)`` optical µm vs the anchor.

    Layer *positions* come from the (data-driven) shape model; only the mu *values* come
    from the real A-lines, so this does not require the real frames to be segmented.
    """
    out = {}
    n = mu_prof.size
    for name, (top_um, bot_um) in layer_bounds_um.items():
        a = int(round(ref_row + top_um / dz_um))
        b = int(round(ref_row + bot_um / dz_um))
        a, b = max(0, min(a, n)), max(0, min(b, n))
        if b - a < 1:
            continue
        seg = mu_prof[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[name] = float(np.median(seg))
    return out
