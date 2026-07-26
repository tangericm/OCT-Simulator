"""Quantitative sim-vs-real distance metrics for retinal B-scans.

The calibration objective in :mod:`octsim.calibrate` scores a **peak-normalised dB
image with no layer segmentation of the real target**. That objective is nearly blind
to the dominant realism defect — a peak-normalised histogram barely moves when 13 dB
of tissue signal is pushed into the noise floor, because the floor dominates the pixel
count either way. This module supplies the statistics that *are* sensitive to it.

Everything here operates on a **dB B-scan in octsim's display domain** (``volume_db``
for simulated frames, :func:`octsim.data.real_maestro.recon_frame` for real ones), so
real and simulated images are measured by literally the same code path.

The measurement protocol is:

1. Flatten to the RPE (brightest row per column, median-filtered, integer roll), so
   curvature does not smear the axial bands.
2. Locate the ILM as the first sustained rise above the vitreous floor.
3. Slice five anatomical bands at fixed **optical-µm** offsets from those two
   landmarks, so the bands land on the same tissue at any pixel pitch.
4. Reduce each band to floor-referenced statistics.

All band edges are optical µm (``Σ n·dz``); divide by ~1.38 for tissue µm.

Metric families (see :class:`RealismMetrics`):

* band medians and **floor-referenced** linear intensity
* contrast pairs (RPE−ONL, inner−vitreous, choroid−ONL)
* how much tissue is buried in the noise floor
* speckle amplitude µ/σ, intensity contrast, axial and lateral grain
* choroid structural contrast and dark-lumen fraction (speckle smoothed away first)
* vessel-shadow count / width / depth distribution
* RPE-band sag across the FOV, ILM→RPE thickness
* RPE-complex A-line shape: how many maxima are resolved and how far apart
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# --- band geometry, optical µm relative to the ILM / RPE landmarks -------------------
#: vitreous sampling window, above the ILM (avoids both the DC band and the ILM itself)
VITREOUS_BAND = (-230.0, -40.0)
#: NFL..OPL, ~115 µm of tissue at n≈1.38
INNER_BAND = (0.0, 159.0)
#: ONL, measured ENTIRELY relative to the anchor. Anchoring its top to the ILM instead made
#: the window span OPL->ONL, so a genuinely dark ONL could be masked by a correctly-bright
#: OPL and the band statistic reported a match while the profile showed a 6 dB deficit.
ONL_FROM_ANCHOR = (-235.0, -65.0)
ONL_TOP = 159.0
#: outer complex: EZ, the dark outer-segment band, IZ and RPE. The anchor is the top of the
#: RPE and the EZ sits ~30 µm optical above it, so the window is asymmetric about it.
RPE_BAND = (-60.0, 40.0)
#: choroid, below the outer complex and above the typical choroid-sclera transition
CHOROID_BAND = (60.0, 360.0)

#: group index used to convert optical µm to tissue µm for reporting
TISSUE_N = 1.38

#: Axial resolution cell (optical µm) — the Maestro3 mirror-measured PSF FWHM. The
#: lamination gradient is reported per resolution cell rather than per pixel so it stays
#: comparable between corpora sampled at different depth pitches.
AXIAL_CELL_UM = 15.5

# --- metric tiers -------------------------------------------------------------------
# Which statistics remain meaningful on a corpus that is not in a linear domain.
#
# A device's display tone curve is an unknown MONOTONE intensity remap. It destroys every
# dB-valued quantity (band medians, contrasts, floor-referenced ratios, speckle amplitude
# statistics) but moves no structure, so geometry and spatial correlation lengths survive
# it intact. JPEG compression additionally corrupts fine texture, which removes the
# speckle-grain statistics as well.
#
# Reporting the full vector on 8-bit public data would produce confident-looking numbers
# that mean nothing, so the harness restricts instead.

#: spatial / geometric statistics — invariant to any monotone intensity remap.
#: ``ilm_row``/``rpe_row`` are deliberately EXCLUDED from every tier: they record where the
#: operator happened to place the retina in the axial window, not anything about the model,
#: and comparing them across corpora produces large meaningless deltas.
TIER_COMPRESSED = frozenset({
    "thickness_tissue_um", "sag_optical_um",
    "rpe_peak_separation_um", "shadow_median_width_um",
})

#: adds fine-texture statistics: valid on uncompressed display-domain data
TIER_DISPLAY = TIER_COMPRESSED | frozenset({
    "speckle_grain_axial_um", "speckle_grain_lateral_um", "choroid_lateral_corr_um",
    "inner_extrema",
})

#: everything (linear domain with a genuine noise floor)
TIER_FULL = None

TIERS = {"full": TIER_FULL, "display": TIER_DISPLAY, "compressed": TIER_COMPRESSED}


def restrict(agg: dict, tier) -> dict:
    """Drop aggregated metrics that the corpus's domain cannot support.

    ``tier`` is a name from :data:`TIERS` or a set of allowed metric names; ``None`` or
    ``"full"`` keeps everything.
    """
    if isinstance(tier, str):
        tier = TIERS.get(tier, TIER_FULL)
    if tier is None:
        return dict(agg)
    return {k: v for k, v in agg.items() if k.split(".")[0] in tier}


def _median_filter1d(a: np.ndarray, size: int) -> np.ndarray:
    """Median filter, SciPy if available, else a strided fallback (no silent no-op)."""
    try:
        from scipy.ndimage import median_filter
        return median_filter(a, size)
    except Exception:
        n = a.size
        half = size // 2
        pad = np.pad(a, half, mode="edge")
        return np.median(np.lib.stride_tricks.sliding_window_view(pad, size), axis=-1)[:n]


def _gaussian_filter(a: np.ndarray, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma, mode="nearest")
    except Exception:  # pragma: no cover - SciPy is a hard dep of the metrics
        raise RuntimeError("realism_metrics requires SciPy for the choroid statistics")


#: rows above this optical depth are excluded from landmark detection. Both real Maestro
#: frames and octsim output carry a bright DC / zero-delay band at the top of the frame
#: (real: residual reference power; sim: the 128-tap box high-pass at reconstruct.py:59).
#: With some optics profiles that band is BRIGHTER than the RPE, so a naive per-column
#: argmax locks onto it and every downstream landmark collapses.
DC_SKIP_UM = 230.0


def find_rpe(db: np.ndarray, dz_um: float = 1.0, skip_um: float = DC_SKIP_UM) -> np.ndarray:
    """Per-column depth row of the RPE complex.

    A plain per-column ``argmax`` is not usable here: it locks onto the DC band when that
    is bright, and — because speckle in a thick bright band throws high maxima — onto the
    inner retina in a large minority of columns. Both failure modes were observed on real
    *and* simulated frames.

    A "deepest strong peak" rule is not usable either, and the reason matters. In real
    B-scans the EZ maximum sits ~5.8 dB above the RPE maximum, i.e. the RPE is at 0.51 of
    the EZ amplitude — right on top of any fixed "at least half the peak" threshold. Such
    a rule therefore resolves to the RPE on one image and the EZ on another, silently
    shifting every band window by ~35 µm between the two images being compared.

    Restricting the search to a single global row range is not an option either: with
    300-600 µm of sag across a B-scan, one scalar cut sits below the outer complex on the
    shallow columns and above it on the deep ones.

    So the curve is found by **cross-correlating each A-line against a template built from
    the whole frame**, which uses the entire axial profile rather than one extreme value
    and is therefore insensitive to any single band being locally over-bright. The template
    is rebuilt once from the aligned columns, and its own maximum defines which structure
    the returned rows point at (the EZ, the brightest band of the aligned mean profile).
    """
    db = np.asarray(db, float)
    n_depth, n_x = db.shape
    skip = min(max(int(round(skip_um / dz_um)), 0), max(n_depth - 8, 0))
    lin = 10.0 ** (db[skip:] / 20.0)
    sm = _gaussian_filter(lin, (max(6.0 / dz_um, 1.0), 3.0))
    n = sm.shape[0]

    # The lag window has to stay UNAMBIGUOUS. The correlation below is circular, so lag L
    # and lag L-n are the same array element; the two slices that re-centre it overlap as
    # soon as 2*max_shift+1 > n, and every lag in the overlap then appears twice. argmax
    # returns the earlier copy, i.e. the negative alias — a genuine +125 px sag is reported
    # as -260 px, which corrupts the RPE curve, the flattening, and every geometry metric
    # derived from them. Measured on a synthetic 500-row frame at 2 µm/px with 250 µm of
    # sag: 375 px of error (≈ the full searchable depth) before this clamp, 12 px after.
    # Deep frames are unaffected — Maestro-like geometry never reaches the bound.
    max_shift = min(int(round(700.0 / dz_um)), max((n - 1) // 2, 1))
    prof = sm.mean(axis=1)
    shifts = np.zeros(n_x)
    for _ in range(2):
        t = prof - prof.mean()
        cols = sm - sm.mean(axis=0, keepdims=True)
        # circular cross-correlation of every column with the template, in one FFT pass
        cc = np.fft.irfft(np.fft.rfft(cols, axis=0)
                          * np.conj(np.fft.rfft(t)[:, None]), n=n, axis=0)
        cc = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]], axis=0)
        shifts = _median_filter1d((np.argmax(cc, axis=0) - max_shift).astype(float), 15)
        # rebuild the template from the aligned columns
        idx = (np.arange(n)[:, None] + shifts[None, :].astype(int)) % n
        prof = np.take_along_axis(sm, idx, axis=0).mean(axis=1)

    # Anchor = the DEEPEST local maximum of the aligned template reaching 10 % of its peak
    # amplitude. Not the global maximum: on a render whose NFL is over-bright the global
    # maximum is the NFL, ~100 rows above the outer complex, and every band would shift by
    # a whole retinal thickness relative to the same measurement on real data. The deepest
    # bright band is the RPE on any image where the outer complex is present at all, so the
    # anchor means the same thing on both.
    base = float(np.percentile(prof, 10))
    amp = prof - base
    thr = 0.10 * float(amp.max())
    is_max = np.zeros(prof.size, bool)
    is_max[1:-1] = (amp[1:-1] >= amp[:-2]) & (amp[1:-1] > amp[2:])
    cand = np.nonzero(is_max & (amp > thr))[0]
    anchor = float(cand[-1]) if cand.size else float(np.argmax(prof))
    return shifts + anchor + skip


def flatten_to_rpe(db: np.ndarray, ref_row: int | None = None,
                   dz_um: float = 1.0, skip_um: float = DC_SKIP_UM):
    """Roll every column so the RPE complex lands on ``ref_row``.

    Returns ``(flat, rpe_rows, ref_row)``. ``rpe_rows`` is the pre-roll RPE row per
    column — the raw material for the curvature-sag metric.
    """
    db = np.asarray(db, float)
    n_depth, n_x = db.shape
    rpe = find_rpe(db, dz_um, skip_um)
    if ref_row is None:
        ref_row = int(np.median(rpe))

    # Reject shift outliers before applying them. A single column whose anchor jumps a band
    # (typically under a dense vessel shadow) otherwise gets rolled by ~90 rows and stamps a
    # large rectangular block into the flattened image.
    smooth = _median_filter1d(rpe, 31)
    bad = np.abs(rpe - smooth) > (40.0 / dz_um)
    rpe = np.where(bad, smooth, rpe)
    # and smooth the curve itself: the cross-correlation estimate is integer-lag, so the raw
    # curve is a staircase and every step edge becomes a visible vertical discontinuity.
    rpe = _gaussian_filter(rpe, 3.0)

    # Sub-pixel shift by interpolation rather than np.roll. np.roll is (a) integer-only, so
    # the staircase survives, and (b) CIRCULAR, so content wraps from the bottom of the frame
    # to the top. Both were visible as blocky notches in flattened real B-scans.
    rows = np.arange(n_depth)
    flat = np.empty_like(db)
    for c in range(n_x):
        src = rows + (rpe[c] - ref_row)
        flat[:, c] = np.interp(src, rows, db[:, c], left=db[0, c], right=db[-1, c])
    return flat, rpe, int(ref_row)


def find_ilm_columns(db: np.ndarray, ez_rows: np.ndarray, dz_um: float,
                     skip_um: float = DC_SKIP_UM) -> np.ndarray:
    """Per-column ILM row.

    Band statistics are taken between the per-column ILM and the per-column EZ, not at
    fixed offsets from one scalar row. Two effects make a scalar offset unusable: 300+ µm
    of curvature moves the whole retina across the frame, and the foveal pit changes the
    ILM-to-EZ distance by ~35 rows between the pit and the parafovea. Anchoring the inner
    band at a constant offset below the EZ fills it ~76 % with vitreous on a 6 mm scan.

    The detector must also survive a low-contrast render. Searching each column
    independently on its own speckle does not: where the inner retina sits ~1 dB above the
    noise floor, vitreous speckle is indistinguishable from tissue, the search locks onto
    noise above the true ILM, the inner band fills with vitreous, and the band statistic
    then "confirms" that the inner retina is dim — a feedback loop that manufactures the
    very deficit it is meant to measure.

    The fix is to smooth **laterally** before searching. A 15-column kernel (~90 µm) averages
    speckle across neighbouring A-lines while staying far narrower than the pit (~700 µm),
    so per-column structure survives and the noise does not.

    The search window is defined **relative to that column's EZ** rather than as a global
    row range, so it follows the curvature instead of drifting off the tissue.
    """
    db = np.asarray(db, float)
    n_depth, n_x = db.shape
    skip = min(max(int(round(skip_um / dz_um)), 0), max(n_depth - 8, 0))
    sm = _gaussian_filter(10.0 ** (db / 20.0), (max(8.0 / dz_um, 1.0), 15.0))
    # The ILM lies between ~620 µm and ~140 µm of optical depth above the anchor. The near
    # edge must clear the whole outer complex: the ELM sits ~50 µm above the RPE, and if it
    # falls inside the search window it becomes the window maximum, inflating the relative
    # threshold so the ILM is "found" tens of rows late (or not at all).
    far = int(round(620.0 / dz_um))
    near = int(round(140.0 / dz_um))

    out = np.empty(n_x)
    for c in range(n_x):
        ez = int(ez_rows[c])
        a = max(skip, ez - far)
        b = max(a + 6, ez - near)
        seg = sm[a:b, c]
        if seg.size < 6:
            out[c] = a
            continue
        floor = float(np.percentile(seg, 10))
        thresh = floor + 0.30 * (float(seg.max()) - floor)
        hit = seg > thresh
        idx = np.nonzero(hit[:-2] & hit[1:-1] & hit[2:])[0]      # sustained 3 rows
        out[c] = (idx[0] + a) if idx.size else a
    return _median_filter1d(out, 15)


def find_ilm(flat: np.ndarray, ref_row: int, dz_um: float,
             skip_um: float = 230.0) -> tuple[int, float]:
    """First sustained rise above the vitreous floor, searching down from ``skip_um``.

    ``skip_um`` skips the DC / zero-delay band at the top of the frame, which is bright
    in real Maestro data (and in sim output, from the box high-pass in
    :mod:`octsim.reconstruct`) and would otherwise be mistaken for the ILM.
    """
    skip = max(int(round(skip_um / dz_um)), 1)
    prof = flat.mean(axis=1)
    hi = max(skip + 8, ref_row - int(round(60.0 / dz_um)))
    # A *median* over [skip, hi) is not a floor estimate: that window spans the whole inner
    # retina, so on a frame where the retina occupies most of it the "floor" comes out ~5 dB
    # into the tissue and the ILM search then only trips at the EZ. A low percentile picks
    # the vitreous regardless of how much of the window is tissue.
    floor = (float(np.percentile(prof[skip:hi], 10)) if hi > skip
             else float(prof[:ref_row].min()))
    # The threshold has to scale with the image's tissue dynamic range. A fixed +6 dB works
    # on real frames (inner retina ~16 dB above floor) but never trips on an under-exposed
    # render whose inner retina sits ~4 dB above floor — the detector would then skip the
    # whole neurosensory retina and "find" the ILM at the EZ. Smoothing first keeps a bumpy
    # inner-retinal profile from dropping back under the threshold between layers.
    sm = _gaussian_filter(prof, 2.0)
    thresh = floor + max(3.0, 0.08 * (float(sm[skip:ref_row + 1].max()) - floor))
    for z in range(skip, ref_row):
        if (sm[z:z + 3] > thresh).all():
            return z, floor
    return skip, floor


def _acf_fwhm(f: np.ndarray, axis: int, max_lag: int = 14) -> float:
    """Half-maximum lag of the normalised autocorrelation along ``axis``, in pixels."""
    n = f.shape[axis]
    max_lag = min(max_lag, n - 1)
    c = []
    for lag in range(max_lag + 1):
        a = np.take(f, range(0, n - lag), axis=axis)
        b = np.take(f, range(lag, n), axis=axis)
        c.append(float((a * b).mean()))
    c = np.asarray(c)
    if c[0] <= 0:
        return 0.0
    c = c / c[0]
    below = np.nonzero(c < 0.5)[0]
    if below.size == 0:
        return float(max_lag)
    k = int(below[0])
    if k == 0:
        return 0.0
    # linear interpolation onto the 0.5 crossing
    return (k - 1) + (c[k - 1] - 0.5) / (c[k - 1] - c[k] + 1e-12)


@dataclass
class RealismMetrics:
    """One B-scan's realism statistics. All dB unless noted."""

    # landmarks
    ilm_row: int = 0
    rpe_row: int = 0
    noise_floor_db: float = 0.0
    thickness_tissue_um: float = 0.0        # ILM -> RPE peak, median across columns
    sag_optical_um: float = 0.0             # RPE-band peak-to-peak across the FOV

    # band medians
    band_db: dict = field(default_factory=dict)
    #: median linear intensity above the noise floor, per band, expressed in dB
    band_above_floor_db: dict = field(default_factory=dict)

    # contrasts
    contrast_rpe_onl: float = 0.0
    contrast_inner_vitreous: float = 0.0
    contrast_choroid_onl: float = 0.0

    # exposure
    frac_tissue_in_floor: float = 0.0       # fraction within 3 dB of the floor
    median_tissue_snr_db: float = 0.0

    # speckle (measured in the inner retina, the best-exposed homogeneous band)
    speckle_amp_mu_over_sigma: float = 0.0  # 1.91 for fully-developed Rayleigh
    speckle_intensity_contrast: float = 0.0  # 1.0 for fully developed
    speckle_grain_axial_um: float = 0.0
    speckle_grain_lateral_um: float = 0.0

    # choroid structure, after smoothing speckle away
    choroid_structure_sd_db: float = 0.0
    choroid_dark_frac_4db: float = 0.0
    choroid_dark_frac_8db: float = 0.0
    choroid_lateral_corr_um: float = 0.0

    # vessel shadows
    shadow_count: int = 0
    shadow_max_depth_db: float = 0.0
    shadow_median_width_um: float = 0.0
    column_energy_sd_db: float = 0.0

    # RPE-complex axial shape
    rpe_peak_count: int = 0
    rpe_peak_separation_um: float = 0.0     # optical µm between the two strongest maxima

    # inner-retinal lamination: is the NFL..OPL block visibly layered, or a flat slab?
    inner_extrema: int = 0                  # resolvable maxima+minima in the ILM->OPL profile
    inner_gradient_db: float = 0.0          # RMS axial gradient there, dB per resolution cell
    inner_swing_db: float = 0.0             # p95-p5 of the profile across the band

    def as_dict(self) -> dict:
        return asdict(self)


def measure(db: np.ndarray, dz_um: float, dx_um: float) -> RealismMetrics:
    """Compute the full metric set for one dB B-scan.

    ``dz_um`` is the **optical** depth pixel pitch, ``dx_um`` the lateral pitch.
    """
    db = np.asarray(db, float)
    flat, rpe_rows, ref = flatten_to_rpe(db, dz_um=dz_um)
    ilm, floor = find_ilm(flat, ref, dz_um)
    m = RealismMetrics(ilm_row=ilm, rpe_row=ref, noise_floor_db=floor)

    px = lambda um: int(round(um / dz_um))
    clip = lambda a, b: (max(0, min(a, flat.shape[0])), max(0, min(b, flat.shape[0])))
    # The vitreous window is the floor reference for every other band, so it must not reach
    # up into the DC / zero-delay residual. When the sample sits shallow in the depth window
    # an unclamped window overlaps it, inflating the "floor" by >10 dB and making every
    # tissue band look as if it were buried in noise.
    dc_row = px(DC_SKIP_UM)

    # --- per-column band masks, taken on the UNFLATTENED image. The prior comes from the
    # flattened column-mean profile (speckle averaged away), then un-flattened per column.
    ilm_rows = find_ilm_columns(db, rpe_rows, dz_um)
    m.ilm_row = int(np.median(ilm_rows))
    rows = np.arange(db.shape[0])[:, None]
    ilm_c = ilm_rows[None, :]
    ez_c = rpe_rows[None, :]
    span = lambda lo, hi: (rows >= lo) & (rows < hi)
    masks = {
        "vitreous": span(np.maximum(ilm_c + px(VITREOUS_BAND[0]), dc_row),
                         ilm_c + px(VITREOUS_BAND[1])),
        "inner": span(ilm_c + px(INNER_BAND[0]), ilm_c + px(INNER_BAND[1])),
        "onl": span(ez_c + px(ONL_FROM_ANCHOR[0]), ez_c + px(ONL_FROM_ANCHOR[1])),
        "rpe_complex": span(ez_c + px(RPE_BAND[0]), ez_c + px(RPE_BAND[1])),
        "choroid": span(ez_c + px(CHOROID_BAND[0]), ez_c + px(CHOROID_BAND[1])),
    }

    raw_i = (10.0 ** (db / 20.0)) ** 2            # linear intensity, unflattened
    vm = masks["vitreous"]
    floor_i = float(np.median(raw_i[vm])) if vm.any() else 1e-12

    for name, msk in masks.items():
        if not msk.any():
            continue
        m.band_db[name] = float(np.median(db[msk]))
        if name == "vitreous":
            continue        # the vitreous IS the floor; the ratio is degenerate
        above = max(float(np.median(raw_i[msk])) - floor_i, 1e-12)
        m.band_above_floor_db[name] = float(10.0 * np.log10(above / max(floor_i, 1e-12)))

    # Flattened windows, used only for the statistics that need axial alignment across
    # columns (speckle autocorrelation, choroid texture, shadow columns).
    lin_i = (10.0 ** (flat / 20.0)) ** 2
    ilm_off = int(np.median(ilm_rows - rpe_rows))
    bands = {
        "inner": clip(ref + ilm_off + px(INNER_BAND[0]), ref + ilm_off + px(INNER_BAND[1])),
        "rpe_complex": clip(ref + px(RPE_BAND[0]), ref + px(RPE_BAND[1])),
        "choroid": clip(ref + px(CHOROID_BAND[0]), ref + px(CHOROID_BAND[1])),
    }

    g = m.band_db.get
    m.contrast_rpe_onl = g("rpe_complex", 0.0) - g("onl", 0.0)
    m.contrast_inner_vitreous = g("inner", 0.0) - g("vitreous", 0.0)
    m.contrast_choroid_onl = g("choroid", 0.0) - g("onl", 0.0)

    # --- exposure: how much neurosensory tissue is buried in the floor
    floor_db_abs = 10.0 * np.log10(max(floor_i, 1e-12))
    tis_mask = masks["inner"] | masks["onl"]
    if tis_mask.any():
        tis = 10.0 * np.log10(raw_i[tis_mask] + 1e-12)
        m.frac_tissue_in_floor = float(np.mean(tis < floor_db_abs + 3.0))
        m.median_tissue_snr_db = float(np.median(tis) - floor_db_abs)

    # --- geometry
    # p2-p98 rather than min-max: a handful of columns where the anchor jumps a band (under
    # a vessel shadow, or at the FOV edge) would otherwise dominate a peak-to-peak sag.
    m.sag_optical_um = float((np.percentile(rpe_rows, 98) - np.percentile(rpe_rows, 2)) * dz_um)
    m.thickness_tissue_um = float(np.median(rpe_rows - ilm_rows) * dz_um / TISSUE_N)

    # --- speckle, inner retina, margins trimmed so band edges don't bias the ACF
    a, b = bands["inner"]
    a, b = a + 2, max(a + 6, b - 2)
    lo, hi = int(0.1 * flat.shape[1]), int(0.9 * flat.shape[1])
    amp = 10.0 ** (flat[a:b, lo:hi] / 20.0)
    if amp.size > 64:
        # Divide out the deterministic per-row layer brightness. Without this the NFL->OPL
        # brightness ladder inflates sigma and the statistic reports layer contrast, not
        # speckle. What is left is the multiplicative speckle fluctuation.
        amp = amp / (np.median(amp, axis=1, keepdims=True) + 1e-12)
        inten = amp ** 2
        m.speckle_amp_mu_over_sigma = float(amp.mean() / (amp.std() + 1e-12))
        m.speckle_intensity_contrast = float(inten.std() / (inten.mean() + 1e-12))
        # detrend on a window well above the expected grain so the ACF is not whitened
        win = (min(41, inten.shape[0] // 2 * 2 + 1), 41)
        fluct = inten / (_gaussian_filter(inten, (win[0] / 4.0, win[1] / 4.0)) + 1e-12) - 1.0
        m.speckle_grain_axial_um = float(_acf_fwhm(fluct, 0) * dz_um)
        m.speckle_grain_lateral_um = float(_acf_fwhm(fluct, 1) * dx_um)

    # --- choroid structure: smooth speckle away, then measure what is left
    a, b = bands["choroid"]
    ch = lin_i[a:b, lo:hi]
    if ch.size > 256:
        sm = _gaussian_filter(ch, (3.0, 3.0))
        smdb = 10.0 * np.log10(np.maximum(sm, 1e-12))
        m.choroid_structure_sd_db = float(smdb.std())
        loc = _gaussian_filter(smdb, (10.0, 20.0))
        m.choroid_dark_frac_4db = float(np.mean(smdb - loc < -4.0))
        m.choroid_dark_frac_8db = float(np.mean(smdb - loc < -8.0))
        # detrend at ~400 µm so the ACF measures lumen-scale structure rather than the
        # slow lateral illumination gradient (which would saturate it at max_lag)
        f = smdb - _gaussian_filter(smdb, (0.0, max(400.0 / dx_um, 1.0)))
        m.choroid_lateral_corr_um = float(_acf_fwhm(f, 1, max_lag=40) * dx_um)

    # --- vessel shadows: per-column energy in the RPE complex vs its lateral trend
    a, b = bands["rpe_complex"]
    col_e = 10.0 * np.log10(lin_i[a:b].mean(axis=0) + 1e-12)
    trend = _median_filter1d(col_e, min(151, (col_e.size // 2) * 2 + 1))
    dip = col_e - trend
    m.column_energy_sd_db = float(dip.std())
    runs, i, n = [], 0, dip.size
    while i < n:
        if dip[i] < -2.0:
            j = i
            while j < n and dip[j] < -2.0:
                j += 1
            if j - i >= 3:
                runs.append((j - i, float(dip[i:j].min())))
            i = j
        else:
            i += 1
    m.shadow_count = len(runs)
    if runs:
        m.shadow_max_depth_db = float(min(r[1] for r in runs))
        m.shadow_median_width_um = float(np.median([r[0] for r in runs]) * dx_um)

    # --- inner-retinal lamination.
    # Band medians alone cannot distinguish a correctly-flat inner-retinal plateau from a
    # featureless one: matching every band statistic is possible while rendering NFL/GCL/
    # IPL/INL/OPL as one homogeneous slab, which is exactly what happened when the inner
    # bright/dark alternation was compressed to match a measured "flat" plateau. These
    # three statistics are sensitive to the presence of boundaries rather than to the mean
    # level, so a fit can no longer erase layer structure without the score noticing.
    # Measured PER COLUMN, then reduced across columns. Averaging the A-lines first is the
    # wrong domain: real retinal thickness varies along the B-scan, so after RPE-flattening
    # the inner layers remain misaligned column-to-column and the mean profile smears its
    # own lamination away. That makes a genuinely laminated real frame score *lower* than a
    # featureless simulated one — the opposite of what the eye sees.
    a, b = clip(ref + ilm_off + px(10.0), ref + ilm_off + px(INNER_BAND[1]))
    lo2, hi2 = int(0.1 * flat.shape[1]), int(0.9 * flat.shape[1])
    if b - a > 6 and hi2 > lo2:
        blk = flat[a:b, lo2:hi2]
        # smooth along depth only, at the resolution cell, and laterally enough to suppress
        # speckle without merging adjacent A-lines' structure
        blk = _gaussian_filter(blk, (max(0.6 * AXIAL_CELL_UM / dz_um, 0.8), 6.0))
        d1 = np.diff(blk, axis=0)
        m.inner_gradient_db = float(np.median(np.sqrt(np.mean(d1 ** 2, axis=0)))
                                    * (AXIAL_CELL_UM / dz_um))
        m.inner_swing_db = float(np.median(np.percentile(blk, 95, axis=0)
                                           - np.percentile(blk, 5, axis=0)))
        dead = 0.15 * float(np.std(d1))
        sig = np.sign(np.where(np.abs(d1) < dead, 0.0, d1))
        counts = [int(np.count_nonzero(np.diff(s[s != 0]) != 0)) for s in sig.T]
        m.inner_extrema = int(np.median(counts)) if counts else 0

    # --- RPE-complex axial shape: count resolved maxima in the mean A-line
    a, b = clip(ref - px(60.0), ref + px(60.0))
    seg = flat[a:b].mean(axis=1)
    if seg.size > 6:
        peaks = [k for k in range(1, seg.size - 1)
                 if seg[k] >= seg[k - 1] and seg[k] > seg[k + 1] and seg[k] > seg.max() - 12.0]
        # merge maxima closer than one axial resolution cell
        merged = []
        for k in peaks:
            if merged and (k - merged[-1]) * dz_um < 12.0:
                if seg[k] > seg[merged[-1]]:
                    merged[-1] = k
            else:
                merged.append(k)
        m.rpe_peak_count = len(merged)
        if len(merged) >= 2:
            top2 = sorted(sorted(merged, key=lambda k: -seg[k])[:2])
            m.rpe_peak_separation_um = float((top2[1] - top2[0]) * dz_um)

    return m


def aggregate(runs: list[RealismMetrics]) -> dict:
    """Mean and sd of every scalar metric across frames/seeds.

    Returns ``{metric: (mean, sd)}``; ``band_db`` / ``band_above_floor_db`` are
    flattened to ``band_db.inner`` style keys.
    """
    flat_runs = []
    for r in runs:
        d = {}
        for k, v in r.as_dict().items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    d[f"{k}.{kk}"] = float(vv)
            else:
                d[k] = float(v)
        flat_runs.append(d)
    keys = sorted(set().union(*(set(d) for d in flat_runs))) if flat_runs else []
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in flat_runs if k in d], float)
        out[k] = (float(vals.mean()), float(vals.std()))
    return out


#: metrics where the sim must match real, with the acceptance tolerance from the plan.
#: ``None`` tolerance means "report the delta, do not gate on it".
GATES = {
    "contrast_inner_vitreous": 3.0,
    "contrast_rpe_onl": 4.0,
    "frac_tissue_in_floor": 0.11,       # real ~0.14, allow up to ~0.25
    "choroid_structure_sd_db": 2.6,     # real ~8.6, allow >= 6.0
    "choroid_dark_frac_8db": 0.036,     # real ~0.066, allow >= 0.03
    "speckle_grain_lateral_um": 3.0,
    "sag_optical_um": 150.0,
    # Guards against a fit flattening the inner retina into a featureless slab while every
    # band median still matches. Measured per column (see `measure`), not on the mean
    # A-line, which smears real lamination away.
    "inner_swing_db": 2.5,
    "inner_gradient_db": 0.8,
}


def compare(real: dict, sim: dict) -> list[tuple]:
    """Signed sim-minus-real deltas.

    Both arguments come from :func:`aggregate`. Returns rows of
    ``(metric, real_mean, real_sd, sim_mean, sim_sd, delta, tolerance, passed)``
    sorted so gated metrics come first.
    """
    rows = []
    for k in sorted(set(real) | set(sim)):
        if k not in real or k not in sim:
            continue
        rm, rs = real[k]
        sm, ss = sim[k]
        tol = GATES.get(k)
        delta = sm - rm
        passed = None if tol is None else bool(abs(delta) <= tol)
        rows.append((k, rm, rs, sm, ss, delta, tol, passed))
    rows.sort(key=lambda r: (r[6] is None, r[0]))
    return rows


def format_table(rows: list[tuple]) -> str:
    """Render :func:`compare` output as a fixed-width table."""
    out = [f"{'metric':<34}{'real':>12}{'sim':>12}{'delta':>10}{'tol':>8}  gate",
           "-" * 84]
    for k, rm, rs, sm, ss, delta, tol, passed in rows:
        gate = "" if passed is None else ("PASS" if passed else "FAIL")
        tol_s = "" if tol is None else f"{tol:.3g}"
        out.append(f"{k:<34}{rm:>8.3f}±{rs:<3.2g}{sm:>8.3f}±{ss:<3.2g}"
                   f"{delta:>+10.3f}{tol_s:>8}  {gate}")
    return "\n".join(out)
