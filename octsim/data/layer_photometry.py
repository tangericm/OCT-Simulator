"""Layer photometry from labelled B-scans — the R1c bridge when unlabeled D cannot ID µ.

Unlabeled texture distance confounds µ↔backscatter and cannot see the choroid (R1a/R1b).
Given a dB B-scan + integer label map (sim labels are free; real device labels come from
a segmenter or manual panel), this module measures per-layer mean dB and local depth slope
as independent photometry targets for backscatter ratios.

Device A does not yet ship layer labels in-repo (OCT Segmentation is empty). Use:

* synthetic labels from ``simulate(...).labels2d`` for sim-side checks;
* OCTA-500 ``GT_Layers`` via :func:`from_octa500_gt` when that corpus is local;
* a future device label folder once the segmenter exists.

Also provides **parafoveal** diagnostics that avoid the full-FOV mean A-line (which washes
out RNFL by mixing the foveal pit with thick-NFL columns):

* :func:`parafoveal_column_mask` — annulus about the ILM pit
* :func:`parafoveal_mean_aline` — mean A-line on that mask (RPE-flattened)
* :func:`layer_band_means` — NFL / inner_stack / ONL / RPE mean dB (labels or landmarks)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

#: Lateral annulus about the ILM pit used for parafoveal tissue diagnostics (mm).
PARAFOVEAL_R_INNER_MM = 0.75
PARAFOVEAL_R_OUTER_MM = 1.50

#: Canonical report bands (order used by bar plots).
BAND_ORDER = ("NFL", "inner_stack", "ONL", "RPE")

#: Sim label names rolled into ``inner_stack`` when a label map is available.
INNER_STACK_LABELS = ("GCL", "IPL", "INL", "OPL")

#: Landmark windows (optical µm) from ILM / RPE when labels are absent.
#: NFL: thin sheet under ILM (~30 µm tissue × n≈1.38).
NFL_FROM_ILM = (0.0, 45.0)
#: GCL..OPL remainder of realism_metrics.INNER_BAND after the NFL window.
INNER_STACK_FROM_ILM = (45.0, 159.0)


@dataclass
class LayerPhotometry:
    """Per-layer brightness / slope fingerprint for one corpus."""

    mean_db: dict = field(default_factory=dict)       # layer -> mean dB
    slope_db_per_mm: dict = field(default_factory=dict)  # layer -> depth slope
    n_frames: int = 0
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LayerPhotometry":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "LayerPhotometry":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def measure_frame(db: np.ndarray, labels: np.ndarray, names: list,
                  *, axial_pitch_um: float = 3.87) -> tuple[dict, dict]:
    """Return ``(mean_db, slope_db_per_mm)`` for one labelled frame.

    Label 0 = background; label i (1-based) = ``names[i-1]``. Slope is a per-column
    linear fit of dB vs depth inside the layer, averaged over columns with ≥4 pixels.
    """
    db = np.asarray(db, float)
    lab = np.asarray(labels, int)
    means, slopes = {}, {}
    depth_um = np.arange(db.shape[0], dtype=float) * float(axial_pitch_um)
    for i, nm in enumerate(names, start=1):
        m = lab == i
        if not np.any(m):
            means[nm] = float("nan")
            slopes[nm] = float("nan")
            continue
        means[nm] = float(db[m].mean())
        col_slopes = []
        for c in range(db.shape[1]):
            rows = np.where(lab[:, c] == i)[0]
            if rows.size < 4:
                continue
            y = db[rows, c]
            x = depth_um[rows]
            # db per mm
            coef = np.polyfit(x, y, 1)[0] * 1000.0
            col_slopes.append(float(coef))
        slopes[nm] = float(np.mean(col_slopes)) if col_slopes else float("nan")
    return means, slopes


def measure_corpus(frames_db, frames_labels, names: list, *,
                   axial_pitch_um: float = 3.87, source: str = "") -> LayerPhotometry:
    """Pool photometry over many frames (nanmean)."""
    acc_m, acc_s = {n: [] for n in names}, {n: [] for n in names}
    n_ok = 0
    for db, lab in zip(frames_db, frames_labels):
        m, s = measure_frame(db, lab, names, axial_pitch_um=axial_pitch_um)
        n_ok += 1
        for n in names:
            if np.isfinite(m.get(n, np.nan)):
                acc_m[n].append(m[n])
            if np.isfinite(s.get(n, np.nan)):
                acc_s[n].append(s[n])
    return LayerPhotometry(
        mean_db={n: float(np.mean(acc_m[n])) if acc_m[n] else float("nan") for n in names},
        slope_db_per_mm={n: float(np.mean(acc_s[n])) if acc_s[n] else float("nan") for n in names},
        n_frames=n_ok, source=source)


def from_simulation(n_frames: int = 4, *, seed: int = 0, instrument: str = "device_a",
                    optics_profile: str = "literature") -> LayerPhotometry:
    """Measure photometry on simulated labelled frames (always available)."""
    from ..anatomy.retina import RETINA_LAYERS
    from ..config import SimConfig
    from ..generate import simulate

    names = [ly.name for ly in RETINA_LAYERS]
    dbs, labs = [], []
    for i in range(n_frames):
        cfg = SimConfig(instrument=instrument, optics_profile=optics_profile,
                        anatomy="macula", shape_model="octa500",
                        lines_per_frame=256, seed=seed + 11 * i, randomize=True)
        s = simulate(cfg)
        dbs.append(s.bscan_db)
        labs.append(s.labels2d)
    return measure_corpus(dbs, labs, names, source=f"sim:{optics_profile}")


def backscatter_targets_from_photometry(photo: LayerPhotometry,
                                        ref_layer: str = "ONL") -> dict:
    """Convert mean-dB ladder to relative linear backscatter vs ``ref_layer``.

    Suitable as a soft target for band backscatter multipliers (not µ).
    """
    ref = photo.mean_db.get(ref_layer)
    if ref is None or not np.isfinite(ref):
        raise ValueError(f"reference layer {ref_layer!r} missing/nan in photometry")
    out = {}
    for nm, db in photo.mean_db.items():
        if not np.isfinite(db):
            continue
        # 20 log10(A) → A ∝ 10**(db/20)
        out[nm] = float(10.0 ** ((db - ref) / 20.0))
    return out


def measure_parafoveal_inner(db: np.ndarray, labels: np.ndarray, names: list,
                             *, dz_um: float = 3.87, dx_um: float = 6000.0 / 512,
                             r_inner_mm: float = PARAFOVEAL_R_INNER_MM,
                             r_outer_mm: float = PARAFOVEAL_R_OUTER_MM,
                             fovea_col: int | None = None) -> dict:
    """Parafoveal NFL / GCL / IPL ladder + NFL−GCL contrast (sim labels).

    Uses :func:`parafoveal_column_mask` about the ILM pit. Returns mean dB per
    named layer inside the annulus, plus ``nfl_gcl_contrast_db``,
    ``nfl_ipl_contrast_db``, and mean NFL thickness in px.
    """
    db = np.asarray(db, float)
    lab = np.asarray(labels, int)
    if fovea_col is None:
        fovea_col = find_fovea_column(db, dz_um, dx_um)
    keep = parafoveal_column_mask(db.shape[1], fovea_col, dx_um,
                                  r_inner_mm=r_inner_mm, r_outer_mm=r_outer_mm)
    cols = np.where(keep)[0]
    out = {"n_cols": int(cols.size), "fovea_col": int(fovea_col),
           "mean_db": {}, "nfl_thick_px": float("nan"),
           "nfl_gcl_contrast_db": float("nan"), "nfl_ipl_contrast_db": float("nan")}
    if cols.size == 0:
        return out
    idx = {nm: i + 1 for i, nm in enumerate(names)}
    thick = []
    for nm in names:
        i = idx[nm]
        vals = []
        for c in cols:
            m = lab[:, c] == i
            if m.any():
                vals.append(float(db[m, c].mean()))
                if nm == "NFL":
                    thick.append(int(m.sum()))
        out["mean_db"][nm] = float(np.mean(vals)) if vals else float("nan")
    if thick:
        out["nfl_thick_px"] = float(np.mean(thick))
    g = out["mean_db"].get
    if np.isfinite(g("NFL", np.nan)) and np.isfinite(g("GCL", np.nan)):
        out["nfl_gcl_contrast_db"] = float(g("NFL") - g("GCL"))
    if np.isfinite(g("NFL", np.nan)) and np.isfinite(g("IPL", np.nan)):
        out["nfl_ipl_contrast_db"] = float(g("NFL") - g("IPL"))
    return out


def from_simulation_parafoveal(n_frames: int = 4, *, seed: int = 0,
                               instrument: str = "device_a",
                               optics_profile: str = "literature",
                               lines: int = 512, fov_um: float = 6000.0) -> dict:
    """Pool :func:`measure_parafoveal_inner` over a few simulated macula frames."""
    from ..anatomy.retina import RETINA_LAYERS
    from ..config import SimConfig
    from ..generate import simulate

    names = [ly.name for ly in RETINA_LAYERS]
    dx = float(fov_um) / max(int(lines), 1)
    acc = {"nfl": [], "gcl": [], "ipl": [], "nfl_gcl": [], "nfl_ipl": [], "thick": []}
    for i in range(n_frames):
        cfg = SimConfig(instrument=instrument, optics_profile=optics_profile,
                        anatomy="macula", shape_model="octa500",
                        lines_per_frame=lines, lateral_fov_um=fov_um,
                        sample_depth_um=640.0, seed=seed + 11 * i, randomize=True)
        s = simulate(cfg)
        dz = float(s.depth_axis_um[1] - s.depth_axis_um[0])
        m = measure_parafoveal_inner(s.bscan_db, s.labels2d, names, dz_um=dz, dx_um=dx)
        g = m["mean_db"].get
        if np.isfinite(g("NFL", np.nan)):
            acc["nfl"].append(g("NFL"))
        if np.isfinite(g("GCL", np.nan)):
            acc["gcl"].append(g("GCL"))
        if np.isfinite(g("IPL", np.nan)):
            acc["ipl"].append(g("IPL"))
        if np.isfinite(m["nfl_gcl_contrast_db"]):
            acc["nfl_gcl"].append(m["nfl_gcl_contrast_db"])
        if np.isfinite(m["nfl_ipl_contrast_db"]):
            acc["nfl_ipl"].append(m["nfl_ipl_contrast_db"])
        if np.isfinite(m["nfl_thick_px"]):
            acc["thick"].append(m["nfl_thick_px"])
    avg = lambda k: float(np.mean(acc[k])) if acc[k] else float("nan")
    return {
        "n_frames": n_frames, "source": f"sim:{optics_profile}:parafoveal",
        "nfl_db": avg("nfl"), "gcl_db": avg("gcl"), "ipl_db": avg("ipl"),
        "nfl_gcl_contrast_db": avg("nfl_gcl"), "nfl_ipl_contrast_db": avg("nfl_ipl"),
        "nfl_thick_px": avg("thick"),
    }


# --------------------------------------------------------------------------- parafoveal
def _ilm_rpe_columns(db: np.ndarray, dz_um: float):
    """Per-column ILM / RPE rows via :mod:`octsim.realism_metrics` landmarks."""
    from .. import realism_metrics as rmet

    db = np.asarray(db, float)
    rpe = rmet.find_rpe(db, dz_um)
    ilm = rmet.find_ilm_columns(db, rpe, dz_um)
    return ilm, rpe


def find_fovea_column(db: np.ndarray, dz_um: float, dx_um: float | None = None) -> int:
    """Column index of the ILM pit (deepest ILM row), searched in the central half-FOV.

    Falls back to the global argmax if the central window is empty / flat.
    """
    ilm, _ = _ilm_rpe_columns(db, dz_um)
    n = int(ilm.size)
    lo, hi = int(0.25 * n), max(int(0.25 * n) + 1, int(0.75 * n))
    seg = ilm[lo:hi]
    if seg.size == 0:
        return int(np.argmax(ilm))
    return lo + int(np.argmax(seg))


def parafoveal_column_mask(n_cols: int, fovea_col: int, dx_um: float,
                           *, r_inner_mm: float = PARAFOVEAL_R_INNER_MM,
                           r_outer_mm: float = PARAFOVEAL_R_OUTER_MM) -> np.ndarray:
    """Boolean mask: columns with |x − fovea| ∈ [r_inner_mm, r_outer_mm]."""
    x_mm = (np.arange(int(n_cols), dtype=float) - float(fovea_col)) * float(dx_um) / 1000.0
    r = np.abs(x_mm)
    return (r >= float(r_inner_mm)) & (r <= float(r_outer_mm))


def parafoveal_mean_aline(db: np.ndarray, dz_um: float, dx_um: float, *,
                          r_inner_mm: float = PARAFOVEAL_R_INNER_MM,
                          r_outer_mm: float = PARAFOVEAL_R_OUTER_MM,
                          fovea_col: int | None = None,
                          flatten: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parafoveal mean A-line.

    Returns ``(depth_um, mean_db, meta)`` where ``meta`` records the fovea column, mask
    count, and annulus. Depth is absolute (row × dz) unless ``flatten`` (default), in which
    case the profile is RPE-flattened and depth is relative to the RPE anchor (µm; negative
    = vitread of RPE).
    """
    from .. import realism_metrics as rmet

    db = np.asarray(db, float)
    if fovea_col is None:
        fovea_col = find_fovea_column(db, dz_um, dx_um)
    mask = parafoveal_column_mask(db.shape[1], fovea_col, dx_um,
                                  r_inner_mm=r_inner_mm, r_outer_mm=r_outer_mm)
    n_cols = int(mask.sum())
    meta = {
        "fovea_col": int(fovea_col),
        "n_cols": n_cols,
        "r_inner_mm": float(r_inner_mm),
        "r_outer_mm": float(r_outer_mm),
        "flattened": bool(flatten),
    }
    if n_cols == 0:
        z = np.arange(db.shape[0], dtype=float) * float(dz_um)
        return z, np.full(db.shape[0], np.nan), meta

    if flatten:
        flat, _rpe, ref = rmet.flatten_to_rpe(db, dz_um=dz_um)
        prof = flat[:, mask].mean(axis=1)
        z = (np.arange(db.shape[0], dtype=float) - float(ref)) * float(dz_um)
        meta["rpe_ref_row"] = int(ref)
    else:
        prof = db[:, mask].mean(axis=1)
        z = np.arange(db.shape[0], dtype=float) * float(dz_um)
    return z, prof, meta


def _nanmean_masked(db: np.ndarray, msk: np.ndarray) -> float:
    if msk is None or not np.any(msk):
        return float("nan")
    return float(np.mean(db[msk]))


def layer_band_means_from_labels(db: np.ndarray, labels: np.ndarray, names: list,
                                 *, col_mask: np.ndarray | None = None) -> dict:
    """Mean dB for NFL / inner_stack / ONL / RPE from a 1-based label map.

    ``inner_stack`` pools GCL+IPL+INL+OPL. ``col_mask`` (optional) restricts columns
    (e.g. parafoveal annulus).
    """
    db = np.asarray(db, float)
    lab = np.asarray(labels, int)
    name_to_idx = {nm: i + 1 for i, nm in enumerate(names)}
    if col_mask is None:
        col_mask = np.ones(db.shape[1], dtype=bool)
    col_mask = np.asarray(col_mask, bool)
    cols_ok = col_mask[None, :]

    def _layer(nm: str) -> np.ndarray:
        idx = name_to_idx.get(nm)
        if idx is None:
            return np.zeros(db.shape, dtype=bool)
        return (lab == idx) & cols_ok

    inner = np.zeros(db.shape, dtype=bool)
    for nm in INNER_STACK_LABELS:
        inner |= _layer(nm)

    return {
        "NFL": _nanmean_masked(db, _layer("NFL")),
        "inner_stack": _nanmean_masked(db, inner),
        "ONL": _nanmean_masked(db, _layer("ONL")),
        "RPE": _nanmean_masked(db, _layer("RPE")),
    }


def layer_band_means_from_landmarks(db: np.ndarray, dz_um: float, *,
                                    col_mask: np.ndarray | None = None) -> dict:
    """Mean dB for NFL / inner_stack / ONL / RPE from ILM/RPE landmark windows.

    Windows match :mod:`octsim.realism_metrics` geometry (optical µm), with NFL split out
    of the inner band. ``col_mask`` restricts columns when provided.
    """
    from .. import realism_metrics as rmet

    db = np.asarray(db, float)
    ilm, rpe = _ilm_rpe_columns(db, dz_um)
    if col_mask is None:
        col_mask = np.ones(db.shape[1], dtype=bool)
    col_mask = np.asarray(col_mask, bool)

    px = lambda um: float(um) / float(dz_um)
    rows = np.arange(db.shape[0], dtype=float)[:, None]
    ilm_c = ilm[None, :]
    ez_c = rpe[None, :]
    cols_ok = col_mask[None, :]

    def span(lo, hi):
        return (rows >= lo) & (rows < hi) & cols_ok

    masks = {
        "NFL": span(ilm_c + px(NFL_FROM_ILM[0]), ilm_c + px(NFL_FROM_ILM[1])),
        "inner_stack": span(ilm_c + px(INNER_STACK_FROM_ILM[0]),
                            ilm_c + px(INNER_STACK_FROM_ILM[1])),
        "ONL": span(ez_c + px(rmet.ONL_FROM_ANCHOR[0]),
                    ez_c + px(rmet.ONL_FROM_ANCHOR[1])),
        "RPE": span(ez_c + px(rmet.RPE_BAND[0]), ez_c + px(rmet.RPE_BAND[1])),
    }
    return {k: _nanmean_masked(db, m) for k, m in masks.items()}


def layer_band_means(db: np.ndarray, dz_um: float, dx_um: float, *,
                     labels: np.ndarray | None = None,
                     names: list | None = None,
                     parafoveal: bool = True,
                     r_inner_mm: float = PARAFOVEAL_R_INNER_MM,
                     r_outer_mm: float = PARAFOVEAL_R_OUTER_MM,
                     fovea_col: int | None = None) -> tuple[dict, dict]:
    """Band mean dB (+ meta). Prefer labels when given; else landmark windows.

    When ``parafoveal`` is True (default), averages are restricted to the ILM-pit annulus.
    """
    db = np.asarray(db, float)
    if fovea_col is None:
        fovea_col = find_fovea_column(db, dz_um, dx_um)
    if parafoveal:
        col_mask = parafoveal_column_mask(db.shape[1], fovea_col, dx_um,
                                          r_inner_mm=r_inner_mm, r_outer_mm=r_outer_mm)
    else:
        col_mask = np.ones(db.shape[1], dtype=bool)

    meta = {
        "fovea_col": int(fovea_col),
        "n_cols": int(col_mask.sum()),
        "r_inner_mm": float(r_inner_mm) if parafoveal else None,
        "r_outer_mm": float(r_outer_mm) if parafoveal else None,
        "parafoveal": bool(parafoveal),
    }
    if labels is not None:
        if names is None:
            from ..anatomy.retina import RETINA_LAYERS
            names = [ly.name for ly in RETINA_LAYERS]
        means = layer_band_means_from_labels(db, labels, names, col_mask=col_mask)
        meta["source"] = "labels"
    else:
        means = layer_band_means_from_landmarks(db, dz_um, col_mask=col_mask)
        meta["source"] = "landmarks"
    return means, meta
