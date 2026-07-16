"""Loader for the Johns Hopkins macular OCT dataset (He et al., Data in Brief 2019).

35 Spectralis macular volumes (49 B-scans each) with **9 manually-delineated surfaces**:
14 **healthy controls** (``hc*``) and 21 multiple-sclerosis (``ms*``). The MS maculae have
no edema (RNFL may thin, but the foveal contour and layer structure are normal), so this is
the clean **normal-eye** geometry prior — unlike the DME set, nothing distorts the curve.

The 9 surfaces bound 8 layers, vitreal → scleral::

    RNFL | GCL+IPL(GCIP) | INL | OPL | ONL | IS(ELM→EZ) | OS | RPE

Delineations are stored as sparse spline **control points** (``control_pts[bscan, surface]``
= ``[n_pts, 2]`` of ``(x_col, y_row)``); this module interpolates them to dense per-column
boundaries. The dataset is NOT redistributed here — download
``OCT_Manual_Delineations-2018_June_29_b.zip`` from ``iacl.ece.jhu.edu`` (or the MEDIC portal)
and point ``root`` at the ``delineation`` folder of ``*_spectralis_macula_*.mat`` files.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ..shape_model import LayerShapeModel, align_and_resample, locate_fovea_um

#: 8 JHU layers (between the 9 traced surfaces).
JHU_LAYERS = ["RNFL", "GCIP", "INL", "OPL", "ONL", "IS", "OS", "RPE"]
#: the 9 non-empty surface columns within the 11-slot ``control_pts`` array.
_SURF_COLS = [0, 1, 3, 4, 5, 6, 7, 8, 9]

AXIAL_PITCH_UM = 3.87
LATERAL_PITCH_UM = 6000.0 / 1024.0        # 6 mm macula scan over 1024 A-lines


def _dense_boundaries(cp_row, axial_pitch_um):
    """Interpolate the 9 control-point surfaces of one B-scan to dense ``[9, nx]`` µm."""
    xmax = 0
    for j in _SURF_COLS:
        e = np.asarray(cp_row[j])
        if e.size and e.ndim == 2:
            xmax = max(xmax, int(e[:, 0].max()))
    if xmax < 8:
        return None, None
    xg = np.arange(1, xmax + 1)
    bnd = np.full((9, xg.size), np.nan)
    for si, j in enumerate(_SURF_COLS):
        e = np.asarray(cp_row[j])
        if e.size and e.ndim == 2 and e.shape[0] >= 2:
            order = np.argsort(e[:, 0])
            bnd[si] = np.interp(xg, e[order, 0], e[order, 1])
    cols_um = xg * LATERAL_PITCH_UM
    return bnd * axial_pitch_um, cols_um


def load_annotated(root: str, *, groups=("hc",)):
    """Yield dense-boundary records for the requested cohort(s) (``hc`` and/or ``ms``)."""
    import scipy.io as sio
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*_spectralis_macula_*.mat"))):
        name = os.path.basename(path)
        if not any(name.startswith(g) for g in groups):
            continue
        cp = sio.loadmat(path).get("control_pts")
        if cp is None:
            continue
        for b in range(cp.shape[0]):
            bnd_um, cols_um = _dense_boundaries(cp[b], AXIAL_PITCH_UM)
            if bnd_um is None or not np.isfinite(bnd_um).all():
                continue
            out.append(dict(subject=name.split("_")[0], frame=int(b),
                            bnd_um=bnd_um, cols_um=cols_um))
    return out


def thickness_stack(records, *, n_ecc: int = 151, half_width_um: float = 1500.0,
                    min_total_um: float = 240.0, max_total_um: float = 360.0,
                    min_pit_depth_um: float = 55.0):
    """Fovea-aligned ``[N, 8, n_ecc]`` thickness stack + eccentricity grid (µm).

    ``min_pit_depth_um`` keeps scans that actually contain the fovea (inner retina at least
    this much thinner at the pit than at the edges), so the mean geometry is a real fovea
    rather than a parafovea-diluted average; 0 keeps every scan.
    """
    ecc = np.linspace(-half_width_um, half_width_um, n_ecc)
    stack, kept = [], []
    for rec in records:
        bnd, cols = rec["bnd_um"], rec["cols_um"]
        thick = np.diff(bnd, axis=0)                       # [8, nx] µm
        total = np.median(bnd[-1] - bnd[0])
        if not (min_total_um <= total <= max_total_um):
            continue
        inner = bnd[4] - bnd[0]                            # ILM→OPL/ONL
        fovea = locate_fovea_um(inner, cols)
        ic = int(np.argmin(np.abs(cols - fovea)))
        edge = np.median(np.concatenate([inner[:inner.size // 5], inner[-inner.size // 5:]]))
        pit_depth = edge - np.median(inner[max(0, ic - 15):ic + 15])
        if pit_depth < min_pit_depth_um:                   # skip parafoveal scans
            continue
        prof = align_and_resample(thick, cols, fovea, ecc)
        if not np.isfinite(prof).all() or (prof <= 0).all():
            continue
        stack.append(prof)
        kept.append(rec)
    return np.array(stack), ecc, kept


def fit_shape_model(root: str, *, groups=("hc",), n_modes: int = 8, **kw) -> LayerShapeModel:
    """Load JHU delineations and fit a normal-eye :class:`LayerShapeModel`."""
    records = load_annotated(root, groups=groups)
    stack, ecc, kept = thickness_stack(records, **kw)
    if stack.shape[0] < 5:
        raise RuntimeError(f"only {stack.shape[0]} usable scans; check root/filters")
    label = "+".join(groups)
    return LayerShapeModel.fit(
        stack, ecc, JHU_LAYERS, n_modes=n_modes, axial_pitch_um=AXIAL_PITCH_UM,
        undulation_rms_um=4.0,   # control-point traces are smooth; use a biological default
        source=f"JHU {label} (He 2019), {stack.shape[0]} B-scans")
