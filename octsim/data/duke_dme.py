"""Loader for the Duke SD-OCT DME dataset (Chiu et al., BOE 2015).

10 subjects × 11 manually-segmented B-scans = 110 scans of ``496×768`` with **8 layer
surfaces** traced by two graders (``manualLayers1/2``). The 8 surfaces bound 7 layers:

    RNFL | GCL+IPL | INL | OPL | ONL | IS(ELM→EZ) | OS+RPE

This module turns those into (a) a fovea-aligned thickness stack for
:class:`~octsim.shape_model.LayerShapeModel`, and (b) an optical
:class:`~octsim.calibration.CalibrationProfile` measured from the same real images.

The dataset is NOT redistributed here. Download ``2015_BOE_Chiu2.zip`` from the Duke page
(``people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm``), unzip, and point ``root`` at the
``2015_BOE_Chiu`` folder of ``Subject_*.mat`` files. **Caveat:** these are diabetic-macular-
edema eyes, so raw thicknesses skew thick; ``max_total_um`` filters toward the more-normal
scans, and the learned shape model is the artifact that travels — refit on normal / Maestro
data with the same API when available.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ..shape_model import LayerShapeModel, align_and_resample, locate_fovea_um

#: 7 Duke layers (between the 8 traced surfaces), vitreal → scleral.
DUKE_LAYERS = ["RNFL", "GCLIPL", "INL", "OPL", "ONL", "IS", "OSRPE"]

# Spectralis geometry for this dataset (axial is the well-known 3.87 µm/px; the lateral
# pitch assumes a 6 mm scan over 768 A-lines and only scales the eccentricity axis).
AXIAL_PITCH_UM = 3.87
LATERAL_PITCH_UM = 6000.0 / 768.0


def load_annotated(root: str, *, grader: int = 1):
    """Yield every manually-segmented B-scan as a record dict.

    Each record: ``subject``, ``frame``, ``image`` ``[496,768]`` uint8, ``bnd_rows``
    ``[8,768]`` (NaN outside the annotated columns), and ``c0``/``c1`` (annotated range).
    """
    import scipy.io as sio
    key = f"manualLayers{grader}"
    out = []
    for path in sorted(glob.glob(os.path.join(root, "Subject_*.mat"))):
        m = sio.loadmat(path)
        if key not in m:
            continue
        imgs, lay = m["images"], m[key]              # [496,768,61], [8,768,61]
        subj = os.path.splitext(os.path.basename(path))[0]
        for f in range(lay.shape[2]):
            b = lay[:, :, f]
            if not np.isfinite(b).any():
                continue
            cols = np.where(np.isfinite(b[0]))[0]
            out.append(dict(subject=subj, frame=int(f), image=imgs[:, :, f],
                            bnd_rows=b, c0=int(cols.min()), c1=int(cols.max())))
    return out


def _record_thicknesses_um(rec, axial_pitch_um):
    """(cols_um, thickness[7,768] µm, inner_thickness[768] µm) for one record."""
    b_um = rec["bnd_rows"] * axial_pitch_um               # [8,768]
    thick = np.diff(b_um, axis=0)                          # [7,768]
    inner = b_um[4] - b_um[0]                              # ILM→OPL/ONL boundary
    cols_um = np.arange(b_um.shape[1]) * LATERAL_PITCH_UM
    return cols_um, thick, inner


def thickness_stack(records, *, n_ecc: int = 151, half_width_um: float = 1500.0,
                    max_total_um: float = 360.0, axial_pitch_um: float = AXIAL_PITCH_UM):
    """Build the fovea-aligned ``[N, 7, n_ecc]`` thickness stack + eccentricity grid.

    Scans whose median total retinal thickness exceeds ``max_total_um`` (heavy edema) are
    dropped. Each kept scan is aligned by its inner-retinal-thickness minimum (the fovea
    when present) and resampled onto ``±half_width_um``.
    """
    ecc = np.linspace(-half_width_um, half_width_um, n_ecc)
    stack, kept = [], []
    for rec in records:
        cols_um, thick, inner = _record_thicknesses_um(rec, axial_pitch_um)
        sl = slice(rec["c0"], rec["c1"] + 1)
        total = np.nanmedian(thick[:, sl].sum(axis=0))
        if not np.isfinite(total) or total > max_total_um:
            continue
        fovea = locate_fovea_um(inner[sl], cols_um[sl])
        prof = align_and_resample(thick[:, sl], cols_um[sl], fovea, ecc)
        if not np.isfinite(prof).all() or (prof <= 0).all():
            continue
        stack.append(prof)
        kept.append(rec)
    return np.array(stack), ecc, kept


def fit_shape_model(root: str, *, grader: int = 1, n_modes: int = 8,
                    max_total_um: float = 360.0, **kw) -> LayerShapeModel:
    """One-call: load the Duke annotations and fit a :class:`LayerShapeModel`."""
    records = load_annotated(root, grader=grader)
    stack, ecc, kept = thickness_stack(records, max_total_um=max_total_um,
                                       axial_pitch_um=AXIAL_PITCH_UM, **kw)
    if stack.shape[0] < 3:
        raise RuntimeError(f"only {stack.shape[0]} usable scans; check root/filters")
    return LayerShapeModel.fit(
        stack, ecc, DUKE_LAYERS, n_modes=n_modes, axial_pitch_um=AXIAL_PITCH_UM,
        undulation_rms_um=_boundary_undulation_rms(kept, AXIAL_PITCH_UM),
        source=f"Duke DME (Chiu 2015), grader {grader}, {stack.shape[0]} scans")


def _boundary_undulation_rms(records, axial_pitch_um) -> float:
    """Median small-scale (detrended) roughness of the traced boundaries, µm."""
    rms = []
    for rec in records[:60]:
        b_um = rec["bnd_rows"] * axial_pitch_um
        sl = slice(rec["c0"], rec["c1"] + 1)
        for k in range(b_um.shape[0]):
            v = b_um[k, sl]
            if not np.isfinite(v).all() or v.size < 20:
                continue
            x = np.arange(v.size)
            trend = np.polyval(np.polyfit(x, v, 4), x)     # remove the smooth layer shape
            rms.append(np.std(v - trend))
    return float(np.median(rms)) if rms else 3.0


def extract_optical_profile(root: str, *, grader: int = 1, max_total_um: float = 360.0):
    """Measure a :class:`~octsim.calibration.CalibrationProfile` from the Duke images.

    Bonus: the same segmented scans yield real per-layer reflectivity/attenuation. The Duke
    exports are 8-bit log-display images, so ``domain='db'`` and reflectivity is treated as a
    relative (log-domain) ratio — coarser than a linear reconstruction, but grounded.
    """
    from .. import calibration as cal
    records = load_annotated(root, grader=grader)
    profs = []
    for rec in records:
        b_um = rec["bnd_rows"] * AXIAL_PITCH_UM
        sl = slice(rec["c0"], rec["c1"] + 1)
        total = np.nanmedian(np.diff(b_um, axis=0)[:, sl].sum(axis=0))
        if not np.isfinite(total) or total > max_total_um:
            continue
        depth = np.arange(rec["image"].shape[0]) * AXIAL_PITCH_UM
        img = rec["image"][:, sl].astype(float)
        img = img / 255.0 * 40.0 - 40.0                    # map 8-bit -> ~[-40,0] dB
        names = _duke_to_canonical()
        try:
            profs.append(cal.extract_profile(img, b_um[:, sl], depth,
                                             layer_names=DUKE_LAYERS, domain="db"))
        except Exception:
            continue
    return cal.average_profiles(profs) if profs else cal.literature_profile()


def _duke_to_canonical():
    return DUKE_LAYERS
