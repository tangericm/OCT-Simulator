"""Loader + analysis for the OCTA-500 dataset (Li et al., Medical Image Analysis 2024).

OCTA-500 ships **real macular OCT volumes** with volumetric layer segmentation — the
uniquely valuable thing here is that it is *3D*: each eye is a contiguous stack of B-scans,
so it validates the simulator's volume model (how the fovea recedes across the slow axis),
not just single-frame appearance.

This module targets the **OCTA_3mm** subset (subjects 10301-10400 in the sample used here):

    OCT/<id>/<b>.bmp        640 x 304  uint8 structural B-scan (304 per eye, 3x3x2 mm)
    GT_Layers/<id>.mat      Layer[6, 304, 304] uint16 — 6 boundary SURFACES as en-face
                            height maps (row index into the 640-deep B-scan). Surface 0 is
                            the ILM, surface 5 the outer boundary (BM/RPE); the four between
                            are OCTA-500's inner boundaries. ILM→BM (0→5) is the neurosensory
                            retina, which is all this module needs.
    GT_FAZ/<id>.bmp         304 x 304 binary en-face foveal avascular zone mask.

The RTVue-XR structural B-scan is the **average of the repeated OCTA acquisitions**, so its
speckle contrast (~0.34) is lower than a single-frame Rayleigh field (~0.52) — an averaged-
acquisition appearance, kept as a SEPARATE texture profile rather than pooled into the
single-frame healthy default (see ``unlabeled.bundled_texture``).

The raw dataset is NOT redistributed here (~4.5 GB). Point ``root`` at an extracted tree with
``OCT/``, ``GT_Layers/``, ``GT_FAZ/`` (``scripts/fit_octa500.py`` also reads the release zip).
Only the tiny fitted artifacts travel with the repo: ``texture_octa500.json`` and
``octa500_macula.json``.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

# --- acquisition geometry (OCTA_3mm) --------------------------------------------------
AXIAL_PITCH_UM = 2000.0 / 640.0        # 2 mm depth over 640 px (~3.125 µm, ~tissue units)
LATERAL_PITCH_UM = 3000.0 / 304.0      # 3 mm over 304 A-lines (~9.87 µm)
DEPTH_PX = 640
N_ENFACE = 304                         # fast + slow axis samples
N_BSCANS = 304

#: the OCTA_3mm subset spans these subject IDs (100 eyes in the sample used here).
ID_LO, ID_HI = 10301, 10400

# --- geometric normal-eye filter ------------------------------------------------------
# OCTA_3mm is ~80% normal / ~20% disease (AMD, DR, CNV). The release ships per-subject
# disease labels in a Text-labels sheet, but that file is NOT part of every partial
# download, so we flag pathology straight from the segmented geometry instead: edema / CNV
# thicken the fovea and obliterate (or invert) the pit, and drusen / RPE elevation roughen
# the outer boundary. These thresholds reproduce the exclusion set below on the sample used
# here (10 of 100 eyes); a subtle non-distorting DR reads as geometrically normal, which is
# correct for a *shape* prior. All lengths are physical µm (× AXIAL_PITCH_UM).
NORMAL_CENTRAL_UM = (150.0, 290.0)     # foveal ILM→BM thickness stays in the normal band
NORMAL_MEAN_UM = (280.0, 360.0)        # whole-field mean thickness
NORMAL_MAX_PIT_UM = -2.5               # ILM must dip ≥2.5 µm deeper at the fovea (a real pit)
NORMAL_MAX_BM_ROUGHNESS_PX = 0.72      # detrended outer-boundary roughness (drusen/CNV flag)

#: eyes the geometric filter rejects on the sample used here (edema/CNV/drusen distortion).
DISEASE_IDS = (10302, 10306, 10308, 10310, 10325, 10335, 10363, 10367, 10373, 10380)


def _bm_roughness_px(bm_px: np.ndarray) -> float:
    """Std of the outer boundary after removing its smooth global shape (drusen/RPE bumps)."""
    from scipy.ndimage import uniform_filter
    return float(np.std(bm_px - uniform_filter(bm_px, size=25, mode="nearest")))


def geometry_metrics(layer_px: np.ndarray) -> dict:
    """Per-eye geometric descriptors from a ``Layer[6, x, y]`` (row-index) segmentation."""
    from scipy.ndimage import gaussian_filter
    L = np.asarray(layer_px, float)
    th = (L[5] - L[0]) * AXIAL_PITCH_UM                 # ILM→BM thickness map, µm
    ths = gaussian_filter(th, 6)
    ny, nx = th.shape
    win = ths[ny // 3:2 * ny // 3, nx // 3:2 * nx // 3]  # fovea near the centre
    fy, fx = np.unravel_index(int(np.argmin(win)), win.shape)
    fy += ny // 3; fx += nx // 3
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - fx, yy - fy)
    ilm = L[0]
    pit = float(ilm[(r >= 8) & (r <= 20)].mean() - ilm[r < 6].mean()) * AXIAL_PITCH_UM
    return dict(central_um=float(th[r < 100 / LATERAL_PITCH_UM].mean()),
                mean_um=float(th.mean()), pit_um=pit,
                bm_roughness_px=_bm_roughness_px(L[5]), fovea_xy=(int(fx), int(fy)))


def is_normal_geometry(layer_px: np.ndarray) -> bool:
    """True if the segmented geometry looks like a healthy fovea (no edema/CNV/drusen)."""
    m = geometry_metrics(layer_px)
    return bool(NORMAL_CENTRAL_UM[0] <= m["central_um"] <= NORMAL_CENTRAL_UM[1]
                and NORMAL_MEAN_UM[0] <= m["mean_um"] <= NORMAL_MEAN_UM[1]
                and m["pit_um"] <= NORMAL_MAX_PIT_UM
                and m["bm_roughness_px"] <= NORMAL_MAX_BM_ROUGHNESS_PX)


# --- reading the extracted tree -------------------------------------------------------
def _layer_dir(root):  return _first_dir(root, "GT_Layers")
def _faz_dir(root):    return _first_dir(root, "GT_FAZ")
def _oct_dir(root):    return _first_dir(root, "OCT")


def _first_dir(root, name):
    """Find ``name`` anywhere under ``root`` (OCTA-500 nests folders redundantly)."""
    if os.path.basename(root.rstrip("/")) == name:
        return root
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else os.path.join(root, name)


def subject_ids(root, *, lo=ID_LO, hi=ID_HI):
    """Subject IDs in ``[lo, hi]`` that have a layer segmentation under ``root``."""
    lay = _layer_dir(root)
    ids = []
    for p in sorted(glob.glob(os.path.join(lay, "*.mat"))):
        try:
            sid = int(os.path.splitext(os.path.basename(p))[0])
        except ValueError:
            continue
        if lo <= sid <= hi:
            ids.append(sid)
    return ids


def load_layer(root, sid):
    """Return one eye's ``Layer[6, 304, 304]`` (row-index) segmentation, or None."""
    import scipy.io as sio
    p = os.path.join(_layer_dir(root), f"{sid}.mat")
    if not os.path.exists(p):
        return None
    L = sio.loadmat(p).get("Layer")
    return None if L is None or L.shape[1:] != (N_ENFACE, N_ENFACE) else np.asarray(L, float)


def normal_ids(root, *, lo=ID_LO, hi=ID_HI):
    """IDs in range whose segmented geometry passes :func:`is_normal_geometry`."""
    out = []
    for sid in subject_ids(root, lo=lo, hi=hi):
        L = load_layer(root, sid)
        if L is not None and is_normal_geometry(L):
            out.append(sid)
    return out


# --- geometry: real radial macular profile + FAZ --------------------------------------
def radial_thickness_profile(root, ids=None, *, ecc_um=None):
    """Fovea-centred, angle-averaged ILM→BM thickness (µm) vs eccentricity over ``ids``."""
    if ecc_um is None:
        ecc_um = np.arange(0.0, 1300.0, 50.0)
    ids = ids if ids is not None else normal_ids(root)
    yy, xx = np.mgrid[0:N_ENFACE, 0:N_ENFACE]
    acc = np.zeros(len(ecc_um)); cnt = np.zeros(len(ecc_um))
    for sid in ids:
        L = load_layer(root, sid)
        if L is None:
            continue
        th = (L[5] - L[0]) * AXIAL_PITCH_UM
        fx, fy = geometry_metrics(L)["fovea_xy"]
        r = np.hypot(xx - fx, yy - fy) * LATERAL_PITCH_UM
        for i, e in enumerate(ecc_um):
            m = (r >= e) & (r < e + 50.0)
            if m.any():
                acc[i] += th[m].mean(); cnt[i] += 1
    return np.asarray(ecc_um, float), acc / np.maximum(cnt, 1)


def faz_area_mm2(root, ids=None):
    """Foveal-avascular-zone area (mm²) per eye from the binary en-face masks."""
    from PIL import Image
    ids = ids if ids is not None else normal_ids(root)
    faz = _faz_dir(root); px2 = (LATERAL_PITCH_UM ** 2) / 1e6
    out = []
    for sid in ids:
        p = os.path.join(faz, f"{sid}.bmp")
        if os.path.exists(p):
            out.append(float((np.array(Image.open(p)) > 127).sum()) * px2)
    return np.asarray(out)


def iter_normal_bscans(root, ids=None, *, per_eye=15):
    """Yield ``per_eye`` evenly spaced structural B-scans (float32) from each normal eye."""
    from PIL import Image
    ids = ids if ids is not None else normal_ids(root)
    oct_dir = _oct_dir(root)
    for sid in ids:
        d = os.path.join(oct_dir, str(sid))
        idx = np.linspace(1, N_BSCANS, per_eye).round().astype(int)
        for b in idx:
            p = os.path.join(d, f"{b}.bmp")
            if os.path.exists(p):
                yield np.asarray(Image.open(p).convert("L"), dtype=np.float32)


def extract_texture(root, ids=None, *, per_eye=15):
    """Fit a :class:`unlabeled.TextureProfile` from the normal eyes' structural volumes."""
    from . import unlabeled as ul
    ids = ids if ids is not None else normal_ids(root)
    imgs = list(iter_normal_bscans(root, ids, per_eye=per_eye))
    return ul.extract_texture(
        imgs, axial_pitch_um=AXIAL_PITCH_UM, lateral_pitch_um=LATERAL_PITCH_UM,
        source=f"OCTA-500 3mm NORMAL ({len(imgs)} B-scans, {len(ids)} eyes)")


# --- compact geometry artifact --------------------------------------------------------
@dataclass
class MaculaProfile:
    """Real macular geometry distilled from OCTA-500 normal eyes (a validation reference)."""

    eccentricity_um: list
    total_retina_thickness_um: list
    central_thickness_um: float
    perifoveal_plateau_um: float
    pit_relative_depth: float            # central / plateau — device-independent pit depth
    foveal_pit_radius_um: float
    faz_area_mm2_median: float
    faz_radius_um_median: float
    curvature_concave_frac: float
    axial_pitch_um: float = AXIAL_PITCH_UM
    lateral_pitch_um: float = LATERAL_PITCH_UM
    n_eyes_normal: int = 0
    n_eyes_disease: int = 0
    disease_ids_excluded: list = field(default_factory=list)
    device: str = "Optovue RTVue-XR Avanti, 3x3 mm, 304x304x640 (~tissue units)"
    source: str = "OCTA-500 3mm"

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def bundled_macula() -> MaculaProfile:
    """Load the bundled OCTA-500 macular geometry reference (``octa500_macula.json``)."""
    return MaculaProfile.load(os.path.join(os.path.dirname(__file__), "octa500_macula.json"))


def build_macula_profile(root) -> MaculaProfile:
    """Measure the macular geometry artifact from an extracted OCTA-500 tree."""
    ids = normal_ids(root)
    ecc, prof = radial_thickness_profile(root, ids)
    faz = faz_area_mm2(root, ids)
    plateau = float(np.mean(prof[ecc >= 1000.0])) if (ecc >= 1000.0).any() else float(prof.max())
    central = float(prof[0])
    half = central + 0.5 * (plateau - central)
    fov_r = float(ecc[int(np.argmax(prof >= half))])
    tex = extract_texture(root, ids)
    all_ids = subject_ids(root)
    return MaculaProfile(
        eccentricity_um=[round(float(e), 1) for e in ecc],
        total_retina_thickness_um=[round(float(v), 1) for v in prof],
        central_thickness_um=round(central, 1), perifoveal_plateau_um=round(plateau, 1),
        pit_relative_depth=round(central / plateau, 3), foveal_pit_radius_um=fov_r,
        faz_area_mm2_median=round(float(np.median(faz)), 3),
        faz_radius_um_median=round(float(np.median(np.sqrt(faz / np.pi)) * 1000.0), 1),
        curvature_concave_frac=round(float(tex.curvature_concave_frac), 3),
        n_eyes_normal=len(ids), n_eyes_disease=len(all_ids) - len(ids),
        disease_ids_excluded=[s for s in all_ids if s not in ids])
