"""Load public, already-displayed OCT corpora (BMP / JPEG) for cross-device validation.

The Maestro ``.raw`` path (:mod:`octsim.data.real_maestro`) reconstructs from the detector
spectra, so its output is in a **linear** domain with a genuine noise floor and octsim's
own dB mapping. Public datasets are not: OCTA-500, OCTID, OCTDL and Kermany ship 8-bit
images that have already been through the device's tone curve (OCTA-500's vitreous sits at
~63/255, not 0), and the JPEG ones are lossily compressed on top.

That does not make them useless — it makes a *subset* of the metric suite valid. Anything
that is a ratio of dB values, referenced to the noise floor, or a speckle amplitude
statistic is destroyed by an unknown monotone tone curve. Anything that is a **geometry**
or a **spatial correlation length** survives it, because a monotone intensity remap moves
no structure. :data:`octsim.realism_metrics.TIER_DISPLAY` encodes exactly that split, and
the validation harness drops the rest rather than reporting numbers it cannot support.

So these corpora answer "does the simulated *anatomy* generalise across devices?" and
deliberately do not answer "does the simulated *photometry* generalise?" — for that, the
only linear-domain data available is the Maestro raw.
"""

from __future__ import annotations

import glob
import os

import numpy as np


def _to_pseudo_db(a: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """8-bit display image -> a dB-like array on octsim's 0 dB = peak convention.

    This is NOT a recovery of the true dB image: the device's tone curve is unknown and
    not invertible from the data. It only puts the array on a comparable scale so the
    same metric code can run; only tone-curve-invariant metrics are then trusted.
    """
    a = np.asarray(a, float)
    peak = float(a.max()) or 1.0
    return np.clip(20.0 * np.log10(np.maximum(a / peak, 1e-6)), floor_db, 0.0)


def load_image_stack(paths, *, limit: int | None = None, stride: int = 1,
                     transpose: bool = False) -> list:
    """Load a list of image files as pseudo-dB arrays (depth on axis 0)."""
    from PIL import Image
    paths = list(paths)[:: max(1, stride)]
    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        a = np.array(Image.open(p).convert("L"))
        if transpose:
            a = a.T
        out.append(_to_pseudo_db(a))
    return out


def load_octa500_volume(oct_dir: str, *, limit: int | None = None, stride: int = 1) -> list:
    """One OCTA-500 structural volume: ``<OCT>/<subject>/*.bmp``, depth already on axis 0.

    Frames are named ``1.bmp`` … ``N.bmp``, so they must be sorted numerically — a
    lexicographic sort interleaves 1, 10, 100, … and scrambles the slow axis.
    """
    files = glob.glob(os.path.join(oct_dir, "*.bmp"))
    if not files:
        raise FileNotFoundError(f"no .bmp in {oct_dir}")
    files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return load_image_stack(files, limit=limit, stride=stride)


def load_flat_corpus(folder: str, *, pattern: str = "*.jpeg", limit: int | None = None,
                     stride: int = 1) -> list:
    """A flat directory of independent B-scans (OCTID, OCTDL, Kermany)."""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {folder}")
    return load_image_stack(files, limit=limit, stride=stride)


#: Known held-out corpora. ``dz``/``dx`` are the published voxel pitches (optical µm);
#: ``tier`` selects which metrics the harness is allowed to report for that source.
#: ``fov_um`` is the lateral field the simulator should be rendered at to match.
#: Normative macular ILM->RPE tissue thickness (µm). Used only to CALIBRATE the axial
#: pitch of corpora that ship no metadata (see :func:`calibrate_axial_pitch`).
NORMATIVE_RETINA_UM = 300.0

#: Re-exported so this module keeps one name for it, but the value lives on the instrument
#: side — a corpus's SNR belongs to the device that acquired it, not to the harness that
#: scores it. See :data:`octsim.instrument.REFERENCE_SNR_DB`.
from ..instrument import REFERENCE_SNR_DB  # noqa: E402,F401


def noise_level_for_snr(snr_db: float, ref_db: float = REFERENCE_SNR_DB) -> float:
    """Scale the noise coefficients to a corpus's own measured SNR.

    SNR is an acquisition property (gain, reference power, integration time), not a tissue
    property. Maestro2 measures 45.3 dB against Maestro3's 52.0 — and that 6.7 dB gap is
    almost exactly the inner-retina-to-vitreous contrast gap seen when scoring the
    Maestro3-tuned simulator against Maestro2. Supplying each corpus's own SNR is therefore
    a device calibration, not a fudge: without it the harness reports a tissue-model failure
    for what is purely an exposure difference.

    Prefer ``SimConfig(instrument=...)``, which applies :attr:`Instrument.noise_scale`
    automatically. This stays for corpora that have a measured SNR but no instrument model.
    """
    return float(10.0 ** ((ref_db - float(snr_db)) / 20.0))


def calibrate_axial_pitch(images, dz_guess: float, dx: float,
                          target_um: float = NORMATIVE_RETINA_UM) -> float:
    """Estimate a corpus's axial pitch from its own retinal thickness.

    Public 8-bit corpora ship no voxel metadata, and a guessed pitch poisons every
    µm-valued metric: assuming OCTA-500's published 3.125 µm/px yields a 449 µm retina,
    which is anatomically impossible, so any thickness/sag/grain number derived from it is
    meaningless rather than merely imprecise.

    Rather than guess, scale the pitch so the corpus's median ILM->RPE distance equals a
    normative macular thickness. The cost is explicit and must be honoured by the caller:
    **thickness is no longer an independent measurement for that corpus** and has to be
    dropped from its reported metrics. Everything else (sag, correlation lengths, grain,
    peak separations) becomes comparable in calibrated units.
    """
    from .. import realism_metrics as rmet
    rows = []
    for im in images:
        m = rmet.measure(im, dz_guess, dx)
        if m.rpe_row > m.ilm_row:
            rows.append(m.rpe_row - m.ilm_row)
    if not rows:
        return dz_guess
    med_rows = float(np.median(rows))
    if med_rows <= 0:
        return dz_guess
    return float(target_um * rmet.TISSUE_N / med_rows)


HOLDOUT_CORPORA = {
    # Same vendor, different unit / gain / operator, and crucially still LINEAR domain:
    # the strongest generalisation test available.
    # ``instrument`` is the single source for this corpus's device numbers: its
    # scan-distortion correction (0.43) and measured SNR (45.33 dB) live in
    # octsim/data/instrument_maestro2.json and reach the renderer through
    # SimConfig(instrument="maestro2"). The corpora below have no instrument model, so they
    # still carry a ``distortion`` here as a fallback.
    "maestro2": dict(kind="raw", tier="full", dz=3.87, dz_known=True, fov_um=6000.0,
                     fov_known=True, instrument="maestro2",
                     path=os.path.join("Maestro2",
                                       "Line_6mm_2048Aline_135degCW_50frame_gain165")),
    # Cross-vendor (Optovue), display domain: geometry and spatial statistics only.
    "octa500": dict(kind="octa500", tier="display", dz=3.125, dz_known=False, fov_um=3000.0,
                    fov_known=True, distortion=0.28,
                    path=os.path.join("OCTA-500", "OCTA_3mm_part1", "OCTA_3mm_part1",
                                      "OCTA_3mm", "OCT", "10301")),
    # Cross-vendor (Cirrus), display domain AND JPEG-compressed: geometry only.
    "octid": dict(kind="flat", tier="compressed", dz=3.9, dz_known=False, fov_um=6000.0,
                  fov_known=False, distortion=0.86,
                  pattern="NORMAL*.jpeg",
                  path=os.path.join("Optical Coherence Tomography Image Retinal Database",
                                    "doi-10.5683-sp-wlw4zt")),
}


def load_holdout(name: str, data_root: str, *, limit: int = 6, stride: int = 8):
    """Load a corpus from :data:`HOLDOUT_CORPORA`; returns ``(images, spec)``."""
    if name not in HOLDOUT_CORPORA:
        raise KeyError(f"unknown holdout {name!r}; have {sorted(HOLDOUT_CORPORA)}")
    spec = dict(HOLDOUT_CORPORA[name])
    full = os.path.join(data_root, spec["path"])
    kind = spec["kind"]
    if kind == "raw":
        from . import real_maestro
        imgs, alines, _skipped = real_maestro.load_corpus(full, limit=limit, stride=stride)
        spec["dx"] = spec["fov_um"] / alines
        return imgs, spec
    if kind == "octa500":
        imgs = load_octa500_volume(full, limit=limit, stride=stride)
    else:
        imgs = load_flat_corpus(full, pattern=spec.get("pattern", "*.jpeg"),
                                limit=limit, stride=stride)
    spec["dx"] = spec["fov_um"] / imgs[0].shape[1]
    if not spec.get("dz_known", True):
        spec["dz"] = calibrate_axial_pitch(imgs, spec["dz"], spec["dx"])
        spec["dz_calibrated"] = True
    return imgs, spec
