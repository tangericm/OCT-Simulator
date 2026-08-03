"""Spectral-domain signal/noise metrics for device grounding (``D_spec``).

Image-domain SNR (~52 dB) cannot separate shot / RIN / read. Real ``bscan*.raw``
spectra + FPN + mirror scans can. This module:

* extracts a :class:`SpectralProfile` from a real scan folder (or from a simulated
  mean spectrum + reconstructed floor);
* compares two profiles via :func:`spectral_distance` (the ``D_spec`` terms);
* provides a small noise-component probe used by offline noise characterization.

No image-space post-filter — everything here is spectrum / reconstructed-floor
physics that the forward model already exposes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class SpectralProfile:
    """Domain-invariant spectral / floor fingerprint of one corpus."""

    #: peak-normalised k-linear (or detector-grid) mean envelope, length n_k
    envelope: list = field(default_factory=list)
    #: electronic / deep tail level as a fraction of the envelope peak (pre-norm)
    tail_frac: float = 0.0
    #: reconstructed deep-floor median (dB, peak-normed frames, typically near floor_db)
    floor_db: float = -60.0
    #: image SNR (dB) — same metric as texture meta / realism ``median_tissue_snr_db``
    snr_db: float = 0.0
    #: mirror-measured 6 dB roll-off depth (µm); 0 if unknown
    falloff_6db_um: float = 0.0
    #: axial PSF FWHM from mirror (µm); 0 if unknown
    axial_fwhm_um: float = 0.0
    n_k: int = 0
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpectralProfile":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectralProfile":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def bundled_spectral(name: str = "device_a") -> SpectralProfile | None:
    """Load ``octsim/data/spectral_<name>.json`` if present."""
    path = os.path.join(_DATA_DIR, f"spectral_{name}.json")
    if not os.path.isfile(path):
        return None
    return SpectralProfile.load(path)


def _relerr2(a: float, b: float) -> float:
    b = float(b)
    if not np.isfinite(b) or abs(b) < 1e-12:
        return 0.0
    return float(((float(a) - b) / b) ** 2)


def spectral_distance(synth: SpectralProfile, target: SpectralProfile,
                      weights: dict | None = None) -> dict:
    """``D_spec`` and its decomposed non-negative terms.

    Default weights emphasise envelope shape and floor/SNR; falloff is low-weight
    because the instrument falloff is Tier-1 frozen and should already match.
    """
    w = {"envelope": 1.0, "tail": 0.3, "floor": 0.5, "snr": 0.5,
         "falloff": 0.2, "fwhm": 0.2, **(weights or {})}

    se = np.asarray(synth.envelope, float)
    te = np.asarray(target.envelope, float)
    n = min(se.size, te.size)
    if n >= 8:
        # compare peak-normed shapes on the overlapping support
        se = se[:n] / (float(np.max(se[:n])) or 1.0)
        te = te[:n] / (float(np.max(te[:n])) or 1.0)
        d_env = float(np.mean((se - te) ** 2))
    else:
        d_env = 0.0

    d_tail = _relerr2(synth.tail_frac, target.tail_frac) if target.tail_frac > 0 else 0.0
    d_floor = _relerr2(abs(synth.floor_db), abs(target.floor_db)) if target.floor_db else 0.0
    d_snr = _relerr2(synth.snr_db, target.snr_db) if target.snr_db > 0 else 0.0
    d_fall = (_relerr2(synth.falloff_6db_um, target.falloff_6db_um)
              if target.falloff_6db_um > 0 else 0.0)
    d_fwhm = (_relerr2(synth.axial_fwhm_um, target.axial_fwhm_um)
              if target.axial_fwhm_um > 0 else 0.0)

    total = (w["envelope"] * d_env + w["tail"] * d_tail + w["floor"] * d_floor
             + w["snr"] * d_snr + w["falloff"] * d_fall + w["fwhm"] * d_fwhm)
    return {
        "D_spec": float(total),
        "D_spec_envelope": float(w["envelope"] * d_env),
        "D_spec_tail": float(w["tail"] * d_tail),
        "D_spec_floor": float(w["floor"] * d_floor),
        "D_spec_snr": float(w["snr"] * d_snr),
        "D_spec_falloff": float(w["falloff"] * d_fall),
        "D_spec_fwhm": float(w["fwhm"] * d_fwhm),
        "raw": {"envelope_mse": d_env, "tail_relerr2": d_tail, "floor_relerr2": d_floor,
                "snr_relerr2": d_snr, "falloff_relerr2": d_fall, "fwhm_relerr2": d_fwhm},
    }


def profile_from_instrument(name: str = "device_a") -> SpectralProfile:
    """Spectral profile seeded from a bundled :class:`~octsim.instrument.Instrument`."""
    from .instrument import load_instrument
    inst = load_instrument(name)
    env = [] if inst.power_envelope is None else [float(x) for x in np.asarray(inst.power_envelope)]
    return SpectralProfile(
        envelope=env, tail_frac=float(inst.read_floor), floor_db=float(inst.floor_db),
        snr_db=float(inst.snr_db), falloff_6db_um=float(inst.falloff_6db_um),
        axial_fwhm_um=float(inst.axial_fwhm_um),
        n_k=len(env), source=f"instrument:{inst.name}")


def profile_from_sim(resolved, bscan_db: np.ndarray, *,
                     floor_rows: int = 80) -> SpectralProfile:
    """Extract a spectral profile from one simulated frame + its resolved config."""
    src = resolved.source
    k = src.k_grid()
    env = np.asarray(src.power_spectrum(k), float)
    env = env / (float(env.max()) or 1.0)
    db = np.asarray(bscan_db, float)
    floor = float(np.median(db[-floor_rows:, :])) if db.size else -60.0
    # tissue SNR ≈ peak − floor (rough; realism_metrics has the gated form)
    peak = float(np.percentile(db, 99.5)) if db.size else 0.0
    snr = float(peak - floor) if np.isfinite(peak) and np.isfinite(floor) else 0.0
    return SpectralProfile(
        envelope=env.tolist(), tail_frac=0.0, floor_db=floor, snr_db=snr,
        falloff_6db_um=float(getattr(resolved, "falloff_6db_um", 0.0) or 0.0),
        axial_fwhm_um=0.0, n_k=int(env.size), source="synthetic")


def measure_folder_envelope(folder: str, *, n_frames: int = 8) -> tuple[np.ndarray, float, float]:
    """Mean detector-grid spectrum from FPN or bscans: ``(env_peaknorm, tail_frac, read_std)``.

    Reuses the Tier-1 measurement recipe from an offline instrument calibration without
    requiring that script on ``sys.path`` for the happy path (FPN present).
    """
    import glob

    pix = 2048

    def _alines(path):
        return int(os.path.getsize(path) // 2 // pix)

    def _read(path, alines):
        d = np.fromfile(path, dtype=np.uint16)
        return d.reshape((pix, alines), order="F").astype(np.float64)

    fpn = None
    for name in ("fpn.bin", "FPN.raw"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            fpn = _read(p, _alines(p))
            break
    if fpn is None:
        paths = sorted(glob.glob(os.path.join(folder, "bscan*.raw")))[:n_frames]
        if not paths:
            raise FileNotFoundError(f"no FPN or bscan*.raw in {folder}")
        stack = np.stack([_read(p, _alines(p)).mean(axis=1) for p in paths], axis=1)
        env_raw = stack.mean(axis=1)
        read_std = float(np.median(stack.std(axis=1)))
    else:
        env_raw = fpn.mean(axis=1)
        read_std = float(np.median(fpn.std(axis=1)))

    tail = float(np.median(np.concatenate([env_raw[:20], env_raw[-20:]])))
    peak = float(np.max(env_raw))
    body = np.maximum(env_raw - tail, 0.0)
    env = body / (float(body.max()) or 1.0)
    tail_frac = float(tail / max(peak, 1.0))
    return env.astype(np.float64), tail_frac, read_std


def estimate_noise_components(clean_spectrum: np.ndarray, noisy_spectrum: np.ndarray,
                              ) -> dict:
    """Estimate shot / RIN / read scales from one clean vs noisy spectrum pair.

    Model: ``n = shot·√|I|·ε + rin·|I|·ε + read·ε``. With a single realisation this is
    under-determined; we report the *variance* of the residual and the three
    least-squares projections onto ``√|I|``, ``|I|``, and ``1`` (useful for comparing
    devices / settings, not as a unique decomposition).
    """
    I = np.abs(np.asarray(clean_spectrum, float).ravel())
    r = np.asarray(noisy_spectrum, float).ravel() - np.asarray(clean_spectrum, float).ravel()
    if I.size != r.size or I.size < 16:
        raise ValueError("clean/noisy spectra must share a length ≥ 16")
    # design matrix for std proxy: |r| ≈ a√I + b I + c
    A = np.column_stack([np.sqrt(I + 1e-12), I, np.ones_like(I)])
    coeff, *_ = np.linalg.lstsq(A, np.abs(r), rcond=None)
    return {
        "shot_proxy": float(max(coeff[0], 0.0)),
        "rin_proxy": float(max(coeff[1], 0.0)),
        "read_proxy": float(max(coeff[2], 0.0)),
        "residual_rms": float(np.sqrt(np.mean(r ** 2))),
        "residual_std": float(np.std(r)),
    }
