"""Reconstruct real Topcon Maestro ``.raw`` frames into octsim's dB B-scan domain.

Shared by ``scripts/build_real_targets.py`` (real ``TextureProfile`` anchors) and
``scripts/calibrate_optics.py`` (real-vs-simulated comparison figures), so both use the
*identical* pipeline the OCT-Denoiser uses:

    uint16 [2048, alines] order='F' → DC-fix + mean-A-line subtract → CLB cubic k-linear
    resample → Hann → IFFT → crop rows (1024, 2048) → magnitude → 20·log10 peak-norm (clip).

Raw OCT data is never redistributed; this only reads a local, hydrated scan folder.
"""

from __future__ import annotations

import glob
import os

import numpy as np

PIX = 2048
_RECALL = 0x00400000       # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS (OneDrive placeholder)
_OFFLINE = 0x00001000      # FILE_ATTRIBUTE_OFFLINE


def is_placeholder(path: str) -> bool:
    """True if the file is a not-locally-present OneDrive placeholder (reading it would hang)."""
    try:
        return bool(os.stat(path).st_file_attributes & (_RECALL | _OFFLINE))
    except (AttributeError, OSError):
        return False


def read_clb(clb_path: str, pixels: int = PIX) -> np.ndarray:
    """CLB k-linearisation LUT: skip the 64-byte header, take the first ``pixels`` float32 ∈ [0,1]."""
    with open(clb_path, "rb") as f:
        f.seek(64)
        arr = np.frombuffer(f.read(), dtype=np.float32)
    return arr[:pixels].astype(np.float64)


def find_clb(folder: str) -> str:
    """Locate the CLB in ``folder`` or its parent (the device-level CLB applies to all scans)."""
    c = sorted(glob.glob(os.path.join(folder, "*.CLB")))
    if not c:
        c = sorted(glob.glob(os.path.join(os.path.dirname(folder.rstrip("/\\")), "*.CLB")))
    if not c:
        raise FileNotFoundError(f"no .CLB near {folder}")
    return c[0]


def klinear_operator(lut: np.ndarray, n: int = PIX):
    """Return a function resampling a [n]/[n,x] spectrum onto the k-linear grid via the CLB LUT
    (natural cubic spline where SciPy is present, else linear)."""
    x = np.linspace(0.0, 1.0, n)
    q = np.clip(lut, 0.0, 1.0)
    try:
        from scipy.interpolate import CubicSpline
        return lambda spec: CubicSpline(x, spec, axis=0, bc_type="natural")(q)
    except Exception:
        return lambda spec: (np.interp(q, x, spec) if spec.ndim == 1
                             else np.stack([np.interp(q, x, spec[:, c])
                                            for c in range(spec.shape[1])], axis=1))


def recon_frame(path: str, alines: int, kop, *, crop=(1024, 2048), floor_db=-60.0,
                orient=True) -> np.ndarray:
    """Raw spectrum file → dB B-scan matching octsim's ``bscan_db`` domain (20·log10 peak-norm)."""
    d = np.fromfile(path, dtype=np.uint16)
    if d.size != PIX * alines:
        raise ValueError(f"{os.path.basename(path)}: {d.size} != {PIX*alines}")
    raw = d.reshape((PIX, alines), order="F").astype(np.float64)
    raw[:3, :] = raw[3:4, :]
    raw[:, 0] = raw[:, 1]
    raw = raw - raw.mean(axis=1, keepdims=True)
    resamp = kop(raw)
    win = np.hanning(PIX)[:, None]
    depth = np.fft.ifft(resamp * win, axis=0)
    mag = np.abs(depth)[crop[0]:crop[1], :]
    peak = float(mag.max()) or 1.0
    db = np.clip(20.0 * np.log10(np.maximum(mag / peak, 1e-12)), floor_db, 0.0)
    return orient_ilm_up(db) if orient else db


def recon_frame_linear(path: str, alines: int, kop, *, crop=(1024, 2048),
                       orient=True) -> np.ndarray:
    """Raw spectrum file → **linear intensity** B-scan (no log, no peak normalisation).

    :func:`recon_frame` returns the dB display image, which is the wrong domain for
    anything quantitative about energy — notably the depth-resolved attenuation estimator
    in :mod:`octsim.attenuation`, whose whole basis is the ratio of a pixel's signal to the
    integrated signal beneath it. Taking that ratio in a log domain, or on a peak-normalised
    image, does not give attenuation.
    """
    d = np.fromfile(path, dtype=np.uint16)
    if d.size != PIX * alines:
        raise ValueError(f"{os.path.basename(path)}: {d.size} != {PIX*alines}")
    raw = d.reshape((PIX, alines), order="F").astype(np.float64)
    raw[:3, :] = raw[3:4, :]
    raw[:, 0] = raw[:, 1]
    raw = raw - raw.mean(axis=1, keepdims=True)
    resamp = kop(raw)
    win = np.hanning(PIX)[:, None]
    depth = np.fft.ifft(resamp * win, axis=0)
    inten = (np.abs(depth) ** 2)[crop[0]:crop[1], :]
    if not orient:
        return inten
    prof = inten.mean(axis=1)
    z = np.arange(prof.size)
    centroid = float((z * prof).sum() / (prof.sum() + 1e-30))
    return inten[::-1, :] if centroid > prof.size / 2 else inten


def orient_ilm_up(img: np.ndarray) -> np.ndarray:
    """Flip depth so the retina reads ILM-up (bright-band centroid in the top half)."""
    prof = img.mean(axis=1)
    prof = prof - prof.min()
    z = np.arange(prof.size)
    centroid = float((z * prof).sum() / (prof.sum() + 1e-9))
    return img[::-1, :] if centroid > prof.size / 2 else img


def alines_of(path: str, pixels: int = PIX) -> int:
    return int(os.path.getsize(path) // 2 // pixels)


def load_corpus(folder: str, *, limit: int | None = None, stride: int = 1):
    """Yield reconstructed dB B-scans from a scan folder, skipping OneDrive placeholders.

    ``stride`` subsamples the frame list (use >1 on a dense volume for spatial diversity).
    Returns (list_of_images, alines, n_skipped).
    """
    lut = read_clb(find_clb(folder))
    bs_all = sorted(glob.glob(os.path.join(folder, "bscan*.raw")))
    if not bs_all:
        raise FileNotFoundError(f"no bscan*.raw in {folder}")
    bs = [p for p in bs_all if not is_placeholder(p)]
    n_skipped = len(bs_all) - len(bs)
    if not bs:
        raise RuntimeError(f"no hydrated bscan*.raw in {folder} (all OneDrive placeholders)")
    bs = bs[::max(1, stride)]
    if limit:
        bs = bs[:limit]
    alines = alines_of(bs[0])
    kop = klinear_operator(lut)
    imgs = []
    for p in bs:
        try:
            imgs.append(recon_frame(p, alines, kop))
        except Exception:
            continue
    return imgs, alines, n_skipped


def snr_cnr(img_db: np.ndarray, sig=(111, 600), bg_rows=20) -> tuple:
    """SNR/CNR (dB) on linear amplitude: signal rows [sig], background = bottom rows."""
    lin = 10.0 ** (img_db / 20.0)
    H, W = lin.shape
    x0, x1 = 10, max(11, W - 10)
    y0, y1 = min(sig[0], H - 2), min(sig[1], H)
    sig_roi = lin[y0:y1, x0:x1]
    bg_roi = lin[max(0, H - bg_rows):H, x0:x1]
    if sig_roi.size == 0 or bg_roi.size == 0:
        return float("nan"), float("nan")
    std_bg = float(np.std(bg_roi)) + 1e-9
    return (20.0 * np.log10(float(np.mean(np.max(sig_roi, axis=0))) / std_bg),
            20.0 * np.log10(float(np.mean(sig_roi)) / std_bg))
