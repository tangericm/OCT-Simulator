"""Build the Tier-1 instrument JSON from a Topcon Maestro scan folder (MEASURED, then frozen).

    python scripts/build_instrument.py \
        --folder "C:/.../OCT Data/Maestro3/6mm_1024Aline" \
        --mirror "C:/.../OCT Data/Maestro3/mirror_6mm_1024Aline" \
        --name maestro3 --out octsim/data/instrument_maestro3.json

Measures — never fits — the device model:

* **P(k)** : ``fpn.bin``/``FPN.raw`` (2048×N uint16) → mean A-line → CLB k-linearise →
  peak-normalised ``power_envelope[2048]`` on octsim's linear-k grid.
* **λ0 / n_k** : 840 nm SD-OCT, 2048 pixels (from ``MainInfo.bin`` / the device spec).
* **bandwidth / k_span** : Gaussian fit of the k-linearised envelope FWHM → ``bandwidth_fwhm_nm``;
  ``k_span_factor`` set so the reconstructed **depth pixel matches the mirror-measured PSF**.
* **reference_amp / read_floor** : DC-to-tail ratio and per-pixel FPN variance.
* **falloff_6db_um / axial_fwhm_um** : from the mirror scans (peak vs depth, PSF width).

Raw OCT data is NEVER committed; only the small JSON this writes is.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

PIX = 2048
LAMBDA0_NM = 840.0      # 3D OCT-1 Maestro SD-OCT
DEPTH_PIXEL_UM = 3.87   # Maestro axial pixel pitch (matches the OCT-Denoiser recon convention)
AXIAL_RES_UM = 8.9      # coherence axial resolution (optical) — calibrated so the synthetic outer
                        # complex (EZ↔RPE) resolves like the real M3 frames (~Maestro 6 µm tissue)
_LN2 = np.log(2.0)
_RECALL = 0x00400000    # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS (OneDrive placeholder)
_OFFLINE = 0x00001000   # FILE_ATTRIBUTE_OFFLINE


def is_placeholder(path: str) -> bool:
    """True if the file is a not-locally-present OneDrive placeholder (reading it would hang)."""
    try:
        return bool(os.stat(path).st_file_attributes & (_RECALL | _OFFLINE))
    except (AttributeError, OSError):
        return False


# --------------------------------------------------------------------------- IO
def read_clb(clb_path: str, pixels: int = PIX) -> np.ndarray:
    """CLB resampling LUT: skip 64-byte header, first ``pixels`` float32 ∈ [0,1]."""
    with open(clb_path, "rb") as f:
        f.seek(64)
        arr = np.frombuffer(f.read(), dtype=np.float32)
    return arr[:pixels].astype(np.float64)


def read_raw(path: str, alines: int, pixels: int = PIX) -> np.ndarray:
    """uint16 [pixels, alines] column-major (order='F')."""
    d = np.fromfile(path, dtype=np.uint16)
    if d.size != pixels * alines:
        raise ValueError(f"{os.path.basename(path)}: {d.size} != {pixels*alines}")
    return d.reshape((pixels, alines), order="F").astype(np.float64)


def find_clb(folder: str) -> str:
    c = sorted(glob.glob(os.path.join(folder, "*.CLB")))
    if not c:                                   # fall back to the parent (device-level CLB)
        c = sorted(glob.glob(os.path.join(os.path.dirname(folder), "*.CLB")))
    if not c:
        raise FileNotFoundError(f"no .CLB near {folder}")
    return c[0]


def klinear(spec: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Resample a [pixels]/[pixels,x] spectrum onto the k-linear grid via the CLB LUT
    (cubic where SciPy is available, else linear). ``lut`` ∈ [0,1] are sample positions."""
    n = spec.shape[0]
    x = np.linspace(0.0, 1.0, n)
    q = np.clip(lut, 0.0, 1.0)
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x, spec, axis=0, bc_type="natural")
        return cs(q)
    except Exception:
        if spec.ndim == 1:
            return np.interp(q, x, spec)
        return np.stack([np.interp(q, x, spec[:, c]) for c in range(spec.shape[1])], axis=1)


# --------------------------------------------------------------------------- measurements
def _alines_of(path: str, pixels: int = PIX) -> int:
    return int(os.path.getsize(path) // 2 // pixels)


def measure_envelope(folder: str, lut: np.ndarray):
    """Measured P(k): k-linearised FPN mean A-line, peak-normalised, + tail floor + read std."""
    fpn = None
    for name in ("fpn.bin", "FPN.raw"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            fpn = read_raw(p, _alines_of(p)); break
    if fpn is None:
        # Fallback (devices with no dedicated FPN file, e.g. Maestro2): the source power spectrum
        # P(k) is the mean raw spectrum. Average a handful of hydrated bscan frames over A-lines —
        # tissue backscatter averages out across the ~2048 A-lines, leaving the source envelope +
        # electronic offset. Concatenate the per-frame A-line stacks so read_std is still meaningful.
        bs = [p for p in sorted(glob.glob(os.path.join(folder, "bscan*.raw")))
              if not is_placeholder(p)][:8]
        if not bs:
            raise FileNotFoundError(f"no fpn.bin/FPN.raw and no hydrated bscan*.raw in {folder}")
        fpn = np.concatenate([read_raw(p, _alines_of(p)) for p in bs], axis=1)
    # DC fix (first 3 rows) then mean over A-lines
    fpn = fpn.copy(); fpn[:3, :] = fpn[3:4, :]
    env_raw = fpn.mean(axis=1)
    tail = float(np.median(np.concatenate([env_raw[:20], env_raw[-20:]])))  # electronic floor
    env_lin = klinear(np.maximum(env_raw - tail, 0.0), lut)
    env_lin = np.maximum(env_lin, 0.0)
    peak = float(env_lin.max()) or 1.0
    env = env_lin / peak
    read_std = float(np.median(fpn.std(axis=1)))          # per-pixel electronic std
    dc = float(env_raw.max())
    reference_amp = float(np.clip(np.sqrt(max(dc / max(tail, 1.0), 1.0)), 2.0, 12.0))
    return env, tail, read_std, reference_amp


def gaussian_fwhm_samples(env: np.ndarray) -> float:
    """FWHM (in samples) of the envelope's central lobe (half-max crossings around the peak)."""
    p = int(np.argmax(env))
    half = env[p] / 2.0
    left = p
    while left > 0 and env[left] > half:
        left -= 1
    right = p
    while right < env.size - 1 and env[right] > half:
        right += 1
    return float(max(right - left, 1))


def measure_mirror(mirror_folder: str, lut: np.ndarray, depth_pixel_um_guess: float):
    """Sensitivity roll-off + axial PSF FWHM from the mirror scans.

    Reconstructs each mirror frame (DC-subtract → k-linear → Hann → IFFT → |·|), finds the
    peak depth + amplitude per column, and returns (falloff_6db_um, axial_fwhm_um). Falls back
    to (0, 0) if the folder is unusable, so the caller can proceed without it.
    """
    try:
        bs = [p for p in sorted(glob.glob(os.path.join(mirror_folder, "bscan*.raw")))
              if not is_placeholder(p)]           # only hydrated mirror frames (no cloud recall)
        if not bs:
            return 0.0, 0.0
        alines = _alines_of(bs[0])
        peaks_depth, peaks_amp, fwhms = [], [], []
        for p in bs[:16]:
            raw = read_raw(p, alines)
            raw[:3, :] = raw[3:4, :]
            raw = raw - raw.mean(axis=1, keepdims=True)
            resamp = klinear(raw, lut)
            win = np.hanning(PIX)[:, None]
            depth = np.fft.ifft(resamp * win, axis=0)
            mag = np.abs(depth[:PIX // 2, :])
            for c in range(0, alines, max(1, alines // 64)):
                col = mag[:, c]
                pk = int(np.argmax(col))
                if pk < 3 or pk > PIX // 2 - 4:
                    continue
                peaks_depth.append(pk); peaks_amp.append(float(col[pk]))
                fwhms.append(_peak_fwhm(col, pk))
        if not peaks_amp:
            return 0.0, 0.0
        axial_fwhm_um = float(np.median(fwhms)) * depth_pixel_um_guess
        # roll-off: amplitude vs depth (dB). 6 dB drop distance.
        d = np.array(peaks_depth, float) * depth_pixel_um_guess
        a_db = 20.0 * np.log10(np.array(peaks_amp, float) / (max(peaks_amp) + 1e-12))
        if d.ptp() > 1e-6:
            slope = np.polyfit(d, a_db, 1)[0]         # dB per µm (negative)
            falloff_6db_um = float(abs(6.0 / slope)) if slope < 0 else 0.0
        else:
            falloff_6db_um = 0.0
        return float(np.clip(falloff_6db_um, 100.0, 5000.0)), axial_fwhm_um
    except Exception:
        return 0.0, 0.0


def _peak_fwhm(col: np.ndarray, pk: int) -> float:
    half = col[pk] / 2.0
    left = pk
    while left > 0 and col[left] > half:
        left -= 1
    right = pk
    while right < col.size - 1 and col[right] > half:
        right += 1
    return float(max(right - left, 1))


def _mirror_peak(folder: str, lut: np.ndarray):
    """Reconstruct the first hydrated mirror frame → (mean_peak_row, mean_peak_amp, psf_fwhm_px).

    The mirror is TILTED (its peak row drifts across A-lines), so the PSF FWHM is the median of the
    PER-A-LINE peak widths — averaging the magnitude over A-lines first would smear a ~2.5 px PSF
    into ~9 px (the source of the earlier spurious 34.8 µm axial FWHM)."""
    bs = [p for p in sorted(glob.glob(os.path.join(folder, "bscan*.raw")))
          if not is_placeholder(p)]
    if not bs:
        return None
    alines = _alines_of(bs[0])
    raw = read_raw(bs[0], alines)
    raw[:3, :] = raw[3:4, :]
    raw = raw - raw.mean(axis=1, keepdims=True)
    depth = np.abs(np.fft.ifft(klinear(raw, lut) * np.hanning(PIX)[:, None], axis=0))[:PIX // 2, :]
    col = depth.mean(1)                                   # for the mean peak DEPTH/AMP (falloff)
    pk = int(np.argmax(col[5:])) + 5
    if pk < 6 or pk > PIX // 2 - 5:
        return None
    widths = []                                          # per-A-line PSF width (no tilt smearing)
    for c in range(0, alines, max(1, alines // 128)):
        v = depth[:, c]; p = int(np.argmax(v[5:])) + 5
        if 6 < p < PIX // 2 - 5:
            widths.append(_peak_fwhm(v, p))
    fwhm = float(np.median(widths)) if widths else _peak_fwhm(col, pk)
    return pk, float(col[pk]), fwhm


def measure_falloff(mirror_folders: list, lut: np.ndarray, depth_pixel_um: float):
    """Sensitivity roll-off + axial PSF FWHM from a set of mirror scans stepped through depth.

    A single mirror folder is at one depth (no roll-off signal); the stepped ``mirror_*`` set
    gives (depth, peak-amplitude) pairs → a dB/µm slope → ``falloff_6db_um``. The axial FWHM is
    taken from the shallowest (sharpest) peak. Returns (falloff_6db_um, axial_fwhm_um).
    """
    pts = []
    for f in mirror_folders:
        if os.path.isdir(f):
            mp = _mirror_peak(f, lut)
            if mp:
                pts.append(mp)
    if len(pts) < 2:
        # one depth only: report the axial PSF but no roll-off
        return 0.0, (pts[0][2] * depth_pixel_um if pts else 0.0)
    pts.sort(key=lambda t: t[0])
    depths = np.array([p[0] for p in pts], float) * depth_pixel_um
    amps = np.array([p[1] for p in pts], float)
    a_db = 20.0 * np.log10(amps / (amps.max() + 1e-12))
    slope = float(np.polyfit(depths - depths[0], a_db, 1)[0])       # dB per µm (negative)
    falloff = float(abs(6.0 / slope)) if slope < 0 else 0.0
    axial_fwhm_um = float(pts[0][2]) * depth_pixel_um               # shallowest = sharpest PSF
    return float(np.clip(falloff, 100.0, 6000.0)), axial_fwhm_um


def sibling_mirror_folders(mirror: str) -> list:
    """From one mirror folder, discover its stepped siblings (``mirror_*`` in the same parent)."""
    if not mirror:
        return []
    parent = os.path.dirname(mirror.rstrip("/\\"))
    base = os.path.basename(mirror.rstrip("/\\"))
    stem = base.split("_")[0] + "_" + "_".join(base.split("_")[1:3]) if "_" in base else base
    sibs = sorted(glob.glob(os.path.join(parent, "mirror_*")))
    sibs = [s for s in sibs if os.path.isdir(s)]
    return sibs or ([mirror] if os.path.isdir(mirror) else [])


# --------------------------------------------------------------------------- main
def build(folder: str, mirror: str | None, name: str, lambda0_nm: float = LAMBDA0_NM,
          depth_pixel_um: float = DEPTH_PIXEL_UM, axial_res_um: float = AXIAL_RES_UM,
          falloff_um: float = 0.0):
    from octsim.instrument import Instrument
    from octsim.source import LightSource

    lut = read_clb(find_clb(folder))
    env, tail, read_std, reference_amp = measure_envelope(folder, lut)

    # Axial RESOLUTION (coherence) sets the bandwidth: Δλ = (2ln2/π)·λ0²/axial_res. This is the
    # AXIAL SHARPNESS of the layers — calibrated (default 8.9 µm optical ≈ Maestro ~6 µm tissue)
    # so the synthetic resolves the outer bright complex (EZ↔RPE) like the real frames rather than
    # over-spreading it. It is set INDEPENDENTLY of the depth pixel (that is the k_span job below).
    # (The old envelope-FWHM×nominal-band heuristic under-estimated Δλ and made layers too spread.)
    bandwidth_fwhm_nm = float((2.0 * _LN2 / np.pi) * (lambda0_nm * 1e-3) ** 2 / axial_res_um * 1e3)

    # Depth calibration: depth_pixel_um ∝ 1/k_span_factor at fixed λ0/BW/n_k, so scale a nominal
    # k_span=1.5 to hit the target 3.87 µm/px (the axial pitch the real frames reconstruct on).
    base = LightSource(center_wavelength_nm=lambda0_nm, bandwidth_fwhm_nm=bandwidth_fwhm_nm,
                       n_k=PIX, k_span_factor=1.5)
    k_span_factor = float(np.clip(1.5 * base.depth_pixel_um / depth_pixel_um, 0.5, 6.0))
    depth_pixel = LightSource(center_wavelength_nm=lambda0_nm, bandwidth_fwhm_nm=bandwidth_fwhm_nm,
                              n_k=PIX, k_span_factor=k_span_factor).depth_pixel_um

    falloff_6db_um, axial_fwhm_um = (0.0, 0.0)
    if mirror:
        folders = sibling_mirror_folders(mirror)      # the stepped mirror_* set (multi-depth)
        falloff_6db_um, axial_fwhm_um = measure_falloff(folders, lut, depth_pixel)
        print(f"  mirror folders used: {len(folders)}  falloff_6db={falloff_6db_um:.0f}um "
              f"axial_fwhm={axial_fwhm_um:.2f}um")
    elif falloff_um > 0:
        # no stepped-mirror set on this device (e.g. Maestro2) — carry an explicit falloff (e.g. the
        # sibling device's MEASURED value) rather than 0 (no roll-off). NOT a measured value: flag it.
        falloff_6db_um = float(falloff_um)
        print(f"  no mirror; using supplied falloff_6db={falloff_6db_um:.0f}um (NOT measured)")

    inst = Instrument(
        name=name, center_wavelength_nm=float(lambda0_nm), n_k=PIX,
        bandwidth_fwhm_nm=bandwidth_fwhm_nm, k_span_factor=k_span_factor,
        power_envelope=env, background=None,
        reference_amp=reference_amp, read_floor=float(np.clip(read_std / 255.0, 1e-3, 0.2)),
        floor_db=-60.0, falloff_6db_um=falloff_6db_um, axial_fwhm_um=axial_fwhm_um,
        source=f"measured:{os.path.basename(folder)}")
    return inst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True, help="scan folder with CLB + fpn.bin")
    ap.add_argument("--mirror", default=None, help="mirror scan folder (falloff + axial PSF)")
    ap.add_argument("--name", default="maestro3")
    ap.add_argument("--lambda0", type=float, default=LAMBDA0_NM)
    ap.add_argument("--depth-pixel-um", type=float, default=DEPTH_PIXEL_UM,
                    help="axial pixel pitch to calibrate k_span to (match the real recon)")
    ap.add_argument("--axial-res-um", type=float, default=AXIAL_RES_UM,
                    help="coherence axial resolution (optical) -> sets bandwidth (layer sharpness)")
    ap.add_argument("--falloff-um", type=float, default=0.0,
                    help="fallback 6 dB falloff when there is no mirror set (e.g. sibling device's "
                         "measured value); NOT a measured value for this device")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    inst = build(args.folder, args.mirror, args.name, args.lambda0, args.depth_pixel_um,
                 args.axial_res_um, args.falloff_um)
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "octsim", "data", f"instrument_{args.name}.json")
    inst.save(out)
    print(f"wrote {out}")
    print(f"  lambda0={inst.center_wavelength_nm:.0f}nm n_k={inst.n_k} "
          f"BW={inst.bandwidth_fwhm_nm:.1f}nm k_span={inst.k_span_factor:.3f}")
    print(f"  reference_amp={inst.reference_amp:.2f} read_floor={inst.read_floor:.4f} "
          f"falloff_6db={inst.falloff_6db_um:.0f}um axial_fwhm={inst.axial_fwhm_um:.2f}um")
    env = np.asarray(inst.power_envelope)
    print(f"  envelope: peak@{int(env.argmax())} nonzero_frac={float((env>0.01).mean()):.2f}")


if __name__ == "__main__":
    main()
