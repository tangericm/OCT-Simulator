"""Emit device-format uint16 spectra from simulated interferograms (Denoiser bridge).

Inverts the forward half of :mod:`octsim.data.real_device`:

    simulated I(k, x)  →  optional inverse-CLB warp  →  uint16 ``bscanNNNNNN.raw``
    (+ copy / write a ``.CLB`` so the Denoiser recon path is unchanged)

Payoff: **noise-free training targets** (``noise_free=True``) that no real corpus can
provide, plus matched noisy spectra from the same seed.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from .data.real_device import PIX, find_clb, read_clb
from .generate import simulate
from .config import SimConfig


def inverse_clb_warp(spectrum_klinear: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map a k-linear spectrum back onto the device detector grid via the CLB LUT.

    ``lut`` is monotone ∈ [0,1] giving the k-linear sample position for each detector
    pixel. We invert by interpolating the k-linear signal onto the detector abscissa.
    """
    n = spectrum_klinear.shape[0]
    q = np.clip(np.asarray(lut[:n], float), 0.0, 1.0)
    # k-linear abscissa is uniform in [0,1]; detector index i samples q[i]
    x_k = np.linspace(0.0, 1.0, n)
    # For each detector pixel, value = signal_klinear(q[i])
    if spectrum_klinear.ndim == 1:
        return np.interp(q, x_k, spectrum_klinear)
    return np.stack([np.interp(q, x_k, spectrum_klinear[:, c])
                     for c in range(spectrum_klinear.shape[1])], axis=1)


def spectra_to_uint16(spectra: np.ndarray, *, peak: float | None = None,
                      max_count: int = 4095) -> np.ndarray:
    """Scale float spectra to device-like 12-bit-in-uint16 counts."""
    s = np.asarray(spectra, float)
    s = np.maximum(s, 0.0)
    pk = float(peak) if peak is not None else float(s.max() or 1.0)
    out = np.clip(np.round(s / pk * max_count), 0, max_count).astype(np.uint16)
    return out


def write_bscan_raw(path: str, spectra_uint16: np.ndarray) -> None:
    """Write ``[n_k, alines]`` uint16 column-major (order='F'), matching device export."""
    a = np.asarray(spectra_uint16, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D spectra, got shape {a.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a.reshape(-1, order="F").tofile(path)


def write_clb(path: str, lut: np.ndarray, *, serial: str = "000003") -> None:
    """Write a minimal device-like CLB (64-byte header + 4096 float32, first n used)."""
    header = bytearray(64)
    header[0:4] = b"2.00"
    header[8:14] = serial.encode("ascii")[:6]
    body = np.zeros(4096, dtype=np.float32)
    n = min(len(lut), 4096)
    body[:n] = np.asarray(lut[:n], dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(body.tobytes())


def export_simulation(out_dir: str, *, n_frames: int = 1, seed: int = 0,
                      instrument: str = "device_a", noise_free: bool = False,
                      clb_folder: str | None = None, lines: int = 1024,
                      fov_um: float = 6000.0) -> list[str]:
    """Simulate and write ``bscan000000.raw`` … plus a ``.CLB``.

    Returns the list of written raw paths. When ``noise_free=True`` the spectra have
    detector noise disabled — the Denoiser clean target.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Prefer a real device CLB so the Denoiser's recon matches training data geometry.
    lut = None
    if clb_folder and os.path.isdir(clb_folder):
        try:
            lut = read_clb(find_clb(clb_folder))
        except Exception:
            lut = None
    if lut is None:
        # identity LUT (already k-linear) — Denoiser still runs, just no warp
        lut = np.linspace(0.0, 1.0, PIX)

    write_clb(os.path.join(out_dir, "sim.CLB"), lut)
    paths = []
    for i in range(n_frames):
        cfg = SimConfig(instrument=instrument, anatomy="macula", shape_model="octa500",
                        lines_per_frame=lines, lateral_fov_um=fov_um,
                        noise_free=noise_free, seed=seed + 17 * i, randomize=True)
        res = simulate(cfg)
        if res.spectra is None:
            raise RuntimeError("simulate() did not retain spectra; cannot export device raw")
        spec = np.asarray(res.spectra, float)  # [n_k, alines] on k-linear grid
        if spec.shape[0] != PIX:
            # resample axially to device pixel count if needed
            x = np.linspace(0.0, 1.0, spec.shape[0])
            q = np.linspace(0.0, 1.0, PIX)
            spec = np.stack([np.interp(q, x, spec[:, c]) for c in range(spec.shape[1])], axis=1)
        det = inverse_clb_warp(spec, lut)
        raw = spectra_to_uint16(det)
        path = os.path.join(out_dir, f"bscan{i:06d}.raw")
        write_bscan_raw(path, raw)
        paths.append(path)
    return paths
