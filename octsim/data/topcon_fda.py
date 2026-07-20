"""Read Topcon ``.fda`` files — the DEVICE-PROCESSED B-scan volume (the clinical "final image").

The ``.fds`` sibling holds raw spectra (what :mod:`octsim.data.real_maestro` reconstructs); the ``.fda``
holds the processed output the device DISPLAYS: log-compressed, tone-mapped to a lifted-gray floor,
frame-averaged/denoised, 8-bit. That processed look — not our raw recon — is what "looks real", so it is
the right calibration reference for appearance.

Format: ``FOCTFDA`` header, then ``[name_len:u8][name][size:u32-LE][data]`` chunks. The B-scan volume is
in ``@IMG_JPEG`` as ``[25-byte header][ per slice: [size:u32-LE][JPEG2000 codestream] ]``. Slices are
JPEG2000 (PIL decodes them; the first carries a JP2 box, the rest are raw codestreams).
"""
from __future__ import annotations

import struct
from io import BytesIO

import numpy as np

_SIG = b"\x0e@FDA_FILE_INFO"     # 1st chunk (name_len 0x0e + name) — anchors chunk-walk start
_JP2_BOX = b"\x00\x00\x00\x0cjP"  # JPEG2000 signature box (marks the first slice)


def _chunks(d: bytes) -> dict:
    i = d.index(_SIG)
    ch = {}
    while i < len(d):
        nl = d[i]; i += 1
        if nl == 0 or i + nl > len(d):
            break
        name = d[i:i + nl]; i += nl
        if i + 4 > len(d):
            break
        size = struct.unpack("<I", d[i:i + 4])[0]; i += 4
        ch[name] = (i, size); i += size
    return ch


def read_fda_volume(path: str) -> np.ndarray:
    """Return the processed B-scan volume ``[n_bscans, depth, width]`` (uint8) from a ``.fda``."""
    d = open(path, "rb").read()
    ch = _chunks(d)
    if b"@IMG_JPEG" not in ch:
        raise ValueError(f"{path}: no @IMG_JPEG chunk")
    loc, size = ch[b"@IMG_JPEG"]
    blob = d[loc:loc + size]
    # header ends 4 bytes before the first JP2 box (that u32 is the first slice's size prefix)
    box = blob.find(_JP2_BOX)
    p = (box - 4) if box >= 4 else 25
    vol = []
    while p + 4 <= len(blob):
        sz = struct.unpack("<i", blob[p:p + 4])[0]; p += 4
        if sz <= 0 or p + sz > len(blob):
            break
        try:
            from PIL import Image
            vol.append(np.asarray(Image.open(BytesIO(blob[p:p + sz])).convert("L")))
        except Exception:
            pass
        p += sz
    if not vol:
        raise ValueError(f"{path}: decoded 0 B-scans from @IMG_JPEG")
    # frames can differ by a row on rare files — crop to the common shape
    h = min(v.shape[0] for v in vol); w = min(v.shape[1] for v in vol)
    return np.stack([v[:h, :w] for v in vol]).astype(np.uint8)


_TONE_PCTILES = (1.0, 5.0, 20.0, 50.0, 80.0, 95.0, 99.0, 99.9)


def calibrate_tone(sim_db: np.ndarray, dev_8bit: np.ndarray):
    """Return (sim_dB_anchors, device_8bit_anchors) — a monotone display LUT mapping the forward
    model's dB onto the device's 8-bit tone (lifted-gray floor, compressed brights). This is the
    device's DISPLAY transfer, calibrated once from a ``.fda`` (every OCT viewer has such a LUT)."""
    return (np.percentile(np.asarray(sim_db, float), _TONE_PCTILES),
            np.percentile(np.asarray(dev_8bit, float), _TONE_PCTILES))


def device_process(db: np.ndarray, tone, *, pitch_ratio: float = 1.56 / 2.6,
                   sigma=(1.0, 0.8)) -> np.ndarray:
    """Forward-model dB B-scan → device-equivalent 8-bit image (the Maestro3 DSP display chain):
    resample depth to the ``.fda`` pixel pitch, a light speckle-reduction (in linear amplitude), then
    the calibrated display tone-curve. Separate from the physics — the forward model stays pure."""
    from scipy.ndimage import gaussian_filter, zoom
    r = zoom(np.asarray(db, float), (pitch_ratio, 1.0), order=1)
    if sigma and (sigma[0] or sigma[1]):
        r = 20.0 * np.log10(np.maximum(gaussian_filter(10.0 ** (r / 20.0), sigma), 1e-6))
    sim_p, dev_p = tone
    return np.interp(r, sim_p, dev_p)


def fda_depth_um_per_px(path: str, default: float = 2.6) -> float:
    """Axial pixel pitch (µm/px) of the processed B-scans, from @PARAM_SCAN_04 if present.

    The scan param block stores the physical depth range; divided by the B-scan height it gives the
    display pitch. Falls back to ``default`` (~Maestro3 2.3 mm / 885 px) if the field can't be read.
    """
    try:
        d = open(path, "rb").read()
        ch = _chunks(d)
        vol_h = None
        if b"@IMG_JPEG" in ch:
            loc, size = ch[b"@IMG_JPEG"]
            box = d[loc:loc + size].find(_JP2_BOX)
            # height lives in the JP2 'ihdr' box: 4-byte height right after 'ihdr'
            ih = d.find(b"ihdr", loc, loc + size)
            if ih > 0:
                vol_h = struct.unpack(">I", d[ih + 4:ih + 8])[0]
        # @PARAM_SCAN_04: try to read a plausible depth-range float (mm) — best-effort
        return default if not vol_h else float(2300.0 / vol_h) if vol_h else default
    except Exception:
        return default
