"""Reconstruct a real Maestro corpus and save its target ``TextureProfile`` (+ SNR/CNR).

    # dense single-location line scan (1024 A-lines → 5.86 µm lateral):
    python scripts/build_real_targets.py --folder ".../Maestro3/6mm_1024Aline" --name maestro3_macula

    # spatially-diverse volume (256 A-lines → 23.4 µm lateral; stride for coverage):
    python scripts/build_real_targets.py --folder ".../Maestro3/M3_Macula_6x6mm" \
        --name maestro3_macula_vol --stride 4 --fov-um 6000

Uses the shared :mod:`octsim.data.real_maestro` pipeline (identical to the comparison-figure
recon), skips OneDrive placeholders, and auto-derives the lateral pitch from FOV/A-lines so the
target is directly comparable to a synthetic render at the same sampling. Raw data is never
committed — only the small JSON this writes is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim.data import real_maestro, unlabeled


def build(folder: str, name: str, *, limit: int | None, stride: int, axial_pitch_um: float,
          fov_um: float | None, lateral_pitch_um: float | None):
    imgs, alines, n_skipped = real_maestro.load_corpus(folder, limit=limit, stride=stride)
    if n_skipped:
        print(f"  skipped {n_skipped} un-hydrated (OneDrive placeholder) frames")
    if not imgs:
        raise RuntimeError("no frames reconstructed")
    if lateral_pitch_um is None:
        lateral_pitch_um = (fov_um / alines) if fov_um else 5.86
    snrs, cnrs = [], []
    for db in imgs:
        s, c = real_maestro.snr_cnr(db)
        if np.isfinite(s):
            snrs.append(s); cnrs.append(c)
    tex = unlabeled.extract_texture(imgs, axial_pitch_um=axial_pitch_um,
                                    lateral_pitch_um=lateral_pitch_um, source=f"real:{name}")
    meta = {
        "snr_db_mean": float(np.mean(snrs)) if snrs else None,
        "snr_db_std": float(np.std(snrs)) if snrs else None,
        "cnr_db_mean": float(np.mean(cnrs)) if cnrs else None,
        "n_frames": len(imgs), "alines": alines, "stride": stride,
        "axial_pitch_um": axial_pitch_um, "lateral_pitch_um": float(lateral_pitch_um),
        "folder": os.path.basename(folder.rstrip("/\\")),
    }
    return tex, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--name", default="maestro3_macula")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="subsample frames (use >1 on a dense volume)")
    ap.add_argument("--axial-pitch-um", type=float, default=3.87)
    ap.add_argument("--fov-um", type=float, default=None, help="lateral FOV (µm); pitch = FOV/A-lines")
    ap.add_argument("--lateral-pitch-um", type=float, default=None, help="override the lateral pitch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tex, meta = build(args.folder, args.name, limit=args.limit, stride=args.stride,
                      axial_pitch_um=args.axial_pitch_um, fov_um=args.fov_um,
                      lateral_pitch_um=args.lateral_pitch_um)
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "octsim", "data", f"texture_{args.name}.json")
    tex.save(out)
    with open(out.replace(".json", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out}")
    print(f"  n={tex.n_images} alines={meta['alines']} lat_pitch={meta['lateral_pitch_um']:.2f}um "
          f"speckle={tex.speckle_contrast:.3f} thick={tex.total_retina_um_mean:.0f}um "
          f"grain(ax/lat)={tex.axial_grain_um:.1f}/{tex.lateral_grain_um:.1f}um")
    print(f"  SNR={meta['snr_db_mean']}dB CNR={meta['cnr_db_mean']}dB frames={meta['n_frames']}")


if __name__ == "__main__":
    main()
