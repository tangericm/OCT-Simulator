#!/usr/bin/env python
"""Fit a label-free TextureProfile from a folder of OCT B-scan images.

Point it at any pile of OCT images (Kermany OCT2017 NORMAL, Srinivasan 2014, OCTID, OCTDL, …);
it auto-detects the ILM+RPE, learns the reflectivity template + speckle/heterogeneity/noise/
thickness/curvature statistics, and writes a small JSON. No labels required.

    python scripts/fit_texture.py --root path/to/NORMAL --out octsim/data/texture_srinivasan.json

Pool several datasets by fitting each and averaging, or just point --root at a parent folder
containing them all (the walker recurses).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder tree of OCT images (recursed)")
    ap.add_argument("--out", default="octsim/data/texture_srinivasan.json")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of images (0 = all)")
    ap.add_argument("--axial-pitch-um", type=float, default=3.87)
    ap.add_argument("--lateral-pitch-um", type=float, default=6.7)
    ap.add_argument("--source", default="unlabeled")
    args = ap.parse_args()

    from octsim.data import unlabeled as ul
    imgs = ul.load_images(args.root, limit=(args.limit or None))
    prof = ul.extract_texture(imgs, axial_pitch_um=args.axial_pitch_um,
                              lateral_pitch_um=args.lateral_pitch_um, source=args.source)
    prof.save(args.out)
    print(f"fit {prof.n_images} B-scans -> {args.out}")
    print(f"  speckle contrast {prof.speckle_contrast:.3f}  heterogeneity {prof.heterogeneity:.3f}")
    print(f"  retina {prof.total_retina_um_mean:.0f}+/-{prof.total_retina_um_std:.0f} um  "
          f"concave {prof.curvature_concave_frac:.2f}  noise {prof.noise_floor_frac:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
