#!/usr/bin/env python
"""Fit a retinal LayerShapeModel from a segmented OCT dataset and save the .npz artifact.

The raw dataset is never redistributed; this reproduces the small (~80 KB) learned model
that the simulator ships with. Duke DME is built in; point ``--root`` at any dataset by
writing a loader that yields a ``[N, n_layer, n_ecc]`` thickness stack (see
``octsim/data/duke_dme.py``).

Duke DME (Chiu et al., BOE 2015):
    1. download 2015_BOE_Chiu2.zip from people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm
    2. unzip -> a 2015_BOE_Chiu/ folder of Subject_*.mat
    3. python scripts/fit_shape_model.py --root path/to/2015_BOE_Chiu \
           --out octsim/data/retina_shape_duke.npz
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="dataset folder (Duke: Subject_*.mat; JHU: *_spectralis_macula_*.mat)")
    ap.add_argument("--dataset", default="jhu_hc", choices=["jhu_hc", "duke_dme"])
    ap.add_argument("--out", default="octsim/data/retina_shape_jhu_hc.npz")
    ap.add_argument("--groups", default="hc", help="JHU cohorts, comma-sep (hc,ms)")
    ap.add_argument("--grader", type=int, default=1, help="Duke grader (1 or 2)")
    ap.add_argument("--n-modes", type=int, default=8)
    ap.add_argument("--max-total-um", type=float, default=360.0,
                    help="drop scans thicker than this (edema filter)")
    args = ap.parse_args()

    if args.dataset == "jhu_hc":
        from octsim.data import jhu_hc
        sm = jhu_hc.fit_shape_model(args.root, groups=tuple(args.groups.split(",")),
                                    n_modes=args.n_modes, max_total_um=args.max_total_um)
    else:
        from octsim.data import duke_dme as dd
        sm = dd.fit_shape_model(args.root, grader=args.grader, n_modes=args.n_modes,
                                max_total_um=args.max_total_um)
    sm.save(args.out)
    ev = sm.explained_variance_ratio
    print(f"fit {sm.n_scans} scans -> {args.out} ({os.path.getsize(args.out)} bytes)")
    print(f"  source: {sm.source}")
    print(f"  undulation RMS: {sm.undulation_rms_um:.2f} um")
    print(f"  explained variance (top 5): {[round(float(v), 3) for v in ev[:5]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
