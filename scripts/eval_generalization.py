"""Evaluate a fitted θ (or bundled optics profile) at a DIFFERENT geometry vs its real target —
the cross-geometry generalization check ("should also produce similar results to the other images").

    python scripts/eval_generalization.py --theta-json runs/<def>/theta.json \
        --target maestro3_disc_6x6 --shape-model onh --anatomy onh --lines 512 --fov 6000

Renders the SAME tissue optics θ at the target's geometry (only the shape model / FOV change) and
reports D + terms vs that geometry's real texture. Prints one RESULT_JSON line. No tracking/writes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import calibrate as C
from octsim.data import unlabeled


def _load_target(name):
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "octsim", "data", f"texture_{name}.json")
    if os.path.exists(p):
        return unlabeled.TextureProfile.load(p)
    return unlabeled.bundled_texture(name)


def _load_theta(theta_json):
    if theta_json and os.path.exists(theta_json):
        return np.array(json.load(open(theta_json))["theta"], float)
    # default: the most recent definitive macula fit
    runs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "runs", "2026*_maestro3_macula_vol_*")), key=os.path.getmtime)
    if runs:
        return np.array(json.load(open(os.path.join(runs[-1], "theta.json")))["theta"], float)
    return C.theta_literature()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta-json", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--shape-model", default="octa500")
    ap.add_argument("--anatomy", default="macula")
    ap.add_argument("--instrument", default="maestro3")
    ap.add_argument("--pixels", type=int, default=2048)
    ap.add_argument("--lines", type=int, default=256)
    ap.add_argument("--fov", type=float, default=6000.0)
    ap.add_argument("--averages", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--wreg", type=float, default=0.1)
    args = ap.parse_args()

    theta = _load_theta(args.theta_json)
    target = _load_target(args.target)
    weights = {**C.DEFAULT_WEIGHTS, "reg": args.wreg}
    seeds = tuple(101 + 111 * i for i in range(args.seeds))
    spec = C.RenderSpec(instrument=args.instrument, shape_model=args.shape_model,
                        anatomy=args.anatomy, center_wavelength_nm=840.0,
                        pixels_per_spectrum=args.pixels, lines_per_frame=args.lines,
                        lateral_fov_um=args.fov, averages=args.averages, seeds=seeds)
    out = C.evaluate(theta, target, spec, weights)
    synth = out.get("synth_profile")
    tgt_sc = float(getattr(target, "speckle_contrast", 0.0) or 0.0)
    sc = None if synth is None else float(synth.speckle_contrast)
    speckle_ok = bool(sc is not None and tgt_sc > 0 and abs(sc - tgt_sc) / tgt_sc <= 0.25
                      and 0.25 <= sc <= 0.60)
    result = {
        "target": args.target, "shape_model": args.shape_model,
        "D": float(out.get("D_total", 1e6)),
        "D_template": float(out.get("D_template", 0)), "D_hist": float(out.get("D_hist", 0)),
        "D_speckle": float(out.get("D_speckle", 0)),
        "speckle": sc, "target_speckle": tgt_sc, "speckle_ok": speckle_ok,
        "thickness_um": None if synth is None else float(synth.total_retina_um_mean),
        "target_thickness_um": float(target.total_retina_um_mean),
    }
    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
