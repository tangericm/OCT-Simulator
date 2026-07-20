"""Run ONE optics fit and emit its θ as JSON — a lightweight, tracking-free fit for stability
sweeps (no manifest/run-folder writes, so many can run in parallel without racing).

    python scripts/fit_theta.py --de-seed 5 --maxiter 10 --popsize 7 --wreg 0.1

Prints a single JSON line: {de_seed, D, bands:{bs_NFL,...}, speckle, thickness, snr}.
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

from octsim import calibrate as C
from octsim.data import unlabeled


def _load_target(name):
    try:
        return unlabeled.bundled_texture(name)
    except Exception:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "octsim", "data", f"texture_{name}.json")
        return unlabeled.TextureProfile.load(p if os.path.exists(p) else name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="maestro3_macula_vol")
    ap.add_argument("--instrument", default="maestro3")
    ap.add_argument("--shape-model", default="octa500")
    ap.add_argument("--pixels", type=int, default=2048)
    ap.add_argument("--lines", type=int, default=256)
    ap.add_argument("--fov", type=float, default=6000.0)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--averages", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=10)
    ap.add_argument("--popsize", type=int, default=7)
    ap.add_argument("--wreg", type=float, default=0.1)
    ap.add_argument("--de-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    target = _load_target(args.target)
    weights = {**C.DEFAULT_WEIGHTS, "reg": args.wreg}
    seeds = tuple(101 + 111 * i for i in range(args.seeds))
    spec = C.RenderSpec(instrument=args.instrument, shape_model=args.shape_model,
                        center_wavelength_nm=840.0, pixels_per_spectrum=args.pixels,
                        lines_per_frame=args.lines, lateral_fov_um=args.fov,
                        averages=args.averages, seeds=seeds)
    theta_hat, D_hat, incumbent, res = C.run_de(
        target, spec, weights=weights, maxiter=args.maxiter, popsize=args.popsize,
        seed=args.de_seed, workers=args.workers)
    out = C.evaluate(theta_hat, target, spec, weights)
    synth = out.get("synth_profile")
    names = [p.name for p in C.theta_space()]

    def val(n):
        i = names.index(n); p = C.theta_space()[i]
        return float(10 ** theta_hat[i] if p.log else theta_hat[i])

    result = {
        "de_seed": args.de_seed, "D": float(D_hat),
        "bands": {b: val(b) for b in ("bs_NFL", "bs_inner", "bs_ONL", "bs_outer", "bs_choroid")},
        "speckle": None if synth is None else float(synth.speckle_contrast),
        "thickness_um": None if synth is None else float(synth.total_retina_um_mean),
    }
    print("RESULT_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
