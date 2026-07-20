"""Pooled optics fit across MULTIPLE M3 geometries — one θ that matches macula + disc + wide at
once, so the fitted optics generalise instead of overfitting a single scan.

    python scripts/fit_pooled.py --maxiter 6 --popsize 6 --seeds 2 --lines 256 --workers -1

Prints per-geometry D before/after and saves octsim/data/retina_optics_maestro3_pooled.json.
The pool geometries are the ones whose ``texture_<name>.json`` targets exist.
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

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "octsim", "data")

# (target name, shape_model, anatomy, lines, fov) — pooled geometries spanning the device's scans.
POOL = [
    ("maestro3_macula_3x3", "octa500", "macula", 320, 3000.0),
    ("maestro3_macula_vol", "octa500", "macula", 256, 6000.0),
    ("maestro3_macula_9x9", "octa500", "macula", 512, 9000.0),
    ("maestro3_disc_6x6", "parametric", "onh", 512, 6000.0),
    ("maestro3_wide_12x9", "octa500", "macula", 512, 12000.0),
]


def _target(name):
    return unlabeled.TextureProfile.load(os.path.join(DATA, f"texture_{name}.json"))


def _spec(shape, anat, lines, fov, seeds, avg):
    return C.RenderSpec(instrument="maestro3", shape_model=shape, anatomy=anat,
                        center_wavelength_nm=840.0, pixels_per_spectrum=2048,
                        lines_per_frame=lines, lateral_fov_um=fov, averages=avg, seeds=seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maxiter", type=int, default=6)
    ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--averages", type=int, default=1)
    ap.add_argument("--wreg", type=float, default=0.1)
    ap.add_argument("--wspeckle", type=float, default=0.1,
                    help="speckle weight during the pooled search (down-weighted: grain is pitch-"
                         "sensitive and the pool renders at a common modest line count)")
    ap.add_argument("--fit-lines", type=int, default=192,
                    help="render every pool geometry at this line count for a tractable search")
    ap.add_argument("--de-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--exclude", default="", help="comma-separated substrings; pool geometries whose "
                    "name contains any are dropped (e.g. 'disc' — the ONH template is a structural "
                    "mismatch the macula-model render can't fit, so it only pulls the pooled mean)")
    ap.add_argument("--out", default="retina_optics_maestro3_pooled.json",
                    help="output profile filename under octsim/data/")
    args = ap.parse_args()

    seeds = tuple(101 + 111 * i for i in range(args.seeds))
    weights = {**C.DEFAULT_WEIGHTS, "reg": args.wreg, "speckle": args.wspeckle}
    excl = [e.strip() for e in args.exclude.split(",") if e.strip()]
    keep = lambda n: os.path.exists(os.path.join(DATA, f"texture_{n}.json")) and not any(e in n for e in excl)
    pool = [(_target(n), _spec(s, a, args.fit_lines, fv, seeds, args.averages))
            for (n, s, a, ln, fv) in POOL if keep(n)]
    names = [n for (n, *_ ) in POOL if keep(n)]
    print(f"pool ({len(pool)}): {names}")

    # baseline: the current definitive macula θ (or literature) per-geometry D
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(DATA)), "runs")
    runs = sorted(glob.glob(os.path.join(runs_dir, "2026*_maestro3_macula_vol_*")),
                  key=os.path.getmtime) if os.path.isdir(runs_dir) else []
    th0 = C.theta_literature()
    for r in reversed(runs):
        tj = os.path.join(r, "theta.json")
        if os.path.exists(tj):
            th0 = np.array(json.load(open(tj))["theta"], float); break
    print("baseline (macula θ) per-geometry D:")
    for (t, s), n in zip(pool, names):
        d = C.evaluate(th0, t, s, weights).get("D_total", 1e6)
        print(f"    {n:22s} D={d:.3f}")

    print(f"[pooled fit] maxiter={args.maxiter} popsize={args.popsize} seeds={args.seeds} "
          f"free={sum(1 for p in C.theta_space() if not p.frozen)} de-seed={args.de_seed}")
    theta_hat, meanD, incumbent, res = C.run_de_pool(
        pool, weights=weights, maxiter=args.maxiter, popsize=args.popsize,
        seed=args.de_seed, workers=args.workers, base_theta=th0)
    print(f"[pooled fit] mean D = {meanD:.4f}")
    print("pooled θ per-geometry D:")
    for (t, s), n in zip(pool, names):
        d = C.evaluate(theta_hat, t, s, weights).get("D_total", 1e6)
        print(f"    {n:22s} D={d:.3f}")

    profile, _globs = C.decode(theta_hat)
    out = os.path.join(DATA, args.out)
    profile.save(out)
    # also persist the FULL θ vector (the saved profile drops the render globals reference_amp/noise/
    # floor, so it can't be re-encoded) → eval_generalization can score the pooled θ at native res.
    theta_out = out.replace(".json", "_theta.json")
    json.dump({"theta": theta_hat.tolist(), "names": [p.name for p in C.theta_space()]},
              open(theta_out, "w"), indent=1)
    print(f"wrote {out}\nwrote {theta_out}")
    bands = {b: float(10 ** theta_hat[C._index(b)]) for b in ("bs_NFL", "bs_inner", "bs_outer")}
    print("pooled retinal bands:", {k: round(v, 3) for k, v in bands.items()})


if __name__ == "__main__":
    main()
