"""Leaderboard over ``runs/manifest.jsonl`` — the 'did it actually improve' view.

Sorts gate-passing runs by ``D_total`` (ascending) and prints the panel that justifies
flipping the ``SimConfig.optics_profile`` default. Non-gate-passing rows are listed separately
so failures stay visible (they are signal about the objective's blind spots).

    python scripts/leaderboard.py [--all] [--target maestro3_macula]
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from octsim.experiment import read_manifest, MANIFEST


def _fmt(v, w=8, p=4):
    if v is None:
        return " " * (w - 2) + "NA"
    try:
        return f"{float(v):{w}.{p}f}"
    except Exception:
        return str(v)[:w].rjust(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--target", default=None, help="filter to one target corpus")
    ap.add_argument("--all", action="store_true", help="include gate-failing rows in the ranking")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.target:
        rows = [r for r in rows if r.get("target_corpus") == args.target]
    if not rows:
        print(f"no runs in {args.manifest}")
        return

    passing = [r for r in rows if r.get("gates_pass")] if not args.all else rows
    passing = sorted(passing, key=lambda r: (r.get("D_total") is None, r.get("D_total", 1e18)))

    hdr = f"{'rank':>4}  {'D_total':>9} {'D_tpl':>8} {'D_hist':>8} {'D_spk':>8} " \
          f"{'speckle':>8} {'snr':>6} {'gates':>6} {'acc':>4}  run_id"
    print(hdr); print("-" * len(hdr))
    for i, r in enumerate(passing, 1):
        print(f"{i:>4}  {_fmt(r.get('D_total'),9)} {_fmt(r.get('D_template'))} "
              f"{_fmt(r.get('D_hist'))} {_fmt(r.get('D_speckle'))} "
              f"{_fmt(r.get('speckle_contrast'),8,3)} {_fmt(r.get('snr_db'),6,1)} "
              f"{str(bool(r.get('gates_pass'))):>6} {str(bool(r.get('accepted'))):>4}  "
              f"{r.get('run_id','?')}")

    failing = [r for r in rows if not r.get("gates_pass")]
    if failing and not args.all:
        print(f"\n{len(failing)} gate-failing run(s) (excluded from ranking):")
        for r in failing:
            print(f"   D={_fmt(r.get('D_total'),9)} gates={r.get('gates')} {r.get('run_id','?')} "
                  f"— {r.get('note','')}")

    best = passing[0] if passing else None
    if best:
        print(f"\nbest gate-passing: {best.get('run_id')}  D_total={best.get('D_total')}")
        print("flip SimConfig.optics_profile default only when this is stable across several runs.")


if __name__ == "__main__":
    main()
