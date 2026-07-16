"""Entry point: ``--gui`` (default), ``--selftest``, or ``--generate N``.

``--selftest`` and ``--generate`` are headless and do **not** import PySide6, so they
work in a display-less environment (CI, packaging smoke tests). Only the GUI path
imports Qt.
"""

from __future__ import annotations

import argparse
import sys


def selftest() -> int:
    """Run a tiny headless generation (B-scan + volume + batch); return 0/1."""
    from ..config import SimConfig
    from ..generate import generate_dataset, simulate
    try:
        cfg = SimConfig(anatomy="macula", pixels_per_spectrum=512, lines_per_frame=48,
                        lateral_fov_um=1000.0)
        s = simulate(cfg, seed=0)
        assert s.bscan.shape[0] == 256 and s.labels2d.max() >= 1
        volcfg = SimConfig(anatomy="macula", scan_pattern="volume", n_frames=3,
                           pixels_per_spectrum=512, lines_per_frame=48, slow_fov_um=300.0)
        v = simulate(volcfg, seed=1)
        assert v.is_volume and v.volume.shape[0] == 3
        ds = list(generate_dataset(cfg, 2, seed=2))
        assert len(ds) == 2
        print("octsim selftest OK")
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"octsim selftest FAILED: {exc}", file=sys.stderr)
        return 1


def run_generate(args) -> int:
    from ..config import SimConfig
    from ..generate import generate_dataset
    cfg = SimConfig(anatomy=args.anatomy, scan_pattern=args.pattern, n_frames=args.frames,
                    seed=args.seed)
    for i, s in enumerate(generate_dataset(cfg, args.generate, seed=args.seed), start=1):
        kind = "volume" if s.is_volume else "bscan"
        print(f"sample {i}: {kind} {tuple(s.volume.shape)}  {s.params}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="octsim", description="OCT simulation platform")
    p.add_argument("--gui", action="store_true", help="launch the GUI (default)")
    p.add_argument("--selftest", action="store_true", help="headless self-test, exit 0/1")
    p.add_argument("--generate", type=int, metavar="N", help="headless: generate N samples")
    p.add_argument("--anatomy", default="macula")
    p.add_argument("--pattern", default="bscan", choices=["bscan", "volume"])
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    if args.generate:
        return run_generate(args)
    from .launcher import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
