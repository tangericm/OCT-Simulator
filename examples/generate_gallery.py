"""Render every preset to a PNG gallery + save the raw spectra and label maps.

Usage (from the ``simulator/`` directory)::

    python examples/generate_gallery.py                 # all presets
    python examples/generate_gallery.py macula cornea   # a subset
    python examples/generate_gallery.py --no-noise --seed 7

Outputs land in ``simulator/examples/output/`` (git-ignored). Each preset gets a
``<name>_bscan.png`` (dB image), and the whole run gets a ``gallery.png`` panel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import LightSource, NoiseModel, presets, simulate_bscan  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", default=list(presets.PRESETS),
                    help="preset names (default: all)")
    ap.add_argument("--n-ascans", type=int, default=384)
    ap.add_argument("--lateral-um", type=float, default=1000.0)
    ap.add_argument("--n-k", type=int, default=1536)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    os.makedirs(OUT, exist_ok=True)
    src = LightSource(n_k=args.n_k)
    print(src.describe())
    noise = None if args.no_noise else NoiseModel.typical()

    names = [n for n in args.names if n in presets.PRESETS]
    results = {}
    for name in names:
        t0 = time.perf_counter()
        res = simulate_bscan(
            presets.PRESETS[name](), src,
            n_ascans=args.n_ascans, lateral_extent_um=args.lateral_um,
            noise=noise, seed=args.seed,
        )
        results[name] = res
        dt = time.perf_counter() - t0
        print(f"  {name:8s}  {res.bscan_db.shape}  {dt:5.1f}s")
        np.save(os.path.join(OUT, f"{name}_spectra.npy"), res.spectra.astype(np.float32))
        np.save(os.path.join(OUT, f"{name}_labels.npy"), res.label_map)
        if plt is not None:
            _save_single(plt, name, res)

    if plt is not None and results:
        _save_panel(plt, results)
    print(f"\nwrote outputs to {OUT}")


def _extent(res):
    return [res.x_axis_um[0], res.x_axis_um[-1],
            res.depth_axis_um[-1], res.depth_axis_um[0]]


def _save_single(plt, name, res):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    ax.imshow(res.bscan_db, cmap="gray", aspect="auto",
              extent=_extent(res), vmin=-55, vmax=0)
    ax.set_title(f"{name} — simulated OCT B-scan")
    ax.set_xlabel("lateral (µm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}_bscan.png"))
    plt.close(fig)


def _save_panel(plt, results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), dpi=130)
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        ax.imshow(res.bscan_db, cmap="gray", aspect="auto",
                  extent=_extent(res), vmin=-55, vmax=0)
        ax.set_title(name)
        ax.set_xlabel("lateral (µm)")
    axes[0].set_ylabel("optical depth (µm)")
    fig.suptitle("octsim — physically-modelled OCT B-scans", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gallery.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
