"""Render the example figures: a macular B-scan, its free label map, and a preset gallery.

    python examples/render_figures.py                  # -> docs/assets/
    python examples/render_figures.py --out /tmp/figs  # elsewhere
    python examples/render_figures.py --save-arrays    # also dump raw spectra + labels

These are the images used in the README. ``--save-arrays`` writes the interferometric
spectra and the integer label map alongside each figure, which is the quickest way to see
what a training sample actually contains.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import SimConfig, simulate  # noqa: E402

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")

#: the four scenes in the gallery panel — two bench targets, then tissue
GALLERY = [
    ("mirror", dict(anatomy="mirror")),
    ("paper", dict(anatomy="paper")),
    ("cornea", dict(anatomy="cornea")),
    ("macula", dict(anatomy="macula", lateral_fov_um=6000.0)),
]


def _extent(res):
    return [res.x_axis_um[0], res.x_axis_um[-1], res.depth_axis_um[-1], res.depth_axis_um[0]]


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _save_arrays(res, out, name):
    if res.spectra is not None:
        np.save(os.path.join(out, f"{name}_spectra.npy"), res.spectra)
    np.save(os.path.join(out, f"{name}_labels.npy"), res.labels2d)
    print(f"wrote {name}_spectra.npy / {name}_labels.npy")


def macula(out: str, save_arrays: bool = False) -> None:
    """A 6 mm macular B-scan and the per-pixel layer labels that ship with it."""
    cfg = SimConfig(anatomy="macula", lateral_fov_um=6000.0, lines_per_frame=768,
                    motion_level=1.0, noise_level=1.0)
    res = simulate(cfg, seed=3)
    ext = _extent(res)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.imshow(res.bscan_db, cmap="gray", aspect="auto", extent=ext, vmin=-55, vmax=0)
    ax.set_xlabel("lateral (µm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    _save(fig, out, "macula-bscan.jpg")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.imshow(res.labels2d, cmap="nipy_spectral", aspect="auto", extent=ext,
              vmin=0, vmax=max(int(res.labels2d.max()), 1))
    ax.set_xlabel("lateral (µm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    _save(fig, out, "macula-labels.jpg")

    if save_arrays:
        _save_arrays(res, out, "macula")


def gallery(out: str, save_arrays: bool = False) -> None:
    """One panel per preset, all on the same display window."""
    fig, axes = plt.subplots(1, len(GALLERY), figsize=(4.2 * len(GALLERY), 4.4), dpi=150)
    for ax, (name, kw) in zip(axes, GALLERY):
        res = simulate(SimConfig(lines_per_frame=512, **kw), seed=1)
        ax.imshow(res.bscan_db, cmap="gray", aspect="auto",
                  extent=_extent(res), vmin=-55, vmax=0)
        ax.set_title(name)
        ax.set_xlabel("lateral (µm)")
        if save_arrays:
            _save_arrays(res, out, name)
    axes[0].set_ylabel("optical depth (µm)")
    fig.tight_layout()
    _save(fig, out, "gallery.jpg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--save-arrays", action="store_true",
                    help="also write the raw spectra and label maps as .npy")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    macula(args.out, args.save_arrays)
    gallery(args.out, args.save_arrays)


if __name__ == "__main__":
    main()
