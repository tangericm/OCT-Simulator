"""One-off: render README images into docs/assets/. Not part of the package or CI."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import SimConfig, simulate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def extent(res):
    return [res.x_axis_um[0], res.x_axis_um[-1], res.depth_axis_um[-1], res.depth_axis_um[0]]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def macula_bscan_and_labels():
    cfg = SimConfig(anatomy="macula", lateral_fov_um=6000.0, lines_per_frame=768,
                     motion_level=1.0, noise_level=1.0)
    res = simulate(cfg, seed=3)
    ext = extent(res)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.imshow(res.bscan_db, cmap="gray", aspect="auto", extent=ext, vmin=-55, vmax=0)
    ax.set_xlabel("lateral (µm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    save(fig, "macula-bscan.jpg")

    n_labels = int(res.labels2d.max())
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    ax.imshow(res.labels2d, cmap="nipy_spectral", aspect="auto", extent=ext,
              vmin=0, vmax=max(n_labels, 1))
    ax.set_xlabel("lateral (µm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    save(fig, "macula-labels.jpg")


def gallery():
    specs = [
        ("mirror", dict(anatomy="mirror")),
        ("paper", dict(anatomy="paper")),
        ("cornea", dict(anatomy="cornea")),
        ("macula", dict(anatomy="macula", lateral_fov_um=6000.0)),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.4), dpi=150)
    for ax, (name, kw) in zip(axes, specs):
        cfg = SimConfig(lines_per_frame=512, **kw)
        res = simulate(cfg, seed=1)
        ax.imshow(res.bscan_db, cmap="gray", aspect="auto", extent=extent(res), vmin=-55, vmax=0)
        ax.set_title(name)
        ax.set_xlabel("lateral (µm)")
    axes[0].set_ylabel("optical depth (µm)")
    fig.tight_layout()
    save(fig, "gallery.jpg")


if __name__ == "__main__":
    macula_bscan_and_labels()
    gallery()
