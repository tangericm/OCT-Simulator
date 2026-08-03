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

#: Render against the MEASURED device model rather than the closed-form Gaussian default.
#: `instrument="device_a"` supplies the measured source spectrum P(k), sensitivity
#: fall-off, SNR, and scan distortion — the difference between a generic-looking frame and
#: one that reads like a real B-scan. Motion is left at its default 0.0: a non-zero
#: motion_level is physically correct but its washout streaks dominate a still figure.
INSTRUMENT = "device_a"

#: display window (dB). -50 matches the contrast a device shows; -55 stretches the noise
#: floor bright enough to grey out the vitreous.
VMIN_DB = -50.0

#: the four scenes in the gallery panel — two bench targets, then tissue
GALLERY = [
    ("mirror", dict(anatomy="mirror")),
    ("paper", dict(anatomy="paper")),
    ("cornea", dict(anatomy="cornea")),
    ("macula", dict(anatomy="macula", lateral_fov_um=6000.0)),
]


def _extent(res):
    return [res.x_axis_um[0], res.x_axis_um[-1], res.depth_axis_um[-1], res.depth_axis_um[0]]


def _tissue_rows(res, pad_um: float = 260.0):
    """Depth rows spanned by labelled tissue, padded. Keeps the frame off the empty window.

    The axial range runs well past the retina, so an uncropped frame renders the tissue as
    a thin strip in a field of noise floor. The label map already says where the tissue is.
    """
    rows = np.nonzero(res.labels2d.any(axis=1))[0]
    dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
    pad = int(round(pad_um / max(dz, 1e-9)))
    n = res.labels2d.shape[0]
    if rows.size == 0:
        return 0, n
    return max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad, n)


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
    cfg = SimConfig(anatomy="macula", instrument=INSTRUMENT, shape_model="octa500",
                    lateral_fov_um=6000.0, lines_per_frame=1024, sample_depth_um=640.0)
    res = simulate(cfg, seed=11)
    r0, r1 = _tissue_rows(res)
    dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
    ext = [res.x_axis_um[0] / 1000.0, res.x_axis_um[-1] / 1000.0, r1 * dz, r0 * dz]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=170)
    ax.imshow(res.bscan_db[r0:r1], cmap="gray", aspect="auto", extent=ext,
              vmin=VMIN_DB, vmax=0)
    ax.set_xlabel("lateral (mm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    _save(fig, out, "macula-bscan.jpg")

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=170)
    ax.imshow(res.labels2d[r0:r1], cmap="nipy_spectral", aspect="auto", extent=ext,
              vmin=0, vmax=max(int(res.labels2d.max()), 1), interpolation="nearest")
    ax.set_xlabel("lateral (mm)")
    ax.set_ylabel("optical depth (µm)")
    fig.tight_layout()
    _save(fig, out, "macula-labels.jpg")

    if save_arrays:
        _save_arrays(res, out, "macula")


def gallery(out: str, save_arrays: bool = False) -> None:
    """One panel per preset, all on the same display window."""
    fig, axes = plt.subplots(1, len(GALLERY), figsize=(4.2 * len(GALLERY), 4.4), dpi=150)
    for ax, (name, kw) in zip(axes, GALLERY):
        res = simulate(SimConfig(lines_per_frame=512, instrument=INSTRUMENT, **kw), seed=1)
        r0, r1 = _tissue_rows(res, pad_um=360.0)
        dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
        ax.imshow(res.bscan_db[r0:r1], cmap="gray", aspect="auto",
                  extent=[res.x_axis_um[0] / 1000.0, res.x_axis_um[-1] / 1000.0,
                          r1 * dz, r0 * dz],
                  vmin=VMIN_DB, vmax=0)
        ax.set_title(name)
        ax.set_xlabel("lateral (mm)")
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
