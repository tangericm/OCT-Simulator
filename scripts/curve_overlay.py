"""Overlay the detected RPE band (real vs simulated) per field-of-view — the curvature-shape check.

Renders the current fitted optics (last accepted loop run) at each panel geometry and plots the
detected RPE band against the reconstructed real band, per FOV, so the posterior-pole curvature
(near-fovea steepness + wide-FOV flattening) can be compared by eye. Writes runs/curve_overlay.png.

    python scripts/curve_overlay.py --data-root "<...>/OCT Data"
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iterate_loop as L
from octsim import calibrate as C
from octsim.data import real_maestro

# (label, folder rel to data-root, fov_um, alines)
CASES = [("3x3", "Maestro3/M3_Macula_3x3mm", 3000, 256),
         ("6mm", "Maestro3/6mm_1024Aline", 6000, 1024),
         ("9x9", "Maestro3/M3_Macula_9x9mm_512x512x2", 9000, 512),
         ("wide_12x9", "Maestro3/M3_Wide_12x9mm", 12000, 512)]


def rpe_trace(db, dp, fov):
    """Smoothed RPE-band depth (µm, relative to the shallowest column) vs lateral position (mm).
    Masks the top DC/zero-delay band and median+gaussian smooths to reject argmax spikes."""
    from scipy.ndimage import median_filter, gaussian_filter1d
    lin = (10.0 ** (db / 20.0)).copy()
    H, W = lin.shape
    lin[:60, :] = 0.0
    rows = np.array([int(np.argmax(lin[:, c])) for c in range(W)], float)
    rows = median_filter(rows, size=max(5, W // 40))
    rows = gaussian_filter1d(rows, W / 60.0)
    x = (np.arange(W) - W / 2) * (fov / W) / 1000.0
    depth = rows * dp
    return x, depth - depth.min()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable"); return

    th, rid = L.find_incumbent_theta()
    print(f"[overlay] fitted θ from {rid}")
    seeds = (101, 212, 323)
    fig, axes = plt.subplots(1, len(CASES), figsize=(4.5 * len(CASES), 4))
    for ax, (nm, folder, fov, alines) in zip(np.atleast_1d(axes), CASES):
        path = os.path.join(args.data_root, folder)
        if not os.path.isdir(path):
            ax.text(0.5, 0.5, f"{nm}\nno data", ha="center", va="center"); ax.axis("off"); continue
        imgs, _, _ = real_maestro.load_corpus(path, limit=6)
        for im in imgs[:3]:
            x, d = rpe_trace(im, 3.87, fov); ax.plot(x, d, "0.6", alpha=0.4, lw=0.8)
        xr, dr = rpe_trace(imgs[len(imgs) // 2], 3.87, fov)
        ax.plot(xr, dr, "k-", lw=2.2, label="real")
        m = dict(zip(["name", "target", "folder", "anatomy", "shape", "alines", "fov", "axpitch", "in_pool"],
                     ["x", "x", folder, "macula", "octa500", alines, float(fov), 3.87, True]))
        spec = L.spec_for(m, seeds); prof, glob = C.decode(th)
        imgs2, res = C.render_crn(prof, glob, spec)
        xs, ds = rpe_trace(imgs2[0], res.source.depth_pixel_um, fov)
        ax.plot(xs, ds, "r-", lw=2.2, label="sim fitted")
        ax.invert_yaxis(); ax.set_title(f"{nm} ({fov/1000:.0f} mm)")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("RPE depth rel. edge (µm)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("RPE band curvature: real (black) vs sim fitted (red) — per FOV", y=1.02)
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "runs", "curve_overlay.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
