"""Measure per-layer attenuation coefficients from real linear-domain OCT.

    python scripts/fit_attenuation.py --folder ".../Maestro3/6mm_1024Aline" --frames 8

Runs the Vermeer depth-resolved estimator (:mod:`octsim.attenuation`) on reconstructed
Maestro A-lines and reports mu per retinal layer in 1/mm, alongside the value currently in
``RETINA_LAYERS`` so the two can be compared directly.

Why this exists: the shipped layer table was built by matching simulated brightness to a
Maestro3 B-scan, so every number in it has absorbed that device's gain, tone curve and
reference power. mu measured this way is a property of the tissue, in physical units, and
is the quantity that should transfer to another wavelength or another device.

Layer *positions* are taken from the simulator's own (data-driven) boundaries and only the
mu *values* come from the real frames, so the real data does not need to be segmented —
which matters because no segmentation ships with the Maestro corpus, and the public
corpora that do have segmentation are 8-bit display-domain and therefore unusable here.
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

from octsim import attenuation, realism_metrics as rmet
from octsim.anatomy.retina import RETINA_LAYERS
from octsim.config import SimConfig
from octsim.data import real_maestro
from octsim.generate import simulate


def layer_bounds_from_sim(alines: int, fov_um: float, dz_um: float) -> dict:
    """``{layer: (top_um, bottom_um)}`` optical µm relative to the RPE anchor.

    Uses the simulator's label map, so the depth mapping is the same one the renderer
    uses and the measured mu lands on the layer it will be applied to.
    """
    res = simulate(SimConfig(instrument="maestro3", optics_profile="literature",
                             lateral_fov_um=fov_um, lines_per_frame=alines,
                             falloff_6db_um=1959.9, sample_depth_um=430.0, seed=7))
    db = res.volume_db[0]
    lbl = res.label_map[0]
    names = res.params["label_names"]
    rows = rmet.find_rpe(db, dz_um)
    n_x = lbl.shape[1]
    out = {}
    for i, nm in enumerate(names):
        tops, bots = [], []
        for c in range(0, n_x, 4):
            rr = np.nonzero(lbl[:, c] == (i + 1))[0]
            if rr.size:
                # PER COLUMN, relative to that column's own anchor. Taking the extent over
                # the whole frame instead spans the full 500+ µm of curvature, so every
                # layer's band overlaps every other and the measured mu becomes a smooth
                # function of depth rather than a property of the layer.
                tops.append((rr.min() - rows[c]) * dz_um)
                bots.append((rr.max() + 1 - rows[c]) * dz_um)
        if tops:
            out[nm] = (float(np.median(tops)), float(np.median(bots)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--fov-um", type=float, default=6000.0)
    p.add_argument("--axial-pitch-um", type=float, default=3.87)
    p.add_argument("--out", default="")
    a = p.parse_args()

    dz = a.axial_pitch_um
    lut = real_maestro.read_clb(real_maestro.find_clb(a.folder))
    kop = real_maestro.klinear_operator(lut)
    files = sorted(glob.glob(os.path.join(a.folder, "bscan*.raw")))
    files = [f for f in files if not real_maestro.is_placeholder(f)][::a.stride][:a.frames]
    if not files:
        raise SystemExit(f"no hydrated bscan*.raw in {a.folder}")
    alines = real_maestro.alines_of(files[0])
    print(f"{len(files)} frames, {alines} A-lines, dz={dz} um")

    lin, anchors = [], []
    for f in files:
        I = real_maestro.recon_frame_linear(f, alines, kop)
        lin.append(I)
        # anchor from the dB view, which is what the landmark detector is tuned on
        db = np.clip(10.0 * np.log10(np.maximum(I / I.max(), 1e-12)), -60.0, 0.0)
        anchors.append(rmet.find_rpe(db, dz))

    # noise floor from the shallow vitreous of the first frame
    floor = float(np.median(lin[0][60:140, :]))
    print(f"noise floor (linear intensity) = {floor:.4g}")

    ref_row = int(np.median(anchors[0]))
    prof = attenuation.mu_profile(lin, dz, anchor_rows=anchors, ref_row=ref_row,
                                  noise_floor=floor)

    # the mu(depth) profile is the primary measurement; the per-layer table below is that
    # profile sampled over each layer's depth range
    print("\nmu(depth) profile, 1/mm, relative to the RPE anchor:")
    for off in range(-420, 400, 60):
        r = int(round(ref_row + off / dz))
        if 0 <= r < prof.size:
            print(f"   {off:+5d} um : {prof[r] * 1e3:7.2f}")

    bounds = layer_bounds_from_sim(alines, a.fov_um, dz)
    measured = attenuation.per_layer_mu(prof, dz, ref_row, bounds)

    current = {ly.name: ly.mu_s_per_mm for ly in RETINA_LAYERS}
    print(f"\n{'layer':10s}{'measured 1/mm':>15s}{'current 1/mm':>15s}{'ratio':>9s}")
    print("-" * 49)
    rows_out = {}
    for nm in [ly.name for ly in RETINA_LAYERS]:
        if nm not in measured:
            continue
        mm = measured[nm] * 1e3          # 1/µm -> 1/mm
        cur = current.get(nm, float("nan"))
        rows_out[nm] = mm
        print(f"{nm:10s}{mm:15.2f}{cur:15.2f}{mm / cur if cur else float('nan'):9.2f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"source": os.path.basename(a.folder), "n_frames": len(files),
                       "axial_pitch_um": dz, "method": "Vermeer 2014 depth-resolved",
                       "mu_per_mm": rows_out}, fh, indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
