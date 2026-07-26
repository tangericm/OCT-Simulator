"""Measure the sim-vs-real distance vector and gate on it.

Reconstructs a real Maestro corpus and renders matched simulated frames through the
*same* metric code (:mod:`octsim.realism_metrics`), then prints a signed delta table.

    # baseline against the dense 6 mm line scan
    python scripts/validate_realism.py --folder ".../Maestro3/6mm_1024Aline" --fov-um 6000

    # after changing optics, compare two profiles head to head
    python scripts/validate_realism.py --folder ... --optics literature --json runs/before.json
    python scripts/validate_realism.py --folder ... --optics maestro3   --json runs/after.json

Real and simulated frames are rendered on the same depth pitch (the Maestro3 instrument
model gives 3.87 µm/px at 2048 spectral samples), so band offsets in optical µm land on
the same tissue in both.
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

from octsim import realism_metrics as rmet
from octsim.config import SimConfig
from octsim.data import display_domain, real_maestro
from octsim.generate import simulate

#: 6 dB roll-off measured from the Maestro3 mirror scans (octsim/data/instrument_maestro3.json)
MAESTRO3_FALLOFF_UM = 1959.9


def _default_data_root() -> str:
    """Where the held-out corpora live. Raw scans are never committed, so this is purely
    a local convenience: ``$OCT_DATA_ROOT`` wins, else a sibling ``OCT Data`` directory."""
    env = os.environ.get("OCT_DATA_ROOT")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(repo), "OCT Data")


def measure_real(folder: str, *, n_frames: int, stride: int, fov_um: float,
                 axial_pitch_um: float):
    imgs, alines, n_skipped = real_maestro.load_corpus(folder, limit=n_frames, stride=stride)
    if not imgs:
        raise RuntimeError(f"no frames reconstructed from {folder}")
    if n_skipped:
        print(f"  skipped {n_skipped} un-hydrated (OneDrive placeholder) frames")
    dx = fov_um / alines
    print(f"  real: {len(imgs)} frames, {alines} A-lines, dz={axial_pitch_um} um, dx={dx:.3f} um")
    return [rmet.measure(db, axial_pitch_um, dx) for db in imgs], alines, dx


#: Optical depth that puts the simulated ILM at roughly the same row as the real Maestro3
#: corpus (real ILM ~210-220 rows at 3.87 µm/px). Matching it keeps the vitreous sampling
#: window clear of the DC band and makes the roll-off act over the same depth range.
SAMPLE_DEPTH_UM = 430.0


def measure_sim(*, n_seeds: int, alines: int, fov_um: float, anatomy: str,
                optics: str, instrument: str, shape_model: str, seed0: int,
                sample_depth_um: float = SAMPLE_DEPTH_UM,
                distortion: float = 0.86, noise_level: float = 1.0):
    out, dz, dx = [], None, fov_um / alines
    for i in range(n_seeds):
        cfg = SimConfig(
            instrument=instrument,
            optics_profile=optics,
            shape_model=shape_model,
            anatomy=anatomy,
            location=anatomy,
            lateral_fov_um=fov_um,
            lines_per_frame=alines,
            falloff_6db_um=MAESTRO3_FALLOFF_UM,
            sample_depth_um=sample_depth_um,
            scan_distortion_correction=distortion,
            noise_level=noise_level,
            seed=seed0 + 7 * i,
            randomize=True,
        )
        res = simulate(cfg)
        db = res.volume_db[0]
        if dz is None:
            dz = float(res.depth_axis_um[1] - res.depth_axis_um[0])
            print(f"  sim:  {n_seeds} seeds, {alines} A-lines, dz={dz:.3f} um, dx={dx:.3f} um "
                  f"(optics={optics}, instrument={instrument or 'gaussian'})")
        out.append(rmet.measure(db, dz, dx))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True, help="real Maestro scan folder (bscan*.raw + .CLB)")
    p.add_argument("--fov-um", type=float, default=6000.0)
    p.add_argument("--axial-pitch-um", type=float, default=3.87)
    p.add_argument("--frames", type=int, default=8, help="real frames to measure")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--seeds", type=int, default=8, help="simulated seeds to measure")
    p.add_argument("--seed0", type=int, default=7)
    p.add_argument("--anatomy", default="macula", choices=["macula", "onh", "periphery"])
    p.add_argument("--optics", default="literature")
    p.add_argument("--instrument", default="maestro3")
    p.add_argument("--shape-model", default="normal")
    p.add_argument("--json", default="", help="write the raw metric vectors here")
    p.add_argument("--holdout", action="append", default=[],
                   help="held-out corpus name (repeatable): "
                        + ", ".join(sorted(display_domain.HOLDOUT_CORPORA)))
    p.add_argument("--data-root", default=_default_data_root(),
                   help="root containing the held-out corpora "
                        "(default: $OCT_DATA_ROOT, else ../OCT Data beside the repo)")
    a = p.parse_args()

    print(f"measuring real corpus: {a.folder}")
    real_runs, alines, dx = measure_real(a.folder, n_frames=a.frames, stride=a.stride,
                                         fov_um=a.fov_um, axial_pitch_um=a.axial_pitch_um)
    print("rendering matched simulated frames")
    sim_runs = measure_sim(n_seeds=a.seeds, alines=alines, fov_um=a.fov_um, anatomy=a.anatomy,
                           optics=a.optics, instrument=a.instrument,
                           shape_model=a.shape_model, seed0=a.seed0)

    real_agg = rmet.aggregate(real_runs)
    sim_agg = rmet.aggregate(sim_runs)
    rows = rmet.compare(real_agg, sim_agg)
    print()
    print(rmet.format_table(rows))

    gated = [r for r in rows if r[7] is not None]
    n_pass = sum(1 for r in gated if r[7])
    print(f"\ngates: {n_pass}/{len(gated)} passing   [TRAINING corpus]")

    # --- held-out corpora: same metric code, no refitting -----------------------------
    holdouts = {}
    for name in a.holdout:
        try:
            imgs, spec = display_domain.load_holdout(name, a.data_root,
                                                     limit=a.frames, stride=a.stride)
        except Exception as exc:
            print(f"\nholdout {name}: SKIPPED ({exc})")
            continue
        dz, dx, tier = spec["dz"], spec["dx"], spec["tier"]
        drop = set()
        if spec.get("dz_calibrated"):
            # the pitch was solved FROM the thickness, so thickness is no longer a
            # measurement for this corpus and must not be reported as one
            drop.add("thickness_tissue_um")
        h_real = [rmet.measure(im, dz, dx) for im in imgs]
        alines_h = imgs[0].shape[1]
        if not spec.get("fov_known", True):
            drop.add("sag_optical_um")      # sag is meaningless without the true lateral FOV
        h_sim = measure_sim(n_seeds=a.seeds, alines=alines_h, fov_um=spec["fov_um"],
                            anatomy=a.anatomy, optics=a.optics, instrument=a.instrument,
                            shape_model=a.shape_model, seed0=a.seed0,
                            distortion=spec.get("distortion", 0.86),
                            noise_level=(display_domain.noise_level_for_snr(spec["snr_db"])
                                         if "snr_db" in spec else 1.0))
        keep = lambda d: {k: v for k, v in rmet.restrict(d, tier).items()
                          if k.split(".")[0] not in drop}
        h_rows = rmet.compare(keep(rmet.aggregate(h_real)), keep(rmet.aggregate(h_sim)))
        print(f"\n===== HOLDOUT {name}  ({len(imgs)} frames, {alines_h} A-lines, "
              f"dz={dz} um, dx={dx:.2f} um, tier={tier}) =====")
        if "snr_db" in spec:
            print(f"  [device SNR {spec['snr_db']:.1f} dB vs the calibration corpus's "
                  f"{display_domain.REFERENCE_SNR_DB:.1f} dB -> noise_level="
                  f"{display_domain.noise_level_for_snr(spec['snr_db']):.2f}]")
        if spec.get("dz_calibrated"):
            print(f"  [no axial metadata: pitch calibrated to dz={dz:.3f} um from a "
                  f"{display_domain.NORMATIVE_RETINA_UM:.0f} um normative retina;")
            print( "   thickness is therefore pinned by construction and is not reported]")
        if tier != "full":
            print(f"  [{tier}-domain source: dB-valued and floor-referenced metrics are")
            print( "   not recoverable through an unknown tone curve and are omitted]")
        print(rmet.format_table(h_rows))
        hg = [r for r in h_rows if r[7] is not None]
        hp = sum(1 for r in hg if r[7])
        print(f"  gates: {hp}/{len(hg)} passing   [GENERALIZATION]")
        holdouts[name] = {"tier": tier, "alines": alines_h,
                          "rows": [{"metric": r[0], "real": r[1], "sim": r[3],
                                    "delta": r[5], "tol": r[6], "pass": r[7]}
                                   for r in h_rows]}

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({
                "folder": a.folder, "optics": a.optics, "instrument": a.instrument,
                "anatomy": a.anatomy, "alines": alines, "fov_um": a.fov_um,
                "real": real_agg, "sim": sim_agg,
                "rows": [{"metric": r[0], "real": r[1], "sim": r[3],
                          "delta": r[5], "tol": r[6], "pass": r[7]} for r in rows],
                "holdouts": holdouts,
            }, f, indent=1)
        print(f"wrote {a.json}")
    return 0 if n_pass == len(gated) else 1


if __name__ == "__main__":
    raise SystemExit(main())
