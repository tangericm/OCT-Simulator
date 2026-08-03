"""Measure the sim-vs-real distance vector and gate on it.

Reconstructs a real device corpus and renders matched simulated frames through the
*same* metric code (:mod:`octsim.realism_metrics`), then prints a signed delta table.

    # baseline against the dense 6 mm line scan
    python scripts/validate_realism.py --folder ".../Device A/6mm_1024Aline" --fov-um 6000

    # after changing optics, compare two profiles head to head
    python scripts/validate_realism.py --folder ... --optics literature --json runs/before.json
    python scripts/validate_realism.py --folder ... --optics device_a   --json runs/after.json

Real and simulated frames are rendered on the same depth pitch (the Device A instrument
model gives 3.87 µm/px at 2048 spectral samples), so band offsets in optical µm land on
the same tissue in both.

Exit status (the held-out corpora count — a cross-device regression must not exit 0)::

    0  every requested corpus was evaluated and every gate passed
    1  a gate failed, training or held-out
    2  all evaluated gates passed but a requested --holdout could not be loaded, so the
       generalization claim went unchecked

A gate listed in :data:`octsim.realism_metrics.EXPECTED_FAIL` is a KNOWN gap with its reason
recorded in the source. It prints as ``XFAIL``, is excluded from the passing count, and does
not set exit 1 — the alternative, widening its tolerance until it passes, would convert a
documented gap into a silent one. A waived gate that starts passing prints as ``XPASS``:
that is a prompt to delete the waiver, not a failure.
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
from octsim.data import display_domain, real_device
from octsim.generate import simulate


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
    imgs, alines, n_skipped = real_device.load_corpus(folder, limit=n_frames, stride=stride)
    if not imgs:
        raise RuntimeError(f"no frames reconstructed from {folder}")
    if n_skipped:
        print(f"  skipped {n_skipped} un-hydrated (OneDrive placeholder) frames")
    dx = fov_um / alines
    print(f"  real: {len(imgs)} frames, {alines} A-lines, dz={axial_pitch_um} um, dx={dx:.3f} um")
    return [rmet.measure(db, axial_pitch_um, dx) for db in imgs], alines, dx


#: Optical depth that puts the simulated ILM at roughly the same row as the real Device A
#: corpus (real ILM ~210-220 rows at 3.87 µm/px). Matching it keeps the vitreous sampling
#: window clear of the DC band and makes the roll-off act over the same depth range.
#: Bumped 430 → 500 (plan C), then 500 → 640 to close the residual
#: ~150 µm mean-A-line / ILM shallow offset vs Device A 6mm_1024Aline (ILM row 176→212
#: vs real 213; tissue peak ~1190→1335 µm vs real ~1310 µm).
SAMPLE_DEPTH_UM = 640.0


def measure_sim(*, n_seeds: int, alines: int, fov_um: float, anatomy: str,
                optics: str, instrument: str, shape_model: str, seed0: int,
                sample_depth_um: float = SAMPLE_DEPTH_UM,
                distortion: float | None = None, noise_level: float = 1.0):
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
            # falloff comes from Instrument JSON via effective_falloff_6db_um() —
            # do not hard-code Device A here (that broke Device B / other devices).
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


def _tally(rows, corpus: str):
    """``(n_pass, n_gated, hard_failures, xfails, xpasses)`` for one corpus's rows.

    A gate listed in :data:`octsim.realism_metrics.EXPECTED_FAIL` is a recorded known gap,
    not a regression: it is counted separately and does not fail the run. Widening its
    tolerance instead would turn a documented gap back into a silent one, which is the
    failure mode this whole command exists to remove.
    """
    gated = [r for r in rows if r[7] is not None]
    n_pass = sum(1 for r in gated if r[7])
    hard, xfail, xpass = [], [], []
    for r in gated:
        out = rmet.gate_outcome(r[7], corpus, r[0])
        if out == "FAIL":
            hard.append(r[0])
        elif out == "XFAIL":
            xfail.append(r[0])
        elif out == "XPASS":
            xpass.append(r[0])
    return n_pass, len(gated), hard, xfail, xpass


def _waiver_note(xfail, xpass) -> str:
    bits = []
    if xfail:
        bits.append(f"{len(xfail)} expected-failing ({', '.join(xfail)})")
    if xpass:
        bits.append(f"{len(xpass)} XPASS - remove the waiver ({', '.join(xpass)})")
    return ("   [" + "; ".join(bits) + "]") if bits else ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True, help="real device scan folder (bscan*.raw + .CLB)")
    p.add_argument("--fov-um", type=float, default=6000.0)
    p.add_argument("--axial-pitch-um", type=float, default=3.87)
    p.add_argument("--frames", type=int, default=8, help="real frames to measure")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--seeds", type=int, default=8, help="simulated seeds to measure")
    p.add_argument("--seed0", type=int, default=7)
    p.add_argument("--anatomy", default="macula", choices=["macula", "onh", "periphery"])
    p.add_argument("--optics", default="literature")
    p.add_argument("--instrument", default="device_a")
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
    print(rmet.format_table(rows, "training"))

    n_pass, n_gated, tr_hard, tr_xfail, tr_xpass = _tally(rows, "training")
    print(f"\ngates: {n_pass}/{n_gated} passing   [TRAINING corpus]"
          + _waiver_note(tr_xfail, tr_xpass))

    # --- held-out corpora: same metric code, no refitting -----------------------------
    holdouts = {}
    skipped = []
    for name in a.holdout:
        try:
            imgs, spec = display_domain.load_holdout(name, a.data_root,
                                                     limit=a.frames, stride=a.stride)
        except Exception as exc:
            print(f"\nholdout {name}: SKIPPED ({exc})")
            skipped.append(name)
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
        # A corpus that has an instrument model supplies its own scan-distortion
        # correction and SNR through it, so the harness must NOT also pass them here --
        # that is the single-source rule R0b exists for. Corpora with no instrument model
        # (the cross-vendor ones) still take a distortion from the table.
        h_inst = spec.get("instrument", a.instrument)
        h_sim = measure_sim(n_seeds=a.seeds, alines=alines_h, fov_um=spec["fov_um"],
                            anatomy=a.anatomy, optics=a.optics, instrument=h_inst,
                            shape_model=a.shape_model, seed0=a.seed0,
                            distortion=spec.get("distortion"))
        keep = lambda d: {k: v for k, v in rmet.restrict(d, tier).items()
                          if k.split(".")[0] not in drop}
        h_rows = rmet.compare(keep(rmet.aggregate(h_real)), keep(rmet.aggregate(h_sim)))
        print(f"\n===== HOLDOUT {name}  ({len(imgs)} frames, {alines_h} A-lines, "
              f"dz={dz} um, dx={dx:.2f} um, tier={tier}) =====")
        if h_inst:
            from octsim.instrument import load_instrument, REFERENCE_SNR_DB
            _i = load_instrument(h_inst)
            # report the distortion actually in force, not the instrument's: a corpus with
            # no instrument model of its own overrides it from HOLDOUT_CORPORA.
            _d = spec.get("distortion")
            _src = "corpus table" if _d is not None else f"instrument {h_inst}"
            if _d is None:
                _d = _i.scan_distortion_correction
            print(f"  [instrument={h_inst}: scan_distortion_correction={_d} ({_src}), "
                  f"SNR {_i.snr_db:.1f} dB vs the calibration corpus's "
                  f"{REFERENCE_SNR_DB:.1f} dB -> noise x{_i.noise_scale:.2f}]")
        if spec.get("dz_calibrated"):
            print(f"  [no axial metadata: pitch calibrated to dz={dz:.3f} um from a "
                  f"{display_domain.NORMATIVE_RETINA_UM:.0f} um normative retina;")
            print( "   thickness is therefore pinned by construction and is not reported]")
        if tier != "full":
            print(f"  [{tier}-domain source: dB-valued and floor-referenced metrics are")
            print( "   not recoverable through an unknown tone curve and are omitted]")
        print(rmet.format_table(h_rows, name))
        hp, hg_n, h_hard, h_xfail, h_xpass = _tally(h_rows, name)
        print(f"  gates: {hp}/{hg_n} passing   [GENERALIZATION]"
              + _waiver_note(h_xfail, h_xpass))
        holdouts[name] = {"tier": tier, "alines": alines_h,
                          "n_pass": hp, "n_gated": hg_n,
                          "n_hard_fail": len(h_hard), "xfail": h_xfail, "xpass": h_xpass,
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

    # Exit status has to cover the HELD-OUT corpora too. Reporting only the training score
    # means a cross-device regression — the exact failure this command exists to catch —
    # still exits 0, and any automation gating on it accepts the regression silently.
    #
    #   0  every requested corpus evaluated and every gate passed
    #   1  a gate failed (training or held-out)
    #   2  everything that ran passed, but a requested holdout could not be evaluated
    #      (missing/unhydrated data). Distinct from 1: nothing is known to be broken, but
    #      the generalization claim was NOT checked, so this must not read as success.
    # A waived (XFAIL) gate is not a failure: it is a known gap recorded in
    # realism_metrics.EXPECTED_FAIL with its reason. Only hard failures set exit 1.
    h_fail = {n: v for n, v in holdouts.items() if v["n_hard_fail"]}
    print(f"\nSUMMARY  training {n_pass}/{n_gated}"
          + "".join(f"   {n} {v['n_pass']}/{v['n_gated']}" for n, v in holdouts.items())
          + (f"   skipped: {', '.join(skipped)}" if skipped else ""))
    if tr_hard or h_fail:
        if h_fail:
            print(f"FAIL: held-out gates failed on {', '.join(sorted(h_fail))}")
        return 1
    if skipped:
        print(f"INCOMPLETE: requested holdout(s) not evaluated: {', '.join(skipped)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
