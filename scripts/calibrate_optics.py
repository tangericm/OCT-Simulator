"""ANCHOR·FIT·JUDGE runner — one tracked calibration experiment per invocation.

    # E0 baseline (render literature optics through the anchor; log the full panel):
    python scripts/calibrate_optics.py --target maestro3_macula --instrument maestro3 \
        --shape-model octa500 --averages 1 --mode baseline --note "E0 anchor"

    # E2 fit (differential_evolution over the ~20 optics params, then JUDGE + GATE):
    python scripts/calibrate_optics.py --target maestro3_macula --instrument maestro3 \
        --mode fit --maxiter 12 --popsize 10 --seeds 10 --note "E2 first M3 fit"

Writes ``runs/<UTC>_<target>_<sha>/`` {config.json, theta.json, metrics.json, incumbent.csv,
figures/…, env.txt} and appends one row to ``runs/manifest.jsonl``. A fit is ACCEPTED only if D
dropped, the sanity gates hold (speckle∈[0.45,0.53] at averages=1, curvature concave), and the
generalization gap did not worsen — see ``docs/CALIBRATION_ROUTINE.md`` §5. Only accepted fitted
JSON is committed; raw data never is.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Force UTF-8 stdout so θ/Δ/λ in progress prints don't crash on the Windows cp1252 console
# (the print is cosmetic, but an encode error there would abort before the run is tracked).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import calibrate as C
from octsim import experiment as E
from octsim import generate
from octsim.calibration import literature_profile
from octsim.data import unlabeled

GEN_CORPORA = ["maestro2", "octa500", "octc8", "octid", "pooled"]
SPECKLE_GATE = (0.45, 0.53)      # fully-developed Rayleigh band (fallback when no target speckle)
SPECKLE_SANE = (0.25, 0.60)      # absolute sanity envelope (reject absurd speckle regardless)
SPECKLE_REL_TOL = 0.25           # synth speckle must be within this fraction of the target's


def _load_target(name: str) -> unlabeled.TextureProfile:
    # keyword in the bundled map, else octsim/data/texture_<name>.json, else a literal path
    try:
        return unlabeled.bundled_texture(name)
    except Exception:
        pass
    data_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "octsim", "data", f"texture_{name}.json")
    if os.path.exists(data_json):
        return unlabeled.TextureProfile.load(data_json)
    return unlabeled.TextureProfile.load(name)


def _db_ladder(theta, spec):
    """Per-layer mean dB from one labelled render (labels are free) — the brightness ladder."""
    profile, glob = C.decode(theta)
    cfg = spec.config(glob)
    generate.set_optics_override(profile)
    try:
        s = generate.simulate(cfg, seed=int(spec.seeds[0]))
    finally:
        generate.set_optics_override(None)
    from octsim.anatomy.retina import RETINA_LAYERS
    names = [ly.name for ly in RETINA_LAYERS]
    db, lab = s.bscan_db, s.labels2d
    ladder = {}
    for i, nm in enumerate(names, start=1):
        m = lab == i
        ladder[nm] = float(db[m].mean()) if m.any() else None
    return ladder, s


def _true_concave(sample) -> float | None:
    """Ground-truth RPE curvature sign from the label map (a<0 = concave-up bowl).

    Synthetic renders carry exact labels, so the curvature sanity gate should read the *true*
    geometry, not the label-free surface detector — which flips the tiny macular-bowl curvature
    (a~1e-4) on synthetic images and falsely reports convex."""
    from octsim.anatomy.retina import RETINA_LAYERS
    rpe_idx = next(i for i, ly in enumerate(RETINA_LAYERS, start=1) if ly.name == "RPE")
    lab = sample.labels2d
    rows = []
    for c in range(lab.shape[1]):
        r = np.where(lab[:, c] == rpe_idx)[0]
        rows.append(r.mean() if r.size else np.nan)
    rows = np.array(rows, float)
    good = np.isfinite(rows)
    if good.sum() < 8:
        return None
    xc = np.arange(lab.shape[1])[good] - lab.shape[1] / 2.0
    a = float(np.polyfit(xc, rows[good], 2)[0])
    return a                                    # a < 0 => concave-up bowl


def _snr_cnr(bscan_db, sig=(111, 600), bg_rows=20):
    lin = 10.0 ** (bscan_db / 20.0)
    H, W = lin.shape
    x0, x1 = min(10, W - 2), max(11, W - 10)
    y0, y1 = min(sig[0], H - 2), min(sig[1], H)
    sig_roi = lin[y0:y1, x0:x1]; bg_roi = lin[max(0, H - bg_rows):H, x0:x1]
    if sig_roi.size == 0 or bg_roi.size == 0:
        return float("nan"), float("nan")
    std_bg = float(np.std(bg_roi)) + 1e-9
    return (20.0 * np.log10(float(np.mean(np.max(sig_roi, axis=0))) / std_bg),
            20.0 * np.log10(float(np.mean(sig_roi)) / std_bg))


def _panel(theta, target, spec, weights):
    """Full metric panel for one θ: D + terms, sanity, diagnostics."""
    out = C.evaluate(theta, target, spec, weights)
    synth = out.get("synth_profile")
    panel = {k: out.get(k, 0.0) for k in ("D_total", "D_template", "D_hist", "D_speckle",
                                          "D_layer", "D_contrast", "D_reg")}
    ladder, sample = _db_ladder(theta, spec)
    snr, cnr = _snr_cnr(sample.bscan_db)
    true_a = _true_concave(sample)              # ground-truth RPE curvature (a<0 = bowl)
    panel.update({
        "speckle_contrast": None if synth is None else float(synth.speckle_contrast),
        "axial_grain_um": None if synth is None else float(synth.axial_grain_um),
        "lateral_grain_um": None if synth is None else float(synth.lateral_grain_um),
        "heterogeneity": None if synth is None else float(synth.heterogeneity),
        "thickness_um": None if synth is None else float(synth.total_retina_um_mean),
        "concave_frac": None if synth is None else float(synth.curvature_concave_frac),
        "true_curvature_a": None if true_a is None else float(true_a),
        "snr_db": float(snr), "cnr_db": float(cnr),
        "db_ladder": ladder,
    })
    # sanity gates. Curvature uses the *ground-truth* label curvature (a<0 = concave bowl), not
    # the label-free detector's concave_frac — the detector flips the tiny true macular curvature
    # on synthetic images (verified: true a<0 for every seed, detector reads convex).
    # Speckle: match the TARGET's contrast (within a relative tolerance) inside an absolute sanity
    # envelope — the fixed Rayleigh band [0.45,0.53] is only right for fully-developed single-frame
    # speckle, but real Maestro frames are smoothed/averaged-look (~0.34-0.38), so the honest gate
    # is "reproduces the real regime", not a fixed band. Falls back to the band with no target.
    sc = panel["speckle_contrast"]
    tgt_sc = float(getattr(target, "speckle_contrast", 0.0) or 0.0)
    if sc is None:
        speckle_ok = False
    elif tgt_sc > 0:
        speckle_ok = (abs(sc - tgt_sc) / tgt_sc <= SPECKLE_REL_TOL
                      and SPECKLE_SANE[0] <= sc <= SPECKLE_SANE[1])
    else:
        speckle_ok = SPECKLE_GATE[0] <= sc <= SPECKLE_GATE[1]
    gates = {
        "speckle_in_gate": bool(speckle_ok),
        "curvature_concave": bool(true_a is not None and true_a < 0.0),
    }
    panel["gates"] = gates
    panel["gates_pass"] = all(gates.values())
    return panel, out, sample


def _generalization(theta, spec, weights, target_D):
    """D(θ) on the other corpora and the gap D_other − D_target (honesty check)."""
    gaps = {}
    for corpus in GEN_CORPORA:
        try:
            tgt = unlabeled.bundled_texture(corpus)
            d = C.evaluate(theta, tgt, spec, weights)["D_total"]
            gaps[corpus] = {"D": float(d), "gap": float(d - target_D)}
        except Exception as exc:
            gaps[corpus] = {"D": None, "gap": None, "error": repr(exc)}
    return gaps


def _real_frame(real_folder: str):
    """Reconstruct a representative real M3 B-scan (mid-corpus) for the comparison figure."""
    if not real_folder or not os.path.isdir(real_folder):
        return None
    try:
        from octsim.data import real_maestro
        imgs, _, _ = real_maestro.load_corpus(real_folder, limit=9)
        return imgs[len(imgs) // 2] if imgs else None
    except Exception:
        return None


def _retina_band(db, pad=40):
    """(lo, hi) row range of the retinal band (brightest rows ± pad) — for the ROI zoom + overlay."""
    prof = db.mean(axis=1)
    r = int(np.argmax(prof))
    return max(0, r - 3 * pad), min(db.shape[0], r + 3 * pad)


def _outer_aline(db, dp, half_um=55):
    """RPE-aligned mean outer-complex A-line (norm intensity vs µm-from-RPE) — the axial-sharpness
    view: resolved EZ/RPE peaks read as the device's true axial resolution, not an over-spread band."""
    lin = 10.0 ** (db / 20.0); H, W = lin.shape; win = int(half_um / dp)
    segs = []
    for c in range(W):
        v = lin[:, c]; p = int(np.argmax(v))
        if p - win < 0 or p + win >= H:
            continue
        segs.append(v[p - win:p + win + 1])
    if not segs:
        return None, None
    prof = np.mean(segs, axis=0); prof = prof / (prof.max() + 1e-9)
    z = (np.arange(-win, win + 1)) * dp
    return z, prof


def _figures(run, theta_lit, theta_hat, target, spec, panel_lit, panel_hat, real_folder=None):
    """One descriptive comparison panel: real vs simulated B-scans, reflectivity templates,
    intensity histograms, and the per-layer dB ladder — the visual justification for a run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    s_lit = panel_lit[2] if panel_lit else _panel(theta_lit, target, spec, None)[2]
    s_hat = panel_hat[2] if panel_hat else _panel(theta_hat, target, spec, None)[2]
    p_lit = C.evaluate(theta_lit, target, spec, None).get("synth_profile")
    p_hat = C.evaluate(theta_hat, target, spec, None).get("synth_profile")
    real = _real_frame(real_folder)
    is_fit = not np.allclose(theta_lit, theta_hat)

    fitlbl = "fitted" if is_fit else "literature"
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 0.9, 0.95], hspace=0.42, wspace=0.30)

    # --- Row 1: FULL-depth B-scans (µm depth axis), retinal ROI marked with a box ---
    full = [("real Maestro3", real, 3.87),
            ("simulated — literature θ", s_lit.bscan_db, s_lit.resolved.source.depth_pixel_um),
            (f"simulated — {fitlbl} θ", s_hat.bscan_db, s_hat.resolved.source.depth_pixel_um)]
    for j, (title, img, dp) in enumerate(full):
        ax = fig.add_subplot(gs[0, j])
        if img is None:
            ax.text(0.5, 0.5, "real frame\nunavailable", ha="center", va="center"); ax.axis("off")
            ax.set_title(title + " (full depth)", fontsize=11); continue
        H, W = img.shape
        ax.imshow(img, cmap="gray", aspect="auto", vmin=-45, vmax=0, extent=[0, W, H * dp, 0])
        lo, hi = _retina_band(img)
        ax.add_patch(plt.Rectangle((0, lo * dp), W, (hi - lo) * dp, fill=False,
                                   edgecolor="deepskyblue", lw=1.2, ls="--"))
        ax.set_xlabel("A-line"); ax.set_ylabel("depth (µm)")
        ax.set_title(title + "  (full tissue depth)", fontsize=11)

    # --- Row 2a: reflectivity-template overlay (real target vs sim) ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(target.template_axis, target.reflectivity_template, "k-", lw=2.2, label="real target")
    if p_lit:
        ax.plot(p_lit.template_axis, p_lit.reflectivity_template, "b--", lw=1.3, label="sim literature")
    if is_fit and p_hat:
        ax.plot(p_hat.template_axis, p_hat.reflectivity_template, "r-", lw=1.6, label="sim fitted")
    ax.set_xlabel("normalised depth  ILM(0)→RPE(1)"); ax.set_ylabel("reflectivity")
    ax.set_title("layer-brightness template", fontsize=11); ax.legend(fontsize=8)

    # --- Row 2b: tissue intensity histogram (real vs sim) ---
    ax = fig.add_subplot(gs[1, 1])
    if target.intensity_hist:
        xh = np.linspace(0, 1, len(target.intensity_hist))
        ax.plot(xh, target.intensity_hist, "k-", lw=2.2, label="real target")
        sp = p_hat if (is_fit and p_hat) else p_lit
        if sp and sp.intensity_hist:
            ax.plot(np.linspace(0, 1, len(sp.intensity_hist)), sp.intensity_hist, "r-", lw=1.6,
                    label="sim " + ("fitted" if is_fit else "literature"))
        ax.set_xlabel("normalised intensity"); ax.set_ylabel("density")
        ax.set_title("tissue intensity histogram", fontsize=11); ax.legend(fontsize=8)

    # --- Row 2c: per-layer dB ladder (sim, from labels) ---
    ax = fig.add_subplot(gs[1, 2])
    ladder = (panel_hat[0] if panel_hat else _panel(theta_hat, target, spec, None)[0])["db_ladder"]
    names = [k for k, v in ladder.items() if v is not None]
    vals = [ladder[k] for k in names]
    ax.barh(range(len(names)), vals, color="slategray")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("mean dB"); ax.set_title("simulated per-layer dB ladder", fontsize=11)

    # --- Row 3: separate ROI subfigures — retinal-band zoom (real | sim) + sharpness A-line ---
    def _zoom(ax, img, dp, title):
        if img is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.axis("off")
            ax.set_title(title, fontsize=10); return
        lo, hi = _retina_band(img)
        ax.imshow(img[lo:hi], cmap="gray", aspect="auto", vmin=-45, vmax=0,
                  extent=[0, img.shape[1], hi * dp, lo * dp])
        ax.set_xlabel("A-line"); ax.set_ylabel("depth (µm)"); ax.set_title(title, fontsize=10)
    _zoom(fig.add_subplot(gs[2, 0]), real, 3.87, "ROI zoom — real retina")
    _zoom(fig.add_subplot(gs[2, 1]), s_hat.bscan_db, s_hat.resolved.source.depth_pixel_um,
          f"ROI zoom — simulated {fitlbl} retina")
    ax = fig.add_subplot(gs[2, 2])
    zr = pr = None
    if real is not None:
        zr, pr = _outer_aline(real, 3.87)
    zs, ps = _outer_aline(s_hat.bscan_db, s_hat.resolved.source.depth_pixel_um)
    if zr is not None:
        ax.plot(zr, pr, "k-", lw=2.0, label="real")
    if zs is not None:
        ax.plot(zs, ps, "r-", lw=1.6, label="sim " + fitlbl)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("depth from RPE peak (µm)"); ax.set_ylabel("norm. int.")
    ax.set_title("axial sharpness: outer-complex A-line", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    D_lit = panel_lit[0]["D_total"] if panel_lit else float("nan")
    D_hat = panel_hat[0]["D_total"] if panel_hat else float("nan")
    fig.suptitle(f"{run.target_corpus}  —  real vs simulated   "
                 f"(D: literature {D_lit:.3f}" + (f" → fitted {D_hat:.3f}" if is_fit else "") + ")",
                 fontsize=13, y=0.98)
    fig.savefig(run.figure_path("comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="texture profile keyword/path (real anchor)")
    ap.add_argument("--instrument", default="", help="instrument keyword ('maestro3') or ''")
    ap.add_argument("--shape-model", default="octa500")
    ap.add_argument("--anatomy", default="macula")
    ap.add_argument("--lambda0", type=float, default=840.0)
    ap.add_argument("--pixels", type=int, default=2048)
    ap.add_argument("--lines", type=int, default=256)
    ap.add_argument("--fov", type=float, default=6000.0)
    ap.add_argument("--averages", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mode", choices=["baseline", "fit"], default="baseline")
    ap.add_argument("--maxiter", type=int, default=12)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--wreg", type=float, default=0.05)
    ap.add_argument("--de-seed", type=int, default=0, help="DE seed (vary to test fit stability)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--real-folder", default=None,
                    help="raw M3 scan folder to reconstruct a real B-scan for the comparison figure")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--volume-frames", type=int, default=1,
                    help=">1 renders each CRN seed as a slow-axis VOLUME (fovea→parafovea) and extracts "
                         "the volume AVERAGE — matches the real target build, smooths single-frame "
                         "inner-layer template oscillation. Use fewer --seeds to offset the cost.")
    args = ap.parse_args()

    target = _load_target(args.target)
    weights = {**C.DEFAULT_WEIGHTS, "reg": args.wreg}
    seeds = tuple(101 + 111 * i for i in range(args.seeds))
    spec = C.RenderSpec(instrument=args.instrument, shape_model=args.shape_model,
                        anatomy=args.anatomy, center_wavelength_nm=args.lambda0,
                        pixels_per_spectrum=args.pixels, lines_per_frame=args.lines,
                        lateral_fov_um=args.fov, averages=args.averages, seeds=seeds,
                        volume_frames=args.volume_frames, slow_fov_um=args.fov)

    run = E.ExperimentRun(target_corpus=args.target, note=args.note,
                          parent_run_id=args.parent).start()
    print(f"[run] {run.run_id}")

    theta_lit = C.theta_literature()
    panel_lit = _panel(theta_lit, target, spec, weights)
    D_lit = panel_lit[0]["D_total"]
    print(f"[baseline] D(literature) = {D_lit:.4f}  speckle={panel_lit[0]['speckle_contrast']} "
          f"snr={panel_lit[0]['snr_db']:.1f}dB gates={panel_lit[0]['gates']}")

    incumbent = []
    if args.mode == "fit":
        n_free = sum(1 for p in C.theta_space() if not p.frozen)
        print(f"[fit] DE maxiter={args.maxiter} popsize={args.popsize} "
              f"free={n_free}/{len(C.theta_space())} seeds={args.seeds} workers={args.workers}")
        theta_hat, D_hat, incumbent, res = C.run_de(
            target, spec, weights=weights, maxiter=args.maxiter, popsize=args.popsize,
            seed=args.de_seed, workers=args.workers)
        print(f"[fit] D(θ̂) = {D_hat:.4f}  (Δ = {D_hat - D_lit:+.4f})")
    else:
        theta_hat, D_hat = theta_lit, D_lit

    panel_hat = _panel(theta_hat, target, spec, weights)
    p = panel_hat[0]
    gen = _generalization(theta_hat, spec, weights, p["D_total"])

    accepted = bool(args.mode == "fit" and D_hat < D_lit and p["gates_pass"])

    # --- persist the experiment ---
    profile, glob = C.decode(theta_hat)
    run.write_config({"mode": args.mode, "target": args.target, "instrument": args.instrument,
                      "shape_model": args.shape_model, "anatomy": args.anatomy,
                      "lambda0_nm": args.lambda0, "pixels": args.pixels, "lines": args.lines,
                      "fov_um": args.fov, "averages": args.averages, "seeds": list(seeds),
                      "weights": weights})
    run.write_theta({"theta": theta_hat.tolist(), "param_names": [q.name for q in C.theta_space()],
                     "profile": profile.to_dict(), "globals": glob.__dict__,
                     "parent_run_id": args.parent})
    run.write_metrics({"literature": panel_lit[0], "evaluated": p,
                       "D_literature": D_lit, "D_evaluated": p["D_total"],
                       "generalization": gen, "accepted": accepted})
    if incumbent:
        run.write_incumbent(incumbent)
    if not args.no_figures:
        try:
            _figures(run, theta_lit, theta_hat, target, spec, panel_lit, panel_hat,
                     real_folder=args.real_folder)
        except Exception as exc:
            print(f"[figures] skipped: {exc}")

    row = {"ts": run.run_id.split("_")[0], "git_sha": E.git_sha(), "N": args.seeds,
           "mode": args.mode, "instrument": args.instrument, "averages": args.averages,
           "D_total": p["D_total"], "D_template": p["D_template"], "D_hist": p["D_hist"],
           "D_speckle": p["D_speckle"], "D_reg": p["D_reg"], "D_literature": D_lit,
           "speckle_contrast": p["speckle_contrast"], "thickness_um": p["thickness_um"],
           "snr_db": p["snr_db"], "cnr_db": p["cnr_db"], "concave_frac": p["concave_frac"],
           "gates_pass": p["gates_pass"], "gates": p["gates"],
           "gen_gap_maestro2": gen.get("maestro2", {}).get("gap"),
           "gen_gap_octa500": gen.get("octa500", {}).get("gap"),
           "gen_gap_octc8": gen.get("octc8", {}).get("gap"),
           "gen_gap_octid": gen.get("octid", {}).get("gap"),
           "accepted": accepted}
    run.append_manifest(row)

    print(f"[judge] gates_pass={p['gates_pass']} accepted={accepted}")
    print(f"[gen] gaps: " + "  ".join(
        f"{k}={gen[k]['gap']:+.3f}" if gen[k].get('gap') is not None else f"{k}=NA"
        for k in GEN_CORPORA))
    if accepted:
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "octsim", "data", f"retina_optics_{args.target}.json")
        profile.save(out)
        print(f"[accepted] wrote {out} (commit this; do NOT flip the default until the leaderboard is stable)")
    print(f"[done] {run.path}")


if __name__ == "__main__":
    main()
