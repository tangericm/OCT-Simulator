"""Multi-corpus ANCHOR·FIT·JUDGE driver — the robust iterate loop across a DIVERSE panel.

Where ``calibrate_optics.py`` runs ONE tracked experiment against ONE corpus, this driver runs
the whole loop over a panel that deliberately spans **instruments** (Maestro2, Maestro3),
**fields-of-view** (3/6/9/12 mm), and **slices/anatomy** (macula line, macula volume, optic
disc, wide), plus the pooled multi-dataset priors (OCTA-500, OCT-C8, OCTID). It:

  1. ANCHOR   — loads each member's frozen instrument + real ``TextureProfile`` anchor.
  2. FIT      — optionally fits ONE shared tissue-optics θ on a POOL of members
                (``run_de_pool``), so the result generalises instead of overfitting a single
                shot. The pool is the anti-overfit path of ``docs/CALIBRATION_ROUTINE.md`` §6.
  3. JUDGE    — evaluates D(θ) + the sanity gates on EVERY panel member, incumbent-vs-fitted.
  4. GENERALIZE — reports the cross-corpus D spread and the gap vs the fit pool's mean.
  5. FIGURE   — one CONTRAST-MATCHED grid (real | sim-incumbent | sim-fitted per corpus), every
                tile rendered with the identical peak-normalised dB window (vmin=-45, vmax=0) so
                real and simulated are compared on exactly the same brightness scale.
  6. TRACK    — writes ``runs/loop_<UTC>/`` {panel.json, FINDINGS.md, figures/*.png} and appends
                one ``mode:"loop"`` row per member to ``runs/manifest.jsonl``.

Physics stays the only image generator; this only orchestrates renders + measurement + figures.

    # baseline sweep of the whole panel (fast; no DE) + the grid figure + findings:
    python scripts/iterate_loop.py --data-root "<...>/OCT Data" --mode baseline --seeds 4

    # shared-optics pooled fit on the Maestro3 macula members, then judge the full panel:
    python scripts/iterate_loop.py --data-root "<...>/OCT Data" --mode fit \
        --maxiter 8 --popsize 8 --seeds 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import calibrate as C
from octsim import experiment as E
from octsim import generate
from octsim.data import real_maestro, unlabeled

# Display window shared by EVERY tile of the comparison grid — real and simulated alike.
# Both domains are peak-normalised dB (0 dB at the brightest pixel; see reconstruct.to_db and
# real_maestro.recon_frame), so a single fixed (vmin,vmax) is a FAIR contrast comparison.
VMIN, VMAX = -45.0, 0.0

# The diverse panel. ``target`` = bundled texture keyword / texture_<name>.json anchor;
# ``folder`` = real raw scan folder (relative to --data-root) for the comparison figure, or None
# for the pooled-prior corpora that have no local raw. ``in_pool`` marks members used for the
# SHARED-optics fit (normal macular tissue across instruments/FOV — the geometry differs but the
# tissue optics are shared). Disc/wide/pathology priors are JUDGE-only (generalization honesty).
PANEL = [
    # name                 target                 folder                                  anatomy  shape    alines  fov     axpitch  in_pool
    ("m3_macula_6mm",     "maestro3_macula",     "Maestro3/6mm_1024Aline",               "macula", "octa500", 1024,  6000.0, 3.87,  True),
    ("m3_macula_3x3",     "maestro3_macula_3x3", "Maestro3/M3_Macula_3x3mm",             "macula", "octa500",  256,  3000.0, 3.87,  True),
    ("m3_macula_9x9",     "maestro3_macula_9x9", "Maestro3/M3_Macula_9x9mm_512x512x2",   "macula", "octa500",  512,  9000.0, 3.87,  True),
    ("m3_disc_6x6",       "maestro3_disc_6x6",   "Maestro3/M3_Disc_6x6mm",               "onh",    "octa500",  512,  6000.0, 3.87,  False),
    # wide is IN the pool so the fit sees the full FOV range (3-12 mm) and the conic K can capture the
    # real curvature SATURATION at wide FOV (~800 µm plateau at 9-12 mm), not just the growing 3-9 mm arm.
    ("m3_wide_12x9",      "maestro3_wide_12x9",  "Maestro3/M3_Wide_12x9mm",              "macula", "octa500",  512, 12000.0, 3.87,  True),
    ("m2_line_6mm",       "maestro2_line_6mm",   "Maestro2/Line_6mm_2048Aline_135degCW_50frame_gain165", "macula", "octa500", 2048, 6000.0, 3.87, True),
    ("octa500",           "octa500",             None,                                    "macula", "octa500",  256,  6000.0, 3.87,  False),
    ("octc8",             "octc8",               None,                                    "macula", "octa500",  256,  6000.0, 3.87,  False),
    ("octid",             "octid",               None,                                    "macula", "octa500",  256,  6000.0, 3.87,  False),
]


def load_target(name: str) -> unlabeled.TextureProfile:
    try:
        return unlabeled.bundled_texture(name)
    except Exception:
        pass
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "octsim", "data", f"texture_{name}.json")
    return unlabeled.TextureProfile.load(p if os.path.exists(p) else name)


def spec_for(m, seeds) -> C.RenderSpec:
    return C.RenderSpec(instrument="maestro3", shape_model=m["shape"], anatomy=m["anatomy"],
                        center_wavelength_nm=840.0, pixels_per_spectrum=2048,
                        lines_per_frame=min(m["alines"], 512), lateral_fov_um=m["fov"],
                        averages=1, seeds=seeds)


def sim_frame(theta, spec, seed):
    """One simulated dB B-scan at this geometry/θ (labels intact — optics never touch geometry)."""
    profile, glob = C.decode(theta)
    cfg = spec.config(glob)
    generate.set_optics_override(profile)
    try:
        s = generate.simulate(cfg, seed=int(seed))
    finally:
        generate.set_optics_override(None)
    return s.bscan_db, s.resolved.source.depth_pixel_um


def real_frame(data_root, folder):
    if not folder:
        return None
    path = os.path.join(data_root, folder)
    if not os.path.isdir(path):
        return None
    try:
        imgs, _, _ = real_maestro.load_corpus(path, limit=9)
        return imgs[len(imgs) // 2] if imgs else None
    except Exception:
        return None


def snr_cnr(db, sig=(111, 600), bg_rows=20):
    lin = 10.0 ** (db / 20.0)
    H, W = lin.shape
    x0, x1 = min(10, W - 2), max(11, W - 10)
    y0, y1 = min(sig[0], H - 2), min(sig[1], H)
    s, b = lin[y0:y1, x0:x1], lin[max(0, H - bg_rows):H, x0:x1]
    if s.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    std = float(np.std(b)) + 1e-9
    return (20 * np.log10(float(np.mean(np.max(s, axis=0))) / std),
            20 * np.log10(float(np.mean(s)) / std))


def true_concave(sample_labels):
    from octsim.anatomy.retina import RETINA_LAYERS
    rpe = next(i for i, ly in enumerate(RETINA_LAYERS, start=1) if ly.name == "RPE")
    rows = []
    for c in range(sample_labels.shape[1]):
        r = np.where(sample_labels[:, c] == rpe)[0]
        rows.append(r.mean() if r.size else np.nan)
    rows = np.array(rows, float)
    good = np.isfinite(rows)
    if good.sum() < 8:
        return None
    xc = np.arange(sample_labels.shape[1])[good] - sample_labels.shape[1] / 2.0
    return float(np.polyfit(xc, rows[good], 2)[0])


SPECKLE_SANE = (0.25, 0.60)
SPECKLE_REL_TOL = 0.25


def gates_for(sc, target_sc, true_a):
    if sc is None:
        speckle_ok = False
    elif target_sc and target_sc > 0:
        speckle_ok = (abs(sc - target_sc) / target_sc <= SPECKLE_REL_TOL
                      and SPECKLE_SANE[0] <= sc <= SPECKLE_SANE[1])
    else:
        speckle_ok = 0.45 <= sc <= 0.53
    g = {"speckle_in_gate": bool(speckle_ok),
         "curvature_concave": bool(true_a is not None and true_a < 0.0)}
    return g, all(g.values())


def judge_member(m, theta, seeds, weights, data_root, want_frames=True):
    """Evaluate D + gates + SNR/CNR for one member at θ, and (optionally) grab real+sim frames."""
    target = load_target(m["target"])
    spec = spec_for(m, seeds)
    out = C.evaluate(theta, target, spec, weights)
    synth = out.get("synth_profile")
    # a single labelled render for gates / SNR-CNR / the sim tile (seed 0 of the CRN set)
    profile, glob = C.decode(theta)
    cfg = spec.config(glob)
    generate.set_optics_override(profile)
    try:
        s = generate.simulate(cfg, seed=int(seeds[0]))
    finally:
        generate.set_optics_override(None)
    sc = None if synth is None else float(synth.speckle_contrast)
    ta = true_concave(s.labels2d)
    snr, cnr = snr_cnr(s.bscan_db)
    gates, gp = gates_for(sc, float(getattr(target, "speckle_contrast", 0.0) or 0.0), ta)
    rec = {
        "name": m["name"], "target": m["target"], "instrument": ("maestro2" if m["name"].startswith("m2") else "maestro3"),
        "fov_um": m["fov"], "anatomy": m["anatomy"], "in_pool": m["in_pool"],
        "D_total": float(out.get("D_total", float("nan"))),
        "D_template": float(out.get("D_template", 0.0)), "D_hist": float(out.get("D_hist", 0.0)),
        "D_speckle": float(out.get("D_speckle", 0.0)), "D_reg": float(out.get("D_reg", 0.0)),
        "speckle_contrast": sc, "target_speckle": float(getattr(target, "speckle_contrast", 0.0) or 0.0),
        "thickness_um": None if synth is None else float(synth.total_retina_um_mean),
        "target_thickness_um": float(getattr(target, "total_retina_um_mean", 0.0) or 0.0),
        "snr_db": float(snr), "cnr_db": float(cnr), "true_curvature_a": ta,
        "gates": gates, "gates_pass": bool(gp),
    }
    frames = {}
    if want_frames:
        frames["sim"] = (s.bscan_db, s.resolved.source.depth_pixel_um)
        frames["real"] = real_frame(data_root, m["folder"])
        frames["real_axpitch"] = m["axpitch"]
    return rec, frames


def make_grid(members, frames_lit, frames_hat, out_png, is_fit):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ncol = 3 if is_fit else 2
    n = len(members)
    fig, axes = plt.subplots(n, ncol, figsize=(4.2 * ncol, 2.7 * n), squeeze=False)
    col_titles = (["real (device)", "sim — incumbent", "sim — fitted"] if is_fit
                  else ["real (device)", "sim — incumbent"])
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=12)

    def show(ax, img, dp, tag):
        if img is None:
            ax.text(0.5, 0.5, "real frame\nunavailable\n(prior-only corpus)",
                    ha="center", va="center", fontsize=8); ax.axis("off"); return
        H, W = img.shape
        # FULL tissue depth, physical µm depth axis; identical vmin/vmax for every tile so real
        # and simulated are on exactly the same brightness scale (matched contrast).
        ax.imshow(img, cmap="gray", aspect="auto", vmin=VMIN, vmax=VMAX,
                  extent=[0, W, H * dp, 0])
        ax.set_xticks([])
        if tag:
            ax.set_ylabel(tag + "\ndepth µm", fontsize=8)

    for i, m in enumerate(members):
        fl = frames_lit[m["name"]]
        show(axes[i][0], fl.get("real"), fl.get("real_axpitch", 3.87), m["name"])
        sdb, sdp = fl["sim"]
        show(axes[i][1], sdb, sdp, "")
        if is_fit:
            fh = frames_hat[m["name"]]
            hdb, hdp = fh["sim"]
            show(axes[i][2], hdb, hdp, "")
    fig.suptitle("octsim — real vs simulated across the diverse panel  "
                 f"(matched contrast: dB, vmin={VMIN:g} vmax={VMAX:g})", fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def d_bar(recs_lit, recs_hat, out_png, is_fit):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = [r["name"] for r in recs_lit]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
    h = 0.38 if is_fit else 0.6
    ax.barh(y + (h / 2 if is_fit else 0), [r["D_total"] for r in recs_lit], height=h,
            color="steelblue", label="incumbent")
    if is_fit:
        ax.barh(y - h / 2, [r["D_total"] for r in recs_hat], height=h, color="firebrick",
                label="fitted (pooled shared-optics)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("D(θ)  — texture distance to real anchor (lower = closer)")
    ax.set_title("cross-corpus D across the panel"); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    return True


def write_findings(path, recs_lit, recs_hat, is_fit, pool_mean_lit, pool_mean_hat, run_id):
    lines = [f"# octsim iterate-loop findings — `{run_id}`", ""]
    lines.append("Physics-only forward model; θ = tissue-optics + a few instrument globals. "
                 "All B-scan tiles share one peak-normalised dB contrast window "
                 f"(vmin={VMIN:g}, vmax={VMAX:g}) so real and simulated are compared on the same scale.")
    lines.append("")
    lines.append(f"**Pool mean D** (shared-optics fit set): incumbent **{pool_mean_lit:.4f}**"
                 + (f" → fitted **{pool_mean_hat:.4f}**  (Δ = {pool_mean_hat - pool_mean_lit:+.4f})"
                    if is_fit else "") + ".")
    lines.append("")
    hdr = "| corpus | instr | FOV | anat | pool | D incumbent"
    sep = "|---|---|---|---|---|---|"
    if is_fit:
        hdr += " | D fitted | Δ"; sep += "---|---|"
    hdr += " | speckle (sim/real) | thick µm (sim/real) | SNR/CNR dB | gates |"
    sep += "---|---|---|---|"
    lines += [hdr, sep]
    by_name_hat = {r["name"]: r for r in recs_hat} if is_fit else {}
    for r in recs_lit:
        row = (f"| {r['name']} | {r['instrument']} | {r['fov_um']/1000:.0f}mm | {r['anatomy']} "
               f"| {'✓' if r['in_pool'] else '·'} | {r['D_total']:.3f}")
        if is_fit:
            h = by_name_hat.get(r["name"], {})
            dh = h.get("D_total", float('nan'))
            row += f" | {dh:.3f} | {dh - r['D_total']:+.3f}"
        sc = r["speckle_contrast"]; tsc = r["target_speckle"]
        th = r["thickness_um"]; tth = r["target_thickness_um"]
        g = "✓" if r["gates_pass"] else "✗(" + ",".join(k for k, v in r["gates"].items() if not v) + ")"
        row += (f" | {sc:.2f}/{tsc:.2f} | {0 if th is None else th:.0f}/{tth:.0f} "
                f"| {r['snr_db']:.0f}/{r['cnr_db']:.0f} | {g} |")
        lines.append(row)
    lines.append("")
    lines.append("## Reading the panel")
    lines.append("- **D incumbent vs fitted** — a fit is honest only if it lowers the *pool* mean "
                 "D **without** blowing up D on the JUDGE-only rows (disc, wide, OCTA-500, OCT-C8, "
                 "OCTID). Watch those rows for the generalization gap.")
    lines.append("- **speckle (sim/real)** — should track the real regime (M3 frame-averaged look "
                 "≈0.34–0.38, not the 0.5 fully-developed band). Gate fails if >25% off.")
    lines.append("- **gates** — `speckle_in_gate` + `curvature_concave` (true RPE-label bowl). A row "
                 "that lowers D but fails a gate is *not* an improvement.")
    lines.append("")
    lines.append("Figures: `figures/panel_grid.png` (contrast-matched real-vs-sim across all "
                 "corpora), `figures/d_bar.png` (cross-corpus D).")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def find_incumbent_theta():
    """Most recent ACCEPTED run's θ (fitted optics), else literature. Matches on the exact
    ``param_names`` list, not just length — the θ-space has both grown and shrunk over time, so two
    runs can share a length while a since-changed slot (e.g. the new ``speckle_shape``) makes the
    vectors semantically incompatible. A name mismatch (or a run with no names) is skipped, so a
    warm-start can never decode a scrambled vector; it falls back to literature until a
    current-layout accepted run exists."""
    cur_names = [p.name for p in C.theta_space()]
    rows = E.read_manifest()
    for row in reversed(rows):
        if row.get("accepted") and row.get("run_id"):
            tp = os.path.join(E.RUNS_ROOT, row["run_id"], "theta.json")
            if os.path.exists(tp):
                try:
                    with open(tp) as fh:
                        d = json.load(fh)
                    th = np.asarray(d["theta"], float)
                    if d.get("param_names") == cur_names and th.shape[0] == len(cur_names):
                        return th, row["run_id"]
                except Exception:
                    pass
    return C.theta_literature(), "literature"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="the local 'OCT Data' folder")
    ap.add_argument("--mode", choices=["baseline", "fit"], default="baseline")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--wreg", type=float, default=0.05)
    ap.add_argument("--de-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--from-incumbent", action="store_true",
                    help="start θ from the last accepted run instead of literature")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    seeds = tuple(101 + 111 * i for i in range(args.seeds))
    weights = {**C.DEFAULT_WEIGHTS, "reg": args.wreg}
    members = [dict(zip(["name", "target", "folder", "anatomy", "shape", "alines", "fov",
                         "axpitch", "in_pool"], row)) for row in PANEL]

    if args.from_incumbent:
        theta_lit, base_id = find_incumbent_theta()
        print(f"[anchor] incumbent θ from {base_id}")
    else:
        theta_lit, base_id = C.theta_literature(), "literature"
        print("[anchor] incumbent θ = literature")

    run = E.ExperimentRun(target_corpus="loop", note=args.note or f"iterate-loop {args.mode}").start()
    print(f"[run] {run.run_id}  mode={args.mode}  members={len(members)}  seeds={args.seeds}")

    # ---- JUDGE the whole panel at the incumbent θ ----
    recs_lit, frames_lit = [], {}
    for m in members:
        rec, fr = judge_member(m, theta_lit, seeds, weights, args.data_root, want_frames=True)
        recs_lit.append(rec); frames_lit[m["name"]] = fr
        print(f"  [incumbent] {m['name']:16s} D={rec['D_total']:.3f} "
              f"speckle={rec['speckle_contrast']} gates={'PASS' if rec['gates_pass'] else 'fail'}")
    pool_lit = [r["D_total"] for r in recs_lit if r["in_pool"]]
    pool_mean_lit = float(np.mean(pool_lit)) if pool_lit else float("nan")

    # ---- FIT shared optics on the pool (anti-overfit) ----
    is_fit = args.mode == "fit"
    recs_hat, frames_hat, theta_hat = recs_lit, frames_lit, theta_lit
    pool_mean_hat = pool_mean_lit
    incumbent_traj = []
    if is_fit:
        pool = [(load_target(m["target"]), spec_for(m, seeds)) for m in members if m["in_pool"]]
        print(f"[fit] pooled DE over {len(pool)} corpora  maxiter={args.maxiter} popsize={args.popsize}")
        theta_hat, dmean, incumbent_traj, _ = C.run_de_pool(
            pool, weights=weights, base_theta=theta_lit, maxiter=args.maxiter,
            popsize=args.popsize, seed=args.de_seed, workers=args.workers)
        print(f"[fit] pool mean D {pool_mean_lit:.4f} → {dmean:.4f}")
        recs_hat, frames_hat = [], {}
        for m in members:
            rec, fr = judge_member(m, theta_hat, seeds, weights, args.data_root, want_frames=True)
            recs_hat.append(rec); frames_hat[m["name"]] = fr
            print(f"  [fitted]    {m['name']:16s} D={rec['D_total']:.3f} "
                  f"gates={'PASS' if rec['gates_pass'] else 'fail'}")
        pool_mean_hat = float(np.mean([r["D_total"] for r in recs_hat if r["in_pool"]]))

    # ---- FIGURES (contrast-matched) ----
    make_grid(members, frames_lit, frames_hat, run.figure_path("panel_grid.png"), is_fit)
    d_bar(recs_lit, recs_hat, run.figure_path("d_bar.png"), is_fit)

    # ---- TRACK ----
    accepted = bool(is_fit and pool_mean_hat < pool_mean_lit
                    and all(r["gates_pass"] for r in recs_hat if r["in_pool"]))
    run.write_json("panel.json", {
        "mode": args.mode, "base_theta": base_id, "seeds": list(seeds), "weights": weights,
        "pool_mean_D_incumbent": pool_mean_lit, "pool_mean_D_fitted": pool_mean_hat,
        "accepted": accepted, "theta_incumbent": theta_lit.tolist(),
        "theta_fitted": theta_hat.tolist(),
        "param_names": [q.name for q in C.theta_space()],
        "incumbent": recs_lit, "fitted": recs_hat if is_fit else None})
    # theta.json lets a later --from-incumbent warm-start from this run (find_incumbent_theta)
    run.write_theta({"theta": theta_hat.tolist(),
                     "param_names": [q.name for q in C.theta_space()], "accepted": accepted})
    if incumbent_traj:
        run.write_incumbent(incumbent_traj)
    write_findings(os.path.join(run.path, "FINDINGS.md"), recs_lit, recs_hat, is_fit,
                   pool_mean_lit, pool_mean_hat, run.run_id)

    for r in (recs_hat if is_fit else recs_lit):
        run.append_manifest({
            "ts": run.run_id.split("_")[0], "git_sha": E.git_sha(), "N": args.seeds,
            "mode": "loop", "instrument": r["instrument"], "averages": 1,
            "D_total": r["D_total"], "D_template": r["D_template"], "D_hist": r["D_hist"],
            "D_speckle": r["D_speckle"], "D_reg": r["D_reg"],
            "speckle_contrast": r["speckle_contrast"], "thickness_um": r["thickness_um"],
            "snr_db": r["snr_db"], "cnr_db": r["cnr_db"],
            "gates_pass": r["gates_pass"], "gates": r["gates"],
            "accepted": accepted, "run_id": run.run_id, "target_corpus": r["target"],
            "parent_run_id": base_id, "note": (args.note or f"loop {args.mode} {r['name']}")})

    print(f"[judge] pool mean D: {pool_mean_lit:.4f}"
          + (f" → {pool_mean_hat:.4f}  accepted={accepted}" if is_fit else ""))
    print(f"[done] {run.path}\n  FINDINGS.md + figures/panel_grid.png + figures/d_bar.png")


if __name__ == "__main__":
    main()
