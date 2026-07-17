#!/usr/bin/env python
"""Distil the OCTA-500 3mm subset into the two bundled artifacts.

OCTA-500 ships real macular OCT *volumes* with a volumetric layer segmentation. This walks
the extracted tree (``OCT/``, ``GT_Layers/``, ``GT_FAZ/``), keeps the geometrically-normal
eyes, and writes:

  * ``octsim/data/texture_octa500.json`` — the frame-averaged RTVue-XR appearance profile
    (kept separate from the single-frame healthy default; see ``data/unlabeled.py``);
  * ``octsim/data/octa500_macula.json``  — the real radial ILM→BM thickness profile, foveal
    pit metrics, and FAZ radius (a geometry validation reference);
  * ``octsim/data/retina_shape_octa500.npz`` — the **default** normal-eye shape model (real
    deep foveal pit), fit from the volumetric layer segmentation.

    python scripts/fit_octa500.py --root path/to/OCTA-500 --out-dir octsim/data

The release zip may be pointed at directly; it is extracted to a temp dir first. The ~4.5 GB
raw dataset is never committed — only the two small JSONs above.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted OCTA-500 tree or the release .zip")
    ap.add_argument("--out-dir", default="octsim/data")
    ap.add_argument("--per-eye", type=int, default=15, help="B-scans sampled per eye for texture")
    args = ap.parse_args()

    from octsim.data import octa500 as o5

    root = args.root
    tmp = None
    if os.path.isfile(root) and root.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="octa500_")
        print(f"extracting {root} -> {tmp} ...")
        with zipfile.ZipFile(root) as zf:
            zf.extractall(tmp)
        root = tmp

    ids = o5.normal_ids(root)
    allids = o5.subject_ids(root)
    print(f"{len(allids)} eyes in range; {len(ids)} pass the normal-geometry filter "
          f"({len(allids) - len(ids)} excluded: {[s for s in allids if s not in ids]})")

    tex = o5.extract_texture(root, ids, per_eye=args.per_eye)
    tex.save(os.path.join(args.out_dir, "texture_octa500.json"))
    print(f"texture: {tex.n_images} B-scans  contrast {tex.speckle_contrast:.3f}  "
          f"concave {tex.curvature_concave_frac:.2f}  thick {tex.total_retina_um_mean:.0f} um")

    mac = o5.build_macula_profile(root)
    mac.save(os.path.join(args.out_dir, "octa500_macula.json"))
    print(f"macula: central {mac.central_thickness_um:.0f} um  plateau "
          f"{mac.perifoveal_plateau_um:.0f} um  rel-depth {mac.pit_relative_depth:.3f}  "
          f"FAZ r {mac.faz_radius_um_median:.0f} um  ({mac.n_eyes_normal} normal eyes)")

    sm = o5.fit_shape_model(root, ids=ids)
    sm.save(os.path.join(args.out_dir, "retina_shape_octa500.npz"))
    e = abs(sm.ecc_grid_um); tot = sm.mean_thickness.sum(axis=0)
    rel = tot[e < 75].mean() / tot[(e >= 1000) & (e <= 1300)].mean()
    print(f"shape model: {sm.n_scans} eyes  mean pit rel-depth {rel:.3f}  "
          f"-> retina_shape_octa500.npz (the default normal geometry)")

    if tmp:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
