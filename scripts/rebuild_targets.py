"""Rebuild every raw-folder real TextureProfile target (captures new depth-envelope + curvature
fields). Run after changing extract_texture. Idempotent; overwrites texture_<name>.json + _meta."""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from octsim.data import real_maestro, unlabeled

# Root of the raw device captures. These are clinical acquisitions and are not
# redistributed with this repo, so the path is an environment variable rather
# than a constant: OCTSIM_DATA_ROOT=/path/to/OCT\ Data python scripts/rebuild_targets.py
DR = os.environ.get("OCTSIM_DATA_ROOT", "")
if not DR:
    raise SystemExit(
        "Set OCTSIM_DATA_ROOT to the directory holding the raw device folders "
        "listed in CORPORA below. The captures are not part of this repository."
    )
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "octsim", "data")

# name, folder(rel to DR), fov_um, stride, limit
CORPORA = [
    ("maestro3_macula",     "Maestro3/6mm_1024Aline",             6000.0, 1, 64),
    ("maestro3_macula_3x3", "Maestro3/M3_Macula_3x3mm",           3000.0, 1, 64),
    ("maestro3_macula_9x9", "Maestro3/M3_Macula_9x9mm_512x512x2", 9000.0, 1, 64),
    ("maestro3_disc_6x6",   "Maestro3/M3_Disc_6x6mm",             6000.0, 1, 64),
    ("maestro3_wide_12x9",  "Maestro3/M3_Wide_12x9mm",           12000.0, 1, 64),
    ("maestro2_line_6mm",   "Maestro2/Line_6mm_2048Aline_135degCW_50frame_gain165", 6000.0, 1, 64),
    ("maestro3_macula_vol", "Maestro3/M3_Macula_6x6mm",           6000.0, 4, 64),
]

for name, folder, fov, stride, limit in CORPORA:
    path = os.path.join(DR, folder)
    if not os.path.isdir(path):
        print(f"SKIP {name}: no folder {folder}"); continue
    imgs, alines, skip = real_maestro.load_corpus(path, limit=limit, stride=stride)
    lat = fov / alines
    tex = unlabeled.extract_texture(imgs, axial_pitch_um=3.87, lateral_pitch_um=lat,
                                    source=f"real:{name}")
    out = os.path.join(DATA, f"texture_{name}.json")
    tex.save(out)
    env_last = tex.depth_intensity_env[-1] if tex.depth_intensity_env else None
    print(f"{name:22s} n={tex.n_images:2d} fov={fov/1000:.0f}mm curv={tex.rpe_curvature_coef:6.1f} "
          f"env_last={env_last:.3f} thick={tex.total_retina_um_mean:.0f} speck={tex.speckle_contrast:.3f}")
print("done")
