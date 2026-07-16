"""Label-free texture learning: extractor recovers structure; bundled profile is sane."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim.data import unlabeled as ul  # noqa: E402


def _synthetic_bscan(rng, ilm=60, rpe=145, nfl=0.9, mid=0.5):
    """A fake normal-ish B-scan: dark vitreous, bright ILM/NFL, dark ONL, bright RPE."""
    im = rng.gamma(1.0, 0.05, (220, 256))
    im[ilm:ilm + 6] += nfl                       # NFL just under the ILM
    im[ilm + 30:ilm + 45] += mid                 # a plexiform band
    im[rpe - 3:rpe + 3] += 1.0                    # RPE (brightest)
    return np.clip(im, 0.0, 3.0)


def test_extract_recovers_surfaces_and_template():
    rng = np.random.default_rng(0)
    imgs = [_synthetic_bscan(rng) for _ in range(12)]
    prof = ul.extract_texture(imgs, axial_pitch_um=3.87, source="synthetic")
    assert prof.n_images == 12
    axis = np.array(prof.template_axis); tpl = np.array(prof.reflectivity_template)
    # brightest band near the RPE (normalised depth ~1.0)
    assert abs(axis[int(np.argmax(tpl))] - 1.0) < 0.15
    # a bright NFL near the ILM (top 20% of depth) exceeds the mid-retina minimum
    top = tpl[axis < 0.2].max()
    assert top > tpl[(axis > 0.55) & (axis < 0.8)].min()
    # thickness ~ (rpe-ilm)*pitch = 85*3.87 ~ 329 um, within tolerance
    assert 250.0 < prof.total_retina_um_mean < 420.0


def test_speckle_contrast_is_rayleigh_ish():
    rng = np.random.default_rng(1)
    # fully-developed speckle within a uniform bright slab -> amplitude contrast ~0.52
    imgs = []
    for _ in range(8):
        im = np.zeros((220, 256))
        im[60:150] = rng.rayleigh(1.0, (90, 256))
        imgs.append(im)
    prof = ul.extract_texture(imgs, source="rayleigh")
    assert 0.4 < prof.speckle_contrast < 0.65


def test_bundled_texture_loads_and_is_sane():
    prof = ul.bundled_texture()                  # Srinivasan NORMAL
    assert prof.n_images > 500
    assert 200.0 < prof.total_retina_um_mean < 420.0    # normal retinal thickness
    assert prof.curvature_concave_frac > 0.6            # real retina images concave-up
    assert 0.4 < prof.speckle_contrast < 0.75
    assert len(prof.reflectivity_template) == ul.N_TEMPLATE


def test_layer_reflectivity_from_template_orders_bright_dark():
    prof = ul.bundled_texture()
    names = ["NFL", "GCL", "IPL", "INL", "OPL", "ONL", "ELM", "EZ", "IZ", "RPE", "choroid"]
    thick = [30, 38, 40, 30, 25, 78, 5, 9, 7, 18, 260]
    refl = ul.layer_reflectivity_from_template(prof, names, thick)
    assert refl["RPE"] > refl["ONL"]             # RPE brighter than the nuclear ONL
    assert refl["NFL"] > refl["INL"]             # NFL brighter than the nuclear INL


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} unlabeled-texture checks passed")
