# Data-driven calibration (Tier 1)

The physics forward model stays the backbone — exact per-pixel labels, controllable
acquisition physics. Calibration only supplies its **parameters** from data instead of
hand-tuned guesses, so calibrated scans stay fully annotated. Module: `octsim/calibration.py`.

## What a profile holds

A `CalibrationProfile` bundles, per retinal layer: relative backscatter, attenuation
(`mu_t`), speckle contrast, thickness (mean/std), and a faint specular-line strength; plus
global fields: **boundary sharpness** (axial transition width — softens the bands),
**curvature** (concave-up radius + saturating sag), foveal-pit radius, ONL bulge, and
choroid contrast. It is JSON-serialisable (`profile.save(path)` / `CalibrationProfile.load`).

## Getting a profile

**No data (works today):**
```python
from octsim import calibration as cal
profile = cal.literature_profile()          # seeded from normative OCT values
```

**From real segmented B-scans (the "learn better" path):**
```python
profile = cal.extract_profile(image, boundaries, depth_axis_um,
                              layer_names=[...], domain="amplitude")
# pool a corpus:
profile = cal.average_profiles([extract_profile(*scan) for scan in corpus])
profile.save("maestro_macula.json")
```

### The data contract for `extract_profile`

You supply, per B-scan:

| arg | shape | meaning |
| --- | --- | --- |
| `image` | `[n_depth, n_x]` | one real B-scan. `domain=` declares units: `"amplitude"` (linear `|recon|`, default), `"intensity"` (`|.|²`), or `"db"`. |
| `boundaries` | `[L+1, n_x]` | optical-depth (µm) of each layer boundary, top→bottom (`boundaries[0]` = ILM), on the same axis as `depth_axis_um`. |
| `depth_axis_um` | `[n_depth]` | depth (µm) of each row. |
| `layer_names` | `L` names | defaults to `NFL, GCL, IPL, INL, OPL, ONL, ELM, EZ, IZ, RPE, choroid`. |

The **boundaries are the only thing you must provide beyond the image** — i.e. a layer
segmentation. Options to get them:
- an existing segmenter (Topcon's own export, or an OCT layer-segmentation model);
- a public dataset that ships boundaries (e.g. Duke SD-OCT / DME, or annotated OCT sets);
- manual tracing for a handful of scans (even 5–10 pins the layer optics well).

No raw OCT ships with this repo. For **Maestro `.raw`**, reconstruct first (the
Line-Field-OCT / OCT-Super-Resolution loaders do this: DC-subtract → k-linearise with the
`.CLB` → FFT), then segment, then call `extract_profile` on the magnitude B-scan.

## Using a profile

```python
import numpy as np
from octsim import calibration as cal
m = cal.retina_from_profile(profile, np.random.default_rng(0),
                            location="macula", lateral_fov_um=6000.0)
# m is a RetinaModel; render it through the normal pipeline (generate.simulate / forward).
```

Optics (reflectivity, attenuation), curvature, and boundary sharpness come straight from the
data; thicknesses and the foveal pit are drawn around the profile's means for per-eye
variation. Labels are unaffected — the simulator still emits exact layer maps.

## Statistical shape model (geometry from data)

Beyond per-layer optics, the *geometry* (layer-thickness profiles, foveal-pit morphology) is
learned from real segmentations by a PCA `octsim.shape_model.LayerShapeModel` and sampled per
eye. Two models ship (raw data is never committed — only the tiny fitted `.npz`):

| bundled name | file | source | use |
| --- | --- | --- | --- |
| `normal` / `jhu` (**default**) | `retina_shape_jhu_hc.npz` (~90 KB) | JHU **healthy controls** (He 2019, 309 foveal B-scans) | normal-eye geometry |
| `duke` / `dme` | `retina_shape_duke.npz` (~80 KB) | Duke **DME** (Chiu 2015, 85 scans) | diabetic / edema geometry |

**The default is the healthy-eye (JHU) model** — DME eyes distort the foveal contour and their
fluid isn't a segmented class, so that geometry is unsafe as a normal prior. From the config /
GUI / API:

```python
from octsim import SimConfig, simulate
res = simulate(SimConfig(anatomy="macula"), seed=2)                       # normal (default)
res = simulate(SimConfig(anatomy="macula", shape_model="dme"), seed=2)    # diabetic geometry
res = simulate(SimConfig(anatomy="macula", shape_model="parametric"))     # hand-tuned pit
```

Or explicitly, geometry from data + optics from a profile, exact labels preserved:

```python
import numpy as np
from octsim import LayerShapeModel, retina_from_shape_model, literature_profile
sm = LayerShapeModel.load("octsim/data/retina_shape_jhu_hc.npz")
m = retina_from_shape_model(sm, np.random.default_rng(0), optics=literature_profile(),
                            location="macula", lateral_fov_um=6000.0)
```

Refit either model from the raw data (deterministic; produces the same `.npz`):

```bash
# JHU healthy controls — download OCT_Manual_Delineations-2018_June_29_b.zip from
# iacl.ece.jhu.edu (or the MEDIC portal), unzip, point --root at the delineation/ folder:
python scripts/fit_shape_model.py --dataset jhu_hc --root path/to/delineation \
    --out octsim/data/retina_shape_jhu_hc.npz
# Duke DME — download 2015_BOE_Chiu2.zip from people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm:
python scripts/fit_shape_model.py --dataset duke_dme --root path/to/2015_BOE_Chiu \
    --out octsim/data/retina_shape_duke.npz
```

The segmentation layers (Duke 7 / JHU 8) are split onto the simulator's 11 by the canonical
thickness ratios (`calibration._layer_split_map`); the choroid stays parametric. The same API
refits on **your Maestro segmentations** — write a loader that yields a `[N, n_layer, n_ecc]`
thickness stack (see `octsim/data/jhu_hc.py`) and call `LayerShapeModel.fit`.

## Learning from UNLABELED B-scans (texture, heterogeneity, reflectivity)

Segmentations tell us *where* the layers are; a pile of plain B-scans tells us how the retina
actually *looks*. `octsim/data/unlabeled.py` extracts, with a label-free ILM+RPE surface finder:
a **reflectivity template** (mean A-line warped onto a normalised ILM→RPE axis — its peaks are
the bright layers), the **speckle contrast** and grain, the intra-layer **heterogeneity**, the
noise floor, the retinal-thickness spread, and the **curvature sign**.

```python
from octsim import extract_texture, bundled_texture
from octsim.data import unlabeled as ul
prof = extract_texture(ul.load_images("path/to/NORMAL"))     # any OCT image folder, no labels
prof = bundled_texture()                                     # Srinivasan-2014 NORMAL, 1407 B-scans
```
```bash
python scripts/fit_texture.py --root path/to/NORMAL --out octsim/data/texture_srinivasan.json
```

The bundled default (`bundled_texture()`) is a **pooled multi-dataset prior** — 1739 NORMAL
B-scans across **two devices**: [Srinivasan-2014](http://www.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)
(Spectralis, 1407) + [OCTDL](https://zenodo.org/records/10839556) (Optovue NO class, 332),
combined with `pool_textures()` (weighted by B-scan count; `bundled_texture("srinivasan")` /
`"octdl"` load the singles). Both devices show the same band structure (NFL peak → ONL dip →
RPE peak); the pool averages their display/contrast differences.

This **validated the physics model**: measured speckle contrast **0.51** (≈ the fully-developed-
speckle Rayleigh value 0.52), **82 % of scans concave-up** (95 % in OCTDL alone — confirming the
posterior-pole bowl curvature), normal retinal thickness, and the correct bright/dark band
ordering (NFL & RPE brightest, ONL darkest). `layer_reflectivity_from_template(prof, names,
thicknesses)` reads relative per-layer brightness off the template — data-grounded optics
**without segmentation**. **Caveat:** the template lives in the log/dB *display* domain, so it
validates layer *ordering* and *shape* but its compressed ratios are not raw linear backscatter;
the speckle/heterogeneity/thickness/curvature numbers are directly comparable. Pool more with
`pool_textures([...])` — point `scripts/fit_texture.py` at Kermany OCT2017 NORMAL, OCTID, etc.

## Field of view and the optic nerve head

The healthy-eye shape model is fit out to **±2.6 mm eccentricity** (using the full width of each
foveal B-scan, not just the central pit), so wide scans get a real foveal→parafoveal→perifoveal
thickness profile (foveal ~250 µm rising to a ~320 µm perifoveal plateau, GCIP peaking
parafoveally, RNFL thickening outward).

`RetinaModel` carries **both** a fovea and an optic disc at their anatomical separation
(`disc_dx_um` ≈ 4.76 mm). When a scan is wide enough to reach the disc (≳ 7–8 mm) or is
ONH-centred, `cross_section` renders a **unified field**: the data-driven macula transitions into
the disc, where the layers terminate at Bruch's-membrane opening, the ILM cups, the RNFL rim
thickens, and a neural-canal fill replaces the neurosensory stack — the hallmark ONH morphology,
on the same B-scan as the fovea. Narrow macula scans keep the fast 11-layer path unchanged.

The **disc geometry is literature-parametric** (BMO half-width, cup depth, rim), not data-fit:
the one public ONH layer+disc dataset ([MGU-Net](https://github.com/Jiaxuan-Li/MGU-Net), 122
radial through-disc B-scans) is Cloudflare-gated (not reachable from this environment) and is
mixed-pathology (high myopia / peripapillary atrophy), so it is unsuitable as a healthy prior
anyway. When ONH segmentations are available locally, the same `LayerShapeModel` machinery can
calibrate the disc dimensions.

## Roadmap

- **Done:** per-layer optics + curvature + boundary sharpness from data; a PCA shape model that
  samples real foveal-pit / thickness geometry from a **healthy-eye** dataset (default, ±2.6 mm)
  or a DME one; a unified fovea+disc field so the ONH appears at wide FOV; all wired into the
  config/GUI with a parametric opt-out.
- **Next:** a foveal-vs-parafoveal sampling knob (bias the mode-1 latent); learn the global
  curvature + ONL bulge as modes; calibrate the disc from local ONH segmentations; extract
  linear-domain optics from Maestro `.raw` reconstructions.
