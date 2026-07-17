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
eye. Three models ship (raw data is never committed — only the tiny fitted `.npz`):

| bundled name | file | source | use |
| --- | --- | --- | --- |
| `normal` / `octa500` (**default**) | `retina_shape_octa500.npz` (~120 KB) | OCTA-500 **normal eyes** (92 real 3D macular volumes) | normal-eye geometry, **deep physiologic pit** |
| `jhu` / `hc` | `retina_shape_jhu_hc.npz` (~90 KB) | JHU **healthy controls** (He 2019, 309 foveal B-scans) | normal geometry, shallower (diluted) pit |
| `duke` / `dme` | `retina_shape_duke.npz` (~80 KB) | Duke **DME** (Chiu 2015, 85 scans) | diabetic / edema geometry |

**The default is the OCTA-500 healthy-eye model.** It is fit from real *3D* macular cubes (dense,
fovea-centred), so its foveal pit is deep and physiologic (rel-depth 0.63) — see the OCTA-500
section below for how it is built and why it replaced the JHU model (whose sparse 49-B-scan
volumes gave a parafovea-diluted, too-shallow pit of ~0.86). The JHU and Duke models remain
selectable. From the config / GUI / API:

```python
from octsim import SimConfig, simulate
res = simulate(SimConfig(anatomy="macula"), seed=2)                        # normal (OCTA-500, default)
res = simulate(SimConfig(anatomy="macula", shape_model="jhu"), seed=2)     # JHU normal (shallower pit)
res = simulate(SimConfig(anatomy="macula", shape_model="dme"), seed=2)     # diabetic geometry
res = simulate(SimConfig(anatomy="macula", shape_model="parametric"))      # hand-tuned pit
```

Or explicitly, geometry from data + optics from a profile, exact labels preserved:

```python
import numpy as np
from octsim import LayerShapeModel, retina_from_shape_model, literature_profile
sm = LayerShapeModel.load("octsim/data/retina_shape_octa500.npz")
m = retina_from_shape_model(sm, np.random.default_rng(0), optics=literature_profile(),
                            location="macula", lateral_fov_um=6000.0)
```

Refit any model from the raw data (deterministic; produces the same `.npz`):

```bash
# OCTA-500 normal eyes (the default) — from the release tree or zip (see the section below):
python scripts/fit_octa500.py --root path/to/OCTA-500        # writes retina_shape_octa500.npz + more
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

The bundled default (`bundled_texture()`) is a **pooled multi-dataset prior** — ~4900 NORMAL
B-scans across **four datasets / multiple devices**:
[Srinivasan-2014](http://www.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm) (Spectralis, 1407),
[OCTDL](https://zenodo.org/records/10839556) (Optovue, 332), [OCTID](https://borealisdata.ca)
(206), and OCT-C8 / RetinalOCT (3000) — combined with `pool_textures()` (weighted by B-scan
count; `bundled_texture("srinivasan"|"octdl"|"octid")` load the singles). They all show the same
band structure (NFL peak → ONL dip → RPE peak); the pool averages device/display differences.

**Disease texture is kept separate.** OCTID and OCT-C8 disease categories (macular hole, CSR,
AMD, CNV, DME, DR, drusen) carry realistic speckle/heterogeneity but *distorted geometry* (fluid,
atrophy, edema), so they are **excluded from the healthy default** and bundled by name
(`bundled_texture("dr"|"amd"|"mh"|"csr"|"cnv"|"dme"|"drusen")`) for deliberately generating
degraded / pathological-looking corpora. Across all of them the measured speckle contrast stays
~0.45–0.53 (Rayleigh) and the curvature stays predominantly concave — an independent, broad
cross-device validation of the physics.

This **validated the physics model**: measured speckle contrast **0.51** (≈ the fully-developed-
speckle Rayleigh value 0.52), **82 % of scans concave-up** (95 % in OCTDL alone — confirming the
posterior-pole bowl curvature), normal retinal thickness, and the correct bright/dark band
ordering (NFL & RPE brightest, ONL darkest). `layer_reflectivity_from_template(prof, names,
thicknesses)` reads relative per-layer brightness off the template — data-grounded optics
**without segmentation**. **Caveat:** the template lives in the log/dB *display* domain, so it
validates layer *ordering* and *shape* but its compressed ratios are not raw linear backscatter;
the speckle/heterogeneity/thickness/curvature numbers are directly comparable. Pool more with
`pool_textures([...])` — point `scripts/fit_texture.py` at Kermany OCT2017 NORMAL, OCTID, etc.

## Volumetric validation: OCTA-500 (real 3D macular cubes)

Every dataset above is a pile of independent 2D B-scans. [OCTA-500](https://ieee-dataport.org/open-access/octa-500)
(Li et al., *Medical Image Analysis* 2024) is the first **volumetric** reference — dense,
fovea-centred 3 mm cubes (304 B-scans × 304 A-lines × 640 px, Optovue RTVue-XR) **with a 3D
layer segmentation** (`GT_Layers[6, x, y]` en-face height maps) and en-face FAZ / vessel masks.
`octsim/data/octa500.py` distils the OCTA_3mm subset; `scripts/fit_octa500.py` reproduces the
two bundled artifacts (the ~4.5 GB raw is never committed).

**Normal-eye filter.** OCTA_3mm is ~80 % normal / 20 % disease (AMD, DR, CNV) and the partial
release used here ships no per-subject disease sheet, so pathology is flagged straight from the
segmented geometry (`is_normal_geometry`): edema/CNV thicken the fovea and obliterate or invert
the pit, drusen roughen the outer boundary. On the 100-eye sample this excludes 10 eyes; the
kept 90 are tight and healthy (foveal 216 ± 18 µm, pit −8.7 ± 2.4 µm).

**What it validated.** The measured normal macula (`octsim.data.octa500.bundled_macula()`,
stored in `octa500_macula.json`) confirms: the posterior pole is **100 % concave-up** (the
strongest confirmation yet of the bowl curvature); a normal **FAZ radius ~324 µm** (median area
0.33 mm²), now the simulator's `retina.FAZ_RADIUS_UM` avascular-zone constant (was a hand-picked
250 µm); and a perifoveal plateau ~338 µm, consistent with the physics-model thickness.

**What it fixed — the default foveal pit (now OCTA-500).** OCTA-500's dense-volume fovea has a
**deep pit** (central 212 µm vs a 338 µm plateau, **rel-depth 0.63**, pit depth ~125 µm). The
former default (the JHU model) rendered a **shallow** pit (rel-depth ~0.86, its fitted mean only
central 250 / plateau 282 µm → a 32 µm pit). Both are valid *healthy* data — the difference is
**sampling fidelity, not biology**: the foveal pit is only ~1 mm wide, so JHU's sparse 49-B-scan,
~6 mm volumes under-sample it (the pit falls between slices / off-centre, and the aggregation
mixes foveal with parafoveal columns), diluting the mean pit to a non-physiologic 32 µm; OCTA-500's
304-B-scan dense cubes, radially averaged about each eye's foveal minimum, resolve the true
~125 µm pit. (Two other differences only shift absolute µm, not the pit ratio: Spectralis-vs-RTVue
calibration, and JHU tracing ILM→RPE vs OCTA-500 ILM→BM — hence the 282 vs 338 plateaus.)

So the **default normal shape model is now `retina_shape_octa500.npz`**, fit from the volumetric
segmentation. OCTA-500's 6 surfaces are not confidently named histologically, but the OPL/ONL
divide is unambiguous from *physiology*: surface 2 is where the inner compartment (ILM→S2)
collapses at the fovea (190→30 µm) while the outer (S2→BM) *bulges* (150→189 µm, the
photoreceptor/ONL foveal bulge). We split the real ILM→BM at that divide and sub-divide each
compartment into the JHU 8-layer scheme by the sim's canonical parafoveal ratios — so total
thickness and pit depth come from OCTA-500 while `calibration._layer_split_map` maps it onto the
11 sim layers unchanged. Rendered pit rel-depth is now ~0.65 (was ~0.86); `test_octa500.py` pins
it. **JHU stays selectable** (`shape_model="jhu"`): it reaches ±2.6 mm (vs OCTA-500's ±1.5 mm,
where the flat plateau is held) and has real 8-surface per-layer structure — a future hybrid could
take OCTA-500's pit inside ±1.5 mm and JHU's finer layers / wider field outside.

**Appearance.** The RTVue-XR *structural* B-scan is the average of the repeated OCTA acquisitions,
so its speckle contrast is **0.34** (≈2.3 effective frames) — a genuinely different, frame-averaged
regime. It is bundled by name (`bundled_texture("octa500")`) but **kept out of the single-frame
healthy default** (which stays at ~0.49, the fully-developed-speckle value); the drop is itself a
cross-check of the averaging model (0.52/0.34 ≈ √2.3).

## Field of view and the optic nerve head

The healthy-eye shape models are sampled on a **±2.6 mm eccentricity** grid, so wide scans get a
real foveal→parafoveal→perifoveal thickness profile. The default OCTA-500 model has real data to
**±1.5 mm** (the 3 mm cube) with the perifoveal plateau (~338 µm, already flat by ~1.05 mm) held
constant beyond that; the JHU model carries real data all the way to ±2.6 mm (foveal ~250 µm
rising to a ~282 µm plateau, GCIP peaking parafoveally, RNFL thickening outward).

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
  samples real foveal-pit / thickness geometry — the **default is now the OCTA-500 volumetric
  model** (real deep pit, rel-depth 0.63), with the JHU and DME models selectable; a unified
  fovea+disc field so the ONH appears at wide FOV; a multi-device texture prior + separate disease
  profiles; and the OCTA-500 reference (real 3D macular cubes) that validated the concave bowl, set
  the FAZ radius, and **refit the default foveal pit**; all wired into the config/GUI with a
  parametric opt-out.
- **Next (highest-value):** a **hybrid macular model** — OCTA-500's deep pit inside ±1.5 mm blended
  with JHU's finer 8-surface per-layer structure and wider ±2.6 mm field. Then: a
  foveal-vs-parafoveal sampling knob; learn the global curvature + ONL bulge as modes; calibrate the
  disc from local ONH segmentations; linear-domain optics from Maestro `.raw`.
