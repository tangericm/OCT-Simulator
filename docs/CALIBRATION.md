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

## Roadmap

- **Now:** per-layer optics + curvature + boundary sharpness from data; literature bootstrap.
- **Next:** a PCA/statistical-shape model over segmented boundaries so the *geometry* (thickness
  profiles, pit morphology, undulation spectrum) is sampled from the data distribution rather
  than parametric priors — the natural extension of `average_profiles`.
