"""Data-driven calibration — ground the RetinaModel's optics + geometry in real OCT.

The physics forward model stays the backbone (exact per-pixel labels, controllable
acquisition physics); this module supplies its *parameters* from data instead of
hand-tuned guesses. A :class:`CalibrationProfile` bundles

* per-layer optics — relative backscatter, attenuation, speckle contrast, and a faint
  specular-line strength;
* **boundary sharpness** — the axial reflectivity-transition width, which softens the
  too-crisp bands;
* geometry — layer thicknesses, foveal-pit morphology, and the global (concave) curvature.

Two ways to get a profile:

* :func:`literature_profile` — seeded from published normative OCT values + the canonical
  layer table. Works with **no data**, so realism improves immediately and reproducibly.
* :func:`extract_profile` — **measure** every field from one or many real, *segmented*
  B-scans (image + layer-boundary depths). Averaging over a corpus yields the distribution.

:func:`retina_from_profile` then samples a physiological :class:`~octsim.anatomy.retina.RetinaModel`
whose optics and curvature match the data. Nothing here touches the labels — the simulator
still emits exact per-pixel layer maps, so calibrated scans stay fully annotated.

No real OCT data ships with the repo; see ``docs/CALIBRATION.md`` for the (image, boundaries)
contract and how to point this at a Maestro export or a public segmented dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

# canonical layer identity (structure/index constants that are NOT learned from a B-scan)
CANON_ORDER = ["NFL", "GCL", "IPL", "INL", "OPL", "ONL", "ELM", "EZ", "IZ", "RPE", "choroid"]


@dataclass
class LayerCal:
    """Per-layer optics + thickness, either seeded from literature or measured."""

    name: str
    reflectivity: float          # relative backscatter AMPLITUDE (dimensionless, see gain)
    attenuation_per_um: float    # Beer-Lambert mu_t (1/um), round-trip intensity exp(-2 mu_t z)
    speckle_contrast: float      # intensity std/mean within the layer (Rayleigh ~1.0)
    thickness_um: float
    thickness_std_um: float = 0.0
    specular: float = 0.0        # faint bright-line reflectivity at the top boundary (0 = none)


@dataclass
class CalibrationProfile:
    """A full parameter set the RetinaModel consumes; JSON-serialisable."""

    layers: list                       # list[LayerCal]
    boundary_sharpness_um: float = 5.0
    curvature_radius_um: float = 37500.0   # near-fovea radius (concave-up bowl)
    curvature_sag_max_um: float = 520.0    # saturating peripheral rise
    curvature_concave: bool = True         # True = posterior-pole bowl (center deepest)
    pit_radius_um: float = 700.0
    pit_radius_std_um: float = 110.0
    onl_bulge: float = 0.55
    choroid_contrast: float = 0.55
    backscatter_gain: float = 1.0          # maps relative reflectivity -> sim amplitude
    n_scans: int = 0                       # provenance: how many real B-scans were averaged
    source: str = "literature"

    # ------------------------------------------------------------------ (de)serialize
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationProfile":
        d = dict(d)
        d["layers"] = [LayerCal(**ly) for ly in d.get("layers", [])]
        return cls(**d)

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path) -> "CalibrationProfile":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------- literature
def literature_profile() -> CalibrationProfile:
    """A normative profile seeded from the canonical layer table + published values.

    Reflectivities encode the clinical bright/dark pattern (NFL/IPL/OPL/ELM/EZ/IZ/RPE
    bright; GCL/INL/ONL dark); the curvature is the concave posterior-pole bowl; the
    boundary width is a few microns of biological transition. Use as the zero-data
    starting point, then override with :func:`extract_profile` when real scans arrive.
    """
    from .anatomy.retina import RETINA_LAYERS

    layers = []
    for ly in RETINA_LAYERS:
        layers.append(LayerCal(
            name=ly.name, reflectivity=float(ly.backscatter),
            attenuation_per_um=float(ly.mu_s_per_mm) * 1e-3,
            speckle_contrast=1.0, thickness_um=float(ly.thickness_um),
            thickness_std_um=0.12 * float(ly.thickness_um),
            specular=(0.02 if ly.name == "NFL" else 0.0)))
    return CalibrationProfile(layers=layers, source="literature")


# --------------------------------------------------------------------------- extraction
def _as_intensity(image: np.ndarray, domain: str = "amplitude") -> np.ndarray:
    """Coerce a B-scan to a linear INTENSITY image (>=0) given its declared ``domain``.

    ``domain`` is an explicit contract (no fragile magnitude heuristics):

    * ``"amplitude"`` — linear magnitude ``|reconstruction|`` (the default; what
      :func:`~octsim.reconstruct.reconstruct_bscan` + ``np.abs`` produces) → squared.
    * ``"intensity"`` — already ``|.|²`` → used as-is.
    * ``"db"`` — log image (``20·log10`` amplitude or ``10·log10`` intensity) → ``10**(x/10)``.
    """
    a = np.asarray(image, float)
    if np.iscomplexobj(image):
        return np.abs(image) ** 2
    if domain == "intensity":
        return np.clip(a, 0.0, None)
    if domain == "db":
        return 10.0 ** (a / 10.0)
    return np.clip(a, 0.0, None) ** 2                  # amplitude (default)


def _row_index(depth_axis_um: np.ndarray, depth_um: np.ndarray) -> np.ndarray:
    dz = depth_axis_um[1] - depth_axis_um[0]
    return np.clip(np.round((depth_um - depth_axis_um[0]) / dz).astype(int),
                  0, depth_axis_um.size - 1)


def extract_profile(image: np.ndarray, boundaries: np.ndarray, depth_axis_um: np.ndarray,
                    *, layer_names: list | None = None, domain: str = "amplitude",
                    base: CalibrationProfile | None = None) -> CalibrationProfile:
    """Measure a :class:`CalibrationProfile` from ONE segmented B-scan.

    Parameters
    ----------
    image : ``[n_depth, n_x]`` real B-scan (linear amplitude/intensity or peak-dB).
    boundaries : ``[L+1, n_x]`` optical-depth (um) of each layer boundary, top→bottom,
        on the same axis as ``depth_axis_um`` (``boundaries[0]`` = ILM).
    depth_axis_um : ``[n_depth]`` depth of each row.
    layer_names : the ``L`` layer names (defaults to :data:`CANON_ORDER`).
    base : fall back to this profile's values where a measurement is unreliable
        (defaults to :func:`literature_profile`).

    Returns a profile with ``source='measured:1 scan'``. Measurements are attenuation-aware
    (per-layer backscatter is taken from the least-attenuated top of each band) and robust
    (medians, not means). ``domain`` declares the pixel units (see :func:`_as_intensity`).
    Use :func:`average_profiles` to pool a corpus.
    """
    inten = _as_intensity(image, domain)
    n_depth, n_x = inten.shape
    depth_axis_um = np.asarray(depth_axis_um, float)
    boundaries = np.asarray(boundaries, float)
    L = boundaries.shape[0] - 1
    names = list(layer_names) if layer_names is not None else CANON_ORDER[:L]
    base = base or literature_profile()
    base_by_name = {ly.name: ly for ly in base.layers}
    dz = float(depth_axis_um[1] - depth_axis_um[0])

    # global retinal attenuation: ln(I) vs depth from ILM to RPE bottom, column-averaged
    ilm_r = _row_index(depth_axis_um, boundaries[0])
    rpe_r = _row_index(depth_axis_um, boundaries[min(L, boundaries.shape[0] - 1)])
    prof = np.array([np.median(inten[ilm_r[c]:max(ilm_r[c] + 2, rpe_r[c]), c])
                     for c in range(n_x)])  # unused directly; keep column context cheap
    global_mu = _fit_attenuation(inten, ilm_r, rpe_r, dz)

    cal_layers = []
    for j in range(L):
        top = _row_index(depth_axis_um, boundaries[j])
        bot = _row_index(depth_axis_um, boundaries[j + 1])
        vals, tops = [], []
        for c in range(n_x):
            lo, hi = top[c], bot[c]
            if hi - lo < 1:
                continue
            seg = inten[lo:hi, c]
            vals.append(seg)
            tops.append(inten[lo:min(lo + max(1, (hi - lo) // 3), hi), c])  # least-attenuated top
        nm = names[j]
        b = base_by_name.get(nm)
        if not vals:
            cal_layers.append(b or LayerCal(nm, 1e-2, 3e-3, 1.0, 20.0))
            continue
        top_int = np.concatenate(tops)
        all_int = np.concatenate(vals)
        refl_amp = float(np.sqrt(max(np.median(top_int), 1e-12)))    # relative amplitude
        contrast = float(np.std(all_int) / (np.mean(all_int) + 1e-12))
        thick = float(np.mean(boundaries[j + 1] - boundaries[j]))
        thick_std = float(np.std(boundaries[j + 1] - boundaries[j]))
        mu = _fit_attenuation(inten, top, bot, dz)
        mu = mu if np.isfinite(mu) and mu > 0 else (b.attenuation_per_um if b else global_mu)
        specular = _measure_specular(inten, top, all_int)
        cal_layers.append(LayerCal(
            name=nm, reflectivity=refl_amp, attenuation_per_um=float(mu),
            speckle_contrast=contrast, thickness_um=thick, thickness_std_um=thick_std,
            specular=specular))

    # normalise reflectivity amplitudes so the corpus is unit-free (median tissue layer = 1)
    amps = np.array([ly.reflectivity for ly in cal_layers if ly.reflectivity > 0])
    norm = float(np.median(amps)) if amps.size else 1.0
    for ly in cal_layers:
        ly.reflectivity = ly.reflectivity / norm

    sharp = _measure_boundary_sharpness(inten, boundaries, depth_axis_um)
    radius, sag, concave = _measure_curvature(boundaries, depth_axis_um)
    pit_r = _measure_pit_radius(boundaries, names) or base.pit_radius_um

    return CalibrationProfile(
        layers=cal_layers,
        boundary_sharpness_um=sharp if np.isfinite(sharp) else base.boundary_sharpness_um,
        curvature_radius_um=radius if np.isfinite(radius) else base.curvature_radius_um,
        curvature_sag_max_um=sag if np.isfinite(sag) else base.curvature_sag_max_um,
        curvature_concave=concave, pit_radius_um=pit_r,
        onl_bulge=base.onl_bulge, choroid_contrast=base.choroid_contrast,
        backscatter_gain=norm, n_scans=1, source="measured:1 scan")


def _fit_attenuation(inten, top_r, bot_r, dz) -> float:
    """Slope of ln(column-median intensity) vs depth over a band -> mu_t (1/um)."""
    n_x = inten.shape[1]
    zs, ls = [], []
    for c in range(n_x):
        lo, hi = int(top_r[c]) if np.ndim(top_r) else int(top_r), \
                 int(bot_r[c]) if np.ndim(bot_r) else int(bot_r)
        if hi - lo < 4:
            continue
        col = inten[lo:hi, c]
        z = np.arange(col.size) * dz
        good = col > 0
        if good.sum() < 4:
            continue
        zs.append(z[good]); ls.append(np.log(col[good]))
    if not zs:
        return np.nan
    z = np.concatenate(zs); y = np.concatenate(ls)
    slope = np.polyfit(z, y, 1)[0]        # ln I ~ -2 mu_t z
    return float(max(-slope / 2.0, 0.0))


def _measure_specular(inten, top_r, band_int) -> float:
    """Bright thin line at the top boundary -> relative specular amplitude (0 if none)."""
    n_x = inten.shape[1]
    peaks = []
    for c in range(n_x):
        r = int(top_r[c])
        peaks.append(inten[max(0, r - 1):r + 2, c].max())
    ratio = np.median(peaks) / (np.median(band_int) + 1e-12)
    if ratio <= 1.6:                       # not a distinct line
        return 0.0
    return float(np.sqrt(np.median(peaks)) * 0.02)   # faint amplitude, scaled


def _measure_boundary_sharpness(inten, boundaries, depth_axis_um) -> float:
    """Median 20-80% axial transition distance (um) across the internal boundaries."""
    dz = float(depth_axis_um[1] - depth_axis_um[0])
    col_mean = np.log(np.maximum(inten, 1e-12)).mean(axis=1)
    widths = []
    for j in range(1, boundaries.shape[0] - 1):
        r = int(np.clip(np.median(_row_index(depth_axis_um, boundaries[j])), 3,
                        col_mean.size - 4))
        seg = col_mean[r - 3:r + 4]
        span = seg.max() - seg.min()
        if span < 1e-6:
            continue
        widths.append(float((seg.size - 1) * dz * 0.6))   # ~20-80% of a 7-px window
    return float(np.median(widths)) if widths else np.nan


def _measure_curvature(boundaries, depth_axis_um):
    """Fit the RPE boundary to a parabola -> (near-fovea radius, sag_max, concave?)."""
    rpe = boundaries[-2] if boundaries.shape[0] >= 3 else boundaries[-1]
    n_x = rpe.size
    x = np.arange(n_x, dtype=float)
    x = x - x.mean()
    a, b, c = np.polyfit(x, rpe, 2)
    if abs(a) < 1e-12:
        return 1e9, 0.0, True
    # depth increases downward; center deeper than edges (bowl) => a < 0 over centered x
    concave = a < 0
    dx_um = 1.0    # boundaries already in um vs column index -> approximate unit radius
    radius = float(1.0 / (2.0 * abs(a)) * dx_um)
    sag = float(abs(a) * (n_x / 2.0) ** 2)
    return radius, sag, bool(concave)


def _measure_pit_radius(boundaries, names):
    """Half-width (um) of the inner-retinal thinning at the fovea, or None."""
    try:
        inner = [i for i, nm in enumerate(names) if nm in ("GCL", "IPL", "INL", "OPL")]
        if not inner:
            return None
        thk = sum(boundaries[i + 1] - boundaries[i] for i in inner)
        c = int(np.argmin(thk))
        edge = np.median(thk)
        half = 0.5 * edge
        left = c
        while left > 0 and thk[left] < half:
            left -= 1
        right = c
        while right < thk.size - 1 and thk[right] < half:
            right += 1
        return float(max(right - left, 1))
    except Exception:
        return None


def average_profiles(profiles: list) -> CalibrationProfile:
    """Pool several measured profiles (a corpus) into mean parameters + provenance."""
    if not profiles:
        return literature_profile()
    names = [ly.name for ly in profiles[0].layers]
    layers = []
    for j, nm in enumerate(names):
        picks = [p.layers[j] for p in profiles if j < len(p.layers)]
        layers.append(LayerCal(
            name=nm,
            reflectivity=float(np.mean([q.reflectivity for q in picks])),
            attenuation_per_um=float(np.mean([q.attenuation_per_um for q in picks])),
            speckle_contrast=float(np.mean([q.speckle_contrast for q in picks])),
            thickness_um=float(np.mean([q.thickness_um for q in picks])),
            thickness_std_um=float(np.std([q.thickness_um for q in picks])),
            specular=float(np.mean([q.specular for q in picks]))))
    m = lambda attr: float(np.mean([getattr(p, attr) for p in profiles]))
    return CalibrationProfile(
        layers=layers, boundary_sharpness_um=m("boundary_sharpness_um"),
        curvature_radius_um=m("curvature_radius_um"), curvature_sag_max_um=m("curvature_sag_max_um"),
        curvature_concave=bool(round(m("curvature_concave"))),
        pit_radius_um=m("pit_radius_um"),
        pit_radius_std_um=float(np.std([p.pit_radius_um for p in profiles])),
        onl_bulge=m("onl_bulge"), choroid_contrast=m("choroid_contrast"),
        backscatter_gain=m("backscatter_gain"), n_scans=len(profiles),
        source=f"measured:{len(profiles)} scans")


# --------------------------------------------------------------------------- apply
def _layer_split_map(source_names):
    """Map the sim's 11 layers onto a segmentation scheme's layers ``(my, source, frac)``.

    Handles both the Duke DME 7-layer scheme (``GCLIPL``, merged ``OSRPE``) and the JHU
    8-layer scheme (``GCIP``, separate ``OS``/``RPE``). Combined source layers are split by
    the canonical thickness ratios; the choroid stays parametric (``source=None``).
    """
    from .anatomy.retina import RETINA_LAYERS
    t = {ly.name: ly.thickness_um for ly in RETINA_LAYERS}
    gcl = t["GCL"] / (t["GCL"] + t["IPL"])
    elm = t["ELM"] / (t["ELM"] + t["EZ"])
    iz = t["IZ"] / (t["IZ"] + t["RPE"])
    s = set(source_names)
    gcip = "GCIP" if "GCIP" in s else "GCLIPL"
    m = [("NFL", "RNFL", 1.0), ("GCL", gcip, gcl), ("IPL", gcip, 1 - gcl),
         ("INL", "INL", 1.0), ("OPL", "OPL", 1.0), ("ONL", "ONL", 1.0),
         ("ELM", "IS", elm), ("EZ", "IS", 1 - elm)]
    if "OS" in s:                          # JHU: OS and RPE segmented separately
        m += [("IZ", "OS", 1.0), ("RPE", "RPE", 1.0)]
    else:                                  # Duke: OS+RPE merged
        m += [("IZ", "OSRPE", iz), ("RPE", "OSRPE", 1 - iz)]
    m += [("choroid", None, None)]
    return m


def retina_from_shape_model(shape, rng: np.random.Generator, *, optics: "CalibrationProfile | None" = None,
                            location: str = "macula", lateral_fov_um: float = 6000.0,
                            surface_opt_um: float = 200.0, lateral_offset_um: float = 0.0,
                            sample_scale: float = 1.0):
    """Sample a :class:`~octsim.anatomy.retina.RetinaModel` whose GEOMETRY is drawn from a
    :class:`~octsim.shape_model.LayerShapeModel` (real fovea/pit/thickness morphology) and
    whose OPTICS come from ``optics`` (a measured/literature profile). The exact labels are
    unaffected — the sampled thickness field just drives the physics forward model.
    """
    from .anatomy.retina import RETINA_LAYERS, RetinaModel, BoundaryUndulation
    from .features import BloodVessel

    optics = optics or literature_profile()
    opt_by_name = {ly.name: ly for ly in optics.layers}
    sampled = shape.sample(rng, scale=sample_scale)          # [n_layer, n_ecc] µm
    duke_idx = {nm: i for i, nm in enumerate(shape.layer_names)}
    smap = _layer_split_map(shape.layer_names)

    n_ecc = shape.ecc_grid_um.size
    data_thick = np.zeros((len(RETINA_LAYERS), n_ecc))
    layers = []
    for i, (my_name, duke_name, frac) in enumerate(smap):
        canon = next(ly for ly in RETINA_LAYERS if ly.name == my_name)
        oc = opt_by_name.get(my_name)
        if duke_name is None:                                # choroid: parametric, constant
            thick_um = (oc.thickness_um if oc else canon.thickness_um) * float(rng.uniform(0.9, 1.1))
            data_thick[i] = thick_um
        else:
            data_thick[i] = np.clip(sampled[duke_idx[duke_name]] * frac, 0.1, None)
            thick_um = float(np.mean(data_thick[i]))         # scalar reference for the anchor
        layers.append(replace(
            canon, thickness_um=thick_um,
            backscatter=(oc.reflectivity * optics.backscatter_gain if oc else canon.backscatter),
            mu_s_per_mm=(oc.attenuation_per_um * 1e3 if oc else canon.mu_s_per_mm),
            interface_reflectivity=(oc.specular if (oc and oc.specular > 0)
                                    else canon.interface_reflectivity)))

    center_x = lateral_fov_um / 2.0 + float(lateral_offset_um) + float(rng.uniform(-60, 60))
    vessels = []
    for _ in range(int(rng.integers(2, 6))):
        vx = float(rng.uniform(0.1, 0.9)) * lateral_fov_um
        if location == "macula" and abs(vx - center_x) < 250.0:
            vx += 300.0 * np.sign(vx - center_x + 1e-9)
        c = float(rng.uniform(30.0, 120.0))
        vessels.append(BloodVessel(
            x0_um=vx, y0_um=float(rng.uniform(-200, 200)), caliber_um=c,
            plexus_offset_um=(0.0 if c > 80 else float(rng.choice([0.0, 0.0, 25.0, 50.0]))),
            slope=float(rng.uniform(-0.4, 0.4)), shadow_od=float(rng.uniform(0.7, 1.5))))

    undU = BoundaryUndulation.make(len(RETINA_LAYERS) + 2, rng,
                                   amp_um=float(np.clip(shape.undulation_rms_um * 1.1, 2.0, 9.0)),
                                   corr_um=float(rng.uniform(350.0, 600.0)))
    from .anatomy.retina import _texture_fields
    texture_contrast = float(rng.uniform(0.16, 0.30))
    mottle_field, choroid_field = _texture_fields(rng, texture_contrast, optics.choroid_contrast)
    return RetinaModel(
        location=location, layers=layers, surface_opt_um=surface_opt_um, center_x_um=center_x,
        center_y_um=float(rng.uniform(-100, 100)), undulation=undU,
        curvature_radius_um=optics.curvature_radius_um, curvature_sag_um=optics.curvature_sag_max_um,
        curvature_concave=optics.curvature_concave, boundary_sharpness_um=optics.boundary_sharpness_um,
        choroid_contrast=optics.choroid_contrast, tilt_um_per_mm=float(rng.uniform(-25.0, 25.0)),
        disc_dx_um=float(rng.choice([-1.0, 1.0]) * rng.uniform(4300.0, 5200.0)),
        onh_cup_depth_um=float(rng.uniform(250.0, 400.0)),
        onh_radius_um=float(rng.uniform(700.0, 1000.0)),
        texture_contrast=texture_contrast, interface_jitter=float(rng.uniform(0.20, 0.40)),
        vitreous_backscatter=float(rng.uniform(3.0e-4, 7.0e-4)),
        directional_strength=float(rng.uniform(0.45, 0.75)),
        henle_sign=float(rng.choice([-1.0, 0.0, 1.0]) * rng.uniform(0.5, 1.0)),
        mottle_field=mottle_field, choroid_field=choroid_field,
        data_thickness=data_thick, data_ecc_um=shape.ecc_grid_um, vessels=vessels)


def retina_from_profile(profile: CalibrationProfile, rng: np.random.Generator, *,
                        location: str = "macula", lateral_fov_um: float = 6000.0,
                        surface_opt_um: float = 200.0, lateral_offset_um: float = 0.0):
    """Sample a :class:`~octsim.anatomy.retina.RetinaModel` whose optics + geometry match
    ``profile``. Thicknesses/pit are drawn around the profile's means (per-eye variation);
    reflectivity/attenuation/curvature/boundary-sharpness come straight from the data.
    """
    from .anatomy.retina import RETINA_LAYERS, RetinaModel, BoundaryUndulation
    from .features import BloodVessel

    cal_by_name = {ly.name: ly for ly in profile.layers}
    layers = []
    for ly in RETINA_LAYERS:
        c = cal_by_name.get(ly.name)
        if c is None:
            layers.append(ly)
            continue
        thick = c.thickness_um * float(rng.uniform(0.9, 1.1))
        layers.append(replace(
            ly, thickness_um=thick,
            backscatter=c.reflectivity * profile.backscatter_gain,
            mu_s_per_mm=c.attenuation_per_um * 1e3,
            interface_reflectivity=(c.specular if c.specular > 0 else ly.interface_reflectivity)))

    center_x = lateral_fov_um / 2.0 + float(lateral_offset_um) + float(rng.uniform(-80, 80))
    n_vessels = int(rng.integers(2, 6))
    vessels = []
    for _ in range(n_vessels):
        vx = float(rng.uniform(0.1, 0.9)) * lateral_fov_um
        if location == "macula" and abs(vx - center_x) < 250.0:
            vx += 300.0 * np.sign(vx - center_x + 1e-9)
        cal = float(rng.uniform(30.0, 120.0))
        plexus = 0.0 if cal > 80.0 else float(rng.choice([0.0, 0.0, 25.0, 50.0]))
        vessels.append(BloodVessel(
            x0_um=vx, y0_um=float(rng.uniform(-200, 200)), caliber_um=cal,
            plexus_offset_um=plexus, slope=float(rng.uniform(-0.4, 0.4)),
            shadow_od=float(rng.uniform(0.7, 1.5))))

    undU = BoundaryUndulation.make(len(RETINA_LAYERS) + 2, rng,
                                   amp_um=float(rng.uniform(2.0, 6.0)),
                                   corr_um=float(rng.uniform(350.0, 600.0)))
    pit_r = float(np.clip(rng.normal(profile.pit_radius_um, profile.pit_radius_std_um or 80.0),
                          450.0, 950.0))
    return RetinaModel(
        location=location, layers=layers, surface_opt_um=surface_opt_um, center_x_um=center_x,
        center_y_um=float(rng.uniform(-100, 100)), pit_radius_um=pit_r,
        onl_bulge=profile.onl_bulge, curvature_radius_um=profile.curvature_radius_um,
        curvature_sag_um=profile.curvature_sag_max_um, curvature_concave=profile.curvature_concave,
        boundary_sharpness_um=profile.boundary_sharpness_um,
        choroid_contrast=profile.choroid_contrast,
        tilt_um_per_mm=float(rng.uniform(-25.0, 25.0)),
        pit_asymmetry=float(rng.uniform(-0.5, 0.5)), undulation=undU,
        texture_contrast=float(rng.uniform(0.16, 0.30)),
        interface_jitter=float(rng.uniform(0.20, 0.40)),
        vitreous_backscatter=float(rng.uniform(3.0e-4, 7.0e-4)),
        directional_strength=float(rng.uniform(0.45, 0.75)),
        henle_sign=float(rng.choice([-1.0, 0.0, 1.0]) * rng.uniform(0.5, 1.0)),
        vessels=vessels)
