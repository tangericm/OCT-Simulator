"""Tier-2 analysis-by-synthesis objective for the ANCHOR·FIT·JUDGE loop.

One evaluation ``D(θ)``:

  θ (≈20 numbers) → decode → a :class:`~octsim.calibration.CalibrationProfile` (tissue optics
  + appearance) + render globals (reference_amp / falloff / noise / floor) → render **N common
  random-number B-scans** through the frozen instrument → :func:`octsim.data.unlabeled.extract_texture`
  → distance to the real target :class:`~octsim.data.unlabeled.TextureProfile`.

Only the domain-invariant texture statistics enter ``D`` (real B-scans are 8-bit log display,
synthetic are dB — the shared per-image percentile normalisation in ``extract_texture`` keeps
only *relative* structure comparable). ``D`` (see ``docs/CALIBRATION_ROUTINE.md`` §4a):

    D(θ) = w_t·MSE(reflectivity_template[120])
         + w_h·χ²(intensity_hist[32])
         + w_s·Σ relerr²(speckle_contrast, axial/lateral_grain_um, heterogeneity, total_retina_um_mean)
         + w_reg·R(θ)                       # ‖θ − θ_lit‖² in normalised param space (anti-overfit)

The optimiser (``scipy.optimize.differential_evolution``) lives in
``scripts/calibrate_optics.py``; this module is the picklable, side-effect-free objective it
drives, and the injected-recovery test drives the same code with a tiny budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import generate
from .calibration import CalibrationProfile, literature_profile
from .config import SimConfig
from .data import unlabeled

# --------------------------------------------------------------------------- θ parameterisation
# Backscatter multiplier bands: the canonical 11 layers grouped into the depth ZONES the
# label-free texture objective can actually distinguish. Fitting all 6 clinical bands is
# over-parameterised — the outer bright complex (ELM/EZ/IZ/RPE) is unresolvable by the aggregate
# template, so independent EZ/IZ↔RPE multipliers swap freely (verified degeneracy). Merging them
# into one "outer" band (intra-band clinical ratios preserved by the shared multiplier) removes
# that flat direction while keeping the inner/nuclear/outer/choroid contrast the objective CAN see.
BACKSCATTER_BANDS = [
    ("NFL", ["NFL"]),
    ("inner", ["GCL", "IPL", "INL", "OPL"]),
    ("ONL", ["ONL"]),
    ("outer", ["ELM", "EZ", "IZ", "RPE"]),
    ("choroid", ["choroid"]),
]
# Attenuation multiplier bands (4): coarser depth zones.
ATTENUATION_BANDS = [
    ("inner", ["NFL", "GCL", "IPL", "INL", "OPL"]),
    ("nuclear", ["ONL", "ELM"]),
    ("outer", ["EZ", "IZ", "RPE"]),
    ("choroid", ["choroid"]),
]


@dataclass(frozen=True)
class ParamSpec:
    """One θ dimension: name, (lo, hi) bound, whether it's a log10-multiplier, and θ_lit."""

    name: str
    lo: float
    hi: float
    log: bool = False          # True: value is a log10 multiplier (0 = literature ×1)
    lit: float = 0.0           # the literature/anchor value in this coordinate
    frozen: bool = False       # measured & frozen (Tier-1): excluded from the fit by default


def theta_space() -> list:
    """The ordered list of ~20 :class:`ParamSpec` the optimiser searches."""
    P = []
    # ONL + choroid backscatter were frozen when the objective was contrast-BLIND (they swapped
    # freely). The un-normalised contrast term now constrains the dark-layer brightness (they set
    # the dynamic range), so they are identifiable again and fit. The tissue attenuations stay frozen
    # (a subtle depth slope the measured instrument falloff already dominates; still weakly identifiable).
    _BS_FROZEN = set()
    for name, _ in BACKSCATTER_BANDS:                 # backscatter multipliers, 0.2–5× (log)
        P.append(ParamSpec(f"bs_{name}", np.log10(0.2), np.log10(5.0), log=True, lit=0.0,
                           frozen=(name in _BS_FROZEN)))
    # attenuation multipliers (log). The DEEP bands (outer, choroid) are UNFROZEN and their upper
    # bound widened to 6× — they are the honest lever for the depth-signal fade (real deep-slab/peak
    # ~0.05-0.17 vs sim ~0.27); the instrument falloff is nearly flat across the shallow retina and
    # stays frozen. The top zones (inner, nuclear) stay frozen (the measured falloff already dominates
    # that shallow slope and they are only weakly identifiable). Gradient comes from D_falloff.
    _AT_FROZEN = {"inner", "nuclear"}
    for name, _ in ATTENUATION_BANDS:
        hi = np.log10(3.0) if name in _AT_FROZEN else np.log10(6.0)
        P.append(ParamSpec(f"at_{name}", np.log10(0.3), hi, log=True, lit=0.0,
                           frozen=(name in _AT_FROZEN)))
    P += [
        ParamSpec("boundary_sharpness_um", 3.0, 15.0, lit=8.0),
        ParamSpec("choroid_contrast", 0.10, 0.60, lit=0.32),
        ParamSpec("texture_contrast", 0.08, 0.45, lit=0.23),
        ParamSpec("interface_jitter", 0.10, 0.60, lit=0.30),
        ParamSpec("vitreous_backscatter", np.log10(1e-4), np.log10(3e-3), log=True,
                  lit=float(np.log10(5e-4))),
        ParamSpec("directional_strength", 0.30, 0.90, lit=0.60),
        # spherical-cap eye radius R_eye driving FOV-dependent curvature (sag = R - sqrt(R²-(FOV/2)²)).
        # Fit against wide-FOV corpora where the global bowl dominates (the 3 mm curvature is foveal-pit,
        # a separate contour). lit MUST equal the retina/JSON default (12000) so the label map stays
        # bit-identical at the anchor. Identified by the geometric D_curve term (intensity-blind).
        ParamSpec("R_eye_um", 8000.0, 40000.0, lit=12000.0),
        # conic constant K of the aspheric posterior-pole sag (peripheral shape at fixed apex R).
        # K=0 sphere (anchor, label-invariant), K<-1 flattens the periphery (real wide-FOV B-scans are
        # flatter than a sphere — device distortion-correction + aspheric anatomy, per deep-research).
        # Lets DE match BOTH the near-fovea curvature (R) and the wide-FOV flattening (K) at once.
        ParamSpec("curvature_conic", -4.0, 1.0, lit=0.0),
        # K-distribution speckle shape M (effective #scatterers/resolution cell), fit in log10.
        # M→hi is numerically Rayleigh (today's look, the literature anchor); low M gives the
        # super-Rayleigh heavy tail real OCT shows in sparse layers — the lever for the intensity
        # histogram (D_hist), which dominates the residual. Lower bound M=2 caps the pre-smoothing
        # intensity contrast at sqrt(1+2/2)=1.41 so the speckle-contrast gate stays reachable.
        ParamSpec("speckle_shape", np.log10(2.0), np.log10(50.0), log=True,
                  lit=float(np.log10(50.0))),
        ParamSpec("reference_amp", 2.0, 12.0, lit=4.0),
        # falloff is MEASURED from the mirror roll-off (~1960 µm) and FROZEN — a fit must not
        # absorb an instrument parameter (guardrail). Kept in θ (so it's rendered) but excluded
        # from the free set by default via `frozen`.
        ParamSpec("falloff_6db_um", 800.0, 3000.0, lit=1960.0, frozen=True),
        # the histogram term (rebuilt from a correctly-segmented real target with a clean CNR~23 dB
        # background) now pins noise to the right level; the earlier hard cap was a band-aid for a
        # corrupted flat target and is no longer needed. Kept a physical upper bound.
        ParamSpec("noise_level", np.log10(0.2), np.log10(2.5), log=True, lit=0.0),
        # upper bound tightened -45→-54: a high floor brightens the deep background and fights the
        # measured depth fade (real deep-slab/peak ~0.05-0.17). Keep the floor genuinely low.
        ParamSpec("floor_db", -70.0, -54.0, lit=-60.0),
    ]
    return P


def bounds() -> list:
    return [(p.lo, p.hi) for p in theta_space()]


def theta_literature() -> np.ndarray:
    """θ that decodes to the literature profile + default globals (the anchor / R(θ)=0 point)."""
    return np.array([p.lit for p in theta_space()], dtype=float)


def _index(name: str) -> int:
    for i, p in enumerate(theta_space()):
        if p.name == name:
            return i
    raise KeyError(name)


@dataclass
class RenderGlobals:
    """The non-tissue θ components injected into the :class:`SimConfig` / dB clip."""

    reference_amp: float = 4.0
    falloff_6db_um: float = 1200.0
    noise_level: float = 1.0
    floor_db: float = -60.0


def decode(theta: np.ndarray, base: CalibrationProfile | None = None):
    """θ vector → (:class:`CalibrationProfile`, :class:`RenderGlobals`).

    Backscatter/attenuation are per-band log10 multipliers on the literature layer values
    (intra-band ratios preserved); the appearance + curvature fields are set directly.
    """
    theta = np.asarray(theta, dtype=float)
    P = theta_space()
    val = {p.name: theta[i] for i, p in enumerate(P)}
    base = base or literature_profile()

    bs_mult = {b: 10.0 ** val[f"bs_{b}"] for b, _ in BACKSCATTER_BANDS}
    at_mult = {b: 10.0 ** val[f"at_{b}"] for b, _ in ATTENUATION_BANDS}
    bs_by_layer, at_by_layer = {}, {}
    for band, names in BACKSCATTER_BANDS:
        for nm in names:
            bs_by_layer[nm] = bs_mult[band]
    for band, names in ATTENUATION_BANDS:
        for nm in names:
            at_by_layer[nm] = at_mult[band]

    layers = []
    for ly in base.layers:
        layers.append(replace(
            ly,
            reflectivity=ly.reflectivity * bs_by_layer.get(ly.name, 1.0),
            attenuation_per_um=ly.attenuation_per_um * at_by_layer.get(ly.name, 1.0)))

    prof = replace(
        base, layers=layers,
        boundary_sharpness_um=float(val["boundary_sharpness_um"]),
        choroid_contrast=float(val["choroid_contrast"]),
        texture_contrast=float(val["texture_contrast"]),
        interface_jitter=float(val["interface_jitter"]),
        vitreous_backscatter=float(10.0 ** val["vitreous_backscatter"]),
        directional_strength=float(val["directional_strength"]),
        curvature_radius_um=float(val["R_eye_um"]),   # apex radius drives FOV-dependent sag
        curvature_conic=float(val["curvature_conic"]),  # conic K: peripheral flatten/steepen
        # map the top of the M range to the None sentinel so the Rayleigh end is an EXACT no-draw
        # render (byte-identical to legacy); below it, decode M = 10**log10(M).
        speckle_shape=(None if 10.0 ** val["speckle_shape"] >= 49.0
                       else float(10.0 ** val["speckle_shape"])),
        source="fitted")
    glob = RenderGlobals(
        reference_amp=float(val["reference_amp"]),
        falloff_6db_um=float(val["falloff_6db_um"]),
        noise_level=float(10.0 ** val["noise_level"]),
        floor_db=float(val["floor_db"]))
    return prof, glob


def encode(profile: CalibrationProfile, glob: RenderGlobals | None = None,
           base: CalibrationProfile | None = None) -> np.ndarray:
    """Inverse of :func:`decode` for a profile whose per-band multipliers are uniform.

    Used by the injected-recovery test to turn a known θ* profile back into a θ vector
    (band multiplier = geometric-mean ratio of profile/base backscatter over the band).
    """
    base = base or literature_profile()
    glob = glob or RenderGlobals()
    b_base = {ly.name: ly.reflectivity for ly in base.layers}
    a_base = {ly.name: ly.attenuation_per_um for ly in base.layers}
    b_prof = {ly.name: ly.reflectivity for ly in profile.layers}
    a_prof = {ly.name: ly.attenuation_per_um for ly in profile.layers}
    P = theta_space()
    out = np.zeros(len(P))
    for i, p in enumerate(P):
        if p.name.startswith("bs_"):
            band = p.name[3:]
            names = dict(BACKSCATTER_BANDS)[band]
            ratios = [b_prof[n] / b_base[n] for n in names if b_base.get(n, 0) > 0]
            out[i] = float(np.log10(np.exp(np.mean(np.log(ratios))))) if ratios else 0.0
        elif p.name.startswith("at_"):
            band = p.name[3:]
            names = dict(ATTENUATION_BANDS)[band]
            ratios = [a_prof[n] / a_base[n] for n in names if a_base.get(n, 0) > 0]
            out[i] = float(np.log10(np.exp(np.mean(np.log(ratios))))) if ratios else 0.0
        elif p.name == "boundary_sharpness_um":
            out[i] = profile.boundary_sharpness_um
        elif p.name == "choroid_contrast":
            out[i] = profile.choroid_contrast
        elif p.name == "texture_contrast":
            out[i] = profile.texture_contrast
        elif p.name == "interface_jitter":
            out[i] = profile.interface_jitter
        elif p.name == "vitreous_backscatter":
            out[i] = float(np.log10(max(profile.vitreous_backscatter, 1e-6)))
        elif p.name == "directional_strength":
            out[i] = profile.directional_strength
        elif p.name == "R_eye_um":
            out[i] = float(getattr(profile, "curvature_radius_um", p.lit))
        elif p.name == "curvature_conic":
            out[i] = float(getattr(profile, "curvature_conic", 0.0))
        elif p.name == "speckle_shape":
            # inverse of decode's sentinel: None (Rayleigh) -> top of range
            out[i] = (p.hi if getattr(profile, "speckle_shape", None) is None
                      else float(np.log10(profile.speckle_shape)))
        elif p.name == "reference_amp":
            out[i] = glob.reference_amp
        elif p.name == "falloff_6db_um":
            out[i] = glob.falloff_6db_um
        elif p.name == "noise_level":
            out[i] = float(np.log10(max(glob.noise_level, 1e-3)))
        elif p.name == "floor_db":
            out[i] = glob.floor_db
    return np.clip(out, [p.lo for p in P], [p.hi for p in P])


# --------------------------------------------------------------------------- render + measure
@dataclass
class RenderSpec:
    """Everything (besides θ) that is FIXED for a fitting session (the ANCHOR)."""

    instrument: str = "maestro3"
    shape_model: str = "octa500"
    anatomy: str = "macula"
    center_wavelength_nm: float = 840.0
    pixels_per_spectrum: int = 2048
    lines_per_frame: int = 256
    lateral_fov_um: float = 6000.0
    averages: int = 1
    seeds: tuple = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
    # VOLUME mode (opt-in, ``volume_frames>1``): each CRN seed renders a small volume spanning
    # ``slow_fov_um`` in the slow axis (fovea→parafovea) instead of one B-scan at the deep pit, so the
    # extracted texture is a volume-AVERAGE — matching how the real target is built (stride over a 6 mm
    # volume). This smooths the single-frame inner-layer template oscillation toward the real (validated:
    # interior-maxima 16→7 vs real 10) and gives the real's shallower parafoveal-averaged pit.
    volume_frames: int = 1
    slow_fov_um: float = 6000.0
    # DEVICE-DSP mode (opt-in): when ``device_process`` and ``tone`` are set, each rendered dB B-scan is
    # pushed through the Maestro3 display chain (resample to the .fda pitch → light speckle-reduction →
    # calibrated tone-curve) so the extracted texture is compared in the CLINICAL .fda domain, not the
    # raw recon. The forward model stays pure; this is the instrument's frozen DSP. ``tone`` is the
    # ``(sim_dB_anchors, dev_8bit_anchors)`` LUT from :func:`topcon_fda.calibrate_tone`.
    device_process: bool = False
    tone: object = None
    fda_pitch_ratio: float = 1.56 / 2.6
    fda_sigma: tuple = (1.0, 0.8)
    fda_axial_pitch_um: float = 2.6

    def config(self, glob: RenderGlobals) -> SimConfig:
        volume = self.volume_frames > 1
        return SimConfig(
            anatomy=self.anatomy, shape_model=self.shape_model, optics_profile="__override__",
            instrument=self.instrument, center_wavelength_nm=self.center_wavelength_nm,
            pixels_per_spectrum=self.pixels_per_spectrum, lines_per_frame=self.lines_per_frame,
            lateral_fov_um=self.lateral_fov_um, averages=self.averages,
            reference_amp=glob.reference_amp, falloff_6db_um=glob.falloff_6db_um,
            noise_level=glob.noise_level,
            scan_pattern=("volume" if volume else "bscan"),
            n_frames=(self.volume_frames if volume else 1),
            slow_fov_um=(self.slow_fov_um if volume else 0.0))


def render_crn(profile: CalibrationProfile, glob: RenderGlobals, spec: RenderSpec):
    """Render the fixed CRN seed set at θ; return (list of dB images, resolved config).

    The dB image is peak-normalised and clipped at ``glob.floor_db`` (θ's floor), matching the
    real-frame reconstruction domain used to build the target.
    """
    cfg = spec.config(glob)
    imgs = []
    resolved = None
    generate.set_optics_override(profile)

    def _to_db(mag):
        peak = float(mag.max()) or 1.0
        return np.clip(20.0 * np.log10(np.maximum(mag / peak, 1e-12)),
                       glob.floor_db, 0.0).astype(np.float32)

    try:
        for sd in spec.seeds:
            s = generate.simulate(cfg, seed=int(sd))
            resolved = s.resolved
            if getattr(s, "volume", None) is not None and s.volume.shape[0] > 1:
                # each slow-axis frame is peak-normalised INDEPENDENTLY (as the real per-frame recon
                # is), then treated as its own B-scan → the extracted texture is the volume average.
                for f in range(s.volume.shape[0]):
                    imgs.append(_to_db(np.abs(s.volume[f])))
            else:
                imgs.append(_to_db(np.abs(s.bscan)))
        if spec.device_process and spec.tone is not None:
            from .data.topcon_fda import device_process as _dproc
            imgs = [_dproc(im, spec.tone, pitch_ratio=spec.fda_pitch_ratio,
                           sigma=spec.fda_sigma).astype(np.float32) for im in imgs]
    finally:
        generate.set_optics_override(None)
    return imgs, resolved


def synth_texture(images, resolved, spec: "RenderSpec | None" = None) -> unlabeled.TextureProfile:
    """Extract a synthetic :class:`TextureProfile` with the target's pitch conventions.

    In device-DSP mode the images are already resampled to the ``.fda`` pixel pitch, so the extract uses
    that pitch instead of the raw recon's ``depth_pixel_um``."""
    dp = (spec.fda_axial_pitch_um if (spec is not None and spec.device_process)
          else float(resolved.source.depth_pixel_um))
    return unlabeled.extract_texture(
        images, axial_pitch_um=dp,
        lateral_pitch_um=float(resolved.derived["fast_pitch_um"]), source="synthetic")


# --------------------------------------------------------------------------- distance
# NOTE: the template-derived per-layer "layer anchor" is DISABLED by default (weight 0). It only
# helps when the target template is peaky enough to localise individual layers; the M3 *volume*
# template is near-flat (per-layer reflectivity all ≈1.0), so the anchor could not discriminate the
# degenerate solutions and merely pulled toward literature. Band-merging (above) resolves the
# degeneracy instead. Kept for a future peaky (line-scan / high-N) target.
# Objective rebalanced after the ILM/RPE detection fix (the real target now reads a correct ~22 dB
# with the true inner-dip/outer-peak structure, not the corrupted 6.6 dB flat). template=1.5 nails the
# layer-brightness SHAPE; contrast=0.3 pins the un-normalised dB range (literature already sits at ~22 dB,
# so this no longer drives noise-gaming — it just prevents drift toward the over-peaked RPE).
# curve/falloff raised (0.2→0.6, 0.4→1.0): at the low weights the geometry/depth terms were outvoted
# by template/hist, so R_eye drifted flat and floor_db rose (brighter deep bkgd) — the fit UNDID the
# realistic FOV curvature. These weights make the loop defend curvature + the depth fade, the aspects
# under focus, while template=1.5 still leads on layer-brightness shape.
DEFAULT_WEIGHTS = {"template": 1.5, "hist": 0.5, "speckle": 0.3, "reg": 0.05, "layer": 0.0,
                   "contrast": 0.3, "curve": 0.6, "falloff": 1.0}


def _target_layer_reflectivity(target: unlabeled.TextureProfile) -> dict:
    """Per-layer relative reflectivity read off the REAL target template at each layer's
    fractional depth (data-grounded, label-free). Anchors *which* layer is bright."""
    from .anatomy.retina import RETINA_LAYERS
    names = [ly.name for ly in RETINA_LAYERS]
    thick = [ly.thickness_um for ly in RETINA_LAYERS]
    try:
        return unlabeled.layer_reflectivity_from_template(target, names, thick)
    except Exception:
        return {}


def _layer_anchor(theta: np.ndarray, target: unlabeled.TextureProfile) -> float:
    """MSE(log) between the DECODED θ per-layer backscatter and the target-template-derived
    per-layer reflectivity — both normalised to unit median. Breaks the outer-band degeneracy
    (photoreceptor↔RPE↔ONL swap) that the aggregate template + reg cannot resolve on their own."""
    tgt = _target_layer_reflectivity(target)
    if not tgt:
        return 0.0
    prof, _ = decode(theta)
    syn = {ly.name: max(float(ly.reflectivity), 1e-6) for ly in prof.layers}
    shared = [n for n in tgt if n in syn and tgt[n] > 0]
    if len(shared) < 4:
        return 0.0
    st = np.array([syn[n] for n in shared]); tt = np.array([tgt[n] for n in shared])
    st = st / (np.median(st) or 1.0)
    tt = tt / (np.median(tt) or 1.0)
    return float(np.mean((np.log(st) - np.log(tt)) ** 2))


def _relerr2(a, b):
    b = float(b)
    if not np.isfinite(b) or abs(b) < 1e-9:
        return 0.0
    return float(((float(a) - b) / b) ** 2)


def reg_term(theta: np.ndarray) -> float:
    """R(θ) = mean squared deviation from θ_lit in bound-normalised coordinates."""
    P = theta_space()
    theta = np.asarray(theta, float)
    acc = 0.0
    for i, p in enumerate(P):
        span = (p.hi - p.lo) or 1.0
        acc += ((theta[i] - p.lit) / span) ** 2
    return float(acc / len(P))


def distance(synth: unlabeled.TextureProfile, target: unlabeled.TextureProfile,
             theta: np.ndarray, weights: dict | None = None) -> dict:
    """Weighted texture distance D(θ) and its decomposed terms (all ≥ 0)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    st = np.asarray(synth.reflectivity_template, float)
    tt = np.asarray(target.reflectivity_template, float)
    n = min(st.size, tt.size)
    d_template = float(np.mean((st[:n] - tt[:n]) ** 2))

    sh = np.asarray(synth.intensity_hist, float)
    th = np.asarray(target.intensity_hist, float)
    if sh.size and th.size and sh.size == th.size:
        d_hist = float(0.5 * np.sum((sh - th) ** 2 / (sh + th + 1e-9)))
    else:
        d_hist = 0.0

    d_speckle = (
        _relerr2(synth.speckle_contrast, target.speckle_contrast)
        + _relerr2(synth.axial_grain_um, target.axial_grain_um)
        + _relerr2(synth.lateral_grain_um, target.lateral_grain_um)
        + _relerr2(synth.heterogeneity, target.heterogeneity)
        + _relerr2(synth.total_retina_um_mean, target.total_retina_um_mean))

    d_reg = reg_term(theta)
    d_layer = _layer_anchor(theta, target)
    # un-normalised inter-layer contrast (dB dynamic range) — the only term that sees the layer
    # brightness SPREAD (all the others are percentile-normalised, hence contrast-blind).
    sc_db = float(getattr(synth, "retinal_contrast_db", 0.0) or 0.0)
    tc_db = float(getattr(target, "retinal_contrast_db", 0.0) or 0.0)
    d_contrast = _relerr2(sc_db, tc_db) if (sc_db > 0 and tc_db > 0) else 0.0

    # GEOMETRIC term: FOV-normalized RPE curvature coefficient (intensity-blind => identifies R_eye
    # only). PHOTOMETRIC term: normalized depth-intensity envelope MSE (deep fade => identifies the
    # deep attenuation only). Both are 0 on un-rebuilt targets (fields empty), so they stay inert until
    # the real targets are rebuilt with the new fields.
    sc_cv = float(getattr(synth, "rpe_curvature_coef", 0.0) or 0.0)
    tc_cv = float(getattr(target, "rpe_curvature_coef", 0.0) or 0.0)
    d_curve = _relerr2(sc_cv, tc_cv) if (sc_cv and tc_cv) else 0.0
    se = np.asarray(getattr(synth, "depth_intensity_env", []) or [], float)
    te = np.asarray(getattr(target, "depth_intensity_env", []) or [], float)
    kk = min(se.size, te.size)
    d_falloff = float(np.mean((se[:kk] - te[:kk]) ** 2)) if kk else 0.0

    total = (w["template"] * d_template + w["hist"] * d_hist + w["speckle"] * d_speckle
             + w.get("layer", 0.0) * d_layer + w.get("contrast", 0.0) * d_contrast
             + w.get("curve", 0.0) * d_curve + w.get("falloff", 0.0) * d_falloff
             + w["reg"] * d_reg)
    return {
        "D_total": float(total),
        "D_template": float(w["template"] * d_template),
        "D_hist": float(w["hist"] * d_hist),
        "D_speckle": float(w["speckle"] * d_speckle),
        "D_layer": float(w.get("layer", 0.0) * d_layer),
        "D_contrast": float(w.get("contrast", 0.0) * d_contrast),
        "D_curve": float(w.get("curve", 0.0) * d_curve),
        "D_falloff": float(w.get("falloff", 0.0) * d_falloff),
        "D_reg": float(w["reg"] * d_reg),
        "raw": {"template_mse": d_template, "hist_chi2": d_hist, "speckle_relerr2": d_speckle,
                "layer_anchor": d_layer, "contrast_relerr2": d_contrast,
                "curve_relerr2": d_curve, "falloff_mse": d_falloff, "reg": d_reg},
    }


def evaluate(theta: np.ndarray, target: unlabeled.TextureProfile, spec: RenderSpec,
             weights: dict | None = None, base: CalibrationProfile | None = None) -> dict:
    """Full D(θ): decode → render CRN → measure → distance. Returns the metric dict plus
    the synthetic profile + resolved config for diagnostics. Robust to a failed render
    (returns a large finite penalty so the optimiser keeps moving)."""
    try:
        profile, glob = decode(theta, base=base)
        images, resolved = render_crn(profile, glob, spec)
        synth = synth_texture(images, resolved, spec)
        out = distance(synth, target, theta, weights)
        out["synth_profile"] = synth
        out["resolved"] = resolved
        out["ok"] = True
        return out
    except Exception as exc:                       # keep the DE population alive
        return {"D_total": 1e6, "D_template": 1e6, "D_hist": 0.0, "D_speckle": 0.0,
                "D_reg": 0.0, "ok": False, "error": repr(exc)}


class FreeObjective:
    """Picklable scalar objective over a *subset* of θ dims (others fixed at literature).

    Expands the free vector into a full θ, evaluates D(θ), and returns the scalar — usable
    with ``differential_evolution(workers=-1)`` because it holds only plain data.
    """

    def __init__(self, target_dict: dict, spec_dict: dict, free_idx, base_theta,
                 weights: dict | None = None):
        self.target_dict = target_dict
        self.spec_dict = spec_dict
        self.free_idx = list(free_idx)
        self.base_theta = np.asarray(base_theta, float)
        self.weights = weights

    def expand(self, x_free) -> np.ndarray:
        full = self.base_theta.copy()
        full[self.free_idx] = np.asarray(x_free, float)
        return full

    def __call__(self, x_free) -> float:
        target = unlabeled.TextureProfile(**self.target_dict)
        spec = RenderSpec(**{**self.spec_dict,
                             "seeds": tuple(self.spec_dict.get("seeds", RenderSpec().seeds))})
        return float(evaluate(self.expand(x_free), target, spec, self.weights).get("D_total", 1e6))


def run_de(target: unlabeled.TextureProfile, spec: RenderSpec, *, free_names=None,
           weights: dict | None = None, base_theta=None, maxiter: int = 8, popsize: int = 8,
           seed: int = 0, workers: int = 1, tol: float = 1e-2, mutation=(0.4, 1.0),
           recombination: float = 0.7, init: str = "sobol", polish: bool = False):
    """Run ``scipy.optimize.differential_evolution`` over the (optionally masked) θ space.

    Returns ``(theta_hat, D_hat, incumbent_rows, result)``; ``incumbent_rows`` is the monotone
    per-generation best-D trajectory (for ``incumbent.csv``). Only ``free_names`` are searched
    (default: all); the rest are held at ``base_theta`` (default: literature).
    """
    from scipy.optimize import differential_evolution

    P = theta_space()
    base_theta = theta_literature() if base_theta is None else np.asarray(base_theta, float)
    names = [p.name for p in P]
    # default free set = all NON-frozen params (measured/frozen instrument params stay at θ_lit)
    default_free = [p.name for p in P if not p.frozen]
    free_idx = [names.index(n) for n in (free_names or default_free)]
    all_bounds = bounds()
    free_bounds = [all_bounds[i] for i in free_idx]

    obj = FreeObjective(target.to_dict(), _spec_to_dict(spec), free_idx, base_theta, weights)

    incumbent, state = [], {"best": np.inf, "gen": 0, "evals": 0}
    import time
    t0 = time.time()

    def _cb(xk, convergence=None):
        state["gen"] += 1
        d = float(obj(xk))
        state["best"] = min(state["best"], d)
        incumbent.append({"generation": state["gen"], "best_D": round(state["best"], 6),
                          "wall_s": round(time.time() - t0, 2), "evals": state["evals"]})
        return False

    res = differential_evolution(
        obj, free_bounds, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation,
        recombination=recombination, seed=seed, polish=polish, init=init,
        updating=("deferred" if workers != 1 else "immediate"), workers=workers, callback=_cb)

    theta_hat = obj.expand(res.x)
    return theta_hat, float(res.fun), incumbent, res


class PooledObjective:
    """Picklable objective = MEAN D(θ) across a POOL of (target, geometry) pairs.

    Fits ONE set of tissue optics that matches *all* the pooled geometries/corpora at once, so the
    result generalises instead of overfitting a single scan — the anti-overfit path of the routine.
    """

    def __init__(self, pool_dicts, free_idx, base_theta, weights: dict | None = None):
        self.pool_dicts = pool_dicts          # [{"target": target_dict, "spec": spec_dict}, ...]
        self.free_idx = list(free_idx)
        self.base_theta = np.asarray(base_theta, float)
        self.weights = weights

    def expand(self, x_free) -> np.ndarray:
        full = self.base_theta.copy()
        full[self.free_idx] = np.asarray(x_free, float)
        return full

    def __call__(self, x_free) -> float:
        theta = self.expand(x_free)
        ds = []
        for pd in self.pool_dicts:
            target = unlabeled.TextureProfile(**pd["target"])
            spec = RenderSpec(**{**pd["spec"],
                                 "seeds": tuple(pd["spec"].get("seeds", RenderSpec().seeds))})
            ds.append(evaluate(theta, target, spec, self.weights).get("D_total", 1e6))
        return float(np.mean(ds))


def run_de_pool(pool, *, free_names=None, weights: dict | None = None, base_theta=None,
                maxiter: int = 10, popsize: int = 8, seed: int = 0, workers: int = 1,
                tol: float = 1e-2, mutation=(0.4, 1.0), recombination: float = 0.7,
                init: str = "sobol", polish: bool = False):
    """Pooled DE fit over a list of ``(TextureProfile, RenderSpec)`` geometry pairs.

    Minimises the mean D across the pool. Returns ``(theta_hat, mean_D, incumbent, result)``.
    """
    from scipy.optimize import differential_evolution
    import time

    P = theta_space()
    base_theta = theta_literature() if base_theta is None else np.asarray(base_theta, float)
    names = [p.name for p in P]
    default_free = [p.name for p in P if not p.frozen]
    free_idx = [names.index(n) for n in (free_names or default_free)]
    free_bounds = [bounds()[i] for i in free_idx]
    pool_dicts = [{"target": t.to_dict(), "spec": _spec_to_dict(s)} for t, s in pool]
    obj = PooledObjective(pool_dicts, free_idx, base_theta, weights)

    incumbent, state, t0 = [], {"best": np.inf, "gen": 0}, time.time()

    def _cb(xk, convergence=None):
        state["gen"] += 1
        d = float(obj(xk)); state["best"] = min(state["best"], d)
        incumbent.append({"generation": state["gen"], "best_D": round(state["best"], 6),
                          "wall_s": round(time.time() - t0, 2), "evals": 0})
        return False

    res = differential_evolution(
        obj, free_bounds, maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation,
        recombination=recombination, seed=seed, polish=polish, init=init,
        updating=("deferred" if workers != 1 else "immediate"), workers=workers, callback=_cb)
    return obj.expand(res.x), float(res.fun), incumbent, res


def _spec_to_dict(spec: RenderSpec) -> dict:
    # ALL render-mode fields must round-trip, else run_de's workers silently render in a different
    # mode than the caller's spec (previously volume_frames + the device_process fields were dropped).
    return {"instrument": spec.instrument, "shape_model": spec.shape_model,
            "anatomy": spec.anatomy, "center_wavelength_nm": spec.center_wavelength_nm,
            "pixels_per_spectrum": spec.pixels_per_spectrum,
            "lines_per_frame": spec.lines_per_frame, "lateral_fov_um": spec.lateral_fov_um,
            "averages": spec.averages, "seeds": list(spec.seeds),
            "volume_frames": spec.volume_frames, "slow_fov_um": spec.slow_fov_um,
            "device_process": spec.device_process, "tone": spec.tone,
            "fda_pitch_ratio": spec.fda_pitch_ratio, "fda_sigma": spec.fda_sigma,
            "fda_axial_pitch_um": spec.fda_axial_pitch_um}
