"""Config-driven generation: scenes, B-scans, volumes, and stochastic datasets.

This is the top layer that ties everything together. From a :class:`~octsim.config.SimConfig`
it resolves the physics, builds a scene (a random or fixed retina, or a preset sample),
and renders either a single B-scan or a **volume of contiguous frames** — applying the
one continuous motion trajectory (:mod:`octsim.motion`) across the whole acquisition so a
volume's frames share correlated bulk motion, phase drift, and washout.

* :func:`simulate` — one sample (a B-scan, or a volume when ``scan_pattern="volume"``).
* :func:`generate_dataset` — an in-memory iterator of **N independent random samples**,
  each a fresh anatomy + speckle + motion realisation, reproducible via ``SeedSequence``.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import numpy as np

from . import presets
from .anatomy.retina import RetinaModel, default_retina, random_retina
from .config import SimConfig, ResolvedConfig
from .forward import synthesize_bscan_spectra
from .motion import build_motion
from .reconstruct import reconstruct_bscan
from .sample import Sample, build_reflectivity, build_reflectivity_from_boundaries

#: integration time of the default acquisition (100 kHz, 0.9 duty) — the signal-scale
#: reference, so the default config has signal_scale == 1.
REFERENCE_INTEGRATION_S = 9e-6

RETINAL = ("macula", "onh", "periphery")


@dataclass
class SampleResult:
    """One generated sample: a B-scan (``n_frames==1``) or a volume."""

    volume: np.ndarray           # complex [n_frames, n_depth, n_x]
    volume_db: np.ndarray        # float   [n_frames, n_depth, n_x], 0 dB at the volume peak
    label_map: np.ndarray        # uint8   [n_frames, n_depth, n_x] layer index
    depth_axis_um: np.ndarray
    x_axis_um: np.ndarray
    y_axis_um: np.ndarray
    resolved: ResolvedConfig
    params: dict
    seed_key: int
    motion_dz_um: np.ndarray | None = None   # [n_frames*lines] trajectory (or None)
    #: the actual detector interferogram I(k, x) of frame 0 — the raw spectra the
    #: displayed B-scan is the FFT reconstruction of (incl. noise). [n_k, n_x] float32.
    spectra: np.ndarray | None = None
    #: the reference spectrum P(k)·a_ref² subtracted before the IFFT. Pass this as
    #: ``reconstruct_bscan(spectra, background=dc_background)`` to reproduce the
    #: displayed B-scan exactly from :attr:`spectra`. [n_k] float64.
    dc_background: np.ndarray | None = None

    @property
    def k_axis_rad_per_um(self) -> np.ndarray:
        """Wavenumber grid (rad/µm) the retained ``spectra`` are sampled on."""
        return self.resolved.source.k_grid()

    def ascan_db(self, col: int) -> np.ndarray:
        """Reconstructed A-scan (dB, 0 dB at the volume peak) for one lateral column."""
        return self.bscan_db[:, int(col)]

    @property
    def is_volume(self) -> bool:
        return self.volume.shape[0] > 1

    @property
    def bscan(self) -> np.ndarray:
        return self.volume[0]

    @property
    def bscan_db(self) -> np.ndarray:
        return self.volume_db[0]

    @property
    def labels2d(self) -> np.ndarray:
        return self.label_map[0]


# --------------------------------------------------------------------------- scenes
def build_scene(config: SimConfig, rng: np.random.Generator):
    """Pick/sample the scene for ``config.anatomy`` (RetinaModel or preset Sample)."""
    scene = _build_scene_inner(config, rng)
    # The fan-beam pivot is scan geometry, not anatomy, so it is owned by the config and
    # stamped onto whichever retina builder ran (each has its own constructor).
    if isinstance(scene, RetinaModel):
        scene.pivot_distance_um = float(config.pivot_distance_um)
        scene.scan_distortion_correction = config.effective_scan_distortion()
    return scene


def _build_scene_inner(config: SimConfig, rng: np.random.Generator):
    a = config.anatomy
    fov = config.lateral_fov_um
    if a in RETINAL:
        sm_name = (config.shape_model or "").lower()
        if sm_name and sm_name not in ("none", "parametric", "off"):
            try:                    # data-driven geometry from a learned shape model
                from . import calibration
                sm = _resolve_shape_model(sm_name)
                optics = (_OPTICS_OVERRIDE if _OPTICS_OVERRIDE is not None
                          else _resolve_optics_profile(config.optics_profile))
                return calibration.retina_from_shape_model(
                    sm, rng, optics=optics, location=a, lateral_fov_um=fov,
                    surface_opt_um=config.sample_depth_um,
                    lateral_offset_um=config.sample_lateral_offset_um)
            except Exception:
                pass                # fall back to the parametric scene if the model is missing
        if config.randomize:
            return random_retina(rng, location=a, surface_opt_um=config.sample_depth_um,
                                 lateral_fov_um=fov,
                                 lateral_offset_um=config.sample_lateral_offset_um)
        return default_retina(location=a, surface_opt_um=config.sample_depth_um,
                              center_x_um=fov / 2 + config.sample_lateral_offset_um,
                              lateral_fov_um=fov)
    if a == "mirror":
        return presets.mirror(depth_um=config.sample_depth_um)
    if a == "paper":
        return presets.paper(surface_um=config.sample_depth_um)
    if a == "cornea":
        return presets.cornea(surface_um=config.sample_depth_um)
    raise ValueError(f"unknown anatomy {a!r}")


#: bundled shape-model artifacts. Default is the OCTA-500 healthy-eye model (fit from real
#: 3D macular volumes, so its foveal pit is deep and physiologic — rel-depth 0.63); the JHU
#: healthy-control model is kept as an alternative (its 49-B-scan volumes give a shallower,
#: parafovea-diluted pit), and the Duke DME model for disease geometry.
_BUNDLED_SHAPE = {
    "octa500": "retina_shape_octa500.npz",
    "normal": "retina_shape_octa500.npz", "default": "retina_shape_octa500.npz",
    "jhu": "retina_shape_jhu_hc.npz", "hc": "retina_shape_jhu_hc.npz",
    "duke": "retina_shape_duke.npz", "dme": "retina_shape_duke.npz",
}


@functools.lru_cache(maxsize=8)
def _resolve_shape_model(name: str):
    """Load a LayerShapeModel by path, or a bundled model by keyword (default = JHU normal)."""
    from .shape_model import LayerShapeModel
    fn = _BUNDLED_SHAPE.get(name)
    if fn is not None:
        name = os.path.join(os.path.dirname(__file__), "data", fn)
    return LayerShapeModel.load(name)


#: bundled fitted tissue-optics profiles (analogous to :data:`_BUNDLED_SHAPE`). ``None`` means
#: "use the literature profile" — the default, so the shipped corpus is unchanged until a fit
#: is committed and its keyword promoted here / set as the ``optics_profile`` default. A path
#: is also accepted. Optics never touch geometry, so switching profile keeps ``label_map``
#: bit-identical at a fixed seed.
_BUNDLED_PROFILE = {
    "literature": None, "parametric": None, "none": None, "": None,
    "maestro3": "retina_optics_maestro3.json",
    "maestro2": "retina_optics_maestro2.json",
    "octa500": "retina_optics_octa500.json",
}


#: process-local in-memory optics override used by the calibration loop to inject a decoded
#: θ profile without round-tripping through a temp JSON. ``None`` = use the named profile.
#: Set via :func:`set_optics_override` (and always cleared in a ``finally``).
_OPTICS_OVERRIDE = None


def set_optics_override(profile) -> None:
    """Inject an in-memory :class:`~octsim.calibration.CalibrationProfile` that
    :func:`build_scene` uses in place of ``config.optics_profile`` (``None`` clears it).
    Optics never touch geometry, so this cannot change ``label_map``."""
    global _OPTICS_OVERRIDE
    _OPTICS_OVERRIDE = profile


@functools.lru_cache(maxsize=8)
def _resolve_optics_profile(name: str):
    """Resolve an optics-profile keyword/path to a :class:`~octsim.calibration.CalibrationProfile`.

    Falls back to the literature profile for unknown/empty names or a missing bundled file,
    so a not-yet-committed fit degrades gracefully to the shipped default."""
    from . import calibration
    key = (name or "literature").lower()
    if key in _BUNDLED_PROFILE:
        fn = _BUNDLED_PROFILE[key]
        if fn is None:
            return calibration.literature_profile()
        path = os.path.join(os.path.dirname(__file__), "data", fn)
        if os.path.exists(path):
            return calibration.CalibrationProfile.load(path)
        return calibration.literature_profile()
    if os.path.exists(name):                    # an explicit path to a fitted profile JSON
        return calibration.CalibrationProfile.load(name)
    return calibration.literature_profile()


def _scene_params(scene) -> dict:
    if isinstance(scene, RetinaModel):
        return {"anatomy": scene.location, "center_x_um": scene.center_x_um,
                "pit_radius_um": scene.pit_radius_um, "n_vessels": len(scene.vessels),
                "layer_thickness_um": [round(ly.thickness_um, 2) for ly in scene.layers]}
    return {"anatomy": getattr(scene, "name", "sample")}


def _label_names(scene, resolved, x_axis, y_um) -> list:
    """Ordered layer names for ``label_map`` (index i+1 -> names[i]).

    Exported because the mapping is NOT fixed: when the optic disc is in view the field
    model inserts a "canal" layer at index 1, so GCL is label 3 on a wide scan and label 2
    on a 6 mm macular scan. A consumer that assumes a constant vocabulary silently mislabels
    every layer below the insertion point.
    """
    if isinstance(scene, RetinaModel):
        return list(scene.cross_section(
            y_um, x_axis, resolved.source.center_wavelength_nm).label_names)
    return [ly.name for ly in getattr(scene, "layers", [])]


#: a layer thinner than this (optical µm) is treated as anatomically absent rather than
#: rounded up to one label row
_MIN_LAYER_UM = 0.05


def _rasterize_labels(opt_b: np.ndarray, depth_axis_um: np.ndarray) -> np.ndarray:
    """Per-pixel layer index (0 = above/below tissue) from optical boundaries.

    Guarantees that every layer with non-zero thickness occupies **at least one row** in
    every column. A plain half-open depth comparison silently drops any layer thinner than
    the depth pixel — at 3.87 µm/px that deleted the IZ from most columns and the RPE from
    hundreds of them, so the exported ground truth showed anatomically continuous layers as
    intermittently missing. The retina is continuous even where the *image* renders a band
    weakly, and the labels have to say so or they are unusable as segmentation targets.
    """
    n_depth = depth_axis_um.size
    n_x = opt_b.shape[1]
    dz = float(depth_axis_um[1] - depth_axis_um[0]) if n_depth > 1 else 1.0
    z0 = float(depth_axis_um[0])

    rows = np.round((opt_b - z0) / dz).astype(np.intp)
    thick = np.diff(opt_b, axis=0)
    # walk down, forcing each present layer to claim a row without reordering boundaries
    for i in range(1, rows.shape[0]):
        rows[i] = np.maximum(rows[i], rows[i - 1] + (thick[i - 1] > _MIN_LAYER_UM))
    np.clip(rows, 0, n_depth, out=rows)

    labels = np.zeros((n_depth, n_x), dtype=np.uint8)
    idx = np.arange(n_depth)[:, None]
    for i in range(rows.shape[0] - 1):
        labels[(idx >= rows[i][None, :]) & (idx < rows[i + 1][None, :])] = i + 1
    return labels


def _frame_reflectivity_builder(scene, resolved: ResolvedConfig, x_axis, y_um,
                                render_depth=None):
    """Return ``(build(rng)->reflectivity, labels)`` — geometry computed once so
    incoherent averages redraw only the speckle."""
    depth = resolved.source.depth_axis_um()
    label_depth = depth
    if render_depth is not None:
        depth = np.asarray(render_depth, float)
    fov = resolved.geometry.lateral_fov_um
    if isinstance(scene, RetinaModel):
        cross = scene.cross_section(y_um, x_axis, resolved.source.center_wavelength_nm)
        labels = _rasterize_labels(cross.opt_b, label_depth)   # visible half only

        def build(rng, speckle=None, speckle_decorr=None):
            return build_reflectivity_from_boundaries(
                cross.opt_b, cross.phys_b, layer_n=cross.layer_n,
                layer_backscatter=cross.layer_backscatter, layer_mu_t=cross.layer_mu_t,
                layer_interface=cross.layer_interface, top_n=cross.top_n,
                depth_axis_um=depth, x_axis_um=x_axis, beam_waist_um=resolved.beam_waist_um,
                lateral_extent_um=fov, rng=rng, features=cross.features, y_um=y_um,
                texture_contrast=cross.texture_contrast,
                interface_jitter=cross.interface_jitter,
                vitreous_backscatter=cross.vitreous_backscatter,
                layer_directional=cross.layer_directional,
                boundary_sharpness_um=cross.boundary_sharpness_um,
                layer_sharpness_um=getattr(cross, "layer_sharpness_um", None),
                speckle_decorrelation=getattr(cross, "speckle_decorrelation", 0.0),
                speckle_corr_lateral_um=resolved.speckle_corr_lateral_um,
                ms_fraction=getattr(cross, "ms_fraction", 0.0),
                ms_length_um=getattr(cross, "ms_length_um", 120.0),
                texture_field=cross.texture_field, speckle=speckle,
                speckle_decorr=speckle_decorr,
                k_shape=getattr(cross, "speckle_shape", None))
    else:
        labels = _rasterize_labels(scene.optical_boundary_depths_um(x_axis), label_depth)

        def build(rng, speckle=None, speckle_decorr=None):
            return build_reflectivity(scene, depth, x_axis, beam_waist_um=resolved.beam_waist_um,
                                      lateral_extent_um=fov, rng=rng)
    return build, labels


# --------------------------------------------------------------------------- simulate
def simulate(config: SimConfig, *, seed=None, rng: np.random.Generator | None = None) -> SampleResult:
    """Render one sample (a B-scan, or a volume when ``config.scan_pattern=='volume'``)."""
    resolved = config.resolve()
    if rng is None:
        rng = np.random.default_rng(config.seed if seed is None else seed)
    scene = build_scene(config, rng)

    geom = resolved.geometry
    timing = resolved.timing
    src = resolved.source
    x_axis = geom.x_axis()
    y_axis = geom.y_axis()
    n_frames = geom.n_frames
    depth = src.depth_axis_um()

    # OUT-OF-RANGE WRAP.
    #
    # The scene is normally rasterised onto exactly the n_k//2 rows the reconstruction
    # returns, so nothing can ever lie beyond z_max and the simulator structurally cannot
    # produce the fold-over that appears in essentially every real disc B-scan (and in any
    # scan where the operator mis-positions the retina).
    #
    # The physics is already there and only the grid was preventing it: the detector
    # spectrum is REAL, so its transform is Hermitian, and a scatterer at depth index
    # p > n_k/2 carries a fringe frequency past Nyquist that aliases to n_k - p. Rendering
    # the scene onto the full n_k-row grid and handing all of it to the forward model
    # therefore folds deep structure back exactly as the device does, with no special-case
    # artifact code.
    render_depth = depth
    if resolved.axial_wrap:
        render_depth = np.arange(src.n_k) * src.depth_pixel_um

    motion = build_motion(timing, motion_level=resolved.motion_level,
                          axial_motion_um=resolved.axial_motion_um,
                          phase_drift_rad=resolved.phase_drift_rad,
                          cardiac=resolved.cardiac, microsaccade=resolved.microsaccade, rng=rng)
    moving = not motion.is_static
    wg_full = motion.washout_gain(src.k0, timing.integration_time_s) if moving else None
    lines = timing.lines_per_frame
    T_int = timing.integration_time_s
    signal_scale = float(np.sqrt(T_int / REFERENCE_INTEGRATION_S)) if T_int > 0 else 1.0

    n_depth = depth.size
    n_x = x_axis.size
    volume = np.empty((n_frames, n_depth, n_x), dtype=np.complex64)
    labels = np.empty((n_frames, n_depth, n_x), dtype=np.uint8)

    navg = max(1, int(resolved.averages))
    # slow-axis speckle correlation: OCT speckle decorrelates over ~ the lateral resolution
    # (the beam waist) in the slow direction too, so a densely-sampled volume shares speckle
    # between adjacent frames instead of boiling. AR(1) carry with correlation length = the beam
    # waist reproduces that (alpha->1 when the slow pitch << the waist, ->0 when it is coarse).
    slow_pitch = float(y_axis[1] - y_axis[0]) if y_axis.size > 1 else 0.0
    waist = max(float(resolved.beam_waist_um), 1e-6)
    speckle_alpha = float(np.exp(-slow_pitch / waist)) if (n_frames > 1 and slow_pitch > 0) else 0.0
    prev_g = prev_g2 = None

    # drawn on the RENDER grid, which is deeper than the visible one when axial_wrap is on
    n_render = int(render_depth.size)

    def _draw_speckle(rng_):
        return (rng_.standard_normal((n_render, n_x))
                + 1j * rng_.standard_normal((n_render, n_x))) / np.sqrt(2.0)

    # Exact reference-spectrum subtraction. The detector sees P(k)·|a_ref + E_s|², whose pure
    # DC part is P(k)·a_ref² — a quantity the forward model *knows*, so it can be removed
    # analytically instead of estimated. The fallback in reconstruct_bscan is a 128-tap box
    # high-pass, which leaves a large low-frequency residual that reconstructs as a bright band
    # at the top of the frame; with a bright optics profile that band becomes the image maximum
    # and, because volume_db is peak-normalised below, it drags the whole B-scan down ~13 dB
    # relative to real data. Subtracting the known DC is what a real device does with a
    # background frame, and it costs one array.
    dc_background = src.power_spectrum(src.k_grid()) * float(resolved.reference_amp) ** 2

    first_spectra = None
    for f in range(n_frames):
        y = float(y_axis[f]) if f < y_axis.size else 0.0
        build, lab = _frame_reflectivity_builder(scene, resolved, x_axis, y, render_depth)
        labels[f] = lab
        if moving:
            dz, _v, phase = motion.frame(f)
            wg = wg_full[f * lines:(f + 1) * lines]
        else:
            dz = phase = wg = None

        # this frame's slow-axis-correlated base speckle (None => builder draws its own).
        # The sub-resolution decorrelated component carries its own AR(1) chain: it is a
        # LATERAL decorrelation, so it must still be shared between adjacent frames or a
        # densely-sampled volume boils regardless of how fine the slow pitch is.
        if speckle_alpha > 0.0:
            fresh = _draw_speckle(rng)
            g0 = fresh if prev_g is None else (
                speckle_alpha * prev_g + np.sqrt(1.0 - speckle_alpha ** 2) * fresh)
            prev_g = g0
            fresh2 = _draw_speckle(rng)
            g0_dec = fresh2 if prev_g2 is None else (
                speckle_alpha * prev_g2 + np.sqrt(1.0 - speckle_alpha ** 2) * fresh2)
            prev_g2 = g0_dec
        else:
            g0 = g0_dec = None

        def one(rng_, speckle, speckle_decorr=None):
            spectra = synthesize_bscan_spectra(
                build(rng_, speckle, speckle_decorr), src, reference_amp=resolved.reference_amp,
                falloff_6db_um=resolved.falloff_6db_um,
                dispersion_coeffs=resolved.dispersion_coeffs, signal_scale=signal_scale,
                motion_dz_um=dz, washout_gain=wg, column_phase=phase)
            noisy = resolved.noise.apply(spectra, rng_)   # the digitised detector spectra
            return reconstruct_bscan(noisy, background=dc_background), noisy

        if navg > 1:  # incoherent (magnitude) averaging — raw speckle reduced, ART-like
            acc = np.zeros((n_depth, n_x))
            for a in range(navg):
                # keep the frame's coherent base for the first average; redraw the rest
                recon, noisy = one(rng, g0 if a == 0 else None,
                                   g0_dec if a == 0 else None)
                acc += np.abs(recon)
                if f == 0 and a == 0:
                    first_spectra = noisy
            volume[f] = (acc / navg).astype(np.complex64)
        else:
            recon, noisy = one(rng, g0, g0_dec)
            volume[f] = recon.astype(np.complex64)
            if f == 0:
                first_spectra = noisy

    peak = np.abs(volume).max() or 1.0
    volume_db = np.clip(20.0 * np.log10(np.maximum(np.abs(volume) / peak, 1e-12)), -60.0, 0.0)

    return SampleResult(
        volume=volume, volume_db=volume_db, label_map=labels, depth_axis_um=depth,
        x_axis_um=x_axis, y_axis_um=y_axis, resolved=resolved, params={**_scene_params(scene),
                "label_names": _label_names(scene, resolved, x_axis, float(y_axis[0]))},
        seed_key=_seed_key(rng), motion_dz_um=(motion.dz_um if moving else None),
        spectra=(first_spectra.astype(np.float32) if first_spectra is not None else None),
        dc_background=dc_background)


def _seed_key(rng: np.random.Generator) -> int:
    """A stable, per-sample-distinct identifier from the RNG's SeedSequence.

    Spawned children share the parent ``entropy`` and differ only by ``spawn_key``,
    so both must enter the key or every sample in a dataset gets the same id.
    """
    try:
        sq = rng.bit_generator.seed_seq  # type: ignore[attr-defined]
        ent = sq.entropy
        base = int(ent if np.isscalar(ent) else ent[0])
        spawn = tuple(int(s) for s in getattr(sq, "spawn_key", ()) or ())
        return int(hash((base, spawn)) & 0xFFFFFFFF)
    except Exception:
        return 0


def generate_dataset(config: SimConfig, n: int | None = None, *, seed=None):
    """Iterate **N independent random samples** (in-memory).

    Each sample gets its own reproducible RNG stream via ``SeedSequence.spawn``, so a
    fresh anatomy, speckle, and motion realisation are drawn per sample while the whole
    run is reproducible from ``seed``.
    """
    n = int(n if n is not None else config.n_samples)
    base = config.seed if seed is None else seed
    for sq in np.random.SeedSequence(base).spawn(n):
        yield simulate(config, rng=np.random.default_rng(sq))
