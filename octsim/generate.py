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
    a = config.anatomy
    fov = config.lateral_fov_um
    if a in RETINAL:
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
        return presets.paper()
    if a == "cornea":
        return presets.cornea()
    raise ValueError(f"unknown anatomy {a!r}")


def _scene_params(scene) -> dict:
    if isinstance(scene, RetinaModel):
        return {"anatomy": scene.location, "center_x_um": scene.center_x_um,
                "pit_radius_um": scene.pit_radius_um, "n_vessels": len(scene.vessels),
                "layer_thickness_um": [round(ly.thickness_um, 2) for ly in scene.layers]}
    return {"anatomy": getattr(scene, "name", "sample")}


def _rasterize_labels(opt_b: np.ndarray, depth_axis_um: np.ndarray) -> np.ndarray:
    """Per-pixel layer index (0 = above/below tissue) from optical boundaries."""
    n_depth = depth_axis_um.size
    n_x = opt_b.shape[1]
    labels = np.zeros((n_depth, n_x), dtype=np.uint8)
    d = depth_axis_um[:, None]
    for i in range(opt_b.shape[0] - 1):
        mask = (d >= opt_b[i][None, :]) & (d < opt_b[i + 1][None, :])
        labels[mask] = i + 1
    return labels


def _frame_reflectivity_builder(scene, resolved: ResolvedConfig, x_axis, y_um):
    """Return ``(build(rng)->reflectivity, labels)`` — geometry computed once so
    incoherent averages redraw only the speckle."""
    depth = resolved.source.depth_axis_um()
    fov = resolved.geometry.lateral_fov_um
    if isinstance(scene, RetinaModel):
        cross = scene.cross_section(y_um, x_axis, resolved.source.center_wavelength_nm)
        labels = _rasterize_labels(cross.opt_b, depth)

        def build(rng):
            return build_reflectivity_from_boundaries(
                cross.opt_b, cross.phys_b, layer_n=cross.layer_n,
                layer_backscatter=cross.layer_backscatter, layer_mu_t=cross.layer_mu_t,
                layer_interface=cross.layer_interface, top_n=cross.top_n,
                depth_axis_um=depth, x_axis_um=x_axis, beam_waist_um=resolved.beam_waist_um,
                lateral_extent_um=fov, rng=rng, features=cross.features, y_um=y_um)
    else:
        labels = _rasterize_labels(scene.optical_boundary_depths_um(x_axis), depth)

        def build(rng):
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
    for f in range(n_frames):
        y = float(y_axis[f]) if f < y_axis.size else 0.0
        build, lab = _frame_reflectivity_builder(scene, resolved, x_axis, y)
        labels[f] = lab
        if moving:
            dz, _v, phase = motion.frame(f)
            wg = wg_full[f * lines:(f + 1) * lines]
        else:
            dz = phase = wg = None

        def one(rng_):
            spectra = synthesize_bscan_spectra(
                build(rng_), src, reference_amp=resolved.reference_amp,
                falloff_6db_um=resolved.falloff_6db_um,
                dispersion_coeffs=resolved.dispersion_coeffs, signal_scale=signal_scale,
                motion_dz_um=dz, washout_gain=wg, column_phase=phase)
            return reconstruct_bscan(resolved.noise.apply(spectra, rng_))

        if navg > 1:  # incoherent (magnitude) averaging — raw speckle reduced, ART-like
            acc = np.zeros((n_depth, n_x))
            for _ in range(navg):
                acc += np.abs(one(rng))
            volume[f] = (acc / navg).astype(np.complex64)
        else:
            volume[f] = one(rng).astype(np.complex64)

    peak = np.abs(volume).max() or 1.0
    volume_db = np.clip(20.0 * np.log10(np.maximum(np.abs(volume) / peak, 1e-12)), -60.0, 0.0)

    return SampleResult(
        volume=volume, volume_db=volume_db, label_map=labels, depth_axis_um=depth,
        x_axis_um=x_axis, y_axis_um=y_axis, resolved=resolved, params=_scene_params(scene),
        seed_key=_seed_key(rng), motion_dz_um=(motion.dz_um if moving else None))


def _seed_key(rng: np.random.Generator) -> int:
    try:
        ent = rng.bit_generator.seed_seq.entropy  # type: ignore[attr-defined]
        return int(ent if np.isscalar(ent) else ent[0])
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
