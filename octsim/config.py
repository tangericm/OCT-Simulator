"""Simulation configuration — one flat, resolvable, serialization-friendly dataclass.

Follows the Line-Field-OCT house style: a single flat ``@dataclass(slots=True)`` grouped
by comment blocks, a chainable :meth:`SimConfig.normalized` validator, and a tolerant
:func:`config_from_mapping` loader. The distinctive piece is **resolvable binding**: the
physically-linked quantities may be specified from *either* side and the config back-solves
the rest. A value of ``0`` (or ``0.0``) on a resolvable field means "derive me":

* ``bandwidth_fwhm_nm`` ↔ ``axial_resolution_um``   (coherence limit)
* ``axial_range_um`` / ``axial_sampling_um`` ↔ ``k_span_factor``  (k-window vs pixels)
* ``lateral_resolution_um`` ↔ ``beam_waist_um``     (Gaussian beam)

:meth:`SimConfig.resolve` returns a :class:`ResolvedConfig` holding the concrete
:class:`~octsim.source.LightSource`, :class:`~octsim.acquisition.ScanGeometry` /
:class:`~octsim.acquisition.Timing`, :class:`~octsim.noise.NoiseModel`, and a
``derived`` report of every specified-vs-derived value for the GUI readout.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .acquisition import ScanGeometry, Timing
from .noise import NoiseModel
from .optics import lateral_fwhm_to_waist_um, waist_to_lateral_fwhm_um
from .source import LightSource

_LN2 = np.log(2.0)
# Maps the intuitive `dispersion` knob to the quadratic phase coefficient a2
# (in (rad/µm)^-2). ~1 gives a mild axial stretch, ~3 a strong one.
DISPERSION_A2_UNIT = 15.0
SCAN_PATTERNS = ("bscan", "volume")
ANATOMIES = ("macula", "onh", "periphery", "mirror", "paper", "cornea")


def _axial_res_from_bandwidth(lambda0_nm: float, bw_nm: float) -> float:
    lam = lambda0_nm * 1e-3
    return (2.0 * _LN2 / np.pi) * lam ** 2 / (bw_nm * 1e-3)


def _bandwidth_from_axial_res(lambda0_nm: float, axial_res_um: float) -> float:
    lam = lambda0_nm * 1e-3
    return ((2.0 * _LN2 / np.pi) * lam ** 2 / axial_res_um) * 1e3  # µm -> nm


@dataclass
class ResolvedConfig:
    """Concrete objects produced by :meth:`SimConfig.resolve`."""

    source: LightSource
    geometry: ScanGeometry
    timing: Timing
    beam_waist_um: float
    reference_amp: float
    dispersion_coeffs: list[float]     # derived from the SimConfig.dispersion knob
    falloff_6db_um: float
    noise: NoiseModel
    motion_level: float
    axial_motion_um: float
    phase_drift_rad: float
    cardiac: bool
    microsaccade: bool
    anatomy: str
    scan_pattern: str
    derived: dict[str, float]


@dataclass(slots=True)
class SimConfig:
    """All simulation parameters (resolvable; 0 = derive on the linked fields)."""

    # --- source / axial ---
    center_wavelength_nm: float = 1050.0
    bandwidth_fwhm_nm: float = 100.0        # >0 specify; 0 -> derive from axial_resolution_um
    axial_resolution_um: float = 0.0        # >0 specify; 0 -> derive from bandwidth
    pixels_per_spectrum: int = 2048
    axial_range_um: float = 0.0             # optical, one-way; >0 specify; 0 -> derive
    axial_sampling_um: float = 0.0          # depth pixel; >0 specify; 0 -> derive
    k_span_factor: float = 1.5              # used when neither axial range nor sampling given

    # --- scan geometry ---
    scan_pattern: str = "bscan"             # bscan | volume
    lateral_fov_um: float = 1000.0
    lines_per_frame: int = 512
    lateral_resolution_um: float = 0.0      # >0 specify; 0 -> use beam_waist_um
    beam_waist_um: float = 15.0
    slow_fov_um: float = 500.0              # slow (y) axis FOV for a volume
    n_frames: int = 32                      # frames in a volume (forced to 1 for bscan)

    # --- timing / speed ---
    line_rate_hz: float = 100_000.0         # A-scan / sweep rate
    duty_cycle: float = 0.9
    flyback_time_s: float = 0.0

    # --- dispersion / fall-off ---
    # Single intuitive tuning knob: 0 = sharp (transform-limited); increasing
    # magnitude = more axial stretch/blur; sign flips the chirp direction.
    dispersion: float = 0.0
    falloff_6db_um: float = 0.0             # 0 -> no roll-off

    # --- noise ---
    noise_free: bool = False
    noise_level: float = 1.0
    shot: float = 0.03
    rin: float = 0.005
    read: float = 0.01
    target_snr_db: float = 0.0              # >0 -> back-solve the read floor

    # --- motion / artifacts ---
    motion_level: float = 0.0               # 0 -> no motion artifacts
    axial_motion_um: float = 2.0            # bulk axial random-walk amplitude scale
    phase_drift_rad: float = 0.5            # slow phase-drift amplitude
    cardiac: bool = True                    # physiological oscillation component
    microsaccade: bool = True               # occasional fast spikes

    # --- scene ---
    anatomy: str = "macula"
    location: str = "macula"                # kept in sync with anatomy for retinal scenes
    reference_amp: float = 4.0

    # --- batch ---
    n_samples: int = 1
    seed: int = 0
    randomize: bool = True

    # ------------------------------------------------------------------ validation
    def normalized(self) -> "SimConfig":
        """Coerce/clamp/enum-snap in place and return self; raise on real conflicts."""
        self.center_wavelength_nm = float(self.center_wavelength_nm)
        self.pixels_per_spectrum = max(16, int(round(self.pixels_per_spectrum)))
        self.k_span_factor = max(0.5, float(self.k_span_factor))
        self.lines_per_frame = max(2, int(round(self.lines_per_frame)))
        self.lateral_fov_um = max(1.0, float(self.lateral_fov_um))
        self.beam_waist_um = max(0.1, float(self.beam_waist_um))
        self.n_frames = max(1, int(round(self.n_frames)))
        self.slow_fov_um = max(0.0, float(self.slow_fov_um))
        self.line_rate_hz = max(1.0, float(self.line_rate_hz))
        self.duty_cycle = min(1.0, max(1e-3, float(self.duty_cycle)))
        self.flyback_time_s = max(0.0, float(self.flyback_time_s))
        self.noise_level = max(0.0, float(self.noise_level))
        self.reference_amp = max(1e-6, float(self.reference_amp))
        self.n_samples = max(1, int(round(self.n_samples)))
        self.seed = int(self.seed)

        for name in ("bandwidth_fwhm_nm", "axial_resolution_um", "axial_range_um",
                     "axial_sampling_um", "lateral_resolution_um", "falloff_6db_um",
                     "target_snr_db", "motion_level", "axial_motion_um", "phase_drift_rad"):
            setattr(self, name, max(0.0, float(getattr(self, name))))

        self.dispersion = float(self.dispersion)   # signed tuning knob
        self.noise_free = bool(self.noise_free)
        self.cardiac = bool(self.cardiac)
        self.microsaccade = bool(self.microsaccade)
        self.randomize = bool(self.randomize)

        if self.scan_pattern not in SCAN_PATTERNS:
            self.scan_pattern = "bscan"
        if self.scan_pattern == "bscan":
            self.n_frames = 1
        if self.anatomy not in ANATOMIES:
            self.anatomy = "macula"
        if self.anatomy in ("macula", "onh", "periphery"):
            self.location = self.anatomy

        # resolvable conflict checks
        if self.bandwidth_fwhm_nm > 0 and self.axial_resolution_um > 0:
            implied = _axial_res_from_bandwidth(self.center_wavelength_nm, self.bandwidth_fwhm_nm)
            if abs(implied - self.axial_resolution_um) / implied > 0.01:
                raise ValueError(
                    f"Over-specified axial: bandwidth={self.bandwidth_fwhm_nm} nm implies "
                    f"axial_resolution={implied:.3f} µm, not {self.axial_resolution_um:.3f}. "
                    "Set one to 0 to derive it.")
        if self.bandwidth_fwhm_nm <= 0 and self.axial_resolution_um <= 0:
            self.bandwidth_fwhm_nm = 100.0
        if self.axial_range_um > 0 and self.axial_sampling_um > 0:
            implied = 2.0 * self.axial_range_um / self.pixels_per_spectrum
            if abs(implied - self.axial_sampling_um) / implied > 0.01:
                raise ValueError(
                    f"Over-specified axial depth: range={self.axial_range_um} µm and "
                    f"pixels={self.pixels_per_spectrum} imply sampling={implied:.4f} µm, not "
                    f"{self.axial_sampling_um:.4f}. Set one to 0 to derive it.")
        return self

    # ------------------------------------------------------------------ resolve
    def resolve(self) -> ResolvedConfig:
        """Back-solve the linked parameters and build the concrete physics objects."""
        self.normalized()
        lam0 = self.center_wavelength_nm

        # 1) bandwidth <-> axial resolution
        if self.bandwidth_fwhm_nm > 0:
            bw = self.bandwidth_fwhm_nm
        else:
            bw = _bandwidth_from_axial_res(lam0, self.axial_resolution_um)

        # 2) axial range / sampling <-> k_span_factor (pixels are primary)
        n_k = self.pixels_per_spectrum
        base = LightSource(center_wavelength_nm=lam0, bandwidth_fwhm_nm=bw, n_k=n_k)
        fwhm_k = base.fwhm_k
        if self.axial_range_um > 0:
            dk = np.pi / (2.0 * self.axial_range_um)
            k_span = dk * (n_k - 1) / (2.0 * fwhm_k)
        elif self.axial_sampling_um > 0:
            axrange = self.axial_sampling_um * n_k / 2.0
            dk = np.pi / (2.0 * axrange)
            k_span = dk * (n_k - 1) / (2.0 * fwhm_k)
        else:
            k_span = self.k_span_factor
        k_span = float(max(0.5, k_span))

        source = LightSource(center_wavelength_nm=lam0, bandwidth_fwhm_nm=bw,
                             n_k=n_k, k_span_factor=k_span)

        # 3) lateral resolution <-> beam waist
        if self.lateral_resolution_um > 0:
            beam_waist = lateral_fwhm_to_waist_um(self.lateral_resolution_um)
        else:
            beam_waist = self.beam_waist_um

        geometry = ScanGeometry(
            lateral_fov_um=self.lateral_fov_um, lines_per_frame=self.lines_per_frame,
            slow_fov_um=(self.slow_fov_um if self.scan_pattern == "volume" else 0.0),
            n_frames=(self.n_frames if self.scan_pattern == "volume" else 1))
        timing = Timing(
            line_rate_hz=self.line_rate_hz, lines_per_frame=self.lines_per_frame,
            n_frames=geometry.n_frames, duty_cycle=self.duty_cycle,
            flyback_time_s=self.flyback_time_s)

        noise = self._resolve_noise()

        derived = {
            "bandwidth_fwhm_nm": bw,
            "axial_resolution_um": source.axial_resolution_um,
            "axial_range_um": source.max_optical_depth_um,
            "axial_sampling_um": source.depth_pixel_um,
            "k_span_factor": k_span,
            "beam_waist_um": beam_waist,
            "lateral_resolution_um": waist_to_lateral_fwhm_um(beam_waist),
            "fast_pitch_um": geometry.fast_pitch_um,
            "slow_pitch_um": geometry.slow_pitch_um,
            "integration_time_us": timing.integration_time_s * 1e6,
            "frame_rate_hz": timing.frame_rate_hz,
            "acquisition_time_s": timing.volume_time_s,
        }

        disp_coeffs = [self.dispersion * DISPERSION_A2_UNIT] if self.dispersion else []
        derived["dispersion"] = self.dispersion

        return ResolvedConfig(
            source=source, geometry=geometry, timing=timing, beam_waist_um=beam_waist,
            reference_amp=self.reference_amp, dispersion_coeffs=disp_coeffs,
            falloff_6db_um=self.falloff_6db_um,
            noise=noise, motion_level=self.motion_level, axial_motion_um=self.axial_motion_um,
            phase_drift_rad=self.phase_drift_rad, cardiac=self.cardiac,
            microsaccade=self.microsaccade, anatomy=self.anatomy,
            scan_pattern=self.scan_pattern, derived=derived)

    def _resolve_noise(self) -> NoiseModel:
        if self.noise_free or self.noise_level <= 0:
            return NoiseModel.none()
        lvl = self.noise_level
        read = self.read
        if self.target_snr_db > 0:
            # background dominated by the read floor: SNR ≈ 20·log10(ref / read)
            read = self.reference_amp * 10.0 ** (-self.target_snr_db / 20.0)
        return NoiseModel(shot=self.shot * lvl, rin=self.rin * lvl, read=read * lvl)

    # ------------------------------------------------------------------ serialize
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_FIELDS = {f.name for f in dataclasses.fields(SimConfig)}


def config_from_mapping(data: dict[str, Any]) -> SimConfig:
    """Tolerant loader: drop unknown keys, coerce, and normalize."""
    clean = {k: v for k, v in (data or {}).items() if k in _FIELDS}
    return SimConfig(**clean).normalized()
