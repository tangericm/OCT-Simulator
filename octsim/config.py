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
    speckle_corr_lateral_um: float
    pivot_distance_um: float
    scan_distortion_correction: float
    reference_amp: float
    dispersion_coeffs: list[float]     # derived from the SimConfig.dispersion knob
    falloff_6db_um: float
    axial_wrap: bool
    noise: NoiseModel
    averages: int
    motion_level: float
    axial_motion_um: float
    phase_drift_rad: float
    cardiac: bool
    microsaccade: bool
    anatomy: str
    scan_pattern: str
    sample_depth_um: float
    sample_lateral_offset_um: float
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
    #: Distance from the scan pivot (near the pupil) to the retina, µm. Drives the
    #: fan-beam sag term. This is SCAN GEOMETRY, not anatomy, and it is device-specific:
    #: measured B-scan sag over a nominal 6 mm scan is ~504 µm on the Maestro3 corpus but
    #: ~249 µm on Maestro2 — less than the anatomic term alone — because vendors apply
    #: different amounts of scan-distortion correction. Fitting it once and baking it into
    #: the retina model made every held-out corpus over-curve. 0 disables the term.
    pivot_distance_um: float = 20000.0
    #: Fraction of the geometric sag (anatomic bowl + fan-beam) that survives the vendor's
    #: scan-distortion correction. 1.0 = raw geometry, 0.0 = fully flattened.
    #: MEASURED per device against a nominal 6 mm B-scan, raw geometric sag ~585 µm:
    #:   Maestro3  504 µm -> 0.86      Maestro2  249 µm -> 0.43
    #:   OCTA-500   41 µm -> 0.28 (at 3 mm)
    #: Devices differ by more than 2x, so this cannot be a universal constant; it is the
    #: single knob that made every held-out corpus over-curve when it was baked in at the
    #: Maestro3 value. Calibrate it per corpus, do not fit it globally.
    #:
    #: ``None`` (the default) means "take it from ``instrument``, or 0.86 if that device
    #: does not carry one". An explicit float always wins. It cannot use the codebase's
    #: usual ``0`` sentinel because 0.0 already means something here — fully flattened.
    scan_distortion_correction: float | None = None
    beam_waist_um: float = 15.0
    #: Lateral correlation length of the sub-resolution speckle decorrelation, µm
    #: (Gaussian σ of the field). An INSTRUMENT property, and a **physical length**: one
    #: value reproduces the measured lateral speckle grain of all seven Maestro corpora,
    #: which span 2.93–23.44 µm of A-line pitch. Fitting the metric directly against those
    #: corpora gives σ ≈ 5 µm; the shipped 4 µm is the value that lands the *rendered*
    #: grain on target, because the beam-filtered component contributes as well. See the
    #: note in :func:`~octsim.sample.build_reflectivity_from_boundaries`. 0 = laterally
    #: white, which makes the grain scale with the A-line pitch (the pre-2026-07 behaviour).
    speckle_corr_lateral_um: float = 4.0
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
    #: render the scene onto the full n_k depth grid so structure deeper than z_max
    #: aliases back into the image, as it does on a real device (disc scans, and any
    #: mis-positioned retina). Costs one extra factor of 2 on the reflectivity build.
    axial_wrap: bool = True

    # --- noise ---
    noise_free: bool = False
    noise_level: float = 1.0
    # Noise coefficients, all calibrated together against the Maestro3 6 mm corpus
    # (scripts/validate_realism.py, and the device's own measured SNR of 52.0 dB recorded
    # in octsim/data/texture_maestro3_macula_meta.json). The previous values put every
    # tissue band ~11 dB closer to the noise floor than real data: the inner retina sat
    # 7 dB above the floor where real sits at 18 dB, and half the neurosensory retina was
    # within 3 dB of the floor. The floor here is shot-dominated, so all three had to move
    # together — cutting the read term alone changes the floor by ~1 dB.
    shot: float = 0.0075
    rin: float = 0.00125
    #: read-noise coefficient, as a fraction of the reference-arm DC level (see
    #: :meth:`_resolve_noise` — it is converted to an absolute std at resolve time).
    read: float = 0.0025
    target_snr_db: float = 0.0              # >0 -> back-solve the read floor
    averages: int = 1                       # incoherent B-scan averages (1 = raw; >1 = ART-like)

    # --- motion / artifacts ---
    motion_level: float = 0.0               # 0 -> no motion artifacts
    axial_motion_um: float = 2.0            # bulk axial random-walk amplitude scale
    phase_drift_rad: float = 0.5            # slow phase-drift amplitude
    cardiac: bool = True                    # physiological oscillation component
    microsaccade: bool = True               # occasional fast spikes

    # --- scene ---
    anatomy: str = "macula"
    location: str = "macula"                # kept in sync with anatomy for retinal scenes
    sample_depth_um: float = 200.0          # optical depth of the anterior surface (sample position)
    sample_lateral_offset_um: float = 0.0   # lateral shift of the anatomy centre within the FOV
    #: data-driven geometry from a learned shape model: "normal"/"jhu" = bundled healthy-eye
    #: model (default), "duke"/"dme" = diabetic-macular-edema model, a .npz path, or
    #: "parametric"/"none" for the hand-tuned pit.
    shape_model: str = "normal"
    #: tissue-optics profile (backscatter/attenuation/appearance): "literature" (default; the
    #: shipped optics) or a fitted keyword/path ("maestro3", …). Optics never touch geometry,
    #: so this leaves ``label_map`` bit-identical at a fixed seed.
    optics_profile: str = "literature"
    #: measured instrument grounding (source envelope P(k)/λ0/n_k/background/read floor):
    #: "" = the closed-form Gaussian source (default, unchanged); "maestro3" or a JSON path
    #: overlays the frozen Tier-1 measurements in :meth:`resolve`.
    instrument: str = ""
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
        self.averages = max(1, int(round(self.averages)))
        self.n_samples = max(1, int(round(self.n_samples)))
        self.seed = int(self.seed)

        for name in ("bandwidth_fwhm_nm", "axial_resolution_um", "axial_range_um",
                     "axial_sampling_um", "lateral_resolution_um", "falloff_6db_um",
                     "target_snr_db", "motion_level", "axial_motion_um", "phase_drift_rad",
                     "sample_depth_um"):
            setattr(self, name, max(0.0, float(getattr(self, name))))
        self.sample_lateral_offset_um = float(self.sample_lateral_offset_um)  # signed
        self.shape_model = str(self.shape_model or "")
        self.optics_profile = str(self.optics_profile or "literature")
        self.instrument = str(self.instrument or "")

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

        # 2b) instrument overlay (Tier 1, measured + frozen): replace the closed-form Gaussian
        #     source with the device's measured envelope P(k) and its grounded λ0/n_k/BW/k-span.
        #     Empty instrument -> unchanged. The measured envelope aligns to this source's k_grid.
        inst = None
        if self.instrument:
            try:
                from .instrument import load_instrument
                inst = load_instrument(self.instrument)
                source = inst.as_source()
                lam0 = source.center_wavelength_nm
                n_k = source.n_k
                bw = source.bandwidth_fwhm_nm
                k_span = source.k_span_factor
            except Exception:
                inst = None             # missing/invalid instrument -> keep the Gaussian source

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

        noise = self._resolve_noise(source, inst)

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
            speckle_corr_lateral_um=self.speckle_corr_lateral_um,
            pivot_distance_um=self.pivot_distance_um,
            scan_distortion_correction=self.effective_scan_distortion(),
            reference_amp=self.reference_amp, dispersion_coeffs=disp_coeffs,
            falloff_6db_um=self.falloff_6db_um, axial_wrap=bool(self.axial_wrap),
            noise=noise, averages=self.averages,
            motion_level=self.motion_level, axial_motion_um=self.axial_motion_um,
            phase_drift_rad=self.phase_drift_rad, cardiac=self.cardiac,
            microsaccade=self.microsaccade, anatomy=self.anatomy,
            scan_pattern=self.scan_pattern, sample_depth_um=self.sample_depth_um,
            sample_lateral_offset_um=self.sample_lateral_offset_um, derived=derived)

    #: Applied when neither the config nor the instrument names a value. It is the Maestro3
    #: number, kept only so a config with no instrument renders as it always has.
    DEFAULT_SCAN_DISTORTION = 0.86

    def effective_scan_distortion(self) -> float:
        """Vendor scan-distortion correction actually in force: config > instrument > default.

        Lives here rather than in :meth:`resolve` because ``generate.build_scene`` stamps
        this onto the retina from the *config*, not from the ``ResolvedConfig`` — overlaying
        it only into the latter would leave the rendered image unchanged while making
        ``resolved.scan_distortion_correction`` report the device value.
        """
        if self.scan_distortion_correction is not None:
            return float(self.scan_distortion_correction)
        if self.instrument:
            try:
                from .instrument import load_instrument
                v = load_instrument(self.instrument).scan_distortion_correction
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return float(self.DEFAULT_SCAN_DISTORTION)

    def _resolve_noise(self, source=None, instrument=None) -> NoiseModel:
        """Build the noise model. ``NoiseModel.read`` is an ABSOLUTE per-pixel std.

        ``self.read`` stays a dimensionless coefficient for backwards compatibility; it is
        converted here against the reference-arm DC level ``mean(P(k))·a_ref²``, which is
        what ``mean(|spectrum|)`` used to stand in for. Doing the conversion once at
        resolve time (rather than per frame, against the scene) makes the floor a property
        of the instrument instead of a property of whatever tissue is in view.

        Note ``Instrument.read_floor`` is deliberately NOT adopted here: it is written as
        ``read_std / 255`` (an 8-bit FPN scale, ``scripts/build_instrument.py``), which is
        not commensurate with the reference-DC units used above. Adopting it directly
        raises the floor ~11x and drives the whole retina into the noise. Grounding the
        floor in it needs that measurement re-expressed in spectrum units first.

        ``Instrument.snr_db`` IS adopted, as a multiplier on all three coefficients. SNR is
        an acquisition property, not a tissue one, and supplying each device's own is what
        took held-out Maestro2 from 4/7 to 9/9 with no tissue refitting. It scales the whole
        model rather than the read term because the floor is shot-dominated — see the note
        on the coefficients above. ``target_snr_db`` is NOT the channel for this: measured
        on a Maestro2 render it moves the deep floor +3.0 dB where the device gap is 6.7 dB,
        because it back-solves ``read`` alone.
        """
        if self.noise_free or self.noise_level <= 0:
            return NoiseModel.none()
        lvl = self.noise_level
        if instrument is not None:
            lvl *= float(getattr(instrument, "noise_scale", 1.0))

        ref_dc = float(self.reference_amp) ** 2
        if source is not None:
            try:
                ref_dc *= float(np.mean(source.power_spectrum(source.k_grid())))
            except Exception:
                pass

        read = float(self.read) * ref_dc
        if self.target_snr_db > 0:
            # background dominated by the read floor: SNR ≈ 20·log10(ref / read)
            read = ref_dc * 10.0 ** (-self.target_snr_db / 20.0)
        return NoiseModel(shot=self.shot * lvl, rin=self.rin * lvl, read=read * lvl)

    # ------------------------------------------------------------------ serialize
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_FIELDS = {f.name for f in dataclasses.fields(SimConfig)}


def config_from_mapping(data: dict[str, Any]) -> SimConfig:
    """Tolerant loader: drop unknown keys, coerce, and normalize."""
    clean = {k: v for k, v in (data or {}).items() if k in _FIELDS}
    return SimConfig(**clean).normalized()
