"""Tier-1 instrument grounding — the *measured*, frozen device model.

Everything in a :class:`CalibrationProfile` is *fit*; everything here is **measured** from
the device's own calibration files and then frozen, so a tissue-optics fit can never absorb
an instrument error. The measurements come from a commercial SD-OCT scan folder:

* ``*.CLB``   — the k-linearisation LUT (first 2048 float32 ∈ [0,1]); confirms 2048 pixels
  and gives the depth calibration used to set ``k_span_factor``.
* ``fpn.bin`` / ``FPN.raw`` — the detector fixed-pattern frame (``2048 × N`` uint16). Its
  mean A-line, k-linearised, is the **measured source envelope** ``P(k)``; its tail is the
  electronic/background floor and its per-pixel variance seeds the read noise.
* ``back*.raw`` — per-volume background acquisitions (reference arm only): a cleaner
  ``background`` for :func:`~octsim.reconstruct.reconstruct_bscan` than FPN alone.
* mirror scans — a point reflector vs depth → the sensitivity **roll-off** ``falloff_6db_um``
  and the **axial PSF FWHM**.
* ``MainInfo.bin`` — device header (a commercial 840 nm SD-OCT, ~840 nm SD-OCT ⇒ ``center_wavelength_nm≈840``).

:func:`load_instrument` reads a bundled/path JSON produced by an offline instrument calibration.
:meth:`Instrument.as_source` builds the concrete :class:`~octsim.source.LightSource` whose
``power_spectrum`` is the measured envelope. Wiring: ``SimConfig.instrument="device_a"`` →
:meth:`SimConfig.resolve` overlays it; ``""`` (default) keeps the closed-form Gaussian source.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .source import LightSource

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

#: Measured image SNR (dB) of the corpus the noise coefficients in :class:`SimConfig` were
#: calibrated against — Device A 6 mm/1024, from
#: ``octsim/data/texture_device_a_macula_meta.json``. A device with its own measured
#: ``snr_db`` is scaled relative to this; see :attr:`Instrument.noise_scale`.
#: Full precision on purpose: it makes ``noise_scale`` exactly 1.0 for the calibration
#: device, so overlaying an instrument cannot nudge the corpus it was calibrated on.
REFERENCE_SNR_DB = 52.04221179196247

#: bundled instrument keywords → JSON filename (raw calibration files are never committed).
_BUNDLED_INSTRUMENT = {
    "device_a": "instrument_device_a.json",
    "device_b": "instrument_device_b.json",
}


@dataclass(frozen=True)
class Instrument:
    """A frozen, measured device model (Tier 1). JSON-serialisable; arrays inline as lists."""

    name: str
    center_wavelength_nm: float = 840.0     # a commercial 840 nm SD-OCT SD-OCT source
    n_k: int = 2048
    bandwidth_fwhm_nm: float = 45.0         # Gaussian-fit width of the measured envelope
    k_span_factor: float = 1.5              # set so the depth pixel matches the CLB calibration
    power_envelope: np.ndarray | None = field(default=None)   # [n_k] measured P(k), peak-norm
    background: np.ndarray | None = field(default=None)       # [n_k] k-linearised background
    reference_amp: float = 4.0              # DC/floor seed for the reference term
    read_floor: float = 0.01               # NoiseModel.read seed (per-pixel electronic std)
    floor_db: float = -60.0                # display/noise floor seed
    falloff_6db_um: float = 0.0            # sensitivity roll-off (0 = none measured)
    axial_fwhm_um: float = 0.0            # measured axial PSF FWHM (diagnostic)
    #: Fraction of the anatomic bowl surviving the vendor's scan-distortion correction.
    #: A DEVICE property: measured against tilt-free quadratic sag on a nominal 6 mm
    #: B-scan (raw anatomic sag at R=12 mm ≈ 381 µm), Device A ~311 µm -> 0.82 and
    #: Device B ~257 µm -> 0.67. Earlier 0.86/0.43 tracked peak-to-peak sag (tilt-
    #: inflated on Device A). ``None`` = not measured; the SimConfig default applies.
    scan_distortion_correction: float | None = None
    #: Measured image SNR of this device's corpus, dB. Scales the noise coefficients
    #: relative to :data:`REFERENCE_SNR_DB`, the corpus they were calibrated against.
    #: 0 = not measured, noise left at the calibration level.
    snr_db: float = 0.0
    source: str = "measured"             # provenance string

    # ------------------------------------------------------------------ (de)serialize
    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "center_wavelength_nm": float(self.center_wavelength_nm),
            "n_k": int(self.n_k),
            "bandwidth_fwhm_nm": float(self.bandwidth_fwhm_nm),
            "k_span_factor": float(self.k_span_factor),
            "power_envelope": (None if self.power_envelope is None
                               else np.asarray(self.power_envelope, float).tolist()),
            "background": (None if self.background is None
                           else np.asarray(self.background, float).tolist()),
            "reference_amp": float(self.reference_amp),
            "read_floor": float(self.read_floor),
            "floor_db": float(self.floor_db),
            "falloff_6db_um": float(self.falloff_6db_um),
            "axial_fwhm_um": float(self.axial_fwhm_um),
            "scan_distortion_correction": (None if self.scan_distortion_correction is None
                                           else float(self.scan_distortion_correction)),
            "snr_db": float(self.snr_db),
            "source": self.source,
        }
        return d

    @property
    def noise_scale(self) -> float:
        """Multiplier taking the calibration corpus's noise to this device's own.

        SNR is an acquisition property — gain, reference power, integration time — not a
        tissue one, so a quieter or noisier unit must not be absorbed into the layer table.
        Supplying each device's own measured SNR is what took held-out Device B from 4/7 to
        9/9 in round 2 with zero tissue refitting.
        """
        if not self.snr_db:
            return 1.0
        return float(10.0 ** ((REFERENCE_SNR_DB - float(self.snr_db)) / 20.0))

    @classmethod
    def from_dict(cls, d: dict) -> "Instrument":
        d = dict(d)
        pe = d.get("power_envelope")
        bg = d.get("background")
        d["power_envelope"] = None if pe is None else np.asarray(pe, dtype=float)
        d["background"] = None if bg is None else np.asarray(bg, dtype=float)
        # tolerate extra/missing keys
        known = {f for f in cls.__dataclass_fields__}          # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path) -> "Instrument":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    # ------------------------------------------------------------------ physics
    def as_source(self) -> LightSource:
        """Build the measured-envelope :class:`~octsim.source.LightSource`.

        The stored ``power_envelope`` is aligned to the ``k_grid`` this source produces
        (same ``center_wavelength_nm``/``n_k``/``bandwidth_fwhm_nm``/``k_span_factor``), so
        ``power_spectrum()`` returns the measured shape rather than the Gaussian.
        """
        env = None if self.power_envelope is None else np.asarray(self.power_envelope, float)
        return LightSource(
            center_wavelength_nm=float(self.center_wavelength_nm), n_k=int(self.n_k),
            bandwidth_fwhm_nm=float(self.bandwidth_fwhm_nm),
            k_span_factor=float(self.k_span_factor), measured_power=env)


def load_instrument(name: str) -> Instrument:
    """Resolve an instrument keyword ("device_a") or JSON path to an :class:`Instrument`."""
    key = (name or "").lower()
    fn = _BUNDLED_INSTRUMENT.get(key)
    path = os.path.join(_DATA_DIR, fn) if fn else name
    return Instrument.load(path)


def has_instrument(name: str) -> bool:
    """True if ``name`` resolves to a committed instrument JSON (keyword or path)."""
    if not name:
        return False
    key = name.lower()
    if key in _BUNDLED_INSTRUMENT:
        return os.path.exists(os.path.join(_DATA_DIR, _BUNDLED_INSTRUMENT[key]))
    return os.path.exists(name)
