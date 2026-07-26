"""A device's own numbers must be reachable from ``SimConfig(instrument=...)``.

Round 2's central result was that the overfitting lived in the DEVICE parameters, not the
tissue model: held-out Maestro2 went 4/7 -> 9/9 on its own scan-distortion correction and
its own SNR, with zero tissue refitting. But those two numbers lived only in the validation
harness, so `simulate(SimConfig(instrument="maestro2"))` still rendered Maestro3 curvature
at Maestro3 noise, and the headline result was not reproducible from the simulator itself.

These tests pin the wiring, not the values: config beats instrument beats default, and an
instrument with no measured value changes nothing.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import SimConfig  # noqa: E402
from octsim.config import config_from_mapping  # noqa: E402
from octsim.instrument import REFERENCE_SNR_DB, Instrument, load_instrument  # noqa: E402

M2_DISTORTION, M3_DISTORTION = 0.43, 0.86


def test_instrument_json_carries_the_device_numbers():
    m3, m2 = load_instrument("maestro3"), load_instrument("maestro2")
    assert m3.scan_distortion_correction == M3_DISTORTION
    assert m2.scan_distortion_correction == M2_DISTORTION
    # Vendors flatten the same anatomy by more than 2x -- that is why this is a device
    # property and cannot be a universal constant.
    assert m2.scan_distortion_correction < m3.scan_distortion_correction / 1.5
    assert 45.0 < m2.snr_db < 46.0 and 51.5 < m3.snr_db < 52.5


def test_scan_distortion_resolves_config_then_instrument_then_default():
    assert SimConfig(instrument="maestro2").effective_scan_distortion() == M2_DISTORTION
    assert SimConfig(instrument="maestro3").effective_scan_distortion() == M3_DISTORTION
    assert SimConfig(instrument="").effective_scan_distortion() == M3_DISTORTION
    # an explicit value always wins, including one that equals another device's
    assert SimConfig(instrument="maestro2",
                     scan_distortion_correction=0.9).effective_scan_distortion() == 0.9
    assert SimConfig(instrument="maestro2",
                     scan_distortion_correction=0.0).effective_scan_distortion() == 0.0


def test_resolved_config_and_scene_agree_on_the_distortion():
    """``generate.build_scene`` stamps this from the CONFIG, so both paths must match.

    Overlaying only into ``ResolvedConfig`` would make ``resolved`` report 0.43 while the
    rendered retina still curved by 0.86 -- a test that passes on an unchanged image.
    """
    from octsim.generate import build_scene
    import numpy as np
    cfg = SimConfig(instrument="maestro2", anatomy="macula", lateral_fov_um=3000.0,
                    lines_per_frame=64, shape_model="")
    assert cfg.resolve().scan_distortion_correction == M2_DISTORTION
    scene = build_scene(cfg, np.random.default_rng(0))
    assert float(scene.scan_distortion_correction) == M2_DISTORTION


def test_device_snr_scales_the_noise_and_the_reference_device_does_not():
    m2 = load_instrument("maestro2")
    quiet = SimConfig(instrument="maestro3").resolve().noise
    noisy = SimConfig(instrument="maestro2").resolve().noise
    plain = SimConfig(instrument="").resolve().noise

    # the corpus the coefficients were calibrated on must be left exactly alone
    assert quiet.shot == plain.shot and quiet.rin == plain.rin
    # ...and a noisier unit scales all three together, because the floor is shot-dominated
    assert noisy.shot / plain.shot == m2.noise_scale
    assert noisy.rin / plain.rin == m2.noise_scale
    assert 2.0 < m2.noise_scale < 2.3            # 52.04 -> 45.33 dB is ~2.17x

    # noise_level stays a user multiplier on top of the device level
    louder = SimConfig(instrument="maestro2", noise_level=2.0).resolve().noise
    assert abs(louder.shot - 2.0 * noisy.shot) < 1e-12


def test_noise_scale_is_one_without_a_measured_snr():
    assert Instrument(name="x").noise_scale == 1.0
    assert Instrument(name="x", snr_db=REFERENCE_SNR_DB).noise_scale == 1.0


def test_new_fields_survive_a_json_round_trip():
    """``to_dict`` is a hand-written key list and ``from_dict`` drops unknown keys."""
    inst = Instrument(name="probe", scan_distortion_correction=0.5, snr_db=40.0)
    back = Instrument.from_dict(inst.to_dict())
    assert back.scan_distortion_correction == 0.5 and back.snr_db == 40.0


def test_config_round_trip_survives_the_none_sentinel():
    cfg = SimConfig(instrument="maestro2")
    assert config_from_mapping(cfg.to_dict()).to_dict() == cfg.normalized().to_dict()
    assert cfg.to_dict()["scan_distortion_correction"] is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} instrument device-number checks passed")
