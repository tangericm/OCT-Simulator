"""Unit tests for spectral-domain D_spec / noise characterization."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import spectral_metrics as sm  # noqa: E402
from octsim.instrument import load_instrument  # noqa: E402


def test_profile_from_instrument_device_a():
    p = sm.profile_from_instrument("device_a")
    assert p.n_k == 2048
    assert 51.5 < p.snr_db < 52.5
    assert p.falloff_6db_um > 1500.0
    assert abs(np.max(p.envelope) - 1.0) < 1e-6


def test_bundled_spectral_device_a_exists():
    p = sm.bundled_spectral("device_a")
    assert p is not None
    assert p.n_k >= 2048 or len(p.envelope) >= 2048


def test_spectral_distance_zero_on_self():
    p = sm.profile_from_instrument("device_a")
    d = sm.spectral_distance(p, p)
    assert d["D_spec"] < 1e-12


def test_spectral_distance_detects_snr_mismatch():
    a = sm.profile_from_instrument("device_a")
    b = sm.SpectralProfile.from_dict({**a.to_dict(), "snr_db": a.snr_db * 0.5})
    d = sm.spectral_distance(a, b)
    assert d["D_spec_snr"] > 0.1


def test_estimate_noise_components_recovers_read_dominated():
    rng = np.random.default_rng(0)
    I = np.linspace(0.1, 1.0, 512) ** 2
    read = 0.05
    noisy = I + read * rng.standard_normal(I.shape)
    proxy = sm.estimate_noise_components(I, noisy)
    assert proxy["read_proxy"] > proxy["shot_proxy"]
    assert proxy["residual_std"] > 0.01


def test_falloff_matches_instrument_json():
    inst = load_instrument("device_a")
    p = sm.profile_from_instrument("device_a")
    assert abs(p.falloff_6db_um - inst.falloff_6db_um) < 1e-9


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
