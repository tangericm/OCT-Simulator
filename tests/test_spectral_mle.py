"""Physics MLE recovers NoiseModel coefficients from synthetic A-lines."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import SimConfig, spectral_mle as mle  # noqa: E402
from octsim.config import DEFAULT_REFERENCE_AMP  # noqa: E402
from octsim.instrument import load_instrument  # noqa: E402
from octsim.noise import NoiseModel  # noqa: E402


def _synth_alines(mu: np.ndarray, shot: float, rin: float, read: float, *,
                  n_alines: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = NoiseModel(shot=shot, rin=rin, read=read)
    cols = [noise.apply(mu, rng) for _ in range(n_alines)]
    return np.stack(cols, axis=1)


def test_mle_recovers_read_dominated():
    mu = np.linspace(0.2, 1.5, 256) ** 2
    shot, rin, read = 0.01, 0.002, 0.08
    spec = _synth_alines(mu, shot, rin, read, n_alines=2000, seed=1)
    s_hat, r_hat, rd_hat, nll = mle.fit_noise_mle(spec, mu=mu)
    assert nll < 2.0
    # read should dominate and be recovered within ~25%
    assert abs(rd_hat - read) / read < 0.25
    assert rd_hat > s_hat * np.sqrt(np.mean(mu)) * 0.5


def test_mle_recovers_shot_rin_mix():
    mu = np.linspace(0.5, 4.0, 512)
    shot, rin, read = 0.05, 0.01, 0.02
    spec = _synth_alines(mu, shot, rin, read, n_alines=4000, seed=2)
    s_hat, r_hat, rd_hat, _ = mle.fit_noise_mle(spec, mu=mu)
    assert abs(s_hat - shot) / shot < 0.35
    assert abs(r_hat - rin) / rin < 0.45
    assert abs(rd_hat - read) / read < 0.35


def test_noise_variance_matches_empirical():
    mu = np.ones(128) * 2.0
    shot, rin, read = 0.03, 0.005, 0.04
    spec = _synth_alines(mu, shot, rin, read, n_alines=3000, seed=3)
    var_emp = np.mean((spec - mu[:, None]) ** 2, axis=1)
    var_hat = mle.noise_variance(mu, shot, rin, read)
    # relative error on the mean variance
    assert abs(float(np.mean(var_emp)) - float(np.mean(var_hat))) / float(np.mean(var_hat)) < 0.1


def test_reference_amp_explicit_or_default():
    """Instrument JSON reference_amp is a seed — not auto-adopted into the sim."""
    assert SimConfig(instrument="device_a").effective_reference_amp() == DEFAULT_REFERENCE_AMP
    assert SimConfig(instrument="").effective_reference_amp() == DEFAULT_REFERENCE_AMP
    assert SimConfig(instrument="device_a",
                     reference_amp=5.5).effective_reference_amp() == 5.5
    assert SimConfig(instrument="device_a").resolve().reference_amp == DEFAULT_REFERENCE_AMP


def test_head_skip_constant_is_positive():
    """Leading background spike bins are excluded — not given a forward-model term."""
    assert mle.HEAD_SKIP >= 3


def test_apply_kwargs_roundtrip():
    res = mle.SpectralMLEResult(
        shot=0.01, rin=0.002, read=0.003, read_abs=0.05, reference_amp=12.0,
        scale=100.0, nll=1.0, n_alines=10, n_k=32, source="unit")
    kw = mle.apply_to_simconfig_kwargs(res)
    cfg = SimConfig(instrument="device_a", shot=kw["shot"], rin=kw["rin"], read=kw["read"],
                    reference_amp=kw["reference_amp"])
    n = cfg.resolve().noise
    assert abs(n.shot - res.shot) < 1e-12
    assert abs(n.rin - res.rin) < 1e-12


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
