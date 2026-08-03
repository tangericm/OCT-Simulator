"""Physics MLE for spectral noise / signal coefficients (A-line i.i.d.).

Fits ``shot``, ``rin``, and absolute ``read`` under the same heteroscedastic Gaussian
model as :class:`~octsim.noise.NoiseModel`:

    I(k,x) ~ N(μ(k,x),  σ²(μ)),
    σ²(μ)  = (shot · √|μ|)² + (rin · |μ|)² + read²

Variance must be **temporal** (across repeated frames at fixed A-line index). Cross-A-line
spread in a single FPN frame is dominated by fixed-pattern structure and must not be used
as the noise residual. Mean envelope μ(k) may still come from FPN / frame average.

Tissue shape is out of scope — freeze P(k) / falloff from the Tier-1 instrument JSON.

Leading detector bins may carry a background/DC spike on device raw data; those bins
are neutralized and excluded from the fit (:data:`HEAD_SKIP`) and are **not** modelled
in the forward path.

Units: real uint16 spectra are scaled into the simulator's interferogram units via the
noise-free reference DC ``P(k)·a_ref²``, so the fitted coefficients drop straight into
:class:`~octsim.config.SimConfig`.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

from .data import real_device
from .instrument import load_instrument
from .noise import NoiseModel

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

#: default bare-instrument reference amplitude when no device JSON is loaded
DEFAULT_REFERENCE_AMP = 4.0

#: Leading detector bins often carry a non-physical DC/background spike on device
#: raw spectra. Neutralize and exclude them from MLE — do **not** put a spike term
#: in the forward / noise model.
HEAD_SKIP = 8


@dataclass
class SpectralMLEResult:
    """Fitted noise coefficients in simulator units (+ diagnostics)."""

    shot: float
    rin: float
    read: float                 # SimConfig.read fraction of reference DC
    read_abs: float             # absolute read std in spectrum units
    reference_amp: float
    scale: float                # real → sim: I_sim ≈ I_real / scale
    nll: float
    n_alines: int
    n_k: int
    source: str = ""
    # moment match diagnostics (device counts, pre-scale)
    var_mse: float = 0.0
    shot_frac: float = 0.0      # mean predicted shot variance / total
    rin_frac: float = 0.0
    read_frac_var: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectralMLEResult":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def as_noise_model(self, *, ref_dc: float) -> NoiseModel:
        """Absolute :class:`NoiseModel` (``read`` already absolute)."""
        return NoiseModel(shot=self.shot, rin=self.rin, read=self.read_abs)


def noise_variance(mu: np.ndarray, shot: float, rin: float, read_abs: float) -> np.ndarray:
    """Predicted per-pixel variance under :class:`NoiseModel` (no draws)."""
    a = np.abs(np.asarray(mu, float))
    return (float(shot) ** 2) * a + (float(rin) ** 2) * (a ** 2) + float(read_abs) ** 2


def gaussian_nll(spectra: np.ndarray, mu: np.ndarray, shot: float, rin: float,
                 read_abs: float) -> float:
    """Mean per-sample NLL (ignore 2π) of i.i.d. A-lines under the noise model."""
    s = np.asarray(spectra, float)
    if s.ndim == 1:
        s = s[:, None]
    m = np.asarray(mu, float).reshape(-1)
    if m.size != s.shape[0]:
        raise ValueError("mu length must match n_k")
    var = np.maximum(noise_variance(m, shot, rin, read_abs), 1e-24)
    r = s - m[:, None]
    return float(np.mean(0.5 * (r ** 2 / var[:, None] + np.log(var)[:, None])))


def _nll_from_moments(mu: np.ndarray, var_emp: np.ndarray,
                      shot: float, rin: float, read_abs: float) -> float:
    """Gaussian NLL (mean per sample) from per-k mean-square residuals ``var_emp``."""
    var = np.maximum(noise_variance(mu, shot, rin, read_abs), 1e-24)
    return float(0.5 * np.mean(var_emp / var + np.log(var)))


def _read_raw_frame(path: str) -> np.ndarray:
    """uint16 device frame → float ``[n_k, n_alines]``, leading spike neutralized.

    The first few detector pixels can spike in background/FPN; that artifact is
    overwritten from the first clean bin so it never enters the noise MLE. The
    forward model does not attempt to reproduce the spike.
    """
    pix = real_device.PIX
    alines = int(os.path.getsize(path) // 2 // pix)
    d = np.fromfile(path, dtype=np.uint16)
    if d.size != pix * alines:
        raise ValueError(f"{os.path.basename(path)}: {d.size} != {pix * alines}")
    raw = d.reshape((pix, alines), order="F").astype(np.float64)
    hs = min(HEAD_SKIP, raw.shape[0] - 1)
    if hs > 0:
        raw[:hs, :] = raw[hs:hs + 1, :]
    return raw


def load_fpn_spectra(folder: str, *, klinear: bool = True,
                     max_alines: int | None = 4096) -> np.ndarray:
    """Load FPN mean-supporting spectra ``[n_k, n_alines]`` (envelope / scale only)."""
    fpn_path = None
    for name in ("fpn.bin", "FPN.raw"):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and not real_device.is_placeholder(p):
            fpn_path = p
            break
    if fpn_path is None:
        paths = [p for p in sorted(glob.glob(os.path.join(folder, "bscan*.raw")))
                 if not real_device.is_placeholder(p)][:1]
        if not paths:
            raise FileNotFoundError(f"no FPN or bscan*.raw in {folder}")
        fpn_path = paths[0]
    spec = _read_raw_frame(fpn_path)
    if max_alines is not None and spec.shape[1] > max_alines:
        idx = np.linspace(0, spec.shape[1] - 1, max_alines).astype(int)
        spec = spec[:, idx]
    if klinear:
        lut = real_device.read_clb(real_device.find_clb(folder))
        spec = np.asarray(real_device.klinear_operator(lut)(spec), float)
    return spec


def _electronic_tail(env: np.ndarray) -> float:
    """Electronic offset from the *high-k* tail only.

    The leading detector edge may still carry residual spike structure even after
    :data:`HEAD_SKIP` neutralization; do not let it set the offset.
    """
    e = np.asarray(env, float).reshape(-1)
    n = max(8, e.size // 50)
    return float(np.median(e[-n:]))


def load_temporal_moments(folder: str, *, n_frames: int = 24, max_alines: int = 256,
                          klinear: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """Temporal mean / variance of repeated bscans: ``(mu[k], var[k], n_obs)``.

    For each A-line index, variance is taken across frames (fixed-pattern cancels).
    ``var[k]`` is then the mean over A-line indices. ``n_obs ≈ n_frames * n_alines``.

    The per-frame electronic offset (extreme-k median) is subtracted before moments so
    the variance model sees optical intensity, matching ``P(k)·a_ref²`` which is ~0 at
    the spectral tails. Without that, shot·√μ absorbs the floor and ``read`` collapses.
    """
    paths = [p for p in sorted(glob.glob(os.path.join(folder, "bscan*.raw")))
             if not real_device.is_placeholder(p)][:n_frames]
    if len(paths) < 2:
        raise FileNotFoundError(
            f"need ≥2 hydrated bscan*.raw in {folder} for temporal noise MLE "
            f"(found {len(paths)})")

    kop = None
    if klinear:
        lut = real_device.read_clb(real_device.find_clb(folder))
        kop = real_device.klinear_operator(lut)

    stack = []
    for p in paths:
        frame = _read_raw_frame(p)
        if kop is not None:
            frame = np.asarray(kop(frame), float)
        if max_alines is not None and frame.shape[1] > max_alines:
            idx = np.linspace(0, frame.shape[1] - 1, max_alines).astype(int)
            frame = frame[:, idx]
        # electronic offset out — optical envelope remains for the shot/RIN predictor
        frame = frame - _electronic_tail(frame.mean(axis=1))
        stack.append(frame)
    n_x = min(f.shape[1] for f in stack)
    n_k = min(f.shape[0] for f in stack)
    arr = np.stack([f[:n_k, :n_x] for f in stack], axis=0)  # [F, K, X]

    # Predictor μ = total optical intensity on the detector (incl. reference DC).
    mu_for_model = np.maximum(arr.mean(axis=(0, 2)), 0.0)

    # Residuals must match the recon path: real_device subtracts the mean A-line
    # (common-mode source RIN / envelope). Fitting raw I(k) variance attributes that
    # common mode to NoiseModel.rin, which the sim applies *after* which analytic DC
    # removal cannot cancel — collapsing B-scan SNR by ~30 dB. Use AC spectra instead.
    ac = arr - arr.mean(axis=2, keepdims=True)
    var = ac.var(axis=0, ddof=1).mean(axis=1)

    # Drop leading bins from the returned vectors so the spike region cannot enter MLE
    # even if k-linearisation smeared a residual. Trailing optical-dark bins stay.
    hs = min(HEAD_SKIP, max(0, mu_for_model.size // 8))
    if hs > 0:
        mu_for_model = mu_for_model[hs:]
        var = var[hs:]
    n_obs = int(arr.shape[0] * arr.shape[2])
    return mu_for_model, var, n_obs


def sim_reference_dc(instrument: str = "device_a",
                     reference_amp: float | None = None) -> tuple[np.ndarray, float, float]:
    """Return ``(μ_sim[k], a_ref, ref_dc_mean)`` for the noise-free reference arm."""
    inst = load_instrument(instrument)
    src = inst.as_source()
    a_ref = float(reference_amp if reference_amp is not None else
                  (inst.reference_amp or DEFAULT_REFERENCE_AMP))
    P = np.asarray(src.power_spectrum(src.k_grid()), float)
    mu = P * (a_ref ** 2)
    return mu, a_ref, float(np.mean(mu))


def fit_noise_moments(mu: np.ndarray, var_emp: np.ndarray, *,
                      x0: tuple[float, float, float] | None = None,
                      bounds_log: tuple[tuple[float, float], ...] | None = None,
                      freeze_rin: float | None = None,
                      ) -> tuple[float, float, float, float]:
    """MLE of ``(shot, rin, read_abs)`` from per-k mean and temporal variance.

    Bright spectral bins on mirror/tissue are dominated by fringe / motion instability
    that looks like RIN (∝μ²) but is not detector RIN. Pass ``freeze_rin`` (e.g. the
    image-calibrated prior) on real data; leave ``None`` for synthetic recovery tests.
    """
    mu_v = np.asarray(mu, float).reshape(-1)
    var_v = np.asarray(var_emp, float).reshape(-1)
    if mu_v.size != var_v.size or mu_v.size < 16:
        raise ValueError("mu/var must share length ≥ 16")

    # Tail-pinned read: extreme-k / low-μ bins are optical-dark after envelope subtract.
    tail = mu_v <= max(np.percentile(mu_v, 12.0), 1e-9)
    read_tail = float(np.sqrt(max(np.median(var_v[tail]), 1e-18)))

    a_abs = np.abs(mu_v) + 1e-12
    if freeze_rin is not None:
        # var - read² - rin²μ² ≈ shot² μ   on low/mid intensity (exclude bright fringes)
        rin_f = float(freeze_rin)
        resid = np.maximum(var_v - read_tail ** 2 - (rin_f ** 2) * (a_abs ** 2), 0.0)
        mid = (mu_v > np.percentile(mu_v, 12.0)) & (mu_v < np.percentile(mu_v, 60.0))
        if not np.any(mid):
            mid = ~tail
        shot0 = float(np.sqrt(max(np.sum(resid[mid]) / max(np.sum(a_abs[mid]), 1e-18), 1e-18)))
        x0 = (shot0, rin_f, read_tail) if x0 is None else x0
        log0 = np.log10(np.maximum(np.asarray([x0[0], x0[2]], float), 1e-18))
        lo, hi = (-6.0, 2.0), (-6.0, 4.0)

        def obj(z):
            shot, read = 10.0 ** float(z[0]), 10.0 ** float(z[1])
            return _nll_from_moments(mu_v, var_v, shot, rin_f, read)

        res = minimize(obj, log0, method="L-BFGS-B", bounds=(lo, hi))
        shot = 10.0 ** float(res.x[0])
        read_abs = 10.0 ** float(res.x[1])
        return float(shot), float(rin_f), float(read_abs), float(res.fun)

    A = np.column_stack([a_abs, a_abs ** 2, np.ones_like(a_abs)])
    coeff, *_ = np.linalg.lstsq(A, np.maximum(var_v, 0.0), rcond=None)
    shot0 = float(np.sqrt(max(coeff[0], 1e-18)))
    rin0 = float(np.sqrt(max(coeff[1], 1e-18)))
    read0 = float(np.sqrt(max(coeff[2], 1e-18)))
    if x0 is None:
        x0 = (shot0, rin0, read0)
    log0 = np.log10(np.maximum(np.asarray(x0, float), 1e-18))
    if bounds_log is None:
        bounds_log = ((-6.0, 2.0), (-6.0, 2.0), (-6.0, 4.0))

    def obj(z):
        shot, rin, read = (10.0 ** float(z[0]), 10.0 ** float(z[1]), 10.0 ** float(z[2]))
        return _nll_from_moments(mu_v, var_v, shot, rin, read)

    res = minimize(obj, log0, method="L-BFGS-B", bounds=bounds_log)
    z = res.x
    shot, rin, read_abs = (10.0 ** float(z[0]), 10.0 ** float(z[1]), 10.0 ** float(z[2]))
    return shot, rin, read_abs, float(res.fun)


def fit_noise_mle(spectra: np.ndarray, *,
                  mu: np.ndarray | None = None,
                  x0: tuple[float, float, float] | None = None,
                  bounds_log: tuple[tuple[float, float], ...] | None = None,
                  ) -> tuple[float, float, float, float]:
    """MLE from a ``[n_k, n_alines]`` stack treated as i.i.d. replicates of μ(k).

    Prefer :func:`fit_noise_moments` with **temporal** variances on real data; this
    entry point is for synthetic recovery tests and controlled replicates.
    """
    s = np.asarray(spectra, float)
    if s.ndim != 2:
        raise ValueError(f"spectra must be [n_k, n_alines], got {s.shape}")
    mu_v = s.mean(axis=1) if mu is None else np.asarray(mu, float).reshape(-1)
    if mu_v.size != s.shape[0]:
        raise ValueError("mu length must match n_k")
    var_emp = np.mean((s - mu_v[:, None]) ** 2, axis=1)
    return fit_noise_moments(mu_v, var_emp, x0=x0, bounds_log=bounds_log)


def fit_folder(folder: str, *, instrument: str = "device_a",
               reference_amp: float | None = None,
               n_frames: int = 24, max_alines: int = 256,
               klinear: bool = True,
               freeze_rin: float | None = 0.00125,
               freeze_shot: float | None = 0.0075) -> SpectralMLEResult:
    """End-to-end: temporal AC moments → scale to sim units → MLE → SimConfig coeffs.

    On real mirror/tissue data, fringe instability masquerades as RIN and inflates
    shot. Defaults freeze both at the image-calibrated priors and fit **read** from
    the optical-dark high-k tail (leading spike bins already excluded).
    """
    mu_real, var_real, n_obs = load_temporal_moments(
        folder, n_frames=n_frames, max_alines=max_alines, klinear=klinear)
    # Map into sim units at the *effective* reference amp (default 4), not the
    # instrument JSON seed (12), so read_frac is commensurate with SimConfig.
    a_ref_use = float(reference_amp if reference_amp is not None else DEFAULT_REFERENCE_AMP)
    mu_sim, a_ref, ref_dc = sim_reference_dc(instrument, reference_amp=a_ref_use)
    # Moments dropped HEAD_SKIP leading bins; align the sim envelope the same way.
    hs = min(HEAD_SKIP, max(0, mu_sim.size // 8))
    if hs > 0:
        mu_sim = mu_sim[hs:]
    n = min(mu_real.size, mu_sim.size)
    mu_real, var_real, mu_sim = mu_real[:n], var_real[:n], mu_sim[:n]

    scale = float(np.mean(mu_real) / (np.mean(mu_sim) + 1e-18))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"bad real→sim scale {scale}")
    mu_scaled = mu_real / scale
    var_scaled = var_real / (scale ** 2)

    if freeze_shot is not None and freeze_rin is not None:
        # read-only MLE from high-k / low-μ tail
        tail = mu_scaled <= max(np.percentile(mu_scaled, 12.0), 1e-9)
        read_abs = float(np.sqrt(max(np.median(var_scaled[tail]), 1e-18)))
        shot, rin = float(freeze_shot), float(freeze_rin)
        nll = _nll_from_moments(mu_scaled, var_scaled, shot, rin, read_abs)
    else:
        shot, rin, read_abs, nll = fit_noise_moments(
            mu_scaled, var_scaled, freeze_rin=freeze_rin)
        if freeze_shot is not None:
            shot = float(freeze_shot)
            nll = _nll_from_moments(mu_scaled, var_scaled, shot, rin, read_abs)

    read_frac = float(read_abs / (ref_dc + 1e-18))

    var_hat = noise_variance(mu_scaled, shot, rin, read_abs)
    shot_v = (shot ** 2) * np.abs(mu_scaled)
    rin_v = (rin ** 2) * (mu_scaled ** 2)
    read_v = np.full_like(mu_scaled, read_abs ** 2)
    tot = np.maximum(var_hat, 1e-24)
    return SpectralMLEResult(
        shot=float(shot), rin=float(rin), read=float(read_frac), read_abs=float(read_abs),
        reference_amp=float(a_ref), scale=float(scale), nll=float(nll),
        n_alines=int(n_obs), n_k=int(n),
        source=f"mle_temporal:{folder}",
        var_mse=float(np.mean((var_scaled - var_hat) ** 2)),
        shot_frac=float(np.mean(shot_v / tot)),
        rin_frac=float(np.mean(rin_v / tot)),
        read_frac_var=float(np.mean(read_v / tot)),
    )


def bundled_mle(name: str = "device_a") -> SpectralMLEResult | None:
    path = os.path.join(_DATA_DIR, f"spectral_mle_{name}.json")
    if not os.path.isfile(path):
        return None
    return SpectralMLEResult.load(path)


def apply_to_simconfig_kwargs(result: SpectralMLEResult) -> dict:
    """Keyword overrides for :class:`~octsim.config.SimConfig` from a fit result."""
    return {
        "shot": float(result.shot),
        "rin": float(result.rin),
        "read": float(result.read),
        "reference_amp": float(result.reference_amp),
    }
