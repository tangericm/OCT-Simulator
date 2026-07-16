"""Statistical shape model of retinal layer geometry, learned from segmented OCT.

A :class:`LayerShapeModel` is a PCA over **fovea-aligned per-layer thickness profiles**
extracted from real, segmented B-scans. It captures the population mean geometry, the
foveal-pit morphology, and the correlated layer-thickness variation across eyes — the
things a hand-tuned parametric pit cannot reproduce. Sampling draws latent coefficients
``z ~ N(0, 1)`` and returns a plausible new thickness field; :func:`fit` builds one from a
``[N, n_layer, n_ecc]`` stack (see :mod:`octsim.data.duke_dme` for a real loader).

The learned model is tiny (a few KB) and travels with the repo as an ``.npz`` artifact; the
raw datasets do not. Nothing here touches labels — the sampled geometry still drives the
exact-label physics forward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerShapeModel:
    """PCA shape model over ``[n_layer, n_ecc]`` thickness profiles (µm)."""

    layer_names: list
    ecc_grid_um: np.ndarray          # [n_ecc] eccentricity from the fovea (signed)
    mean_thickness: np.ndarray       # [n_layer, n_ecc]
    modes: np.ndarray                # [K, n_layer, n_ecc] unit eigenvectors
    mode_std: np.ndarray             # [K] sqrt(eigenvalue) (µm-scale weight per mode)
    undulation_rms_um: float = 3.0   # residual small-scale boundary roughness
    axial_pitch_um: float = 3.87
    n_scans: int = 0
    source: str = "unknown"

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, thickness: np.ndarray, ecc_grid_um: np.ndarray, layer_names: list,
            *, n_modes: int = 8, undulation_rms_um: float = 3.0,
            axial_pitch_um: float = 3.87, source: str = "") -> "LayerShapeModel":
        """PCA over a ``[N, n_layer, n_ecc]`` thickness stack (µm)."""
        t = np.asarray(thickness, float)
        n, n_layer, n_ecc = t.shape
        flat = t.reshape(n, -1)
        mean = flat.mean(axis=0)
        X = flat - mean
        # SVD PCA (rows = samples); singular values -> per-mode std
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = int(min(n_modes, S.size))
        comp_std = S[:k] / np.sqrt(max(n - 1, 1))
        modes = Vt[:k].reshape(k, n_layer, n_ecc)
        return cls(
            layer_names=list(layer_names), ecc_grid_um=np.asarray(ecc_grid_um, float),
            mean_thickness=mean.reshape(n_layer, n_ecc), modes=modes, mode_std=comp_std,
            undulation_rms_um=float(undulation_rms_um), axial_pitch_um=float(axial_pitch_um),
            n_scans=int(n), source=source)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        v = self.mode_std ** 2
        # total variance isn't stored; report the modes' share of the retained variance
        return v / (v.sum() + 1e-12)

    # ------------------------------------------------------------------ sample
    def sample(self, rng: np.random.Generator, *, scale: float = 1.0,
               n_modes: int | None = None) -> np.ndarray:
        """Draw a plausible ``[n_layer, n_ecc]`` thickness field (µm, non-negative).

        ``scale`` widens/narrows the sampled variation; ``n_modes`` truncates to the top
        modes. The result is clamped at a small floor so no layer collapses to negative.
        """
        k = self.mode_std.size if n_modes is None else min(n_modes, self.mode_std.size)
        z = rng.standard_normal(k) * scale
        t = self.mean_thickness + np.einsum("k,klm->lm", z * self.mode_std[:k], self.modes[:k])
        return np.clip(t, 0.1, None)

    def mean_profile(self) -> np.ndarray:
        return self.mean_thickness.copy()

    # ------------------------------------------------------------------ (de)serialize
    def save(self, path) -> None:
        np.savez(path, layer_names=np.array(self.layer_names, dtype=object),
                 ecc_grid_um=self.ecc_grid_um, mean_thickness=self.mean_thickness,
                 modes=self.modes, mode_std=self.mode_std,
                 undulation_rms_um=self.undulation_rms_um, axial_pitch_um=self.axial_pitch_um,
                 n_scans=self.n_scans, source=self.source)

    @classmethod
    def load(cls, path) -> "LayerShapeModel":
        d = np.load(path, allow_pickle=True)
        return cls(
            layer_names=list(d["layer_names"]), ecc_grid_um=d["ecc_grid_um"],
            mean_thickness=d["mean_thickness"], modes=d["modes"], mode_std=d["mode_std"],
            undulation_rms_um=float(d["undulation_rms_um"]),
            axial_pitch_um=float(d["axial_pitch_um"]),
            n_scans=int(d["n_scans"]), source=str(d["source"]))


# --------------------------------------------------------------------------- helpers
def align_and_resample(thicknesses: np.ndarray, cols_um: np.ndarray, fovea_um: float,
                       ecc_grid_um: np.ndarray) -> np.ndarray:
    """Resample per-layer thickness columns onto a fovea-centred eccentricity grid.

    ``thicknesses`` ``[n_layer, n_col]`` (µm) sampled at ``cols_um`` (µm, absolute), the
    fovea at ``fovea_um``. Edge-clamped interpolation onto ``ecc_grid_um`` (signed µm from
    the fovea). Columns with NaN are dropped per layer before interpolation.
    """
    n_layer = thicknesses.shape[0]
    out = np.zeros((n_layer, ecc_grid_um.size))
    ecc = cols_um - fovea_um
    for j in range(n_layer):
        v = thicknesses[j]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        out[j] = np.interp(ecc_grid_um, ecc[ok], v[ok])   # np.interp clamps at the edges
    return out


def locate_fovea_um(inner_thickness_um: np.ndarray, cols_um: np.ndarray,
                    smooth: int = 15) -> float:
    """Fovea = column of minimum (smoothed) inner-retinal thickness."""
    v = np.asarray(inner_thickness_um, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return float(np.mean(cols_um))
    vv = np.interp(cols_um, cols_um[ok], v[ok])
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        vv = np.convolve(vv, ker, mode="same")
    return float(cols_um[int(np.argmin(vv))])
