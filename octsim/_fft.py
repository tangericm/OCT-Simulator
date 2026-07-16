"""FFT dispatch — multi-threaded ``scipy.fft`` when available, numpy fallback.

``scipy.fft`` and ``numpy.fft`` share the same pocketfft kernel, so the results are
bit-identical (verified 0.0 max-abs difference); the only change is that scipy runs
the transform across ``workers`` threads. Every heavy transform in the forward model
and the reconstruction routes through here so large corpora scale across cores while
the numerics stay pinned by the existing reproducibility tests.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly by every reconstruction
    import scipy.fft as _sfft

    def ifft(a: np.ndarray, axis: int = -1) -> np.ndarray:
        return _sfft.ifft(a, axis=axis, workers=-1)

    def fft(a: np.ndarray, axis: int = -1) -> np.ndarray:
        return _sfft.fft(a, axis=axis, workers=-1)

    HAVE_SCIPY_FFT = True
except Exception:  # pragma: no cover - numpy-only environments
    def ifft(a: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.fft.ifft(a, axis=axis)

    def fft(a: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.fft.fft(a, axis=axis)

    HAVE_SCIPY_FFT = False
