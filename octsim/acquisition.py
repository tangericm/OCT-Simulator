"""Scan geometry and acquisition timing.

These two objects turn "how the beam is scanned and how fast" into the concrete
quantities the rest of the simulator needs:

* :class:`ScanGeometry` — the fast axis (x, within a B-scan) and the slow axis
  (y, across the frames of a volume): field of view, sample count, and pitch on each.
* :class:`Timing` — the camera line-rate / swept-source sweep-rate turned into a
  per-A-scan dwell, an **integration time** ``T_int = dwell · duty_cycle`` (which
  drives collected signal → shot-noise SNR, and fringe washout), a frame period
  (including flyback), and a **timestamp for every (frame, line)** spanning the whole
  volume acquisition. That single time axis is what makes bulk motion and phase drift
  correlated across A-scans *and across the frames of a volume*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScanGeometry:
    """Fast-axis (x) and slow-axis (y) sampling of a B-scan or volume."""

    lateral_fov_um: float
    lines_per_frame: int
    slow_fov_um: float = 0.0
    n_frames: int = 1

    @property
    def fast_pitch_um(self) -> float:
        return self.lateral_fov_um / (self.lines_per_frame - 1) if self.lines_per_frame > 1 else self.lateral_fov_um

    @property
    def slow_pitch_um(self) -> float:
        return self.slow_fov_um / (self.n_frames - 1) if self.n_frames > 1 else 0.0

    def x_axis(self) -> np.ndarray:
        """Fast-axis positions (µm), length ``lines_per_frame``."""
        return np.linspace(0.0, self.lateral_fov_um, self.lines_per_frame)

    def y_axis(self) -> np.ndarray:
        """Slow-axis positions (µm), length ``n_frames`` (centred at 0 for a volume)."""
        if self.n_frames <= 1:
            return np.zeros(1)
        return np.linspace(-self.slow_fov_um / 2.0, self.slow_fov_um / 2.0, self.n_frames)


@dataclass(frozen=True)
class Timing:
    """Acquisition speed → integration time and per-A-scan timestamps.

    ``line_rate_hz`` is the A-scan rate (SD-OCT camera line rate or SS-OCT sweep
    rate). ``duty_cycle`` is the fraction of the dwell actually integrating light.
    """

    line_rate_hz: float
    lines_per_frame: int
    n_frames: int = 1
    duty_cycle: float = 0.9
    flyback_time_s: float = 0.0

    @property
    def dwell_s(self) -> float:
        return 1.0 / self.line_rate_hz if self.line_rate_hz > 0 else 0.0

    @property
    def integration_time_s(self) -> float:
        """Per-A-scan light-collection time — drives signal level and washout."""
        return self.dwell_s * self.duty_cycle

    @property
    def frame_period_s(self) -> float:
        return self.lines_per_frame * self.dwell_s + self.flyback_time_s

    @property
    def frame_rate_hz(self) -> float:
        return 1.0 / self.frame_period_s if self.frame_period_s > 0 else 0.0

    @property
    def volume_time_s(self) -> float:
        return self.n_frames * self.frame_period_s

    def frame_line_times(self, frame: int) -> np.ndarray:
        """Timestamps (s) of each A-scan in ``frame`` on the volume time axis."""
        t0 = frame * self.frame_period_s
        return t0 + np.arange(self.lines_per_frame) * self.dwell_s

    def all_times(self) -> np.ndarray:
        """Flat timestamps for the whole acquisition, shape ``[n_frames·lines]``."""
        return np.concatenate([self.frame_line_times(f) for f in range(self.n_frames)])
