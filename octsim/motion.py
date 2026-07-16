"""Bulk axial motion, phase drift, and fringe washout — one continuous trajectory.

Patient/system motion during an OCT acquisition is a *single continuous process* in
time, not an independent perturbation per A-scan. This module builds one axial
displacement trajectory ``Δz(t)`` over the whole acquisition time axis (from
:class:`octsim.acquisition.Timing`) and samples it at every A-scan's timestamp, so
the resulting artifacts are **correlated across A-scans within a frame and across the
frames of a volume** — a volume shows continuous inter-frame warp and coherent drift,
exactly the physical behaviour.

Components of ``Δz(t)`` (all scaled by ``motion_level``):
* **slow drift** — a normalised random walk (low-frequency baseline wander);
* **physiological oscillation** — cardiac (~1 Hz) + breathing (~0.3 Hz) sinusoids;
* **microsaccades** — a few fast Gaussian transients.

From the trajectory:
* **bulk axial motion** enters the forward model as a per-A-scan depth phase
  ``exp(i·2·k·Δz)`` (:mod:`octsim.forward`, ``motion_dz_um``);
* **phase drift** is a separate slow, k-independent phase random walk (``column_phase``);
* **fringe washout** comes from the instantaneous velocity ``v = dΔz/dt``: motion during
  the integration time ``T_int`` averages the fringe down by ``sinc(k0·v·T_int)``
  (``washout_gain``). Faster acquisition (smaller ``T_int``) ⇒ less washout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import Timing


@dataclass
class MotionTrajectory:
    """Sampled ``Δz(t)`` and derived quantities over the whole acquisition."""

    times_s: np.ndarray          # [n_total] A-scan timestamps
    dz_um: np.ndarray            # [n_total] axial displacement
    velocity_um_s: np.ndarray    # [n_total] dΔz/dt
    phase_drift_rad: np.ndarray  # [n_total] slow k-independent phase
    lines_per_frame: int
    n_frames: int

    @property
    def is_static(self) -> bool:
        return not np.any(self.dz_um) and not np.any(self.phase_drift_rad)

    def washout_gain(self, k0: float, integration_time_s: float) -> np.ndarray:
        """Per-A-scan fringe-contrast factor ``|sinc(k0·|v|·T_int)|`` in [0, 1].

        The contrast is the magnitude of the fringe averaged over the integration
        window, so the (sign-oscillating) ``sinc`` is taken in absolute value.
        """
        x = k0 * np.abs(self.velocity_um_s) * integration_time_s / np.pi
        return np.abs(np.sinc(x))

    def frame(self, f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(dz, velocity, phase_drift)`` for frame ``f`` (length ``lines_per_frame``)."""
        s = slice(f * self.lines_per_frame, (f + 1) * self.lines_per_frame)
        return self.dz_um[s], self.velocity_um_s[s], self.phase_drift_rad[s]


def build_motion(
    timing: Timing,
    *,
    motion_level: float,
    axial_motion_um: float,
    phase_drift_rad: float,
    cardiac: bool,
    microsaccade: bool,
    rng: np.random.Generator,
) -> MotionTrajectory:
    """Build the acquisition-wide motion trajectory (all zeros if ``motion_level<=0``)."""
    t = timing.all_times()
    n = t.size
    zeros = np.zeros(n)
    if motion_level <= 0 or n == 0:
        return MotionTrajectory(t, zeros, zeros, zeros, timing.lines_per_frame, timing.n_frames)

    amp = float(axial_motion_um) * float(motion_level)

    # slow drift: normalised random walk (RMS ~ amp over the acquisition)
    walk = np.cumsum(rng.standard_normal(n))
    walk -= walk.mean()
    if walk.std() > 0:
        walk *= amp / walk.std()
    dz = walk

    # physiological oscillation
    if cardiac:
        f_card = rng.uniform(0.8, 1.4)      # Hz (~heart rate)
        f_breath = rng.uniform(0.2, 0.35)   # Hz (~breathing)
        ph_c, ph_b = rng.uniform(0.0, 2.0 * np.pi, size=2)
        dz = dz + 0.4 * amp * np.sin(2 * np.pi * f_card * t + ph_c)
        dz = dz + 0.7 * amp * np.sin(2 * np.pi * f_breath * t + ph_b)

    # microsaccades: a few fast Gaussian transients
    if microsaccade and t[-1] > t[0]:
        for _ in range(int(rng.integers(0, 4))):
            t0 = rng.uniform(t[0], t[-1])
            width = rng.uniform(5e-3, 2e-2)
            a = rng.uniform(-3.0, 3.0) * amp
            dz = dz + a * np.exp(-0.5 * ((t - t0) / width) ** 2)

    velocity = np.gradient(dz, t) if n > 1 and t[-1] > t[0] else zeros.copy()

    # phase drift: a separate slow phase random walk
    pw = np.cumsum(rng.standard_normal(n))
    pw -= pw.mean()
    if pw.std() > 0:
        pw *= (float(phase_drift_rad) * float(motion_level)) / pw.std()

    return MotionTrajectory(t, dz, velocity, pw, timing.lines_per_frame, timing.n_frames)
