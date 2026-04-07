"""Layer 3: Shock detection and adjustment.

Detects and fits discrete events that caused the time series to diverge
from its predicted behavior (Layer 1 + Layer 2). The model doesn't need
to know *what* happened — it detects that reality diverged from prediction
in a structured way and fits the shape of that divergence.

Shock shapes:
- Step: permanent level change (e.g., price regime shift)
- Spike with decay: temporary disruption, exponential return (e.g., supply shock)
- Ramp: accelerating linear pressure (e.g., emerging trend)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ShockShape(Enum):
    STEP = "step"
    SPIKE_DECAY = "spike_decay"
    RAMP = "ramp"


@dataclass
class ShockComponent:
    """A detected shock event."""

    onset_idx: int  # index in the original series where the shock begins
    shape: ShockShape
    magnitude: float  # peak deviation from prediction
    decay_rate: float  # 0 for step, positive for spike_decay, slope for ramp
    aic: float  # model quality

    def evaluate(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        """Reconstruct this shock's contribution at time indices t."""
        t = np.asarray(t, dtype=np.float64)
        result = np.zeros_like(t)
        active = t >= self.onset_idx

        if self.shape == ShockShape.STEP:
            result[active] = self.magnitude

        elif self.shape == ShockShape.SPIKE_DECAY:
            dt = t[active] - self.onset_idx
            result[active] = self.magnitude * np.exp(-self.decay_rate * dt)

        elif self.shape == ShockShape.RAMP:
            dt = t[active] - self.onset_idx
            result[active] = self.magnitude + self.decay_rate * dt

        return result


@dataclass
class ShockResult:
    """Full result of shock detection."""

    shocks: list[ShockComponent]
    adjusted_residual: NDArray[np.floating]


def _aic(n: int, k: int, rss: float) -> float:
    """Akaike Information Criterion. Lower is better."""
    if n <= 0:
        return float("inf")
    if rss <= 0:
        # Perfect fit — use a very negative but finite value
        return -1e15 + 2 * k
    return n * np.log(rss / n) + 2 * k


def _fit_step(
    delta: NDArray[np.floating], onset: int
) -> tuple[float, float, float]:
    """Fit a step function starting at onset. Returns (magnitude, 0.0, aic)."""
    n = len(delta)
    active = delta[onset:]
    if len(active) == 0:
        return 0.0, 0.0, float("inf")
    magnitude = float(np.mean(active))
    pred = np.zeros(n)
    pred[onset:] = magnitude
    rss = float(np.sum((delta - pred) ** 2))
    return magnitude, 0.0, _aic(n, 2, rss)  # 2 params: onset, magnitude


def _fit_spike_decay(
    delta: NDArray[np.floating], onset: int
) -> tuple[float, float, float]:
    """Fit a spike with exponential decay. Returns (magnitude, decay_rate, aic)."""
    n = len(delta)
    active = delta[onset:]
    n_active = len(active)
    if n_active < 3:
        return 0.0, 0.0, float("inf")

    magnitude = float(active[0])
    if abs(magnitude) < 1e-10:
        return 0.0, 0.0, float("inf")

    # Estimate decay rate from log of absolute ratios
    abs_active = np.abs(active)
    sign = np.sign(magnitude)
    # Find where the signal hasn't crossed zero yet
    same_sign = np.where(sign * active > abs(magnitude) * 0.01)[0]
    if len(same_sign) < 3:
        return 0.0, 0.0, float("inf")

    # Fit log(|active|) = log(|magnitude|) - decay_rate * t
    log_vals = np.log(np.maximum(np.abs(active[same_sign]), 1e-30))
    t_vals = same_sign.astype(np.float64)
    if len(t_vals) < 2:
        return 0.0, 0.0, float("inf")

    X = np.column_stack([t_vals, np.ones(len(t_vals))])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, log_vals, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, float("inf")

    decay_rate = float(-coeffs[0])
    if decay_rate <= 0:
        return 0.0, 0.0, float("inf")

    # Recompute magnitude from the intercept
    magnitude = float(sign * np.exp(coeffs[1]))

    pred = np.zeros(n)
    dt = np.arange(n_active, dtype=np.float64)
    pred[onset:] = magnitude * np.exp(-decay_rate * dt)
    rss = float(np.sum((delta - pred) ** 2))
    return magnitude, decay_rate, _aic(n, 3, rss)  # 3 params: onset, magnitude, decay


def _fit_ramp(
    delta: NDArray[np.floating], onset: int
) -> tuple[float, float, float]:
    """Fit a linear ramp starting at onset. Returns (intercept, slope, aic)."""
    n = len(delta)
    active = delta[onset:]
    n_active = len(active)
    if n_active < 3:
        return 0.0, 0.0, float("inf")

    t_active = np.arange(n_active, dtype=np.float64)
    X = np.column_stack([np.ones(n_active), t_active])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, active, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, float("inf")

    intercept, slope = float(coeffs[0]), float(coeffs[1])

    pred = np.zeros(n)
    pred[onset:] = intercept + slope * t_active
    rss = float(np.sum((delta - pred) ** 2))
    return intercept, slope, _aic(n, 3, rss)  # 3 params: onset, intercept, slope


def detect_shocks(
    actual: NDArray[np.floating],
    predicted: NDArray[np.floating],
    noise_std: float,
    min_sigma: float = 2.0,
    lookback_window: int | None = None,
    max_shocks: int = 3,
) -> ShockResult:
    """Detect and fit shock components from prediction-actual divergence.

    Args:
        actual: Recent actual values.
        predicted: Predicted values from Layer 1 + Layer 2.
        noise_std: Estimated noise standard deviation from prior layers.
        min_sigma: Minimum deviation (in noise sigmas) to trigger shock detection.
        lookback_window: How far back to look for shocks. None = full series.
        max_shocks: Maximum number of shocks to detect.

    Returns:
        ShockResult with detected shocks and the adjusted residual.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}"
        )

    delta = actual - predicted
    n = len(delta)

    if lookback_window is not None:
        # Only analyze the most recent window
        start = max(0, n - lookback_window)
    else:
        start = 0

    threshold = min_sigma * max(noise_std, 1e-10)
    shocks: list[ShockComponent] = []
    working_delta = delta.copy()

    for _ in range(max_shocks):
        # Find the largest deviation in the analysis window
        window = working_delta[start:]
        abs_window = np.abs(window)
        max_dev_local = int(np.argmax(abs_window))

        if abs_window[max_dev_local] < threshold:
            break

        # Find the onset: the first point in the window that exceeds threshold.
        # argmax gives the peak deviation, but the shock may have started earlier.
        exceeds = np.where(abs_window > threshold)[0]
        onset_local = int(exceeds[0]) if len(exceeds) > 0 else max_dev_local
        max_dev_idx = start + onset_local

        # Try all three shapes, select by AIC
        mag_step, _, aic_step = _fit_step(working_delta, max_dev_idx)
        mag_spike, decay_spike, aic_spike = _fit_spike_decay(working_delta, max_dev_idx)
        mag_ramp, slope_ramp, aic_ramp = _fit_ramp(working_delta, max_dev_idx)

        candidates = [
            (ShockShape.STEP, mag_step, 0.0, aic_step),
            (ShockShape.SPIKE_DECAY, mag_spike, decay_spike, aic_spike),
            (ShockShape.RAMP, mag_ramp, slope_ramp, aic_ramp),
        ]

        # Pick best by AIC (lower is better), excluding infinite
        valid = [c for c in candidates if np.isfinite(c[3])]
        if not valid:
            break

        best = min(valid, key=lambda c: c[3])
        shape, magnitude, decay_rate, aic = best

        shock = ShockComponent(
            onset_idx=max_dev_idx,
            shape=shape,
            magnitude=magnitude,
            decay_rate=decay_rate,
            aic=aic,
        )

        # Subtract the shock from the working delta
        t = np.arange(n, dtype=np.float64)
        working_delta = working_delta - shock.evaluate(t)
        shocks.append(shock)

    return ShockResult(shocks=shocks, adjusted_residual=working_delta)
