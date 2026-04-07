"""Layer 2: Long-period trend fitting.

Fits non-periodic structure from the residual after Fourier extraction:
compounding growth, secular trends, structural drift. These are effects
with periods longer than the observation window — they look like drift,
not oscillation.

Uses BIC model selection among linear, quadratic, and exponential fits.
Sliding window weighting gives recency bias for forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


class TrendType(Enum):
    NONE = "none"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"


@dataclass
class TrendModel:
    """Fitted trend model."""

    trend_type: TrendType
    params: dict[str, float]
    bic: float
    residual_std: float

    def predict(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate the trend at time indices t."""
        t = np.asarray(t, dtype=np.float64)
        if self.trend_type == TrendType.NONE:
            return np.zeros_like(t)
        elif self.trend_type == TrendType.LINEAR:
            return self.params["a"] * t + self.params["b"]
        elif self.trend_type == TrendType.QUADRATIC:
            return self.params["a"] * t**2 + self.params["b"] * t + self.params["c"]
        elif self.trend_type == TrendType.EXPONENTIAL:
            return self.params["a"] * np.exp(self.params["b"] * t) + self.params["c"]
        raise ValueError(f"Unknown trend type: {self.trend_type}")

    def predict_damped(
        self, t: NDArray[np.floating], t_boundary: float, damping_halflife: float
    ) -> NDArray[np.floating]:
        """Evaluate trend with damping beyond the training boundary.

        For nonlinear trends (quadratic, exponential), extrapolation far beyond
        the training window is unreliable. This transitions from the full model
        to its tangent line (local slope at t_boundary) using exponential damping.

        Linear and none trends are unaffected — linear extrapolation is linear,
        and none is zero everywhere.

        Args:
            t: Time indices to evaluate.
            t_boundary: Last time index of training data.
            damping_halflife: Distance (in samples) at which the nonlinear
                component is reduced by half. Smaller = faster transition to
                tangent line.
        """
        t = np.asarray(t, dtype=np.float64)

        # Linear and none don't need damping
        if self.trend_type in (TrendType.NONE, TrendType.LINEAR):
            return self.predict(t)

        # Full model prediction and tangent line at boundary
        full_pred = self.predict(t)
        boundary_val = self.predict(np.array([t_boundary]))[0]

        # Compute slope at boundary via finite difference
        eps = 0.5
        slope = (
            self.predict(np.array([t_boundary + eps]))[0]
            - self.predict(np.array([t_boundary - eps]))[0]
        ) / (2 * eps)
        tangent = boundary_val + slope * (t - t_boundary)

        # Blend: within training window, use full model.
        # Beyond boundary, exponentially transition to tangent.
        dt = np.maximum(t - t_boundary, 0.0)
        decay = np.log(2) / max(damping_halflife, 1.0)
        blend = np.exp(-decay * dt)  # 1.0 at boundary, decays toward 0

        return blend * full_pred + (1 - blend) * tangent


@dataclass
class TrendResult:
    """Full result of trend fitting."""

    model: TrendModel
    residual: NDArray[np.floating]
    candidates_tried: int


def _bic(n: int, k: int, rss: float) -> float:
    """Bayesian Information Criterion. Lower is better."""
    if n <= 0:
        return float("inf")
    if rss <= 0:
        return -1e15 + k * np.log(max(n, 2))
    return n * np.log(rss / n) + k * np.log(n)


def _weighted_lstsq(
    X: NDArray[np.floating], y: NDArray[np.floating], w: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Solve weighted least squares: min_c sum(w_i * (y_i - X_i @ c)^2).

    Uses O(n*k) memory via broadcasting instead of O(n^2) from np.diag(w).
    """
    # Weight X and y by sqrt(w) to convert to ordinary least squares
    sqrt_w = np.sqrt(w)[:, np.newaxis]
    Xw = X * sqrt_w
    yw = y * np.sqrt(w)
    coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coeffs


def _fit_linear(
    t: NDArray[np.floating], y: NDArray[np.floating], w: NDArray[np.floating]
) -> tuple[dict[str, float], float]:
    """Weighted least squares linear fit."""
    X = np.column_stack([t, np.ones_like(t)])
    try:
        coeffs = _weighted_lstsq(X, y, w)
    except np.linalg.LinAlgError:
        return {"a": 0.0, "b": float(np.mean(y))}, float(np.sum(w * y**2))
    pred = X @ coeffs
    rss = float(np.sum(w * (y - pred) ** 2))
    return {"a": float(coeffs[0]), "b": float(coeffs[1])}, rss


def _fit_quadratic(
    t: NDArray[np.floating], y: NDArray[np.floating], w: NDArray[np.floating]
) -> tuple[dict[str, float], float]:
    """Weighted least squares quadratic fit."""
    X = np.column_stack([t**2, t, np.ones_like(t)])
    try:
        coeffs = _weighted_lstsq(X, y, w)
    except np.linalg.LinAlgError:
        return {"a": 0.0, "b": 0.0, "c": float(np.mean(y))}, float(np.sum(w * y**2))
    pred = X @ coeffs
    rss = float(np.sum(w * (y - pred) ** 2))
    return {"a": float(coeffs[0]), "b": float(coeffs[1]), "c": float(coeffs[2])}, rss


def _fit_exponential(
    t: NDArray[np.floating], y: NDArray[np.floating], w: NDArray[np.floating]
) -> tuple[dict[str, float], float]:
    """Weighted nonlinear exponential fit: a * exp(b * t) + c."""
    # Normalize t to [0, 1] for numerical stability
    t_max = max(float(t[-1]), 1.0)
    t_norm = t / t_max

    def model(t_val, a, b, c):
        return a * np.exp(b * t_val) + c

    y_range = float(np.ptp(y))
    y_mean = float(np.mean(y))

    try:
        popt, _ = curve_fit(
            model,
            t_norm,
            y,
            p0=[y_range * 0.1, 1.0, y_mean],
            sigma=1.0 / np.sqrt(np.maximum(w, 1e-10)),
            maxfev=2000,
            bounds=([-np.inf, -10, -np.inf], [np.inf, 10, np.inf]),
        )
        # Convert b back to original time scale
        a, b, c = popt
        b_original = b / t_max
        pred = a * np.exp(b_original * t) + c
        rss = float(np.sum(w * (y - pred) ** 2))
        return {"a": float(a), "b": float(b_original), "c": float(c)}, rss
    except (RuntimeError, ValueError):
        # Exponential fit failed — return infinite RSS so it loses BIC comparison
        return {"a": 0.0, "b": 0.0, "c": float(np.mean(y))}, float("inf")


def fit_trend(
    residual: NDArray[np.floating],
    recency_halflife: float | None = None,
) -> TrendResult:
    """Fit a trend model to the residual from Fourier extraction.

    Args:
        residual: 1D array — the signal after periodic components are removed.
        recency_halflife: Half-life for exponential recency weighting (in samples).
            If None, defaults to len(residual) / 2 — mild recency bias.

    Returns:
        TrendResult with the best model (by BIC), its residual, and metadata.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1:
        raise ValueError(f"Expected 1D residual, got shape {residual.shape}")

    n = len(residual)
    t = np.arange(n, dtype=np.float64)

    # Recency weighting: exponential decay from end of series
    if recency_halflife is None:
        recency_halflife = n / 2.0
    decay_rate = np.log(2) / max(recency_halflife, 1.0)
    w = np.exp(-decay_rate * (n - 1 - t))
    w = w / w.sum() * n  # normalize so sum(w) = n

    # Fit candidates
    candidates: list[tuple[TrendType, dict[str, float], float, int]] = []

    # None model (just weighted mean)
    wmean = float(np.sum(w * residual) / np.sum(w))
    rss_none = float(np.sum(w * (residual - wmean) ** 2))
    candidates.append((TrendType.NONE, {}, rss_none, 1))

    # Linear
    params_lin, rss_lin = _fit_linear(t, residual, w)
    candidates.append((TrendType.LINEAR, params_lin, rss_lin, 2))

    # Quadratic
    params_quad, rss_quad = _fit_quadratic(t, residual, w)
    candidates.append((TrendType.QUADRATIC, params_quad, rss_quad, 3))

    # Exponential
    params_exp, rss_exp = _fit_exponential(t, residual, w)
    candidates.append((TrendType.EXPONENTIAL, params_exp, rss_exp, 3))

    # Select by BIC
    best = min(candidates, key=lambda c: _bic(n, c[3], c[2]))
    trend_type, params, rss, k = best

    model = TrendModel(
        trend_type=trend_type,
        params=params,
        bic=_bic(n, k, rss),
        residual_std=float(np.sqrt(rss / n)),
    )

    trend_values = model.predict(t)
    trend_residual = residual - trend_values

    return TrendResult(
        model=model,
        residual=trend_residual,
        candidates_tried=len(candidates),
    )
