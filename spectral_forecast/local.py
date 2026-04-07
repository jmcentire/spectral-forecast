"""Layer 3b: Local residual correction via autoregressive modeling.

After Layers 1 (Fourier) and 2 (Trend) predict the signal, the residual
contains local structure that isn't periodic or trend-like: momentum,
mean-reversion, regime-specific autocorrelation.

This layer fits an AR(p) model on the recent residuals using Yule-Walker
equations, selects p by AIC, and forecasts the correction forward. The
correction captures whatever systematic pattern exists in the prediction
errors — the "pressure" from recent events that the static model doesn't know.

The AR forecast naturally decays toward zero as the horizon extends beyond
the lag structure, which is the right behavior: local effects are most
informative near-term and uninformative far-term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class LocalModel:
    """Fitted autoregressive model on residuals."""

    order: int  # AR order (p)
    coefficients: NDArray[np.floating]  # AR coefficients [a1, a2, ..., ap]
    intercept: float  # constant term
    aic: float
    residual_std: float  # std of the AR model's own residuals
    window_size: int  # how many recent points were used for fitting

    def describe(self) -> str:
        if self.order == 0:
            return "Local: none (residuals are white noise)"
        coeff_str = ", ".join("%.4f" % c for c in self.coefficients)
        return (
            "Local: AR(%d) coeffs=[%s] intercept=%.4f std=%.4f"
            % (self.order, coeff_str, self.intercept, self.residual_std)
        )


@dataclass
class LocalResult:
    """Full result of local correction fitting."""

    model: LocalModel
    fitted_values: NDArray[np.floating]  # correction values over training window
    recent_residuals: NDArray[np.floating]  # last p residuals (needed for forecasting)


def _yule_walker(residuals: NDArray[np.floating], order: int) -> tuple[NDArray, float]:
    """Solve Yule-Walker equations for AR(p) coefficients.

    Returns (coefficients, innovation_variance).
    """
    n = len(residuals)
    if order == 0 or n <= order:
        return np.array([]), float(np.var(residuals))

    # Compute autocorrelation
    mean = np.mean(residuals)
    centered = residuals - mean
    autocov = np.correlate(centered, centered, mode="full")[n - 1 :]
    autocov = autocov / n  # biased estimator (more stable)

    if autocov[0] <= 0:
        return np.array([]), float(np.var(residuals))

    # Build Toeplitz system: R @ a = r
    R = np.zeros((order, order))
    r = np.zeros(order)
    for i in range(order):
        r[i] = autocov[i + 1]
        for j in range(order):
            R[i, j] = autocov[abs(i - j)]

    try:
        coeffs = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return np.array([]), float(np.var(residuals))

    # Innovation variance
    innovation_var = autocov[0] - float(np.dot(coeffs, r))
    innovation_var = max(innovation_var, 1e-30)

    return coeffs, innovation_var


def _ar_aic(n: int, p: int, innovation_var: float) -> float:
    """AIC for AR(p) model."""
    if innovation_var <= 0 or n <= 0:
        return float("inf")
    return n * np.log(innovation_var) + 2 * (p + 1)


def fit_local(
    residuals: NDArray[np.floating],
    max_order: int = 12,
    window_size: int | None = None,
) -> LocalResult:
    """Fit a local AR model on recent residuals.

    Args:
        residuals: Full residual array (after Layer 1 + Layer 2 extraction).
        max_order: Maximum AR order to consider.
        window_size: How many recent residuals to use for fitting.
            None = min(len(residuals), 256). Using a window focuses the
            model on recent behavior rather than the full history.

    Returns:
        LocalResult with fitted model and recent residuals for forecasting.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    n_total = len(residuals)

    if window_size is None:
        window_size = min(n_total, 256)
    window_size = min(window_size, n_total)

    # Use the most recent residuals for fitting
    recent = residuals[-window_size:]
    n = len(recent)

    # Limit max_order to something reasonable for the window size
    max_order = min(max_order, n // 4, 24)

    # Try AR(0) through AR(max_order), select by AIC
    best_order = 0
    best_coeffs = np.array([])
    best_aic = _ar_aic(n, 0, float(np.var(recent)))
    best_innov_var = float(np.var(recent))

    for p in range(1, max_order + 1):
        coeffs, innov_var = _yule_walker(recent, p)
        if len(coeffs) == 0:
            continue
        aic = _ar_aic(n, p, innov_var)
        if aic < best_aic:
            best_aic = aic
            best_order = p
            best_coeffs = coeffs
            best_innov_var = innov_var

    # Compute intercept (for non-zero-mean residuals)
    mean_resid = float(np.mean(recent))
    if best_order > 0:
        intercept = mean_resid * (1 - float(np.sum(best_coeffs)))
    else:
        intercept = mean_resid

    # Compute fitted values over the training window
    fitted = np.zeros(n_total)
    for t in range(best_order, n_total):
        fitted[t] = intercept
        for j in range(best_order):
            fitted[t] += best_coeffs[j] * residuals[t - j - 1]

    model = LocalModel(
        order=best_order,
        coefficients=best_coeffs,
        intercept=intercept,
        aic=best_aic,
        residual_std=float(np.sqrt(best_innov_var)),
        window_size=window_size,
    )

    return LocalResult(
        model=model,
        fitted_values=fitted,
        recent_residuals=residuals[-max(best_order, 1) :].copy(),
    )


def forecast_local(
    model: LocalModel,
    recent_residuals: NDArray[np.floating],
    horizon: int,
) -> NDArray[np.floating]:
    """Forecast the local correction h steps ahead.

    The AR model forecasts recursively: each future value depends on
    the previous values (initially from actual residuals, then from
    the model's own predictions). The forecast naturally reverts toward
    the intercept/(1-sum(coeffs)) as the horizon extends.

    Args:
        model: Fitted AR model.
        recent_residuals: Last p residual values.
        horizon: Number of steps to forecast.

    Returns:
        1D array of local correction values for each forecast step.
    """
    if model.order == 0:
        # No AR structure — just return the mean correction
        return np.full(horizon, model.intercept)

    p = model.order
    # Pad recent residuals if needed
    history = np.zeros(p)
    n_avail = min(len(recent_residuals), p)
    history[-n_avail:] = recent_residuals[-n_avail:]

    forecast = np.zeros(horizon)
    for h in range(horizon):
        val = model.intercept
        for j in range(p):
            if h - j - 1 >= 0:
                val += model.coefficients[j] * forecast[h - j - 1]
            else:
                idx = p + (h - j - 1)
                if 0 <= idx < p:
                    val += model.coefficients[j] * history[idx]
        forecast[h] = val

    return forecast
