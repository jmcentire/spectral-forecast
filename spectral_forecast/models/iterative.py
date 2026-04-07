"""
Iterative Decomposition Forecaster.

Core idea: speculative self-referential forecasting through iterative
trend/seasonal/residual decomposition.

Algorithm:
  1. Decompose the training series into trend + seasonal + residual.
  2. Forecast each component independently (trend: linear extrapolation;
     seasonal: phase-continuation; residual: AR(p)).
  3. Combine to form an initial speculative forecast.
  4. ITERATE: extend the series with the speculative future values, re-decompose
     the extended series, re-forecast, and check convergence.
  5. The loop converges when the forecast changes by less than `tol` in RMSE.

Why does iteration help?
  - The extended window gives the decomposition more context for seasonal phase.
  - Trend fitting is more stable when future values are anchored.
  - The AR residual model sees a longer window, reducing edge effects.
  - In practice: 3-5 iterations typically suffice; beyond 7 yields diminishing
    returns. On chirp signals, convergence slows as the model chases phase drift.

Decomposition method:
  - Trend: centered moving average (period-length window).
  - Seasonal: average of detrended values at each phase index.
  - Residual: series minus trend minus seasonal.

Limitations noted in FINDINGS.md:
  - Does not outperform AR on pure AR processes (correct behavior).
  - Can diverge on very noisy signals if noise_threshold is not enforced.
"""
import numpy as np
from typing import Optional, List, Tuple

from .baseline import LinearARForecaster


def _centered_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Compute centered moving average with `window` size.

    Returns array same length as y with NaN at the edges.
    """
    half = window // 2
    trend = np.full_like(y, np.nan)
    for i in range(half, len(y) - half):
        trend[i] = np.mean(y[i - half : i + half + 1])
    return trend


def _extrapolate_trend(trend: np.ndarray, horizon: int) -> np.ndarray:
    """Extrapolate trend using linear regression; forecast starts at len(trend)."""
    return _extrapolate_trend_from(trend, len(trend), horizon)


def _extrapolate_trend_from(
    trend: np.ndarray, start_idx: int, horizon: int
) -> np.ndarray:
    """Extrapolate trend via linear regression on the non-NaN portion of trend,
    then evaluate at positions start_idx .. start_idx+horizon-1.

    This allows forecasting from an interior position when trend was estimated
    on a longer (extended) series.
    """
    valid = ~np.isnan(trend)
    if not np.any(valid):
        return np.zeros(horizon)
    t = np.where(valid)[0]
    v = trend[valid]
    # Use only the last 25% of valid trend points to capture recent slope
    tail_len = max(20, len(t) // 4)
    t_tail = t[-tail_len:]
    v_tail = v[-tail_len:]
    coeffs = np.polyfit(t_tail, v_tail, 1)
    future_t = np.arange(start_idx, start_idx + horizon)
    return np.polyval(coeffs, future_t)


def _extract_seasonal(residual_from_trend: np.ndarray, period: int) -> np.ndarray:
    """Estimate seasonal component by averaging over each phase index."""
    n = len(residual_from_trend)
    seasonal_pattern = np.zeros(period)
    for i in range(period):
        indices = np.arange(i, n, period)
        valid_vals = residual_from_trend[indices]
        valid_vals = valid_vals[~np.isnan(valid_vals)]
        if len(valid_vals) > 0:
            seasonal_pattern[i] = np.mean(valid_vals)
    # Project pattern across n+horizon points starting from offset 0
    return seasonal_pattern


def _project_seasonal(pattern: np.ndarray, start_idx: int, length: int) -> np.ndarray:
    """Project a seasonal pattern of length `period` starting at `start_idx`."""
    period = len(pattern)
    indices = (np.arange(start_idx, start_idx + length)) % period
    return pattern[indices]


class IterativeDecompositionForecaster:
    """Speculative iterative decomposition forecaster.

    Parameters
    ----------
    period : int
        Dominant seasonal period. If None, estimated from FFT.
    ar_lags : int
        Number of AR lags for residual modeling.
    max_iter : int
        Maximum number of speculative refinement iterations.
    tol : float
        Convergence threshold: stop when RMSE(pred_new - pred_old) / std(pred_old) < tol.
    """

    def __init__(
        self,
        period: Optional[int] = None,
        ar_lags: int = 12,
        max_iter: int = 10,
        tol: float = 1e-4,
    ) -> None:
        self.period = period
        self.ar_lags = ar_lags
        self.max_iter = max_iter
        self.tol = tol
        self._fitted: bool = False
        self._train: Optional[np.ndarray] = None
        self.convergence_history: List[float] = []

    def _estimate_period(self, y: np.ndarray) -> int:
        """Estimate dominant period using FFT power spectrum."""
        n = len(y)
        # Remove linear trend before FFT to avoid spectral leakage
        t = np.arange(n)
        coeffs = np.polyfit(t, y, 1)
        detrended = y - np.polyval(coeffs, t)
        spectrum = np.abs(np.fft.rfft(detrended)) ** 2
        freqs = np.fft.rfftfreq(n)
        # Exclude DC (index 0) and very high frequencies
        min_period = 4
        valid = (freqs > 0) & (freqs < 1 / min_period)
        if not np.any(valid):
            return 12
        peak_idx = np.argmax(spectrum[valid])
        dominant_freq = freqs[valid][peak_idx]
        period = max(4, int(round(1.0 / dominant_freq)))
        return min(period, n // 4)  # Cap at n/4 to ensure enough full cycles

    def _decompose(
        self, y: np.ndarray, period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose y into (trend, seasonal, residual)."""
        trend = _centered_moving_average(y, period)
        # For trend-removed component, fill NaN edges with first/last valid
        trend_filled = trend.copy()
        valid_mask = ~np.isnan(trend_filled)
        if np.any(valid_mask):
            first_valid = trend_filled[valid_mask][0]
            last_valid = trend_filled[valid_mask][-1]
            trend_filled[: np.argmax(valid_mask)] = first_valid
            trend_filled[len(valid_mask) - np.argmax(valid_mask[::-1]) :] = last_valid
        detrended = y - trend_filled
        seasonal_pattern = _extract_seasonal(detrended, period)
        seasonal = _project_seasonal(seasonal_pattern, 0, len(y))
        residual = y - trend_filled - seasonal
        return trend, seasonal_pattern, residual

    def fit(self, y: np.ndarray) -> "IterativeDecompositionForecaster":
        """Fit to training data y."""
        self._train = y.copy()
        if self.period is None:
            self.period = self._estimate_period(y)
        self._fitted = True
        return self

    def _forecast_from_series(
        self, y: np.ndarray, horizon: int
    ) -> np.ndarray:
        """Generate a forecast of `horizon` steps from the end of series y.

        Decomposes y, then extrapolates each component forward from len(y).
        """
        return self._forecast_from_position(y, len(y), horizon)

    def _forecast_from_position(
        self, y: np.ndarray, n_orig: int, horizon: int
    ) -> np.ndarray:
        """Decompose y, then forecast `horizon` steps starting at position n_orig.

        When y == y[:n_orig] this is identical to _forecast_from_series.
        When y is extended (len(y) > n_orig), the decomposition benefits from
        the extra context but predictions still start at n_orig — this is the
        speculative refinement: more context → better component estimates.
        """
        period = self.period
        trend, seasonal_pattern, residual = self._decompose(y, period)

        # Trend: extrapolate from n_orig (not len(y))
        trend_pred = _extrapolate_trend_from(trend, n_orig, horizon)

        # Seasonal: project forward from n_orig
        seasonal_pred = _project_seasonal(seasonal_pattern, n_orig, horizon)

        # Residual: fit AR on residual up to n_orig only (no peeking at future)
        residual_to_fit = residual[:n_orig]
        ar = LinearARForecaster(p=min(self.ar_lags, len(residual_to_fit) // 4))
        if len(residual_to_fit) > ar.p + 5:
            ar.fit(residual_to_fit)
            residual_pred = ar.predict(residual_to_fit, horizon)
        else:
            residual_pred = np.zeros(horizon)

        return trend_pred + seasonal_pred + residual_pred

    def predict(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast `horizon` steps with iterative speculative refinement.

        Each iteration extends the series with the current speculative forecast,
        re-decomposes using the extended context, and re-forecasts from the
        original endpoint. The decomposition benefits from longer windows;
        predictions remain anchored at len(y).

        Returns the converged forecast array of shape (horizon,).
        Convergence history (relative RMSE change per iteration) is in
        `self.convergence_history`.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        self.convergence_history = []
        n_orig = len(y)

        # Initial forecast using only the training data
        pred = self._forecast_from_series(y, horizon)

        for _ in range(self.max_iter):
            # Extend series with speculative future values
            y_extended = np.concatenate([y, pred])
            # Re-forecast FROM n_orig using the extended context for decomposition
            pred_new = self._forecast_from_position(y_extended, n_orig, horizon)

            scale = np.std(pred) if np.std(pred) > 1e-10 else 1.0
            relative_change = float(np.sqrt(np.mean((pred_new - pred) ** 2)) / scale)
            self.convergence_history.append(relative_change)

            pred = pred_new
            if relative_change < self.tol:
                break

        return pred
