"""Forecaster: superposes all three layers with error bounds.

The forecast at any future time t is:
    y(t) = sum(Layer1 components at t) + Layer2 trend at t + sum(Layer3 shocks at t)
    error_bounds = quantiles of the final residual noise distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.extraction import (
    ExtractionResult,
    ExtractedComponent,
    extract,
)
from spectral_forecast.shock import ShockComponent, ShockResult, detect_shocks
from spectral_forecast.trend import TrendModel, TrendResult, fit_trend


@dataclass
class ForecastResult:
    """Complete forecast with decomposition."""

    point_forecast: NDArray[np.floating]
    lower_bound: NDArray[np.floating]  # e.g., 10th percentile
    upper_bound: NDArray[np.floating]  # e.g., 90th percentile
    components: list[ExtractedComponent]
    trend: TrendModel
    shocks: list[ShockComponent]
    noise_std: float
    residual_quantiles: tuple[float, float]  # (lower_q, upper_q) of residual

    def describe(self) -> str:
        """Human-readable description of the forecast decomposition."""
        lines = []
        lines.append(f"Forecast: {len(self.point_forecast)} steps ahead")
        lines.append(f"  Periodic components: {len(self.components)}")
        for i, c in enumerate(self.components):
            period = 1.0 / c.frequency if c.frequency > 0 else float("inf")
            lines.append(
                f"    [{i}] freq={c.frequency:.6f} period={period:.1f} "
                f"amp={c.amplitude:.4f} snr={c.snr:.1f}"
            )
        lines.append(f"  Trend: {self.trend.trend_type.value} {self.trend.params}")
        lines.append(f"  Shocks: {len(self.shocks)}")
        for i, s in enumerate(self.shocks):
            lines.append(
                f"    [{i}] {s.shape.value} at t={s.onset_idx} "
                f"mag={s.magnitude:.4f} decay={s.decay_rate:.4f}"
            )
        lines.append(f"  Noise std: {self.noise_std:.6f}")
        lines.append(
            f"  Error bounds: [{self.residual_quantiles[0]:.4f}, "
            f"{self.residual_quantiles[1]:.4f}]"
        )
        return "\n".join(lines)


class SpectralForecaster:
    """Three-layer spectral decomposition forecaster."""

    def __init__(
        self,
        sample_rate: float = 1.0,
        max_components: int = 50,
        min_snr: float = 2.0,
        noise_threshold_ratio: float = 0.5,
        recency_halflife: float | None = None,
        shock_lookback: int | None = None,
        shock_min_sigma: float = 2.0,
        confidence_level: float = 0.8,
    ):
        """
        Args:
            sample_rate: Samples per time unit.
            max_components: Max Fourier components to extract.
            min_snr: Min SNR for Fourier extraction.
            noise_threshold_ratio: Stopping threshold for extraction.
            recency_halflife: Half-life for trend recency weighting.
            shock_lookback: Window for shock detection (None = auto).
            shock_min_sigma: Min sigma for shock detection.
            confidence_level: Width of error bounds (0.8 = 10th-90th).
        """
        self.sample_rate = sample_rate
        self.max_components = max_components
        self.min_snr = min_snr
        self.noise_threshold_ratio = noise_threshold_ratio
        self.recency_halflife = recency_halflife
        self.shock_lookback = shock_lookback
        self.shock_min_sigma = shock_min_sigma
        self.confidence_level = confidence_level

        # Stored after fit
        self._extraction: ExtractionResult | None = None
        self._trend: TrendResult | None = None
        self._shocks: ShockResult | None = None
        self._n_train: int = 0
        self._residual: NDArray[np.floating] | None = None
        self._detrend_slope: float = 0.0
        self._detrend_intercept: float = 0.0

    def fit(self, signal: NDArray[np.floating]) -> None:
        """Decompose the signal into three layers."""
        signal = np.asarray(signal, dtype=np.float64)
        if not np.all(np.isfinite(signal)):
            raise ValueError(
                "Input signal contains NaN or Inf values. "
                "Clean the data before forecasting."
            )
        self._n_train = len(signal)
        t = np.arange(self._n_train, dtype=np.float64)

        # Pre-detrend: remove linear trend before FFT to prevent trend energy
        # from leaking into the frequency domain. Standard in spectral analysis.
        X = np.column_stack([t, np.ones(self._n_train)])
        coeffs, _, _, _ = np.linalg.lstsq(X, signal, rcond=None)
        self._detrend_slope = float(coeffs[0])
        self._detrend_intercept = float(coeffs[1])
        linear_trend = self._detrend_slope * t + self._detrend_intercept
        detrended = signal - linear_trend

        # Layer 1: Fourier extraction on detrended signal
        self._extraction = extract(
            detrended,
            sample_rate=self.sample_rate,
            max_components=self.max_components,
            min_snr=self.min_snr,
            noise_threshold_ratio=self.noise_threshold_ratio,
        )

        # Layer 2: Trend fitting on extraction residual + pre-removed trend.
        # The residual still contains any nonlinear trend structure.
        # We add back the linear trend so Layer 2 can fit the full trend shape.
        trend_input = self._extraction.residual + linear_trend
        self._trend = fit_trend(
            trend_input,
            recency_halflife=self.recency_halflife,
        )

        # Predicted signal from Layer 1 + Layer 2
        periodic = np.zeros(self._n_train)
        for comp in self._extraction.components:
            periodic += comp.evaluate(t)
        trend_vals = self._trend.model.predict(t)
        predicted = periodic + trend_vals

        # Layer 3: Shock detection on the full prediction vs actual
        lookback = self.shock_lookback or max(self._n_train // 4, 20)
        noise_for_shock = max(self._extraction.noise_std, self._trend.model.residual_std)
        self._shocks = detect_shocks(
            actual=signal,
            predicted=predicted,
            noise_std=noise_for_shock,
            min_sigma=self.shock_min_sigma,
            lookback_window=lookback,
        )

        # Final residual after all three layers
        shock_vals = np.zeros(self._n_train)
        for shock in self._shocks.shocks:
            shock_vals += shock.evaluate(t)
        self._residual = signal - predicted - shock_vals

    def forecast(self, horizon: int) -> ForecastResult:
        """Generate forecast for `horizon` steps ahead.

        Args:
            horizon: Number of future time steps to forecast.

        Returns:
            ForecastResult with point forecast, bounds, and full decomposition.
        """
        if self._extraction is None or self._trend is None or self._shocks is None:
            raise RuntimeError("Must call fit() before forecast()")

        t_future = np.arange(
            self._n_train, self._n_train + horizon, dtype=np.float64
        )

        # Layer 1: periodic components
        periodic = np.zeros(horizon)
        for comp in self._extraction.components:
            periodic += comp.evaluate(t_future)

        # Layer 2: trend extrapolation
        trend_vals = self._trend.model.predict(t_future)

        # Layer 3: shock extrapolation (shocks continue their shape)
        shock_vals = np.zeros(horizon)
        for shock in self._shocks.shocks:
            shock_vals += shock.evaluate(t_future)

        point_forecast = periodic + trend_vals + shock_vals

        # Error bounds from residual distribution (possibly asymmetric)
        residual = self._residual
        alpha = (1 - self.confidence_level) / 2
        lower_q = float(np.quantile(residual, alpha))
        upper_q = float(np.quantile(residual, 1 - alpha))
        noise_std = float(np.std(residual))

        # Error grows with forecast horizon (sqrt scaling)
        horizon_factor = np.sqrt(1 + np.arange(horizon, dtype=np.float64) / self._n_train)
        lower_bound = point_forecast + lower_q * horizon_factor
        upper_bound = point_forecast + upper_q * horizon_factor

        return ForecastResult(
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            components=self._extraction.components,
            trend=self._trend.model,
            shocks=self._shocks.shocks,
            noise_std=noise_std,
            residual_quantiles=(lower_q, upper_q),
        )

    def fit_forecast(
        self, signal: NDArray[np.floating], horizon: int
    ) -> ForecastResult:
        """Convenience: fit and forecast in one call."""
        self.fit(signal)
        return self.forecast(horizon)
