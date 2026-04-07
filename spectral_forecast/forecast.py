"""Forecaster: superposes all layers with error bounds.

The forecast at any future time t is:
    y(t) = Layer1 periodic + Layer2 trend + Layer3a shocks + Layer3b local correction
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
from spectral_forecast.local import LocalModel, LocalResult, fit_local, forecast_local
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
    local: LocalModel
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
        lines.append(f"  Local: {self.local.describe()}")
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
        trend_damping_halflife: float | None = None,
        amplitude_damping: bool = True,
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
            trend_damping_halflife: Nonlinear trends transition to tangent
                line beyond training boundary. This is the half-life in
                samples. None = auto (context_length / 4).
            amplitude_damping: Dampen Fourier component amplitudes over
                forecast horizon based on phase uncertainty from SNR.
        """
        self.sample_rate = sample_rate
        self.max_components = max_components
        self.min_snr = min_snr
        self.noise_threshold_ratio = noise_threshold_ratio
        self.recency_halflife = recency_halflife
        self.shock_lookback = shock_lookback
        self.shock_min_sigma = shock_min_sigma
        self.confidence_level = confidence_level
        self.trend_damping_halflife = trend_damping_halflife
        self.amplitude_damping = amplitude_damping

        # Stored after fit
        self._extraction: ExtractionResult | None = None
        self._trend: TrendResult | None = None
        self._shocks: ShockResult | None = None
        self._local: LocalResult | None = None
        self._n_train: int = 0
        self._residual: NDArray[np.floating] | None = None
        self._detrend_slope: float = 0.0
        self._detrend_intercept: float = 0.0
        self._periodic_range: tuple[float, float] = (0.0, 0.0)
        self._signal_range: tuple[float, float] = (0.0, 0.0)

    def fit(self, signal: NDArray[np.floating]) -> None:
        """Decompose the signal into three layers."""
        signal = np.asarray(signal, dtype=np.float64)
        if not np.all(np.isfinite(signal)):
            raise ValueError(
                "Input signal contains NaN or Inf values. "
                "Clean the data before forecasting."
            )
        self._n_train = len(signal)
        # Record observed signal range for forecast clamping
        margin = float(np.std(signal))
        self._signal_range = (float(signal.min()) - margin, float(signal.max()) + margin)
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
        # Record observed periodic range — used to clamp forecast extrapolation
        margin = self._extraction.noise_std
        self._periodic_range = (
            float(periodic.min()) - margin,
            float(periodic.max()) + margin,
        )
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

        # Residual after Layer 1 + Layer 2 + Layer 3a (shocks)
        shock_vals = np.zeros(self._n_train)
        for shock in self._shocks.shocks:
            shock_vals += shock.evaluate(t)
        residual_after_shocks = signal - predicted - shock_vals

        # Layer 3b: Local correction — AR model on recent residuals.
        # This captures momentum, mean-reversion, and local autocorrelation
        # that the Fourier + trend model misses.
        self._local = fit_local(residual_after_shocks)

        # Final residual after all layers including local correction
        self._residual = residual_after_shocks - self._local.fitted_values

    def forecast(self, horizon: int) -> ForecastResult:
        """Generate forecast for `horizon` steps ahead.

        Args:
            horizon: Number of future time steps to forecast.

        Returns:
            ForecastResult with point forecast, bounds, and full decomposition.
        """
        if self._extraction is None or self._trend is None or self._shocks is None or self._local is None:
            raise RuntimeError("Must call fit() before forecast()")

        t_future = np.arange(
            self._n_train, self._n_train + horizon, dtype=np.float64
        )

        # Layer 1: periodic components with optional amplitude damping.
        # Phase uncertainty grows with horizon: sigma_phase = 2*h / (N*sqrt(2*snr/3))
        # When sigma_phase is large, the component's contribution is unreliable.
        # Dampen amplitude by exp(-sigma_phase^2 / 2).
        periodic = np.zeros(horizon)
        h_indices = np.arange(1, horizon + 1, dtype=np.float64)
        for comp in self._extraction.components:
            comp_signal = comp.evaluate(t_future)
            if self.amplitude_damping and comp.snr > 0:
                sigma_phase = 2.0 * h_indices / (
                    self._n_train * np.sqrt(max(2.0 * comp.snr / 3.0, 1e-10))
                )
                damping = np.exp(-0.5 * sigma_phase**2)
                comp_signal = comp_signal * damping
            periodic += comp_signal

        # Clamp periodic to observed range. Constructive interference from
        # phase drift can produce values far outside what the data showed.
        # This is data-driven (no tuning parameters): the periodic component
        # of the signal can't be larger than what was observed in context.
        lo, hi = self._periodic_range
        periodic = np.clip(periodic, lo, hi)

        # Layer 2: trend extrapolation with damping for nonlinear trends
        damping_hl = self.trend_damping_halflife
        if damping_hl is None:
            damping_hl = self._n_train / 4.0
        trend_vals = self._trend.model.predict_damped(
            t_future, t_boundary=float(self._n_train - 1), damping_halflife=damping_hl
        )

        # Layer 3a: shock extrapolation (shocks continue their shape)
        shock_vals = np.zeros(horizon)
        for shock in self._shocks.shocks:
            shock_vals += shock.evaluate(t_future)

        # Layer 3b: local correction from AR model on recent residuals
        local_correction = forecast_local(
            self._local.model,
            self._local.recent_residuals,
            horizon,
        )

        point_forecast = periodic + trend_vals + shock_vals + local_correction

        # Clamp forecast to observed signal range. The model should not
        # extrapolate to values far outside what the context data showed.
        # This is data-driven: margin is 1 std of the training signal.
        sig_lo, sig_hi = self._signal_range
        point_forecast = np.clip(point_forecast, sig_lo, sig_hi)

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
            local=self._local.model,
            noise_std=noise_std,
            residual_quantiles=(lower_q, upper_q),
        )

    def fit_forecast(
        self, signal: NDArray[np.floating], horizon: int
    ) -> ForecastResult:
        """Convenience: fit and forecast in one call."""
        self.fit(signal)
        return self.forecast(horizon)
