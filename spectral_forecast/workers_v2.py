"""v2.0 worker variants for the ForecastEngine ensemble.

These workers implement three Gemini-identified improvements that previously
regressed when applied to the decomposition pipeline directly. As ensemble
workers, the confidence mechanism auto-downweights them when they hurt:

1. ShockAwareFourierWorker: pre-pass shock detection before Fourier extraction
2. NormalityAwareFourierWorker: Shapiro-Wilk confidence modifier
3. AggressiveDampingFourierWorker: faster trend/amplitude damping for long horizons
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.engine import Worker


class ShockAwareFourierWorker(Worker):
    """Fourier decomposition with pre-pass level shift detection.

    Runs detect_level_shifts() before Fourier extraction. If level shifts
    are found, removes them from the signal, decomposes the cleaned signal,
    and adds the shifts back to the forecast.

    This failed as a pipeline change because it can't distinguish periodic
    components from level shifts. As a worker variant, the ensemble's
    confidence tracking downweights it on periodic data where the pre-pass
    hurts, but it may help on data with genuine discontinuities.
    """

    name = "ShockFourier"
    min_observations = 64

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.forecast import SpectralForecaster
        from spectral_forecast.prepass import detect_level_shifts

        arr = np.array(self._history, dtype=np.float64)
        self._prepass_result = detect_level_shifts(arr)
        self._model = SpectralForecaster()
        self._model.fit(self._prepass_result.cleaned_signal)

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        forecast = self._model.forecast(horizon).point_forecast
        # Add back any detected level shifts (they persist into the future)
        if len(self._prepass_result.level_shifts) > 0:
            shift_level = self._prepass_result.shift_signal[-1]
            forecast = forecast + shift_level
        return forecast

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.forecast import SpectralForecaster
        from spectral_forecast.prepass import detect_level_shifts

        prepass = detect_level_shifts(data)
        m = SpectralForecaster()
        m.fit(prepass.cleaned_signal)
        forecast = m.forecast(horizon).point_forecast
        if len(prepass.level_shifts) > 0:
            forecast = forecast + prepass.shift_signal[-1]
        return forecast


class NormalityAwareFourierWorker(Worker):
    """Fourier decomposition with Shapiro-Wilk confidence modifier.

    Runs the full decomposition identically to FourierWorker, but tests
    the residual for normality. If the residual fails Shapiro-Wilk (p < 0.01),
    the worker's confidence is reduced by 50%.

    The insight: non-Gaussian residuals indicate the decomposition may be
    mining tails for spurious structure. This doesn't change the forecast,
    just how much the ensemble trusts it.
    """

    name = "NormFourier"
    min_observations = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._normality_penalty: float = 1.0

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.forecast import SpectralForecaster
        from scipy.stats import shapiro

        arr = np.array(self._history, dtype=np.float64)
        self._model = SpectralForecaster()
        self._model.fit(arr)

        # Test residual normality
        residual = self._model._residual
        if residual is not None and len(residual) >= 20:
            sample = residual[:5000] if len(residual) > 5000 else residual
            _, p_value = shapiro(sample)
            self._normality_penalty = 1.0 if p_value > 0.01 else 0.5
        else:
            self._normality_penalty = 1.0

    @property
    def confidence(self) -> float:
        """Base confidence modified by normality test."""
        base = super().confidence
        return base * self._normality_penalty

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        return self._model.forecast(horizon).point_forecast

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.forecast import SpectralForecaster
        m = SpectralForecaster()
        m.fit(data)
        return m.forecast(horizon).point_forecast


class AggressiveDampingFourierWorker(Worker):
    """Fourier decomposition with aggressive trend and amplitude damping.

    Uses faster damping than the default FourierWorker:
    - Trend damping halflife = N/8 instead of N/4
    - Amplitude damping with 2x phase uncertainty factor

    This worker should win at long horizons (H=336, 720) where trend
    extrapolation diverges. At short horizons it may slightly underperform
    the default FourierWorker because it over-damps components that are
    still reliable.
    """

    name = "AggrFourier"
    min_observations = 64

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.forecast import SpectralForecaster

        arr = np.array(self._history, dtype=np.float64)
        self._model = SpectralForecaster(
            trend_damping_halflife=len(arr) / 8.0,
            amplitude_damping=True,
        )
        self._model.fit(arr)

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        # Override the forecast to apply extra amplitude damping
        result = self._model.forecast(horizon)
        # Apply additional damping factor (2x the default phase uncertainty)
        h_indices = np.arange(1, horizon + 1, dtype=np.float64)
        extra_damping = np.exp(-0.5 * (h_indices / max(len(self._history), 1)) ** 0.5)
        # Blend: weighted toward the damped version
        damped = result.point_forecast * extra_damping
        undamped = result.point_forecast
        # The extra damping increases with horizon: near-term is mostly undamped
        blend = np.exp(-h_indices / (len(self._history) / 4.0))
        return blend * undamped + (1 - blend) * damped

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.forecast import SpectralForecaster
        m = SpectralForecaster(
            trend_damping_halflife=len(data) / 8.0,
            amplitude_damping=True,
        )
        m.fit(data)
        result = m.forecast(horizon)
        h_indices = np.arange(1, horizon + 1, dtype=np.float64)
        extra_damping = np.exp(-0.5 * (h_indices / max(len(data), 1)) ** 0.5)
        blend = np.exp(-h_indices / (len(data) / 4.0))
        return blend * result.point_forecast + (1 - blend) * result.point_forecast * extra_damping
