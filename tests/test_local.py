"""Tests for local residual correction."""

import numpy as np

from spectral_forecast.local import LocalModel, forecast_local


def test_forecast_local_bounds_explosive_ar_recursion():
    model = LocalModel(
        order=1,
        coefficients=np.array([2.0], dtype=np.float64),
        intercept=0.0,
        aic=0.0,
        residual_std=0.5,
        window_size=4,
    )

    forecast = forecast_local(
        model,
        recent_residuals=np.array([1.0], dtype=np.float64),
        horizon=2048,
    )

    assert np.all(np.isfinite(forecast))
    assert float(np.max(forecast)) <= 1.5
