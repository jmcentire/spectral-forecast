"""Tests for Layer 2: long-period trend fitting."""

import numpy as np
import pytest

from spectral_forecast.trend import TrendType, fit_trend


class TestLinearTrend:
    def test_pure_linear(self):
        t = np.arange(200, dtype=np.float64)
        signal = 0.5 * t + 10.0
        result = fit_trend(signal)

        assert result.model.trend_type in (TrendType.LINEAR, TrendType.QUADRATIC)
        # Check that prediction at the endpoints is close
        pred = result.model.predict(t)
        r2 = 1 - np.sum((signal - pred) ** 2) / np.sum((signal - np.mean(signal)) ** 2)
        assert r2 > 0.95, f"R² = {r2}"

    def test_linear_with_noise(self):
        rng = np.random.default_rng(42)
        t = np.arange(500, dtype=np.float64)
        signal = 0.3 * t + 5.0 + rng.normal(0, 2.0, 500)
        result = fit_trend(signal)

        # Should detect a trend (not NONE)
        assert result.model.trend_type != TrendType.NONE
        pred = result.model.predict(t)
        # Should capture the slope direction
        slope = pred[-1] - pred[0]
        assert slope > 100, f"Expected positive slope, got {slope}"


class TestExponentialTrend:
    def test_pure_exponential(self):
        t = np.arange(100, dtype=np.float64)
        signal = 2.0 * np.exp(0.03 * t) + 1.0
        result = fit_trend(signal)

        pred = result.model.predict(t)
        r2 = 1 - np.sum((signal - pred) ** 2) / np.sum((signal - np.mean(signal)) ** 2)
        assert r2 > 0.95, f"R² = {r2}, type = {result.model.trend_type}"

    def test_compound_growth(self):
        """Compounding growth (like stock prices)."""
        t = np.arange(200, dtype=np.float64)
        signal = 100 * (1.005 ** t)  # 0.5% per period
        result = fit_trend(signal)

        # Should pick exponential or quadratic (which approximates exponential)
        assert result.model.trend_type in (TrendType.EXPONENTIAL, TrendType.QUADRATIC)
        pred = result.model.predict(t)
        r2 = 1 - np.sum((signal - pred) ** 2) / np.sum((signal - np.mean(signal)) ** 2)
        assert r2 > 0.90, f"R² = {r2}"


class TestNoTrend:
    def test_flat_signal(self):
        signal = np.full(200, 5.0)
        result = fit_trend(signal)

        # BIC should prefer NONE or nearly-zero coefficients
        pred = result.model.predict(np.arange(200, dtype=np.float64))
        variation = float(np.ptp(pred))
        assert variation < 1.0, f"Predicted variation {variation} on flat signal"

    def test_pure_noise(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 500)
        result = fit_trend(signal)

        # Should not find a strong trend in noise
        pred = result.model.predict(np.arange(500, dtype=np.float64))
        assert float(np.std(pred)) < 2.0, "Found strong trend in pure noise"


class TestRecencyWeighting:
    def test_recent_shift(self):
        """Signal that changes behavior recently — recency should help."""
        t = np.arange(500, dtype=np.float64)
        signal = np.zeros(500)
        signal[:400] = 0.1 * t[:400]  # mild upward
        signal[400:] = 0.5 * t[400:]  # steeper recently

        # With recency weighting, the trend should reflect recent slope more
        result = fit_trend(signal, recency_halflife=50)
        pred_end = result.model.predict(np.array([499.0]))
        pred_start = result.model.predict(np.array([400.0]))
        recent_slope = (pred_end[0] - pred_start[0]) / 99

        # The fitted recent slope should be closer to 0.5 than to 0.1
        assert recent_slope > 0.2, f"Recent slope {recent_slope} too low"
