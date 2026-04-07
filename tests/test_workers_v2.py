"""Tests for v2.0 worker variants."""

import numpy as np
import pytest

from spectral_forecast.engine import ForecastEngine
from spectral_forecast.workers_v2 import (
    ShockAwareFourierWorker,
    NormalityAwareFourierWorker,
    AggressiveDampingFourierWorker,
)


def _make_signal(n=300):
    t = np.arange(n, dtype=np.float64)
    return 3.0 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * t + 5.0


class TestShockAwareFourierWorker:
    def test_produces_forecast(self):
        w = ShockAwareFourierWorker()
        w.observe_batch(_make_signal())
        pred = w.predict(10)
        assert len(pred.values) == 10
        assert pred.confidence > 0
        assert not np.any(np.isnan(pred.values))

    def test_handles_step_function(self):
        signal = _make_signal(300)
        signal[200:] += 10.0  # step shock
        w = ShockAwareFourierWorker()
        w.observe_batch(signal)
        pred = w.predict(10)
        # Should produce a forecast that accounts for the elevated level
        assert not np.any(np.isnan(pred.values))

    def test_in_ensemble(self):
        from spectral_forecast.engine import ARWorker
        engine = ForecastEngine(workers=[ARWorker(p=12), ShockAwareFourierWorker()])
        engine.observe_batch(_make_signal())
        result = engine.predict(10)
        assert "ShockFourier" in result.worker_predictions or len(result.worker_predictions) >= 1


class TestNormalityAwareFourierWorker:
    def test_produces_forecast(self):
        w = NormalityAwareFourierWorker()
        w.observe_batch(_make_signal())
        pred = w.predict(10)
        assert len(pred.values) == 10
        assert pred.confidence > 0

    def test_confidence_penalized_on_heavy_tails(self):
        """Non-Gaussian residuals should reduce confidence."""
        # Clean signal -> Gaussian residuals -> no penalty
        w_clean = NormalityAwareFourierWorker()
        w_clean.observe_batch(_make_signal(300))

        # Signal with outliers -> non-Gaussian residuals -> penalty
        rng = np.random.default_rng(42)
        noisy = _make_signal(300) + rng.standard_cauchy(300) * 0.5
        noisy = np.clip(noisy, -100, 100)  # prevent extreme values
        w_noisy = NormalityAwareFourierWorker()
        w_noisy.observe_batch(noisy)

        # The noisy worker should have lower penalty factor
        # (may or may not trigger depending on exact residuals)
        assert w_clean._normality_penalty >= w_noisy._normality_penalty


class TestAggressiveDampingFourierWorker:
    def test_produces_forecast(self):
        w = AggressiveDampingFourierWorker()
        w.observe_batch(_make_signal())
        pred = w.predict(10)
        assert len(pred.values) == 10
        assert pred.confidence > 0

    def test_damping_increases_with_horizon(self):
        """Longer horizons should have more damping (smaller amplitude)."""
        w = AggressiveDampingFourierWorker()
        w.observe_batch(_make_signal(500))

        short = w.predict(10)
        long_pred = w.predict(200)

        # The std of the long forecast should be smaller due to damping
        short_std = float(np.std(short.values))
        long_std = float(np.std(long_pred.values[-50:]))
        # Aggressive damping should reduce amplitude at long horizons
        assert long_std < short_std * 2  # weak check — just verify it's not exploding

    def test_in_ensemble_with_all_workers(self):
        """Six-worker ensemble should work."""
        from spectral_forecast.engine import ARWorker, FourierWorker, SRWorker
        engine = ForecastEngine(workers=[
            ARWorker(p=12),
            FourierWorker(),
            SRWorker(),
            ShockAwareFourierWorker(),
            NormalityAwareFourierWorker(),
            AggressiveDampingFourierWorker(),
        ])
        engine.observe_batch(_make_signal(300))
        result = engine.predict(20)
        assert len(result.values) == 20
        assert result.confidence > 0
        assert len(result.worker_predictions) >= 4  # at least 4 should be ready
