"""Tests for ForecastEngine ensemble."""

import numpy as np
import pytest

from spectral_forecast.engine import (
    ForecastEngine,
    ARWorker,
    FourierWorker,
    SRWorker,
    EnsembleResult,
)


class TestForecastEngine:
    def test_basic_observe_predict(self):
        """Engine should produce a forecast after observing enough data."""
        engine = ForecastEngine(workers=[ARWorker(p=12)])
        t = np.arange(200, dtype=np.float64)
        signal = np.sin(2 * np.pi * 0.05 * t) + 5.0
        engine.observe_batch(signal)

        result = engine.predict(10)
        assert isinstance(result, EnsembleResult)
        assert len(result.values) == 10
        assert result.confidence > 0
        assert not np.any(np.isnan(result.values))

    def test_multi_worker_ensemble(self):
        """Multiple workers should produce blended forecast."""
        engine = ForecastEngine(workers=[
            ARWorker(p=12),
            FourierWorker(),
        ])
        rng = np.random.default_rng(42)
        t = np.arange(300, dtype=np.float64)
        signal = 3.0 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * t + rng.normal(0, 0.5, 300)
        engine.observe_batch(signal)

        result = engine.predict(20)
        assert len(result.worker_predictions) >= 2
        assert result.agreement > 0
        assert len(result.lower) == 20
        assert len(result.upper) == 20
        assert np.all(result.lower <= result.upper)

    def test_streaming_observe(self):
        """Observe one value at a time should work."""
        engine = ForecastEngine(workers=[ARWorker(p=12)])
        t = np.arange(100, dtype=np.float64)
        signal = np.sin(2 * np.pi * 0.1 * t) + 3.0

        for v in signal:
            engine.observe(float(v))

        result = engine.predict(5)
        assert len(result.values) == 5
        assert result.confidence > 0

    def test_not_enough_data(self):
        """Should return NaN with zero confidence when insufficient data."""
        engine = ForecastEngine(workers=[ARWorker(p=24)])
        engine.observe_batch(np.array([1.0, 2.0, 3.0]))
        result = engine.predict(5)
        assert result.confidence == 0.0

    def test_describe(self):
        """Describe should produce readable output."""
        engine = ForecastEngine(workers=[ARWorker(p=12)])
        engine.observe_batch(np.sin(np.arange(100, dtype=np.float64)))
        result = engine.predict(5)
        desc = result.describe()
        assert "confidence" in desc
        assert "AR" in desc

    def test_confidence_tracks_accuracy(self):
        """Worker on predictable signal should have higher confidence than on noise."""
        # Predictable signal
        engine_good = ForecastEngine(workers=[ARWorker(p=12)])
        t = np.arange(200, dtype=np.float64)
        engine_good.observe_batch(np.sin(2 * np.pi * 0.05 * t) + 5.0)
        # Feed a few more observations to build confidence
        for v in np.sin(2 * np.pi * 0.05 * np.arange(200, 220)) + 5.0:
            engine_good.observe(float(v))
        result_good = engine_good.predict(5)

        # Noisy signal
        engine_bad = ForecastEngine(workers=[ARWorker(p=12)])
        rng = np.random.default_rng(42)
        engine_bad.observe_batch(rng.normal(0, 1, 200))
        for v in rng.normal(0, 1, 20):
            engine_bad.observe(float(v))
        result_bad = engine_bad.predict(5)

        # Predictable should have higher confidence
        good_conf = result_good.worker_predictions.get("AR")
        bad_conf = result_bad.worker_predictions.get("AR")
        if good_conf and bad_conf:
            assert good_conf.confidence > bad_conf.confidence
