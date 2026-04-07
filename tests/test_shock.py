"""Tests for Layer 3: shock detection and adjustment."""

import numpy as np
import pytest

from spectral_forecast.shock import ShockShape, detect_shocks


class TestStepShock:
    def test_detect_step_up(self):
        """Prediction is flat, actual jumps up mid-series."""
        n = 200
        predicted = np.zeros(n)
        actual = np.zeros(n)
        actual[100:] = 5.0  # step up at t=100

        result = detect_shocks(actual, predicted, noise_std=0.5)

        assert len(result.shocks) >= 1
        shock = result.shocks[0]
        assert shock.shape == ShockShape.STEP
        assert abs(shock.magnitude - 5.0) < 1.0
        assert abs(shock.onset_idx - 100) < 10

    def test_detect_step_down(self):
        n = 200
        predicted = np.full(n, 10.0)
        actual = np.full(n, 10.0)
        actual[150:] = 3.0  # step down

        result = detect_shocks(actual, predicted, noise_std=0.5)

        assert len(result.shocks) >= 1
        assert result.shocks[0].magnitude < 0  # negative step


class TestSpikeDecayShock:
    def test_detect_spike_decay(self):
        """Sudden spike that decays exponentially."""
        n = 200
        predicted = np.zeros(n)
        actual = np.zeros(n)
        t = np.arange(n, dtype=np.float64)
        # Spike at t=100, decays with rate 0.05
        active = t >= 100
        actual[active] = 10.0 * np.exp(-0.05 * (t[active] - 100))

        result = detect_shocks(actual, predicted, noise_std=0.3)

        assert len(result.shocks) >= 1
        shock = result.shocks[0]
        assert shock.shape == ShockShape.SPIKE_DECAY
        assert shock.magnitude > 5.0  # should be close to 10

    def test_no_shock_in_noise(self):
        """No shock should be detected in pure noise within threshold."""
        rng = np.random.default_rng(42)
        n = 200
        predicted = np.zeros(n)
        actual = rng.normal(0, 0.5, n)

        result = detect_shocks(actual, predicted, noise_std=1.0, min_sigma=3.0)
        assert len(result.shocks) == 0


class TestRampShock:
    def test_detect_ramp(self):
        """Linear ramp starting mid-series."""
        n = 200
        predicted = np.zeros(n)
        actual = np.zeros(n)
        # Ramp starting at t=120
        for i in range(120, 200):
            actual[i] = 0.1 * (i - 120)

        result = detect_shocks(actual, predicted, noise_std=0.3)

        assert len(result.shocks) >= 1
        # Should find something near t=120
        shock = result.shocks[0]
        assert abs(shock.onset_idx - 120) < 30


class TestMultipleShocks:
    def test_two_shocks(self):
        """Two distinct shocks: step then spike."""
        n = 300
        predicted = np.zeros(n)
        actual = np.zeros(n)
        actual[80:] += 3.0  # step at 80
        t = np.arange(n, dtype=np.float64)
        spike_mask = t >= 200
        actual[spike_mask] += 8.0 * np.exp(-0.1 * (t[spike_mask] - 200))

        result = detect_shocks(actual, predicted, noise_std=0.5, max_shocks=3)

        assert len(result.shocks) >= 2


class TestShockEvaluation:
    def test_step_evaluation(self):
        from spectral_forecast.shock import ShockComponent

        shock = ShockComponent(
            onset_idx=50, shape=ShockShape.STEP, magnitude=3.0,
            decay_rate=0.0, aic=0.0,
        )
        t = np.arange(100, dtype=np.float64)
        vals = shock.evaluate(t)

        assert np.allclose(vals[:50], 0.0)
        assert np.allclose(vals[50:], 3.0)

    def test_spike_decay_evaluation(self):
        from spectral_forecast.shock import ShockComponent

        shock = ShockComponent(
            onset_idx=10, shape=ShockShape.SPIKE_DECAY, magnitude=5.0,
            decay_rate=0.1, aic=0.0,
        )
        t = np.arange(50, dtype=np.float64)
        vals = shock.evaluate(t)

        assert vals[10] == pytest.approx(5.0, rel=0.01)
        assert vals[20] == pytest.approx(5.0 * np.exp(-0.1 * 10), rel=0.01)
        assert np.allclose(vals[:10], 0.0)
