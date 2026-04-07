"""Integration tests: full pipeline on synthetic data."""

import numpy as np
import pytest

from spectral_forecast.forecast import SpectralForecaster


class TestFullPipeline:
    def test_sinusoid_plus_trend(self):
        """Forecast a sinusoid with linear trend."""
        n = 500
        t = np.arange(n, dtype=np.float64)
        signal = 3.0 * np.cos(2 * np.pi * 0.05 * t) + 0.1 * t + 10.0

        forecaster = SpectralForecaster()
        result = forecaster.fit_forecast(signal, horizon=50)

        # Forecast should continue the pattern
        t_future = np.arange(n, n + 50, dtype=np.float64)
        expected = 3.0 * np.cos(2 * np.pi * 0.05 * t_future) + 0.1 * t_future + 10.0

        # Check that forecast is reasonably close to expected
        mae = float(np.mean(np.abs(result.point_forecast - expected)))
        assert mae < 2.0, f"MAE = {mae}"

    def test_sinusoid_plus_trend_plus_noise(self):
        """Forecast with noise — error bounds should contain truth."""
        rng = np.random.default_rng(42)
        n = 1000
        t = np.arange(n, dtype=np.float64)
        clean = 5.0 * np.cos(2 * np.pi * 0.03 * t) + 0.05 * t
        signal = clean + rng.normal(0, 1.0, n)

        forecaster = SpectralForecaster(confidence_level=0.9)
        result = forecaster.fit_forecast(signal, horizon=30)

        t_future = np.arange(n, n + 30, dtype=np.float64)
        true_future = 5.0 * np.cos(2 * np.pi * 0.03 * t_future) + 0.05 * t_future

        # Most true values should be within error bounds
        in_bounds = (true_future >= result.lower_bound) & (true_future <= result.upper_bound)
        coverage = float(np.mean(in_bounds))
        assert coverage > 0.5, f"Coverage = {coverage}"

    def test_describe_output(self):
        """The describe() method should return interpretable text."""
        t = np.arange(200, dtype=np.float64)
        signal = 2.0 * np.cos(2 * np.pi * 0.1 * t) + 5.0

        forecaster = SpectralForecaster()
        result = forecaster.fit_forecast(signal, horizon=10)
        desc = result.describe()

        assert "Periodic components:" in desc
        assert "Trend:" in desc
        assert "Shocks:" in desc
        assert "Noise std:" in desc

    def test_with_shock(self):
        """Signal with a step change — Layer 3 should detect it."""
        n = 400
        t = np.arange(n, dtype=np.float64)
        signal = 2.0 * np.cos(2 * np.pi * 0.05 * t)
        signal[300:] += 10.0  # step shock at t=300

        forecaster = SpectralForecaster()
        result = forecaster.fit_forecast(signal, horizon=50)

        # Forecast should maintain the elevated level
        assert result.point_forecast.mean() > 5.0, (
            f"Forecast mean {result.point_forecast.mean()} should reflect shock"
        )

    def test_pure_noise_gives_wide_bounds(self):
        """Pure noise should produce wide error bounds and near-zero forecast."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 500)

        forecaster = SpectralForecaster(min_snr=5.0)
        result = forecaster.fit_forecast(signal, horizon=20)

        # Point forecast should be near zero
        assert abs(result.point_forecast.mean()) < 2.0
        # Bounds should be wide relative to the signal
        width = float(np.mean(result.upper_bound - result.lower_bound))
        assert width > 0.5, f"Bounds too narrow for noise: width={width}"


class TestFitForecastSeparate:
    def test_fit_then_forecast(self):
        """fit() and forecast() should work separately."""
        t = np.arange(300, dtype=np.float64)
        signal = np.sin(2 * np.pi * 0.1 * t) + 5.0

        forecaster = SpectralForecaster()
        forecaster.fit(signal)
        r1 = forecaster.forecast(10)
        r2 = forecaster.forecast(20)

        assert len(r1.point_forecast) == 10
        assert len(r2.point_forecast) == 20
        # First 10 of r2 should match r1
        np.testing.assert_allclose(r1.point_forecast, r2.point_forecast[:10])

    def test_forecast_without_fit_raises(self):
        forecaster = SpectralForecaster()
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.forecast(10)
