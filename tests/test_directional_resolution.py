"""Tests for independent directional-resolution calibration."""

from spectral_forecast.directional_resolution import directional_resolution_calibration


def test_resolution_calibration_rejects_too_few_observations() -> None:
    report = directional_resolution_calibration(n=4, series_count=8, trials=2)

    assert not report.usable
    assert report.reason == "too_few_observations"


def test_resolution_calibration_reports_supported_mechanisms() -> None:
    report = directional_resolution_calibration(
        n=128,
        series_count=8,
        trials=4,
        null_repeats=24,
        seed=22,
        min_detection_rate=0.5,
        max_false_positive_rate=0.25,
    )

    assert report.false_positive_controlled
    assert "coactivation" in report.supported_mechanisms
