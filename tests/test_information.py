"""Tests for finite-window information readiness diagnostics."""

import numpy as np

from spectral_forecast.information import (
    information_readiness,
    scan_information_readiness,
)


def test_sinusoid_has_higher_entropy_deficit_than_noise():
    rng = np.random.default_rng(42)
    t = np.arange(512, dtype=np.float64)
    sinusoid = 3.0 * np.cos(2 * np.pi * 0.08 * t) + rng.normal(0, 0.3, len(t))
    noise = rng.normal(0, 1.0, len(t))

    structured = information_readiness(sinusoid)
    random = information_readiness(noise, min_snr=3.0)

    assert structured.entropy_deficit > random.entropy_deficit
    assert structured.peak_surprise > random.peak_surprise
    assert structured.peak_p_value < random.peak_p_value
    assert structured.ready
    assert not random.ready


def test_readiness_scan_finds_stable_informative_prefix():
    rng = np.random.default_rng(7)
    t = np.arange(1024, dtype=np.float64)
    signal = 2.5 * np.cos(2 * np.pi * 0.05 * t + 0.2) + rng.normal(0, 0.2, len(t))

    scan = scan_information_readiness(
        signal,
        min_size=64,
        step=64,
        stable_windows=2,
    )

    assert scan.first_ready_n is not None
    assert scan.stable_ready_n is not None
    assert scan.stable_ready_n <= 512


def test_diffuse_noise_scan_has_no_stable_ready_prefix():
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1.0, 768)

    scan = scan_information_readiness(
        noise,
        min_snr=3.0,
        min_size=64,
        step=64,
        stable_windows=2,
    )

    assert scan.stable_ready_n is None
    assert scan.points[-1].reason in {"entropy_diffuse", "peak_below_noise_extreme", "too_few_bins"}
