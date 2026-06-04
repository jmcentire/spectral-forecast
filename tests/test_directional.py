"""Tests for label-free directional structure diagnostics."""

import numpy as np

from spectral_forecast.directional import directional_quality


def _noise(seed: int, *, n: int = 768, count: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"n{i}": rng.normal(0.0, 1.0, n) for i in range(count)}


def _coactivation(seed: int) -> dict[str, np.ndarray]:
    series = _noise(seed)
    for start in range(24, 744, 72):
        for values in series.values():
            values[start : start + 6] += 6.0
    return series


def _exclusion(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 768
    series = {f"e{i}": np.zeros(n, dtype=np.float64) for i in range(8)}
    for index in range(n):
        series[f"e{int(rng.integers(0, 8))}"][index] = 8.0
    return series


def _succession() -> dict[str, np.ndarray]:
    n = 768
    series = {f"s{i}": np.zeros(n, dtype=np.float64) for i in range(8)}
    for start in range(12, n - 24, 32):
        for channel in range(8):
            series[f"s{channel}"][start + 2 * channel] = 8.0
    return series


def _phase_offset(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(768, dtype=np.float64)
    return {
        f"p{i}": np.sin(2.0 * np.pi * t / 48.0 + i * np.pi / 8.0)
        + rng.normal(0.0, 0.05, len(t))
        for i in range(8)
    }


def _diagnose(series: dict[str, np.ndarray], seed: int):
    return directional_quality(
        series,
        null_repeats=32,
        seed=seed,
        max_lag=16,
        null_block_size=8,
    )


def test_directional_quality_does_not_invent_strong_noise_mechanism() -> None:
    result = _diagnose(_noise(1), 11)

    assert result.verdict != "structured"


def test_directional_quality_identifies_coactivation() -> None:
    result = _diagnose(_coactivation(2), 12)

    assert result.strongest_mechanism == "coactivation"
    assert result.metrics["coactivation"].detected


def test_directional_quality_identifies_exclusion() -> None:
    result = _diagnose(_exclusion(3), 13)

    assert result.strongest_mechanism == "exclusion"
    assert result.metrics["exclusion"].detected


def test_directional_quality_identifies_lagged_succession() -> None:
    result = _diagnose(_succession(), 14)

    assert result.strongest_mechanism == "lagged_succession"
    assert result.metrics["lagged_succession"].detected


def test_directional_quality_identifies_phase_offset() -> None:
    result = _diagnose(_phase_offset(3), 15)

    assert result.strongest_mechanism == "phase_offset"
    assert result.metrics["phase_offset"].detected
