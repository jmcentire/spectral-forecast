"""Tests for unsupervised observation tuning."""

import numpy as np

from spectral_forecast.autotune import (
    AutoTuneConfig,
    default_autotune_configs,
    score_autotune_config,
    tune_observation,
)


def _shared_shift_series(n: int = 360) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9)
    t = np.arange(n, dtype=np.float64)
    base = np.sin(2 * np.pi * 0.04 * t)
    out = {}
    for i in range(4):
        y = base + rng.normal(0.0, 0.08, n)
        y[220:] += np.linspace(0.0, 3.0 + i * 0.15, n - 220)
        out[f"s{i}"] = y
    return out


def test_score_autotune_config_reports_null_exceedances_and_quality_terms():
    series = _shared_shift_series()
    config = AutoTuneConfig(
        baseline_size=144,
        adaptive_window=72,
        stride=12,
        emission_threshold=2.0,
        decay=0.8,
        min_active_series=2,
    )

    score = score_autotune_config(series, config, null_repeats=12, seed=3)

    assert score.null_summary.null_repeats == 12
    assert 0 <= score.null_summary.null_exceedances <= 12
    assert score.null_summary.observed_total > 0.0
    assert score.readiness_score >= 0.0
    assert score.null_lift_score >= 0.0
    assert score.quality > -1.0


def test_tune_observation_prefers_non_saturating_threshold():
    series = _shared_shift_series()
    saturated = AutoTuneConfig(
        baseline_size=144,
        adaptive_window=72,
        stride=12,
        emission_threshold=0.0,
        decay=0.8,
        min_active_series=1,
    )
    selective = AutoTuneConfig(
        baseline_size=144,
        adaptive_window=72,
        stride=12,
        emission_threshold=2.5,
        decay=0.8,
        min_active_series=2,
    )

    result = tune_observation(
        series,
        configs=[saturated, selective],
        null_repeats=12,
        seed=5,
    )

    assert result.best is not None
    assert result.best.config == selective
    assert result.scores[0].saturation_penalty <= result.scores[1].saturation_penalty


def test_default_autotune_configs_are_valid_for_common_length():
    configs = default_autotune_configs(512)

    assert configs
    for config in configs:
        config.validate()
        assert config.baseline_size < 512
