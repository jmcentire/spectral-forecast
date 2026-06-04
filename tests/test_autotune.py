"""Tests for unsupervised observation tuning."""

import numpy as np

from spectral_forecast.autotune import (
    AutoTuneConfig,
    build_autotune_observation,
    default_autotune_configs,
    observation_null_totals_for_config,
    score_autotune_config,
    score_autotune_observation,
    summarize_observed_vs_null_totals,
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


def test_cached_observation_matches_direct_score_for_same_null_seed():
    series = _shared_shift_series()
    config = AutoTuneConfig(
        baseline_size=144,
        adaptive_window=72,
        stride=12,
        emission_threshold=2.0,
        decay=0.8,
        min_active_series=2,
    )

    observation = build_autotune_observation(series, config)
    cached = score_autotune_observation(observation, config, null_repeats=12, seed=7)
    direct = score_autotune_config(series, config, null_repeats=12, seed=7)

    assert cached.null_summary.observed_total == direct.null_summary.observed_total
    assert cached.null_summary.null_mean == direct.null_summary.null_mean
    assert cached.quality == direct.quality


def test_score_autotune_config_supports_shift_and_block_nulls():
    series = _shared_shift_series()
    config = AutoTuneConfig(
        baseline_size=144,
        adaptive_window=72,
        stride=12,
        emission_threshold=2.0,
        decay=0.8,
        min_active_series=2,
    )

    shift = score_autotune_config(series, config, null_repeats=8, seed=3, null_mode="shift")
    blocks = score_autotune_config(
        series,
        config,
        null_repeats=8,
        seed=3,
        null_mode="block-permute",
        null_block_size=3,
    )

    assert shift.null_summary.null_repeats == 8
    assert blocks.null_summary.null_repeats == 8
    assert shift.null_summary.unique_null_totals > 1
    assert blocks.null_summary.unique_null_totals > 1


def test_observation_null_totals_can_be_summarized_as_batch_distribution():
    series = _shared_shift_series()
    config = AutoTuneConfig(
        baseline_size=144,
        adaptive_window=72,
        stride=12,
        emission_threshold=2.0,
        decay=0.8,
        min_active_series=2,
    )
    observation = build_autotune_observation(series, config)
    score = score_autotune_observation(observation, config, null_repeats=8, seed=11)
    null_totals = observation_null_totals_for_config(
        observation,
        config,
        null_repeats=8,
        seed=11,
    )

    summary = summarize_observed_vs_null_totals(
        anchors=score.null_summary.anchors,
        observed_total=score.null_summary.observed_total,
        observed_active_windows=score.null_summary.observed_active_windows,
        null_totals=null_totals,
    )

    assert summary.null_repeats == 8
    assert summary.null_mean == score.null_summary.null_mean
    assert summary.observed_minus_null == score.null_summary.observed_minus_null


def test_null_summary_reports_both_empirical_tails() -> None:
    summary = summarize_observed_vs_null_totals(
        anchors=4,
        observed_total=0.0,
        observed_active_windows=1,
        null_totals=[1.0, 2.0, 3.0],
    )

    assert summary.null_exceedances == 3
    assert summary.null_below_or_equal == 0
    assert summary.empirical_p_ge_observed == 1.0
    assert summary.empirical_p_le_observed == 0.25
    assert summary.empirical_p_two_sided == 0.5


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
