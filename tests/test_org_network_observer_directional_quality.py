"""Tests for organizational observer-score directional diagnostics."""

import argparse

import numpy as np

from experiments.org_network_observer_directional_quality import (
    layer_comparison,
    observer_directional_segment,
    observer_replication_summary,
)
from spectral_forecast.autotune import AutoTuneConfig


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        sample_rate=1.0,
        min_observer_anchors=16,
        null_repeats=16,
        null_block_size=4,
        active_z_threshold=1.5,
        max_lag=6,
        aggregate_quantile=0.9,
        max_score_series=16,
        significance_level=0.1,
        min_z_effect=1.0,
        resolution_trials=2,
        resolution_null_repeats=4,
        resolution_min_detection_rate=0.5,
        resolution_max_false_positive_rate=0.5,
        seed=19,
    )


def test_observer_segment_reports_insufficient_anchor_resolution() -> None:
    series = {
        f"s{i}": np.sin(np.arange(80, dtype=np.float64) / (5.0 + i))
        for i in range(4)
    }
    config = AutoTuneConfig(
        baseline_size=48,
        adaptive_window=16,
        stride=8,
        min_active_series=1,
    )

    result = observer_directional_segment(
        series,
        input_feature_names=list(series),
        config=config,
        args=_args(),
        seed_offset=0,
    )

    assert result["status"] == "insufficient_observer_anchors"
    assert result["observer_anchor_count"] == 4
    assert result["positive_coactivation_aggregate"]["coherence_direction"]
    assert result["resolution_calibration"]["reason"] == "too_few_observations"


def test_observer_segment_diagnoses_dense_score_matrix() -> None:
    rng = np.random.default_rng(20)
    n = 160
    latent = np.sin(np.arange(n, dtype=np.float64) / 5.0)
    series = {
        f"s{i}": latent + rng.normal(0.0, 0.05, n)
        for i in range(6)
    }
    config = AutoTuneConfig(
        baseline_size=48,
        adaptive_window=16,
        stride=2,
        min_active_series=1,
    )

    result = observer_directional_segment(
        series,
        input_feature_names=list(series),
        config=config,
        args=_args(),
        seed_offset=0,
    )

    assert result["status"] == "ok"
    assert result["activation_mode"] == "positive"
    assert result["observer_anchor_count"] >= 16
    assert result["positive_coactivation_aggregate"]["coherence_direction"]
    assert result["resolution_calibration"] is not None


def test_observer_replication_and_layer_comparison_report_loss() -> None:
    segment = {
        "status": "ok",
        "detected_mechanisms": ["phase_offset"],
        "strongest_mechanism": "phase_offset",
        "quality_score": 0.8,
        "resolution_calibration": {
            "supported_mechanisms": ["coactivation", "phase_offset"],
        },
    }
    replication = observer_replication_summary(segment, segment)
    comparison = layer_comparison(
        {
            "replication": {
                "stable_detected_mechanisms": ["coactivation", "phase_offset"],
            }
        },
        replication,
    )

    assert replication["stable_detected_mechanisms"] == ["phase_offset"]
    assert comparison["survived_surface_to_observer"] == ["phase_offset"]
    assert comparison["not_observed_at_observer"] == ["coactivation"]
    assert comparison["interpretable_absences"] == ["coactivation"]
