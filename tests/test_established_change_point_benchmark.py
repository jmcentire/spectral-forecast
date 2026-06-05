"""Known-answer checks for the established change-point benchmark."""

import json
from argparse import Namespace

import numpy as np

from experiments.established_change_point_benchmark import (
    checkpoint_signature,
    heldout_gain_gate,
    load_checkpoint,
    robust_standardize,
    run_changeforest,
    summarize_trials,
    synthetic_matrix,
    transform_view,
    wilson_interval,
)


def test_robust_standardize_handles_constant_columns() -> None:
    matrix = np.column_stack((np.arange(20), np.ones(20)))

    standardized = robust_standardize(matrix)

    assert np.all(np.isfinite(standardized))
    assert np.allclose(standardized[:, 1], 0.0)


def test_changeforest_detects_obvious_mean_shift() -> None:
    matrix, split = synthetic_matrix(
        "mean_shift", samples=240, dimensions=6, seed=11
    )

    result = run_changeforest(
        matrix,
        method="random_forest",
        view="raw",
        seed=12,
        permutations=99,
        estimators=50,
        jobs=2,
    )

    assert split is not None
    assert result["detected"]
    assert min(abs(point - split) for point in result["split_points"]) <= 12


def test_ar1_residual_view_removes_one_sample_and_stays_finite() -> None:
    rng = np.random.default_rng(9)
    matrix = np.cumsum(rng.normal(size=(100, 4)), axis=0)

    residuals = transform_view(matrix, "ar1_residual")

    assert residuals.shape == (99, 4)
    assert np.all(np.isfinite(residuals))


def test_summary_counts_misses_against_all_trials() -> None:
    rows = [
        {"detected": True, "split_points": [48], "root_best_split": 48},
        {"detected": True, "split_points": [50, 70], "root_best_split": 70},
        {"detected": False, "split_points": [], "root_best_split": 20},
    ]

    summary = summarize_trials(rows, expected_split=50, tolerance=5)

    assert summary["detection_rate"] == 2 / 3
    assert summary["localized_trials"] == 2
    assert summary["localized_rate_all_trials"] == 2 / 3
    assert summary["root_localized_trials"] == 1
    assert summary["root_localized_rate_all_trials"] == 1 / 3


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(5, 10)

    assert lower < 0.5 < upper


def test_checkpoint_rejects_argument_mismatch(tmp_path) -> None:
    args = Namespace(
        pmubage_root=tmp_path,
        pmu_file_limit=1,
        pmu_events_per_file=2,
        pmu_sensor_limit=10,
        pmu_event_onset=300,
        permutations=19,
        estimators=20,
        seed=7,
        checkpoint=tmp_path / "checkpoint.jsonl",
    )
    args.checkpoint.write_text(
        json.dumps({"signature": checkpoint_signature(args) | {"seed": 8}}) + "\n",
        encoding="utf-8",
    )

    try:
        load_checkpoint(args)
    except ValueError as exc:
        assert "signature" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint should fail")


def test_heldout_gain_gate_uses_early_files_only() -> None:
    baseline = []
    events = []
    for file_index in range(4):
        path = f"frequency_{file_index}.npy"
        for event_index in range(2):
            baseline.append(
                {
                    "file": path,
                    "event_index": event_index,
                    "root_gain": float(file_index + event_index),
                    "root_best_split": 100,
                    "split_points": [100],
                    "detected": True,
                }
            )
            events.append(
                {
                    "file": path,
                    "event_index": event_index,
                    "root_gain": 20.0,
                    "root_best_split": 300,
                    "split_points": [300],
                    "detected": True,
                }
            )

    gate = heldout_gain_gate(
        events,
        baseline,
        expected_split=300,
        tolerance=5,
        quantile=1.0,
    )

    assert gate["calibration_files"] == 2
    assert gate["validation_files"] == 2
    assert gate["threshold"] == 2.0
    assert gate["validation_events"]["root_localized_rate_all_trials"] == 1.0
