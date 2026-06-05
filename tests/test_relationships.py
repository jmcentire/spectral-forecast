"""Tests for explicit label-free relationship-hypothesis discovery."""

import numpy as np

from spectral_forecast.relationships import (
    calibrate_relationship_geometry,
    discover_relationships,
    residualize_relationship_view,
)


def _evidence(result, family):
    return [row for row in result.evidence if row.family == family]


def test_relationship_discovery_finds_shared_spectral_shape_without_labels() -> None:
    rng = np.random.default_rng(10)
    n = 2048
    t = np.arange(n, dtype=np.float64)
    frequencies = (0.03, 0.071, 0.12)
    matrix = np.column_stack(
        [
            sum(
                np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
                for frequency in frequencies
            )
            + rng.normal(0.0, 0.15, n)
            for _ in range(3)
        ]
    )

    result = discover_relationships(
        matrix,
        ("a", "b", "c"),
        sample_rate=1.0,
        null_repeats=19,
        seed=11,
        nperseg=256,
    )

    assert any(row.detected for row in _evidence(result, "spectral_alignment"))


def test_relationship_discovery_finds_lag_and_relation_change() -> None:
    rng = np.random.default_rng(20)
    n = 2048
    midpoint = n // 2
    source = rng.normal(0.0, 1.0, n)
    delayed = np.roll(source, 18) + rng.normal(0.0, 0.05, n)
    changing = rng.normal(0.0, 1.0, n)
    changing[midpoint:] = source[midpoint:] + rng.normal(0.0, 0.05, n - midpoint)

    result = discover_relationships(
        np.column_stack([source, delayed, changing]),
        ("source", "delayed", "changing"),
        sample_rate=1.0,
        null_repeats=19,
        seed=21,
        nperseg=256,
        max_lag=32,
    )

    lagged = [
        row
        for row in _evidence(result, "lagged_dependence")
        if set(row.entities) == {"source", "delayed"}
    ][0]
    changed = [
        row
        for row in _evidence(result, "relation_change")
        if set(row.entities) == {"source", "changing"}
    ][0]
    assert lagged.detected
    assert abs(lagged.attributes["best_lag_samples"]) == 18
    assert changed.detected


def test_common_mode_residualization_removes_first_shared_component() -> None:
    rng = np.random.default_rng(30)
    n = 1024
    latent = rng.normal(0.0, 1.0, n)
    matrix = np.column_stack(
        [latent + rng.normal(0.0, 0.1, n) for _ in range(4)]
    )

    raw = discover_relationships(
        matrix,
        ("a", "b", "c", "d"),
        sample_rate=1.0,
        null_repeats=9,
        seed=31,
    )
    residual = discover_relationships(
        residualize_relationship_view(matrix, common_mode_fraction=1.0),
        ("a", "b", "c", "d"),
        sample_rate=1.0,
        null_repeats=9,
        seed=32,
    )

    raw_common = _evidence(raw, "common_mode")[0].observed
    residual_common = _evidence(residual, "common_mode")[0].observed
    assert residual_common < raw_common * 0.5


def test_exact_geometry_calibration_reports_supported_families() -> None:
    calibration = calibrate_relationship_geometry(
        n=1024,
        series_count=3,
        trials=2,
        null_repeats=19,
        seed=40,
        nperseg=128,
        max_lag=32,
        min_detection_rate=0.5,
        max_false_positive_rate=0.5,
    )

    assert calibration.family_detection_rates["common_mode"] >= 0.5
    assert calibration.family_detection_rates["spectral_alignment"] >= 0.5
