"""Tests for relationship-specific grouped-matrix nulls."""

import numpy as np

from spectral_forecast.relationship_nulls import grouped_matrix_null_summary


def test_trajectory_regroup_preserves_shared_anchor_template() -> None:
    rng = np.random.default_rng(10)
    template = np.asarray([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    matrices = [
        np.stack(
            [template + rng.normal(0.0, 0.05, len(template)) for _ in range(3)],
            axis=1,
        )
        for _ in range(80)
    ]

    anchor = grouped_matrix_null_summary(
        matrices,
        threshold=2.0,
        min_active_series=2,
        null_mode="anchor-permute",
        null_repeats=40,
        seed=11,
    )
    regroup = grouped_matrix_null_summary(
        matrices,
        threshold=2.0,
        min_active_series=2,
        null_mode="trajectory-regroup",
        null_repeats=40,
        seed=12,
    )

    assert anchor.observed_minus_null > 0.0
    assert anchor.empirical_p_ge_observed <= 1 / 41
    assert abs(regroup.observed_minus_null) < anchor.observed_minus_null * 0.05


def test_trajectory_regroup_detects_group_specific_alignment() -> None:
    rng = np.random.default_rng(20)
    matrices = []
    for _ in range(120):
        anchor = int(rng.integers(0, 12))
        matrix = rng.normal(0.0, 0.05, (12, 3))
        matrix[anchor, :] += 6.0
        matrices.append(matrix)

    regroup = grouped_matrix_null_summary(
        matrices,
        threshold=2.0,
        min_active_series=2,
        null_mode="trajectory-regroup",
        null_repeats=40,
        seed=21,
    )

    assert regroup.observed_minus_null > 0.0
    assert regroup.empirical_p_ge_observed <= 1 / 41


def test_within_slot_regroup_preserves_slot_identity_but_breaks_groups() -> None:
    rng = np.random.default_rng(30)
    matrices = []
    slot_offsets = np.asarray([0.0, 1.0, 2.0])
    for _ in range(120):
        anchor = int(rng.integers(0, 12))
        matrix = rng.normal(0.0, 0.05, (12, 3)) + slot_offsets
        matrix[anchor, :] += 6.0
        matrices.append(matrix)

    regroup = grouped_matrix_null_summary(
        matrices,
        threshold=3.0,
        min_active_series=2,
        null_mode="trajectory-regroup-within-slot",
        null_repeats=40,
        seed=31,
    )

    assert regroup.observed_minus_null > 0.0
    assert regroup.empirical_p_ge_observed <= 1 / 41


def test_grouped_matrix_null_rejects_mixed_shapes() -> None:
    matrices = [np.zeros((8, 3)), np.zeros((9, 3))]

    try:
        grouped_matrix_null_summary(
            matrices,
            threshold=1.0,
            min_active_series=2,
            null_mode="trajectory-regroup",
        )
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("expected a shape mismatch")
