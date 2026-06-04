"""Relationship-specific nulls for grouped observer-score matrices."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.autotune import AutoTuneNullSummary, summarize_observed_vs_null_totals

GroupedMatrixNullMode = Literal[
    "anchor-permute",
    "trajectory-regroup",
    "trajectory-regroup-within-slot",
]


def _stack_grouped_matrices(
    matrices: Sequence[NDArray[np.floating]],
) -> NDArray[np.float64]:
    if not matrices:
        raise ValueError("at least one grouped matrix is required")
    grouped = np.stack(
        [np.asarray(matrix, dtype=np.float64) for matrix in matrices],
        axis=0,
    )
    if grouped.ndim != 3:
        raise ValueError("grouped matrices must have shape (groups, anchors, series)")
    if grouped.shape[1] < 2:
        raise ValueError("grouped matrices need at least two anchors")
    if grouped.shape[2] < 2:
        raise ValueError("grouped matrices need at least two series")
    if not np.all(np.isfinite(grouped)):
        raise ValueError("grouped matrices must contain only finite values")
    return grouped


def grouped_emission_total(
    matrices: Sequence[NDArray[np.floating]] | NDArray[np.floating],
    *,
    threshold: float,
    min_active_series: int,
) -> tuple[float, int]:
    """Return total threshold excess and active anchor count across groups."""

    grouped = (
        np.asarray(matrices, dtype=np.float64)
        if isinstance(matrices, np.ndarray) and matrices.ndim == 3
        else _stack_grouped_matrices(matrices)  # type: ignore[arg-type]
    )
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not 1 <= min_active_series <= grouped.shape[2]:
        raise ValueError("min_active_series must be between 1 and series count")
    excess = np.maximum(grouped - threshold, 0.0)
    active = np.sum(excess > 0.0, axis=2)
    emissions = np.sum(excess, axis=2)
    emissions[active < min_active_series] = 0.0
    return float(np.sum(emissions)), int(np.sum(emissions > 0.0))


def _anchor_permute(
    grouped: NDArray[np.float64],
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    groups, anchors, series = grouped.shape
    trajectories = grouped.transpose(0, 2, 1).reshape(groups * series, anchors)
    order = np.argsort(rng.random(trajectories.shape), axis=1)
    permuted = np.take_along_axis(trajectories, order, axis=1)
    return permuted.reshape(groups, series, anchors).transpose(0, 2, 1)


def _trajectory_regroup(
    grouped: NDArray[np.float64],
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    groups, anchors, series = grouped.shape
    trajectories = grouped.transpose(0, 2, 1).reshape(groups * series, anchors)
    regrouped = trajectories[rng.permutation(len(trajectories))]
    return regrouped.reshape(groups, series, anchors).transpose(0, 2, 1)


def _trajectory_regroup_within_slot(
    grouped: NDArray[np.float64],
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    regrouped = np.empty_like(grouped)
    for slot in range(grouped.shape[2]):
        regrouped[:, :, slot] = grouped[rng.permutation(grouped.shape[0]), :, slot]
    return regrouped


def grouped_matrix_null_summary(
    matrices: Sequence[NDArray[np.floating]],
    *,
    threshold: float,
    min_active_series: int,
    null_mode: GroupedMatrixNullMode,
    null_repeats: int = 100,
    seed: int = 20260604,
) -> AutoTuneNullSummary:
    """Compare grouped emissions with a relationship-specific empirical null.

    ``anchor-permute`` preserves each trajectory's score distribution while
    independently destroying anchor alignment.

    ``trajectory-regroup`` preserves every complete trajectory, including any
    shared anchor-position profile, while destroying original group membership.

    ``trajectory-regroup-within-slot`` additionally preserves each series
    slot's identity/distribution while destroying only shared group membership.
    """

    if null_repeats < 1:
        raise ValueError("null_repeats must be >= 1")
    if null_mode not in (
        "anchor-permute",
        "trajectory-regroup",
        "trajectory-regroup-within-slot",
    ):
        raise ValueError(f"unknown grouped matrix null mode: {null_mode}")

    grouped = _stack_grouped_matrices(matrices)
    observed_total, observed_active = grouped_emission_total(
        grouped,
        threshold=threshold,
        min_active_series=min_active_series,
    )
    rng = np.random.default_rng(seed)
    null_totals = []
    for _ in range(null_repeats):
        if null_mode == "anchor-permute":
            null = _anchor_permute(grouped, rng=rng)
        elif null_mode == "trajectory-regroup":
            null = _trajectory_regroup(grouped, rng=rng)
        else:
            null = _trajectory_regroup_within_slot(grouped, rng=rng)
        null_total, _ = grouped_emission_total(
            null,
            threshold=threshold,
            min_active_series=min_active_series,
        )
        null_totals.append(null_total)

    return summarize_observed_vs_null_totals(
        anchors=int(grouped.shape[0] * grouped.shape[1]),
        observed_total=observed_total,
        observed_active_windows=observed_active,
        null_totals=null_totals,
    )
