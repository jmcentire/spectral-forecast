"""Directional, label-free diagnostics for multichannel structure.

The existing stigmergic objective rewards simultaneous positive activity. This
module complements it by testing four distinct forms of organization against an
independent block-permutation null:

- simultaneous coactivation;
- mutual exclusion;
- directed lagged succession; and
- nonzero-lag phase offset.

The diagnostics identify statistical organization, not its meaning or utility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DirectionalMetricEvidence:
    """Observed-versus-null evidence for one structural mechanism."""

    name: str
    observed: float
    null_mean: float
    null_std: float
    observed_minus_null: float
    z_effect: float | None
    null_exceedances: int
    empirical_p_ge_observed: float
    empirical_p_floor: float
    unique_null_values: int
    detected: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalQuality:
    """Label-free evidence for several possible multichannel organizations."""

    n: int
    input_series_count: int
    usable_series_count: int
    selected_series: list[str]
    active_z_threshold: float
    max_lag: int
    aggregate_quantile: float
    null_repeats: int
    requested_null_block_size: int
    effective_null_block_size: int
    significance_level: float
    min_z_effect: float
    metrics: dict[str, DirectionalMetricEvidence]
    detected_mechanisms: list[str]
    strongest_mechanism: str | None
    strongest_z_effect: float | None
    quality_score: float
    verdict: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metrics"] = {
            name: evidence.to_dict()
            for name, evidence in self.metrics.items()
        }
        return payload


def _robust_standardize(values: NDArray[np.floating]) -> NDArray[np.float64]:
    y = np.asarray(values, dtype=np.float64)
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return np.zeros_like(y, dtype=np.float64)
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if scale < 1e-10:
        scale = float(np.std(finite))
    if scale < 1e-10:
        return np.zeros_like(y, dtype=np.float64)
    return np.where(np.isfinite(y), (y - center) / scale, 0.0).astype(np.float64)


def _matrix_from_series(
    series: Mapping[str, NDArray[np.floating]],
    *,
    max_series: int,
) -> tuple[NDArray[np.float64], list[str]]:
    if not series:
        raise ValueError("at least one series is required")
    if max_series < 2:
        raise ValueError("max_series must be >= 2")
    n = min(len(np.asarray(values)) for values in series.values())
    if n < 16:
        raise ValueError("series are too short for directional diagnostics")

    candidates: list[tuple[float, str, NDArray[np.float64]]] = []
    for name, values in series.items():
        y = np.asarray(values, dtype=np.float64)[:n]
        standardized = _robust_standardize(y)
        variance = float(np.var(standardized))
        if variance < 1e-10:
            continue
        candidates.append((variance, name, standardized))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:max_series]
    if len(selected) < 2:
        raise ValueError("fewer than two non-constant series are available")
    return np.column_stack([item[2] for item in selected]), [item[1] for item in selected]


def _tail_mean(values: NDArray[np.floating], *, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    cutoff = float(np.quantile(finite, quantile))
    tail = finite[finite >= cutoff]
    return float(np.mean(tail)) if len(tail) else 0.0


def directional_metric_values(
    matrix: NDArray[np.floating],
    *,
    active_z_threshold: float = 1.5,
    max_lag: int = 12,
    aggregate_quantile: float = 0.9,
) -> dict[str, float]:
    """Compute competing structural-mechanism statistics for one matrix."""

    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("matrix must be 2D with at least two series")
    if x.shape[0] < 16:
        raise ValueError("matrix is too short for directional diagnostics")
    if active_z_threshold <= 0:
        raise ValueError("active_z_threshold must be positive")
    if not 0.5 <= aggregate_quantile < 1.0:
        raise ValueError("aggregate_quantile must be in [0.5, 1.0)")
    effective_max_lag = min(max_lag, max(1, x.shape[0] // 4))
    if effective_max_lag < 1:
        raise ValueError("max_lag must be positive")

    active = np.asarray(np.abs(x) >= active_z_threshold, dtype=np.float64)
    n, series_count = active.shape
    upper = np.triu(np.ones((series_count, series_count), dtype=bool), k=1)
    off_diagonal = ~np.eye(series_count, dtype=bool)

    activity = np.mean(active, axis=0)
    joint_zero = (active.T @ active) / float(n)
    expected_zero = np.outer(activity, activity)
    zero_lift = joint_zero - expected_zero

    coactivation = _tail_mean(
        np.maximum(zero_lift[upper], 0.0),
        quantile=aggregate_quantile,
    )
    exclusion = _tail_mean(
        np.maximum(-zero_lift[upper], 0.0),
        quantile=aggregate_quantile,
    )

    best_succession = np.zeros((series_count, series_count), dtype=np.float64)
    for lag in range(1, effective_max_lag + 1):
        left = active[:-lag]
        right = active[lag:]
        joint = (left.T @ right) / float(n - lag)
        expected = np.outer(np.mean(left, axis=0), np.mean(right, axis=0))
        best_succession = np.maximum(best_succession, joint - expected)
    succession_gain = np.maximum(best_succession - np.maximum(zero_lift, 0.0), 0.0)
    lagged_succession = _tail_mean(
        succession_gain[off_diagonal],
        quantile=aggregate_quantile,
    )

    centered = x - np.mean(x, axis=0, keepdims=True)
    scales = np.std(centered, axis=0, keepdims=True)
    normalized = centered / np.where(scales > 1e-10, scales, 1.0)
    zero_correlation = np.abs((normalized.T @ normalized) / float(n))
    best_nonzero_correlation = np.zeros_like(zero_correlation)
    for lag in range(1, effective_max_lag + 1):
        lagged = np.abs((normalized[:-lag].T @ normalized[lag:]) / float(n - lag))
        best_nonzero_correlation = np.maximum(best_nonzero_correlation, lagged)
        best_nonzero_correlation = np.maximum(best_nonzero_correlation, lagged.T)
    phase_gain = np.maximum(best_nonzero_correlation - zero_correlation, 0.0)
    phase_offset = _tail_mean(
        phase_gain[upper],
        quantile=aggregate_quantile,
    )

    return {
        "coactivation": coactivation,
        "exclusion": exclusion,
        "lagged_succession": lagged_succession,
        "phase_offset": phase_offset,
    }


def _effective_block_size(requested: int, n: int) -> int:
    if requested < 1:
        raise ValueError("null_block_size must be >= 1")
    return min(requested, max(1, n // 2))


def _block_permute_matrix(
    matrix: NDArray[np.float64],
    *,
    block_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    n, series_count = matrix.shape
    blocks = [np.arange(start, min(start + block_size, n)) for start in range(0, n, block_size)]
    null = np.empty_like(matrix)
    for column in range(series_count):
        order = rng.permutation(len(blocks))
        indices = np.concatenate([blocks[int(index)] for index in order])
        null[:, column] = matrix[indices, column]
    return null


def _metric_evidence(
    name: str,
    observed: float,
    null_values: NDArray[np.float64],
    *,
    significance_level: float,
    min_z_effect: float,
) -> DirectionalMetricEvidence:
    null_mean = float(np.mean(null_values))
    null_std = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
    delta = float(observed - null_mean)
    z_effect = delta / null_std if null_std > 1e-12 else None
    exceedances = int(np.sum(null_values >= observed))
    p_ge = float((exceedances + 1) / (len(null_values) + 1))
    detected = bool(
        delta > 0.0
        and p_ge <= significance_level
        and z_effect is not None
        and z_effect >= min_z_effect
    )
    return DirectionalMetricEvidence(
        name=name,
        observed=float(observed),
        null_mean=null_mean,
        null_std=null_std,
        observed_minus_null=delta,
        z_effect=z_effect,
        null_exceedances=exceedances,
        empirical_p_ge_observed=p_ge,
        empirical_p_floor=float(1.0 / (len(null_values) + 1)),
        unique_null_values=int(len(np.unique(np.round(null_values, decimals=12)))),
        detected=detected,
    )


def directional_quality(
    series: Mapping[str, NDArray[np.floating]],
    *,
    null_repeats: int = 100,
    seed: int = 20260604,
    active_z_threshold: float = 1.5,
    max_lag: int = 12,
    aggregate_quantile: float = 0.9,
    null_block_size: int = 8,
    max_series: int = 64,
    significance_level: float = 0.05,
    min_z_effect: float = 2.0,
) -> DirectionalQuality:
    """Test several forms of multichannel organization against a block null."""

    if null_repeats < 1:
        raise ValueError("null_repeats must be >= 1")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be in (0, 1)")
    matrix, names = _matrix_from_series(series, max_series=max_series)
    effective_block_size = _effective_block_size(null_block_size, matrix.shape[0])
    observed = directional_metric_values(
        matrix,
        active_z_threshold=active_z_threshold,
        max_lag=max_lag,
        aggregate_quantile=aggregate_quantile,
    )

    null_by_metric: dict[str, list[float]] = {name: [] for name in observed}
    rng = np.random.default_rng(seed)
    for _ in range(null_repeats):
        null = _block_permute_matrix(matrix, block_size=effective_block_size, rng=rng)
        values = directional_metric_values(
            null,
            active_z_threshold=active_z_threshold,
            max_lag=max_lag,
            aggregate_quantile=aggregate_quantile,
        )
        for name, value in values.items():
            null_by_metric[name].append(value)

    metrics = {
        name: _metric_evidence(
            name,
            value,
            np.asarray(null_by_metric[name], dtype=np.float64),
            significance_level=significance_level,
            min_z_effect=min_z_effect,
        )
        for name, value in observed.items()
    }
    ranked = sorted(
        metrics.values(),
        key=lambda evidence: (
            evidence.z_effect if evidence.z_effect is not None else float("-inf"),
            evidence.observed_minus_null,
        ),
        reverse=True,
    )
    detected_mechanisms = [evidence.name for evidence in ranked if evidence.detected]
    strongest = ranked[0] if ranked and ranked[0].z_effect is not None else None
    strongest_positive_z = (
        max(0.0, float(strongest.z_effect))
        if strongest is not None and strongest.z_effect is not None
        else 0.0
    )
    quality_score = float(min(1.0, strongest_positive_z / 5.0))
    if detected_mechanisms:
        verdict = "structured"
    elif strongest_positive_z >= min_z_effect:
        verdict = "suggestive_but_unresolved"
    else:
        verdict = "no_detected_directional_structure"

    return DirectionalQuality(
        n=matrix.shape[0],
        input_series_count=len(series),
        usable_series_count=len(names),
        selected_series=names,
        active_z_threshold=active_z_threshold,
        max_lag=min(max_lag, max(1, matrix.shape[0] // 4)),
        aggregate_quantile=aggregate_quantile,
        null_repeats=null_repeats,
        requested_null_block_size=null_block_size,
        effective_null_block_size=effective_block_size,
        significance_level=significance_level,
        min_z_effect=min_z_effect,
        metrics=metrics,
        detected_mechanisms=detected_mechanisms,
        strongest_mechanism=strongest.name if strongest is not None else None,
        strongest_z_effect=strongest.z_effect if strongest is not None else None,
        quality_score=quality_score,
        verdict=verdict,
    )
