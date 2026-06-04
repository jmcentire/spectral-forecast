"""Agnostic multichannel structure-readiness diagnostics.

This module asks a preflight question: does a multichannel dataset contain
cross-component or temporal structure beyond what independent shuffled columns
would produce? It does not use labels and it does not identify what the
structure means.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StructureReadiness:
    """Entropy-like preflight metrics for multichannel exploitable structure."""

    n: int
    series_count: int
    usable_series_count: int
    null_repeats: int
    active_z_threshold: float
    active_fraction: float
    covariance_entropy: float
    covariance_null_mean: float
    covariance_null_std: float
    covariance_entropy_deficit: float
    covariance_z_effect: float | None
    pattern_entropy: float
    pattern_null_mean: float
    pattern_null_std: float
    pattern_entropy_deficit: float
    pattern_z_effect: float | None
    temporal_memory: float
    temporal_null_mean: float
    temporal_null_std: float
    temporal_memory_lift: float
    temporal_z_effect: float | None
    structure_score: float
    ready: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize metrics for experiment reports."""

        return asdict(self)


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
) -> tuple[NDArray[np.float64], list[str]]:
    if not series:
        raise ValueError("at least one series is required")
    lengths = [len(np.asarray(values)) for values in series.values()]
    n = min(lengths)
    if n < 4:
        raise ValueError("series are too short for structure readiness")

    rows: list[NDArray[np.float64]] = []
    names: list[str] = []
    for name, values in series.items():
        standardized = _robust_standardize(np.asarray(values, dtype=np.float64)[:n])
        if float(np.std(standardized)) < 1e-10:
            continue
        rows.append(standardized)
        names.append(name)
    if not rows:
        raise ValueError("no non-constant series available for structure readiness")
    matrix = np.column_stack(rows)
    return matrix, names


def _normalized_entropy(weights: NDArray[np.floating]) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) <= 1:
        return 0.0
    probs = values / float(np.sum(values))
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(entropy / np.log(len(probs)))


def covariance_entropy(matrix: NDArray[np.floating]) -> float:
    """Normalized entropy of correlation eigenvalues.

    Values near one mean cross-series variance is diffuse. Lower values mean a
    few shared modes explain more of the cross-series variance.
    """

    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2D")
    if x.shape[1] <= 1:
        return 0.0
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return _normalized_entropy(eigenvalues)


def _pattern_entropy(matrix: NDArray[np.floating], *, active_z_threshold: float) -> tuple[float, float]:
    active = np.abs(np.asarray(matrix, dtype=np.float64)) >= active_z_threshold
    active_fraction = float(np.mean(active)) if active.size else 0.0
    active_rows = [tuple(bool(value) for value in row) for row in active if np.any(row)]
    if len(active_rows) <= 1:
        return 0.0, active_fraction
    _, counts = np.unique(active_rows, return_counts=True, axis=0)
    return _normalized_entropy(counts.astype(np.float64)), active_fraction


def _temporal_memory(matrix: NDArray[np.floating]) -> float:
    values: list[float] = []
    for column in range(matrix.shape[1]):
        y = np.asarray(matrix[:, column], dtype=np.float64)
        if len(y) < 3 or float(np.std(y)) < 1e-10:
            continue
        corr = float(np.corrcoef(y[:-1], y[1:])[0, 1])
        if np.isfinite(corr):
            values.append(abs(corr))
    return float(np.mean(values)) if values else 0.0


def _z_effect(observed: float, null_values: NDArray[np.float64]) -> tuple[float, float, float | None]:
    mean = float(np.mean(null_values)) if len(null_values) else 0.0
    std = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
    return mean, std, (observed - mean) / std if std > 1e-9 else None


def _score_from_z(z: float | None) -> float:
    if z is None or z <= 0:
        return 0.0
    return float(min(1.0, z / 3.0))


def _support_score(active_fraction: float) -> float:
    if active_fraction <= 0.0:
        return 0.0
    if active_fraction < 0.01:
        return float(active_fraction / 0.01)
    if active_fraction <= 0.35:
        return 1.0
    if active_fraction >= 0.75:
        return 0.0
    return float(1.0 - (active_fraction - 0.35) / 0.40)


def _saturation_score(active_fraction: float) -> float:
    if active_fraction < 0.75:
        return 1.0
    if active_fraction >= 0.95:
        return 0.0
    return float(1.0 - (active_fraction - 0.75) / 0.20)


def structure_readiness(
    series: Mapping[str, NDArray[np.floating]],
    *,
    null_repeats: int = 100,
    seed: int = 20260604,
    active_z_threshold: float = 1.5,
    min_series: int = 3,
    min_score: float = 0.35,
) -> StructureReadiness:
    """Estimate whether multichannel data contains exploitable structure.

    The null independently permutes each component through time. That preserves
    each series' marginal distribution while destroying temporal memory and
    cross-series alignment.
    """

    if null_repeats < 1:
        raise ValueError("null_repeats must be >= 1")
    if active_z_threshold <= 0:
        raise ValueError("active_z_threshold must be positive")
    if min_series < 1:
        raise ValueError("min_series must be >= 1")

    matrix, names = _matrix_from_series(series)
    cov_entropy = covariance_entropy(matrix)
    pat_entropy, active_fraction = _pattern_entropy(
        matrix,
        active_z_threshold=active_z_threshold,
    )
    temporal = _temporal_memory(matrix)

    rng = np.random.default_rng(seed)
    cov_null: list[float] = []
    pat_null: list[float] = []
    temporal_null: list[float] = []
    for _ in range(null_repeats):
        null = matrix.copy()
        for column in range(null.shape[1]):
            null[:, column] = rng.permutation(null[:, column])
        cov_null.append(covariance_entropy(null))
        null_pattern, _ = _pattern_entropy(null, active_z_threshold=active_z_threshold)
        pat_null.append(null_pattern)
        temporal_null.append(_temporal_memory(null))

    cov_null_array = np.asarray(cov_null, dtype=np.float64)
    pat_null_array = np.asarray(pat_null, dtype=np.float64)
    temporal_null_array = np.asarray(temporal_null, dtype=np.float64)

    cov_mean, cov_std, cov_z = _z_effect(-cov_entropy, -cov_null_array)
    cov_mean = -cov_mean
    covariance_deficit = float(cov_mean - cov_entropy)

    pat_mean, pat_std, pat_z = _z_effect(-pat_entropy, -pat_null_array)
    pat_mean = -pat_mean
    pattern_deficit = float(pat_mean - pat_entropy)

    temporal_mean, temporal_std, temporal_z = _z_effect(temporal, temporal_null_array)
    temporal_lift = float(temporal - temporal_mean)

    pattern_support = _support_score(active_fraction)
    saturation = _saturation_score(active_fraction)
    covariance_score = _score_from_z(cov_z)
    pattern_score = _score_from_z(pat_z)
    temporal_score = _score_from_z(temporal_z)
    best_structure = max(covariance_score, pattern_score, temporal_score)
    structure_score = float(
        saturation
        * min(
            1.0,
            0.45 * covariance_score
            + 0.30 * pattern_support * pattern_score
            + 0.25 * temporal_score
        )
    )

    ready = bool(
        len(names) >= min_series
        and saturation > 0.0
        and structure_score >= min_score
        and best_structure > 0.0
    )
    if len(names) < min_series:
        reason = "too_few_series"
    elif saturation <= 0.0:
        reason = "active_fraction_saturated"
    elif best_structure <= 0.0:
        reason = "no_null_separation"
    elif structure_score < min_score:
        reason = "weak_structure"
    else:
        reason = "ready"

    return StructureReadiness(
        n=matrix.shape[0],
        series_count=len(series),
        usable_series_count=len(names),
        null_repeats=null_repeats,
        active_z_threshold=active_z_threshold,
        active_fraction=active_fraction,
        covariance_entropy=cov_entropy,
        covariance_null_mean=cov_mean,
        covariance_null_std=cov_std,
        covariance_entropy_deficit=covariance_deficit,
        covariance_z_effect=cov_z,
        pattern_entropy=pat_entropy,
        pattern_null_mean=pat_mean,
        pattern_null_std=pat_std,
        pattern_entropy_deficit=pattern_deficit,
        pattern_z_effect=pat_z,
        temporal_memory=temporal,
        temporal_null_mean=temporal_mean,
        temporal_null_std=temporal_std,
        temporal_memory_lift=temporal_lift,
        temporal_z_effect=temporal_z,
        structure_score=structure_score,
        ready=ready,
        reason=reason,
    )
