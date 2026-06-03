"""Unsupervised tuning for spectral+stigmergy observation.

The tuner does not use labels. It searches for settings that produce
reproducible cross-series structure above a domain-preserving timing null while
penalizing saturation and fragile threshold effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.information import information_readiness
from spectral_forecast.observation import ObservationResult, ScoreName, observe_series


@dataclass(frozen=True)
class AutoTuneConfig:
    """One candidate observation/stigmergy configuration."""

    baseline_size: int
    adaptive_window: int
    stride: int
    score: ScoreName = "max"
    emission_threshold: float = 3.0
    decay: float = 0.9
    min_active_series: int = 2

    def validate(self) -> None:
        if self.baseline_size <= self.adaptive_window:
            raise ValueError("baseline_size must be greater than adaptive_window")
        if self.adaptive_window <= 8:
            raise ValueError("adaptive_window must be greater than 8")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.emission_threshold < 0:
            raise ValueError("emission_threshold must be >= 0")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if self.min_active_series < 1:
            raise ValueError("min_active_series must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AutoTuneNullSummary:
    """Observed-vs-null aggregate for one candidate."""

    anchors: int
    observed_total: float
    observed_active_windows: int
    null_repeats: int
    null_mean: float
    null_std: float
    observed_minus_null: float
    z_effect: float | None
    null_exceedances: int
    empirical_p_ge_observed: float
    empirical_p_floor: float
    unique_null_totals: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AutoTuneScore:
    """Scored candidate with decomposed quality terms."""

    config: AutoTuneConfig
    quality: float
    readiness_score: float
    null_lift_score: float
    stability_score: float
    compression_score: float
    residual_activity_score: float
    saturation_penalty: float
    fragility_penalty: float
    null_summary: AutoTuneNullSummary

    @property
    def accepted(self) -> bool:
        """Whether this config clears the label-free tuning gate."""
        return self.quality > 0.0 and self.null_lift_score > 0.0 and self.saturation_penalty < 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "accepted": self.accepted,
            "quality": self.quality,
            "readiness_score": self.readiness_score,
            "null_lift_score": self.null_lift_score,
            "stability_score": self.stability_score,
            "compression_score": self.compression_score,
            "residual_activity_score": self.residual_activity_score,
            "saturation_penalty": self.saturation_penalty,
            "fragility_penalty": self.fragility_penalty,
            "null_summary": self.null_summary.to_dict(),
        }


@dataclass(frozen=True)
class AutoTuneResult:
    """Sorted label-free tuning results."""

    scores: list[AutoTuneScore]
    skipped: list[dict[str, object]]

    @property
    def best(self) -> AutoTuneScore | None:
        return self.scores[0] if self.scores else None

    def to_dict(self) -> dict[str, object]:
        return {
            "best": self.best.to_dict() if self.best is not None else None,
            "scores": [score.to_dict() for score in self.scores],
            "skipped": self.skipped,
        }


def default_autotune_configs(
    n: int,
    *,
    thresholds: Sequence[float] = (2.5, 3.0, 3.5),
    decays: Sequence[float] = (0.75, 0.9),
    min_active_series_options: Sequence[int] = (2, 3),
) -> list[AutoTuneConfig]:
    """Generate conservative candidate configs for a common series length."""

    if n <= 32:
        raise ValueError("series length must be greater than 32")

    baseline_candidates = [
        max(16, int(round(n * fraction)))
        for fraction in (0.2, 0.3)
    ]
    adaptive_candidates = [
        max(12, int(round(n * fraction)))
        for fraction in (0.1, 0.15)
    ]
    stride_candidates = [
        max(1, int(round(n * fraction)))
        for fraction in (0.025, 0.05)
    ]

    configs: list[AutoTuneConfig] = []
    seen: set[tuple[object, ...]] = set()
    for baseline in baseline_candidates:
        for adaptive in adaptive_candidates:
            if baseline <= adaptive or n <= baseline:
                continue
            for stride in stride_candidates:
                for threshold in thresholds:
                    for decay in decays:
                        for min_active in min_active_series_options:
                            config = AutoTuneConfig(
                                baseline_size=baseline,
                                adaptive_window=adaptive,
                                stride=stride,
                                emission_threshold=float(threshold),
                                decay=float(decay),
                                min_active_series=int(min_active),
                            )
                            key = tuple(config.to_dict().values())
                            if key not in seen:
                                seen.add(key)
                                configs.append(config)
    if not configs:
        raise ValueError("no valid autotune configs for series length")
    return configs


def _score_matrix(
    results: Sequence[ObservationResult],
    score: ScoreName,
) -> tuple[list[int], NDArray[np.float64]]:
    if not results:
        raise ValueError("at least one observation result is required")
    by_series = [{point.index: point.score(score) for point in result.points} for result in results]
    common = sorted(set.intersection(*(set(row) for row in by_series)))
    if not common:
        raise ValueError("observation results have no common anchors")
    matrix = np.asarray([[row[index] for row in by_series] for index in common], dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return common, matrix


def _active_matrix(
    matrix: NDArray[np.float64],
    *,
    threshold: float,
) -> NDArray[np.bool_]:
    return np.asarray(matrix > threshold, dtype=np.bool_)


def _emissions_from_matrix(
    matrix: NDArray[np.float64],
    *,
    threshold: float,
    min_active_series: int,
) -> NDArray[np.float64]:
    excess = np.maximum(matrix - threshold, 0.0)
    active = np.sum(excess > 0.0, axis=1)
    emissions = np.sum(excess, axis=1)
    emissions[active < min_active_series] = 0.0
    return emissions.astype(np.float64)


def _decayed_total(emissions: NDArray[np.float64], *, decay: float) -> float:
    pheromone = 0.0
    total = 0.0
    for emission in emissions:
        pheromone = decay * pheromone + float(emission)
        total += pheromone
    return float(total)


def _null_summary(
    matrix: NDArray[np.float64],
    *,
    config: AutoTuneConfig,
    null_repeats: int,
    seed: int,
) -> AutoTuneNullSummary:
    if null_repeats < 1:
        raise ValueError("null_repeats must be >= 1")

    observed_emissions = _emissions_from_matrix(
        matrix,
        threshold=config.emission_threshold,
        min_active_series=config.min_active_series,
    )
    observed_total = _decayed_total(observed_emissions, decay=config.decay)

    rng = np.random.default_rng(seed)
    null_totals = []
    for _ in range(null_repeats):
        permuted = matrix.copy()
        for column in range(permuted.shape[1]):
            permuted[:, column] = rng.permutation(permuted[:, column])
        emissions = _emissions_from_matrix(
            permuted,
            threshold=config.emission_threshold,
            min_active_series=config.min_active_series,
        )
        null_totals.append(_decayed_total(emissions, decay=config.decay))

    null = np.asarray(null_totals, dtype=np.float64)
    null_mean = float(np.mean(null))
    null_std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    exceedances = int(np.sum(null >= observed_total))
    return AutoTuneNullSummary(
        anchors=int(matrix.shape[0]),
        observed_total=observed_total,
        observed_active_windows=int(np.sum(observed_emissions > 0.0)),
        null_repeats=null_repeats,
        null_mean=null_mean,
        null_std=null_std,
        observed_minus_null=observed_total - null_mean,
        z_effect=(observed_total - null_mean) / null_std if null_std > 0 else None,
        null_exceedances=exceedances,
        empirical_p_ge_observed=(exceedances + 1) / (null_repeats + 1),
        empirical_p_floor=1 / (null_repeats + 1),
        unique_null_totals=int(len(set(float(value) for value in null))),
    )


def _top_anchor_set(emissions: NDArray[np.float64], *, top_fraction: float = 0.2) -> set[int]:
    positive = np.flatnonzero(emissions > 0.0)
    if len(positive) == 0:
        return set()
    top_n = max(1, int(np.ceil(len(positive) * top_fraction)))
    ordered = positive[np.argsort(emissions[positive])[::-1]]
    return set(int(index) for index in ordered[:top_n])


def _stability_score(
    matrix: NDArray[np.float64],
    *,
    config: AutoTuneConfig,
) -> float:
    if matrix.shape[1] < 2:
        return 0.0
    left = matrix[:, 0::2]
    right = matrix[:, 1::2]
    left_min = max(1, int(np.ceil(config.min_active_series * left.shape[1] / matrix.shape[1])))
    right_min = max(1, int(np.ceil(config.min_active_series * right.shape[1] / matrix.shape[1])))
    left_emissions = _emissions_from_matrix(
        left,
        threshold=config.emission_threshold,
        min_active_series=left_min,
    )
    right_emissions = _emissions_from_matrix(
        right,
        threshold=config.emission_threshold,
        min_active_series=right_min,
    )
    left_top = _top_anchor_set(left_emissions)
    right_top = _top_anchor_set(right_emissions)
    if not left_top and not right_top:
        return 0.0
    union = left_top | right_top
    if not union:
        return 0.0
    return float(len(left_top & right_top) / len(union))


def _compression_score(active: NDArray[np.bool_]) -> float:
    rows = [tuple(bool(value) for value in row) for row in active if np.any(row)]
    if not rows:
        return 0.0
    unique = len(set(rows))
    return float(max(0.0, 1.0 - unique / len(rows)))


def _saturation_penalty(active: NDArray[np.bool_], active_windows: int) -> float:
    if active.size == 0:
        return 0.0
    active_window_fraction = active_windows / max(active.shape[0], 1)
    active_cell_fraction = float(np.mean(active))
    window_penalty = max(0.0, (active_window_fraction - 0.35) / 0.65)
    cell_penalty = max(0.0, (active_cell_fraction - 0.25) / 0.75)
    return float(min(1.0, max(window_penalty, cell_penalty)))


def _fragility_penalty(
    matrix: NDArray[np.float64],
    *,
    config: AutoTuneConfig,
    relative_jitter: float = 0.1,
) -> float:
    if config.emission_threshold <= 0:
        return 1.0
    base = _emissions_from_matrix(
        matrix,
        threshold=config.emission_threshold,
        min_active_series=config.min_active_series,
    )
    low = _emissions_from_matrix(
        matrix,
        threshold=config.emission_threshold * (1.0 - relative_jitter),
        min_active_series=config.min_active_series,
    )
    high = _emissions_from_matrix(
        matrix,
        threshold=config.emission_threshold * (1.0 + relative_jitter),
        min_active_series=config.min_active_series,
    )
    base_top = _top_anchor_set(base)
    low_top = _top_anchor_set(low)
    high_top = _top_anchor_set(high)
    if not base_top:
        return 1.0

    def jaccard(other: set[int]) -> float:
        union = base_top | other
        return len(base_top & other) / len(union) if union else 0.0

    stability = min(jaccard(low_top), jaccard(high_top))
    return float(1.0 - stability)


def _readiness_score(
    series: Mapping[str, NDArray[np.floating]],
    *,
    config: AutoTuneConfig,
    sample_rate: float,
) -> float:
    scores = []
    for values in series.values():
        y = np.asarray(values, dtype=np.float64)
        if len(y) < config.baseline_size:
            scores.append(0.0)
            continue
        baseline = information_readiness(
            y[: config.baseline_size],
            sample_rate=sample_rate,
        )
        adaptive = information_readiness(
            y[: config.adaptive_window],
            sample_rate=sample_rate,
            min_usable_bins=min(16, max(1, config.adaptive_window // 8)),
        )
        scores.append(min(baseline.readiness_score, adaptive.readiness_score))
    return float(np.mean(scores)) if scores else 0.0


def _null_lift_score(summary: AutoTuneNullSummary) -> float:
    if summary.observed_total <= 0 or summary.observed_minus_null <= 0:
        return 0.0
    fractional_lift = summary.observed_minus_null / max(abs(summary.null_mean), 1e-9)
    z = summary.z_effect if summary.z_effect is not None else 0.0
    frac_score = max(0.0, min(1.0, fractional_lift))
    z_score = max(0.0, min(1.0, z / 5.0))
    empirical_score = 1.0 - summary.empirical_p_ge_observed
    return float(0.4 * frac_score + 0.4 * z_score + 0.2 * empirical_score)


def score_autotune_config(
    series: Mapping[str, NDArray[np.floating]],
    config: AutoTuneConfig,
    *,
    sample_rate: float = 1.0,
    null_repeats: int = 100,
    seed: int = 20260603,
) -> AutoTuneScore:
    """Score one candidate without using labels."""

    config.validate()
    if len(series) < 1:
        raise ValueError("at least one series is required")
    if config.min_active_series > len(series):
        raise ValueError("min_active_series cannot exceed number of series")

    results = [
        observe_series(
            np.asarray(values, dtype=np.float64),
            series=name,
            baseline_size=config.baseline_size,
            adaptive_window=config.adaptive_window,
            stride=config.stride,
            sample_rate=sample_rate,
        )
        for name, values in series.items()
    ]
    _, matrix = _score_matrix(results, config.score)
    null_summary = _null_summary(
        matrix,
        config=config,
        null_repeats=null_repeats,
        seed=seed,
    )
    active = _active_matrix(matrix, threshold=config.emission_threshold)
    readiness = _readiness_score(series, config=config, sample_rate=sample_rate)
    null_lift = _null_lift_score(null_summary)
    stability = _stability_score(matrix, config=config)
    compression = _compression_score(active)
    residual_activity = min(
        1.0,
        null_summary.observed_active_windows / max(1.0, 0.15 * null_summary.anchors),
    )
    saturation = _saturation_penalty(active, null_summary.observed_active_windows)
    fragility = _fragility_penalty(matrix, config=config)
    structure_quality = (
        0.25 * readiness
        + 0.25 * stability
        + 0.20 * compression
        + 0.15 * residual_activity
        + 0.15 * (1.0 - fragility)
    )
    quality = null_lift * structure_quality - 0.30 * saturation - 0.10 * fragility

    return AutoTuneScore(
        config=config,
        quality=float(quality),
        readiness_score=readiness,
        null_lift_score=null_lift,
        stability_score=stability,
        compression_score=compression,
        residual_activity_score=float(residual_activity),
        saturation_penalty=saturation,
        fragility_penalty=fragility,
        null_summary=null_summary,
    )


def tune_observation(
    series: Mapping[str, NDArray[np.floating]],
    configs: Sequence[AutoTuneConfig] | None = None,
    *,
    sample_rate: float = 1.0,
    null_repeats: int = 100,
    seed: int = 20260603,
) -> AutoTuneResult:
    """Rank candidate configs by label-free structure quality."""

    if not series:
        raise ValueError("at least one series is required")
    lengths = [len(np.asarray(values)) for values in series.values()]
    common_length = min(lengths)
    candidates = list(configs) if configs is not None else default_autotune_configs(common_length)

    scores: list[AutoTuneScore] = []
    skipped: list[dict[str, object]] = []
    for index, config in enumerate(candidates):
        try:
            score = score_autotune_config(
                series,
                config,
                sample_rate=sample_rate,
                null_repeats=null_repeats,
                seed=seed + index,
            )
        except Exception as exc:  # noqa: BLE001 - skip invalid search points with reasons.
            skipped.append({"config": config.to_dict(), "reason": str(exc)})
            continue
        scores.append(score)

    scores.sort(key=lambda item: item.quality, reverse=True)
    return AutoTuneResult(scores=scores, skipped=skipped)
