"""Validate label-free autotune on bounded CDIP aligned buoy windows.

This script uses CDIP as a calibration domain, not as a target-label source. It
chooses observation settings from raw aligned buoy windows using the generic
autotune objective, then freezes the selected settings and evaluates the next
aligned windows as heldout validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.cdip_batch import (
    BatchWindow,
    _discover_windows,
    _series_from_records,
)
from experiments.cdip_observe import PREPROCESS_MODES, load_cdip_raw_record
from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneObservation,
    AutoTuneScore,
    build_autotune_observation,
    null_lift_score,
    observation_null_totals_for_config,
    observation_total_for_config,
    score_autotune_observation,
    summarize_observed_vs_null_totals,
)


@dataclass(frozen=True)
class CdipAutoTuneCandidate:
    """One CDIP candidate: preprocessing plus core autotune settings."""

    preprocess: str
    config: AutoTuneConfig

    def to_dict(self) -> dict[str, object]:
        row = {"preprocess": self.preprocess}
        row.update(self.config.to_dict())
        return row

    def matrix_key(self, *, preprocess: str | None = None) -> dict[str, object]:
        return {
            "preprocess": preprocess or self.preprocess,
            "baseline_size": self.config.baseline_size,
            "adaptive_window": self.config.adaptive_window,
            "stride": self.config.stride,
            "score": self.config.score,
        }


@dataclass
class MatrixCache:
    """In-memory and optional on-disk cache for expensive observation matrices."""

    directory: Path | None = None
    entries: dict[str, AutoTuneObservation] | None = None
    hits: int = 0
    misses: int = 0
    disk_hits: int = 0
    writes: int = 0

    def __post_init__(self) -> None:
        if self.entries is None:
            self.entries = {}

    def get(self, key: dict[str, object]) -> AutoTuneObservation | None:
        digest = _cache_digest(key)
        assert self.entries is not None
        if digest in self.entries:
            self.hits += 1
            return self.entries[digest]
        if self.directory is not None:
            path = self.directory / f"{digest}.npz"
            if path.exists():
                observation = _load_observation_npz(path)
                self.entries[digest] = observation
                self.hits += 1
                self.disk_hits += 1
                return observation
        self.misses += 1
        return None

    def put(self, key: dict[str, object], observation: AutoTuneObservation) -> None:
        digest = _cache_digest(key)
        assert self.entries is not None
        self.entries[digest] = observation
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{digest}.npz"
        if path.exists():
            return
        _write_observation_npz(path, observation)
        self.writes += 1

    def to_dict(self) -> dict[str, object]:
        assert self.entries is not None
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "disk_hits": self.disk_hits,
            "writes": self.writes,
            "directory": str(self.directory) if self.directory is not None else None,
        }


@dataclass(frozen=True)
class CdipWindowScore:
    """One window score plus the repeat-aligned null totals used for batch aggregation."""

    score: AutoTuneScore
    null_totals: list[float]


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _cache_digest(key: dict[str, object]) -> str:
    payload = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _write_observation_npz(path: Path, observation: AutoTuneObservation) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            anchors=np.asarray(observation.anchors, dtype=np.int64),
            matrix=observation.matrix,
            readiness_score=np.asarray([observation.readiness_score], dtype=np.float64),
            series_count=np.asarray([observation.series_count], dtype=np.int64),
        )
    tmp.replace(path)


def _load_observation_npz(path: Path) -> AutoTuneObservation:
    with np.load(path, allow_pickle=False) as payload:
        anchors = [int(value) for value in payload["anchors"]]
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        readiness_score = float(np.asarray(payload["readiness_score"]).reshape(-1)[0])
        series_count = int(np.asarray(payload["series_count"]).reshape(-1)[0])
    return AutoTuneObservation(
        anchors=anchors,
        matrix=matrix,
        readiness_score=readiness_score,
        series_count=series_count,
    )


def build_candidates(args: argparse.Namespace) -> list[CdipAutoTuneCandidate]:
    candidates = []
    for preprocess in _parse_str_list(args.preprocess_grid):
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"Unknown preprocess mode: {preprocess}")
        for baseline in _parse_int_list(args.baselines):
            for adaptive in _parse_int_list(args.adaptive_windows):
                for stride in _parse_int_list(args.strides):
                    for threshold in _parse_float_list(args.thresholds):
                        for decay in _parse_float_list(args.decays):
                            for min_active in _parse_int_list(args.min_active_series):
                                candidates.append(
                                    CdipAutoTuneCandidate(
                                        preprocess=preprocess,
                                        config=AutoTuneConfig(
                                            baseline_size=baseline,
                                            adaptive_window=adaptive,
                                            stride=stride,
                                            emission_threshold=threshold,
                                            decay=decay,
                                            min_active_series=min_active,
                                        ),
                                    )
                                )
    return candidates


def _window_identity(window: BatchWindow) -> dict[str, object]:
    return {
        "group": list(window.platforms),
        "paths": list(window.source_paths),
        "start": _iso(window.start_time),
        "end": _iso(window.start_time + (window.n_samples - 1) / window.target_rate),
        "sample_rate": window.target_rate,
        "samples": window.n_samples,
        "group_index": window.group_index,
        "window_index": window.window_index,
    }


def _window_cache_key(
    window: BatchWindow,
    candidate: CdipAutoTuneCandidate,
    *,
    preprocess: str,
    phase_surrogate_seed: int,
) -> dict[str, object]:
    return {
        "version": 1,
        "window": {
            "paths": list(window.source_paths),
            "start_time": window.start_time,
            "n_samples": window.n_samples,
            "target_rate": window.target_rate,
        },
        "candidate": candidate.matrix_key(preprocess=preprocess),
        "phase_surrogate_seed": phase_surrogate_seed,
    }


def _phase_surrogate_preprocess(preprocess: str) -> str:
    if preprocess == "none":
        return "phase-randomize"
    if preprocess == "highpass":
        return "highpass-phase-randomize"
    if preprocess == "dominant-mask":
        return "dominant-mask-phase-randomize"
    if preprocess == "highpass-dominant-mask":
        return "highpass-dominant-mask-phase-randomize"
    if preprocess in {
        "phase-randomize",
        "highpass-phase-randomize",
        "dominant-mask-phase-randomize",
        "highpass-dominant-mask-phase-randomize",
    }:
        return preprocess
    raise ValueError(f"Unknown preprocess mode: {preprocess}")


def _observation_for_window(
    window: BatchWindow,
    candidate: CdipAutoTuneCandidate,
    *,
    cache: MatrixCache,
    channels: Sequence[str],
    highpass_period_minutes: float,
    mask_dominant_bins: int,
    mask_bin_radius: int,
    phase_surrogate_seed: int,
    preprocess: str | None = None,
) -> AutoTuneObservation:
    selected_preprocess = preprocess or candidate.preprocess
    key = _window_cache_key(
        window,
        candidate,
        preprocess=selected_preprocess,
        phase_surrogate_seed=phase_surrogate_seed,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    series = _series_from_records(
        window.records,
        channels,
        start_time=window.start_time,
        n_samples=window.n_samples,
        target_rate=window.target_rate,
        preprocess=selected_preprocess,
        highpass_period_seconds=highpass_period_minutes * 60.0,
        mask_dominant_bins=mask_dominant_bins,
        mask_bin_radius=mask_bin_radius,
        phase_surrogate_seed=phase_surrogate_seed,
    )
    values = {item.name: item.values for item in series}
    observation = build_autotune_observation(
        values,
        candidate.config,
        sample_rate=window.target_rate,
    )
    cache.put(key, observation)
    return observation


def _phase_surrogate_null_totals(
    window: BatchWindow,
    candidate: CdipAutoTuneCandidate,
    *,
    cache: MatrixCache,
    channels: Sequence[str],
    args: argparse.Namespace,
    seed: int,
) -> list[float]:
    preprocess = _phase_surrogate_preprocess(candidate.preprocess)
    totals = []
    for repeat in range(args.null_repeats):
        observation = _observation_for_window(
            window,
            candidate,
            cache=cache,
            channels=channels,
            highpass_period_minutes=args.highpass_period_minutes,
            mask_dominant_bins=args.mask_dominant_bins,
            mask_bin_radius=args.mask_bin_radius,
            phase_surrogate_seed=seed + repeat,
            preprocess=preprocess,
        )
        totals.append(
            observation_total_for_config(
                observation,
                candidate.config,
                total_kind="emission",
            )
        )
    return totals


def _score_window(
    window: BatchWindow,
    candidate: CdipAutoTuneCandidate,
    *,
    cache: MatrixCache,
    channels: Sequence[str],
    args: argparse.Namespace,
    null_mode: str,
    seed: int,
) -> CdipWindowScore:
    observation = _observation_for_window(
        window,
        candidate,
        cache=cache,
        channels=channels,
        highpass_period_minutes=args.highpass_period_minutes,
        mask_dominant_bins=args.mask_dominant_bins,
        mask_bin_radius=args.mask_bin_radius,
        phase_surrogate_seed=args.phase_surrogate_seed,
    )
    if null_mode == "phase-surrogate":
        null_totals = _phase_surrogate_null_totals(
            window,
            candidate,
            cache=cache,
            channels=channels,
            args=args,
            seed=args.phase_surrogate_seed + seed * 1000,
        )
        score = score_autotune_observation(
            observation,
            candidate.config,
            null_totals=null_totals,
            total_kind="emission",
        )
        return CdipWindowScore(score=score, null_totals=null_totals)

    null_totals = observation_null_totals_for_config(
        observation,
        candidate.config,
        null_repeats=args.null_repeats,
        seed=seed,
        null_mode=null_mode,  # type: ignore[arg-type]
        null_block_size=args.null_block_size,
        total_kind="emission",
    )
    score = score_autotune_observation(
        observation,
        candidate.config,
        null_totals=null_totals,
        total_kind="emission",
    )
    return CdipWindowScore(score=score, null_totals=null_totals)


def _aggregate_scores(
    candidate: CdipAutoTuneCandidate,
    scores: Sequence[AutoTuneScore],
    *,
    skipped: Sequence[dict[str, object]],
    min_accepted_fraction: float,
    min_positive_window_fraction: float = 0.5,
    min_z_effect: float,
    null_totals_by_window: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    if not scores:
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "windows_scored": 0,
            "windows_skipped": len(skipped),
            "quality": float("-inf"),
            "skipped": list(skipped)[:5],
        }

    observed = float(sum(score.null_summary.observed_total for score in scores))
    if null_totals_by_window is not None:
        null_matrix = np.asarray(null_totals_by_window, dtype=np.float64)
        if null_matrix.ndim != 2:
            raise ValueError("null_totals_by_window must be a 2D collection")
        if null_matrix.shape[0] != len(scores):
            raise ValueError("null total window count must match score count")
        if null_matrix.shape[1] < 1:
            raise ValueError("at least one null repeat is required")
        aggregate_null_totals = np.sum(null_matrix, axis=0)
        aggregate_summary = summarize_observed_vs_null_totals(
            anchors=sum(score.null_summary.anchors for score in scores),
            observed_total=observed,
            observed_active_windows=sum(
                score.null_summary.observed_active_windows for score in scores
            ),
            null_totals=aggregate_null_totals,
        )
        null_mean = aggregate_summary.null_mean
        z_effect = aggregate_summary.z_effect
        null_total_repeats = aggregate_summary.null_repeats
        null_total_exceedances = aggregate_summary.null_exceedances
        null_total_p_ge_observed = aggregate_summary.empirical_p_ge_observed
        null_total_empirical_p_floor = aggregate_summary.empirical_p_floor
        null_total_unique_repeats = aggregate_summary.unique_null_totals
        null_total_std = aggregate_summary.null_std
        aggregate_lift = null_lift_score(aggregate_summary)
    else:
        null_mean = float(sum(score.null_summary.null_mean for score in scores))
        variance = float(sum(score.null_summary.null_std**2 for score in scores))
        z_effect = (observed - null_mean) / math.sqrt(variance) if variance > 0 else None
        null_total_std = math.sqrt(variance)
        null_total_repeats = int(sum(score.null_summary.null_repeats for score in scores))
        null_total_exceedances = int(
            sum(score.null_summary.null_exceedances for score in scores)
        )
        null_total_p_ge_observed = None
        null_total_empirical_p_floor = None
        null_total_unique_repeats = None
        aggregate_lift = None
    accepted_windows = sum(1 for score in scores if score.accepted)
    accepted_fraction = accepted_windows / len(scores)
    positive_windows = sum(1 for score in scores if score.null_summary.observed_minus_null > 0.0)
    positive_window_fraction = positive_windows / len(scores)
    mean_quality = float(np.mean([score.quality for score in scores]))
    mean_saturation = float(np.mean([score.saturation_penalty for score in scores]))
    mean_lift = float(np.mean([score.null_lift_score for score in scores]))
    mean_readiness = float(np.mean([score.readiness_score for score in scores]))
    effective_lift = aggregate_lift if aggregate_lift is not None else mean_lift
    quality = mean_quality + 0.20 * effective_lift + 0.05 * positive_window_fraction
    aggregate_z = z_effect if z_effect is not None else 0.0
    accepted = (
        quality > 0.0
        and observed > null_mean
        and accepted_fraction >= min_accepted_fraction
        and positive_window_fraction >= min_positive_window_fraction
        and aggregate_z >= min_z_effect
        and mean_saturation < 1.0
        and effective_lift > 0.0
    )

    return {
        "candidate": candidate.to_dict(),
        "accepted": accepted,
        "windows_scored": len(scores),
        "windows_skipped": len(skipped),
        "accepted_windows": accepted_windows,
        "accepted_fraction": accepted_fraction,
        "positive_windows": positive_windows,
        "positive_window_fraction": positive_window_fraction,
        "quality": quality,
        "mean_quality": mean_quality,
        "mean_readiness": mean_readiness,
        "mean_null_lift": mean_lift,
        "aggregate_null_lift": aggregate_lift,
        "mean_saturation": mean_saturation,
        "mean_stability": float(np.mean([score.stability_score for score in scores])),
        "mean_fragility": float(np.mean([score.fragility_penalty for score in scores])),
        "observed_total": observed,
        "null_mean_total": null_mean,
        "null_std_total": null_total_std,
        "observed_minus_null_total": observed - null_mean,
        "z_effect": z_effect,
        "min_accepted_fraction": min_accepted_fraction,
        "min_positive_window_fraction": min_positive_window_fraction,
        "min_z_effect": min_z_effect,
        "null_repeats_per_window": scores[0].null_summary.null_repeats,
        "null_exceedance_sum": int(sum(score.null_summary.null_exceedances for score in scores)),
        "null_repeat_sum": int(sum(score.null_summary.null_repeats for score in scores)),
        "null_total_repeats": null_total_repeats,
        "null_total_exceedances": null_total_exceedances,
        "null_total_empirical_p_ge_observed": null_total_p_ge_observed,
        "null_total_empirical_p_floor": null_total_empirical_p_floor,
        "null_total_unique_repeats": null_total_unique_repeats,
        "window_scores": [
            {
                "quality": score.quality,
                "accepted": score.accepted,
                "observed": score.null_summary.observed_total,
                "null_mean": score.null_summary.null_mean,
                "z_effect": score.null_summary.z_effect,
                "exceedances": score.null_summary.null_exceedances,
                "repeats": score.null_summary.null_repeats,
                "saturation": score.saturation_penalty,
                "null_lift": score.null_lift_score,
            }
            for score in scores
        ],
        "skipped": list(skipped)[:5],
    }


def _score_candidate_windows_for_null_mode(
    windows: Sequence[BatchWindow],
    candidate: CdipAutoTuneCandidate,
    *,
    cache: MatrixCache,
    channels: Sequence[str],
    args: argparse.Namespace,
    seed_offset: int,
    null_mode: str,
) -> dict[str, Any]:
    window_scores: list[CdipWindowScore] = []
    skipped: list[dict[str, object]] = []
    started = time.time()
    last_progress = 0.0
    for index, window in enumerate(windows):
        try:
            window_scores.append(
                _score_window(
                    window,
                    candidate,
                    cache=cache,
                    channels=channels,
                    args=args,
                    null_mode=null_mode,
                    seed=args.seed + seed_offset + index,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report invalid candidate/window pairs.
            skipped.append(
                {
                    "window": _window_identity(window),
                    "reason": str(exc),
                }
            )
        if args.progress and args.progress_every > 0:
            now = time.time()
            if now - last_progress >= args.progress_every:
                elapsed = max(now - started, 1e-9)
                processed = index + 1
                rate = processed / elapsed
                remaining = max(len(windows) - processed, 0)
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    (
                        "cdip_autotune progress preprocess=%s baseline=%d "
                        "adaptive=%d stride=%d threshold=%.3f null_mode=%s "
                        "windows=%d/%d skipped=%d elapsed=%.1fs eta=%.1fs "
                        "cache_hits=%d cache_misses=%d"
                    )
                    % (
                        candidate.preprocess,
                        candidate.config.baseline_size,
                        candidate.config.adaptive_window,
                        candidate.config.stride,
                        candidate.config.emission_threshold,
                        null_mode,
                        processed,
                        len(windows),
                        len(skipped),
                        elapsed,
                        eta,
                        cache.hits,
                        cache.misses,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now
    scores = [item.score for item in window_scores]
    report = _aggregate_scores(
        candidate,
        scores,
        skipped=skipped,
        min_accepted_fraction=args.min_accepted_fraction,
        min_positive_window_fraction=args.min_positive_window_fraction,
        min_z_effect=args.min_z_effect,
        null_totals_by_window=[item.null_totals for item in window_scores],
    )
    report["null_mode"] = null_mode
    return report


def _combine_null_mode_reports(
    candidate: CdipAutoTuneCandidate,
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not reports:
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "quality": float("-inf"),
            "null_mode_reports": [],
        }

    def _number(row: dict[str, Any], key: str, default: float) -> float:
        value = row.get(key, default)
        return float(value) if value is not None else default

    quality = min(_number(row, "quality", float("-inf")) for row in reports)
    delta = min(_number(row, "observed_minus_null_total", float("-inf")) for row in reports)
    z_values = [_number(row, "z_effect", float("-inf")) for row in reports]
    accepted = all(bool(row.get("accepted", False)) for row in reports)
    worst = min(
        reports,
        key=lambda row: (
            bool(row.get("accepted", False)),
            _number(row, "quality", float("-inf")),
            _number(row, "observed_minus_null_total", float("-inf")),
        ),
    )
    combined = {
        "candidate": candidate.to_dict(),
        "accepted": accepted,
        "windows_scored": min(int(row.get("windows_scored", 0)) for row in reports),
        "windows_skipped": sum(int(row.get("windows_skipped", 0)) for row in reports),
        "accepted_windows": min(int(row.get("accepted_windows", 0)) for row in reports),
        "accepted_fraction": min(_number(row, "accepted_fraction", 0.0) for row in reports),
        "positive_windows": min(int(row.get("positive_windows", 0)) for row in reports),
        "positive_window_fraction": min(
            _number(row, "positive_window_fraction", 0.0) for row in reports
        ),
        "quality": quality,
        "mean_quality": min(_number(row, "mean_quality", float("-inf")) for row in reports),
        "mean_readiness": min(_number(row, "mean_readiness", 0.0) for row in reports),
        "mean_null_lift": min(_number(row, "mean_null_lift", 0.0) for row in reports),
        "aggregate_null_lift": min(
            _number(row, "aggregate_null_lift", 0.0) for row in reports
        ),
        "mean_saturation": max(_number(row, "mean_saturation", 1.0) for row in reports),
        "mean_stability": min(_number(row, "mean_stability", 0.0) for row in reports),
        "mean_fragility": max(_number(row, "mean_fragility", 1.0) for row in reports),
        "observed_total": worst.get("observed_total", 0.0),
        "null_mean_total": worst.get("null_mean_total", 0.0),
        "null_std_total": worst.get("null_std_total", 0.0),
        "observed_minus_null_total": delta,
        "z_effect": min(z_values) if z_values else None,
        "null_total_repeats": worst.get("null_total_repeats"),
        "null_total_exceedances": worst.get("null_total_exceedances"),
        "null_total_empirical_p_ge_observed": worst.get(
            "null_total_empirical_p_ge_observed"
        ),
        "null_total_empirical_p_floor": worst.get("null_total_empirical_p_floor"),
        "null_total_unique_repeats": worst.get("null_total_unique_repeats"),
        "null_modes": [str(row.get("null_mode")) for row in reports],
        "worst_null_mode": worst.get("null_mode"),
        "null_mode_reports": list(reports),
        "skipped": [
            item
            for row in reports
            for item in row.get("skipped", [])
        ][:5],
    }
    return combined


def score_candidate_windows(
    windows: Sequence[BatchWindow],
    candidate: CdipAutoTuneCandidate,
    *,
    cache: MatrixCache,
    channels: Sequence[str],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    reports = [
        _score_candidate_windows_for_null_mode(
            windows,
            candidate,
            cache=cache,
            channels=channels,
            args=args,
            seed_offset=seed_offset + 100000 * mode_index,
            null_mode=null_mode,
        )
        for mode_index, null_mode in enumerate(_parse_str_list(args.null_modes))
    ]
    return _combine_null_mode_reports(candidate, reports)


def discover_cdip_windows(
    args: argparse.Namespace,
    *,
    channels: Sequence[str],
    window_offset_per_group: int,
    max_windows_per_group: int,
) -> list[BatchWindow]:
    records = _load_records(args, channels)
    min_clean_samples = _min_clean_samples(args)
    return _discover_windows(
        records,
        group_size=args.group_size,
        min_clean_samples=min_clean_samples,
        window_samples=args.window_samples,
        window_step=args.window_step,
        max_groups=args.max_groups,
        max_windows_per_group=max_windows_per_group,
        window_offset_per_group=window_offset_per_group,
        group_strategy=args.group_strategy,
    )


def _load_records(args: argparse.Namespace, channels: Sequence[str]) -> list[Any]:
    return [
        load_cdip_raw_record(path, channels, keep_flags=set(args.keep_flags))
        for path in args.files
    ]


def _min_clean_samples(args: argparse.Namespace) -> int:
    return args.min_clean_samples or (
        max(_parse_int_list(args.baselines))
        + max(_parse_int_list(args.adaptive_windows))
        + max(_parse_int_list(args.strides))
    )


def _take_group_windows(
    windows: Sequence[BatchWindow],
    *,
    group_start: int,
    group_count: int,
    window_offset: int,
    windows_per_group: int,
) -> list[BatchWindow]:
    selected: list[BatchWindow] = []
    taken_by_group: dict[int, int] = {}
    group_end = group_start + group_count
    for window in windows:
        if not group_start <= window.group_index < group_end:
            continue
        if window.window_index < window_offset:
            continue
        taken = taken_by_group.get(window.group_index, 0)
        if taken >= windows_per_group:
            continue
        selected.append(window)
        taken_by_group[window.group_index] = taken + 1
    return selected


def discover_cdip_split_windows(
    args: argparse.Namespace,
    *,
    channels: Sequence[str],
) -> tuple[list[BatchWindow], list[BatchWindow]]:
    if args.validation_split == "next-window":
        calibration_windows = discover_cdip_windows(
            args,
            channels=channels,
            window_offset_per_group=args.calibration_window_offset,
            max_windows_per_group=args.calibration_windows_per_group,
        )
        validation_windows = discover_cdip_windows(
            args,
            channels=channels,
            window_offset_per_group=args.validation_window_offset,
            max_windows_per_group=args.validation_windows_per_group,
        )
        return calibration_windows, validation_windows

    validation_groups = args.validation_groups or args.max_groups
    total_groups = args.max_groups + validation_groups
    max_window_count = max(
        args.calibration_window_offset + args.calibration_windows_per_group,
        args.validation_window_offset + args.validation_windows_per_group,
    )
    all_windows = _discover_windows(
        _load_records(args, channels),
        group_size=args.group_size,
        min_clean_samples=_min_clean_samples(args),
        window_samples=args.window_samples,
        window_step=args.window_step,
        max_groups=total_groups,
        max_windows_per_group=max_window_count,
        window_offset_per_group=0,
        group_strategy=args.group_strategy,
    )
    return (
        _take_group_windows(
            all_windows,
            group_start=0,
            group_count=args.max_groups,
            window_offset=args.calibration_window_offset,
            windows_per_group=args.calibration_windows_per_group,
        ),
        _take_group_windows(
            all_windows,
            group_start=args.max_groups,
            group_count=validation_groups,
            window_offset=args.validation_window_offset,
            windows_per_group=args.validation_windows_per_group,
        ),
    )


def run_cdip_autotune(args: argparse.Namespace) -> dict[str, Any]:
    channels = [channel.lower() for channel in args.channels]
    candidates = build_candidates(args)
    calibration_windows, validation_windows = discover_cdip_split_windows(args, channels=channels)
    cache = MatrixCache(directory=args.matrix_cache_dir)
    started = time.time()
    candidate_reports = []
    for index, candidate in enumerate(candidates):
        if args.progress:
            print(
                "cdip_autotune candidate %d/%d preprocess=%s config=%s"
                % (index + 1, len(candidates), candidate.preprocess, candidate.config.to_dict()),
                file=sys.stderr,
                flush=True,
            )
        candidate_reports.append(
            score_candidate_windows(
                calibration_windows,
                candidate,
                cache=cache,
                channels=channels,
                args=args,
                seed_offset=1000 * index,
            )
        )

    candidate_reports = sorted(
        candidate_reports,
        key=lambda row: (
            bool(row.get("accepted", False)),
            float(row.get("observed_minus_null_total", 0.0)) > 0.0,
            float(row.get("quality", float("-inf"))),
            float(row.get("observed_minus_null_total", float("-inf"))),
        ),
        reverse=True,
    )
    best_report = candidate_reports[0] if candidate_reports else None
    best_candidate = None
    validation_report = None
    if best_report is not None:
        candidate_row = best_report["candidate"]
        best_candidate = CdipAutoTuneCandidate(
            preprocess=str(candidate_row["preprocess"]),
            config=AutoTuneConfig(
                baseline_size=int(candidate_row["baseline_size"]),
                adaptive_window=int(candidate_row["adaptive_window"]),
                stride=int(candidate_row["stride"]),
                score=str(candidate_row["score"]),  # type: ignore[arg-type]
                emission_threshold=float(candidate_row["emission_threshold"]),
                decay=float(candidate_row["decay"]),
                min_active_series=int(candidate_row["min_active_series"]),
            ),
        )
        validation_report = score_candidate_windows(
            validation_windows,
            best_candidate,
            cache=cache,
            channels=channels,
            args=args,
            seed_offset=900000,
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "method": {
            "label_use": "none",
            "selection": "rank candidate settings by label-free aggregate autotune quality on calibration windows",
            "validation": "freeze selected candidate and score heldout windows",
        },
        "parameters": {
            "files": [str(path) for path in args.files],
            "channels": channels,
            "keep_flags": list(args.keep_flags),
            "group_size": args.group_size,
            "group_strategy": args.group_strategy,
            "max_groups": args.max_groups,
            "window_samples": args.window_samples,
            "window_step": args.window_step,
            "calibration_window_offset": args.calibration_window_offset,
            "calibration_windows_per_group": args.calibration_windows_per_group,
            "validation_window_offset": args.validation_window_offset,
            "validation_windows_per_group": args.validation_windows_per_group,
            "validation_split": args.validation_split,
            "validation_groups": args.validation_groups,
            "candidate_count": len(candidates),
            "null_repeats": args.null_repeats,
            "null_modes": _parse_str_list(args.null_modes),
            "null_block_size": args.null_block_size,
            "min_accepted_fraction": args.min_accepted_fraction,
            "min_positive_window_fraction": args.min_positive_window_fraction,
            "min_z_effect": args.min_z_effect,
            "preprocess_grid": _parse_str_list(args.preprocess_grid),
            "baselines": _parse_int_list(args.baselines),
            "adaptive_windows": _parse_int_list(args.adaptive_windows),
            "strides": _parse_int_list(args.strides),
            "thresholds": _parse_float_list(args.thresholds),
            "decays": _parse_float_list(args.decays),
            "min_active_series": _parse_int_list(args.min_active_series),
            "progress_every": args.progress_every,
        },
        "matrix_cache": cache.to_dict(),
        "calibration": {
            "windows": [_window_identity(window) for window in calibration_windows],
            "best": best_report,
            "candidates": candidate_reports[: args.top_candidates],
        },
        "validation": {
            "windows": [_window_identity(window) for window in validation_windows],
            "selected_candidate": best_candidate.to_dict() if best_candidate else None,
            "result": validation_report,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print("CDIP autotune calibration")
    print(
        "  files=%d candidates=%d calibration_windows=%d validation_windows=%d elapsed=%.1fs"
        % (
            len(report["parameters"]["files"]),
            report["parameters"]["candidate_count"],
            len(report["calibration"]["windows"]),
            len(report["validation"]["windows"]),
            report["elapsed_seconds"],
        )
    )
    cache = report.get("matrix_cache", {})
    print(
        "  split=%s null_modes=%s matrix_cache entries=%s hits=%s misses=%s disk_hits=%s writes=%s"
        % (
            report["parameters"].get("validation_split", "next-window"),
            ",".join(report["parameters"].get("null_modes", [])),
            cache.get("entries", 0),
            cache.get("hits", 0),
            cache.get("misses", 0),
            cache.get("disk_hits", 0),
            cache.get("writes", 0),
        )
    )
    best = report["calibration"]["best"]
    if not best:
        print("  no valid candidates")
        return
    candidate = best["candidate"]
    print(
        "  best accepted=%s quality=%.4f delta=%.3f z=%s windows=%d/%d positive=%d/%d"
        % (
            best["accepted"],
            best["quality"],
            best.get("observed_minus_null_total", 0.0),
            "None" if best.get("z_effect") is None else "%.2f" % best["z_effect"],
            best.get("accepted_windows", 0),
            best.get("windows_scored", 0),
            best.get("positive_windows", 0),
            best.get("windows_scored", 0),
        )
    )
    if best.get("null_total_repeats") is not None:
        print(
            "  best null repeats=%s exceedances=%s p_ge=%s p_floor=%s null_std=%.3f"
            % (
                best.get("null_total_repeats"),
                best.get("null_total_exceedances"),
                best.get("null_total_empirical_p_ge_observed"),
                best.get("null_total_empirical_p_floor"),
                best.get("null_std_total", 0.0),
            )
        )
    if best.get("worst_null_mode"):
        print("  best worst_null_mode=%s" % best["worst_null_mode"])
    print(
        "  config preprocess=%s baseline=%d adaptive=%d stride=%d threshold=%.3f decay=%.3f min_active=%d"
        % (
            candidate["preprocess"],
            candidate["baseline_size"],
            candidate["adaptive_window"],
            candidate["stride"],
            candidate["emission_threshold"],
            candidate["decay"],
            candidate["min_active_series"],
        )
    )
    validation = report["validation"]["result"]
    if validation:
        print(
            "  validation accepted=%s quality=%.4f delta=%.3f z=%s windows=%d/%d positive=%d/%d"
            % (
                validation["accepted"],
                validation["quality"],
                validation.get("observed_minus_null_total", 0.0),
                "None" if validation.get("z_effect") is None else "%.2f" % validation["z_effect"],
                validation.get("accepted_windows", 0),
                validation.get("windows_scored", 0),
                validation.get("positive_windows", 0),
                validation.get("windows_scored", 0),
            )
        )
        if validation.get("null_total_repeats") is not None:
            print(
                "  validation null repeats=%s exceedances=%s p_ge=%s p_floor=%s null_std=%.3f"
                % (
                    validation.get("null_total_repeats"),
                    validation.get("null_total_exceedances"),
                    validation.get("null_total_empirical_p_ge_observed"),
                    validation.get("null_total_empirical_p_floor"),
                    validation.get("null_std_total", 0.0),
                )
            )
        if validation.get("worst_null_mode"):
            print("  validation worst_null_mode=%s" % validation["worst_null_mode"])
    print("  top candidates")
    for row in report["calibration"]["candidates"]:
        cfg = row["candidate"]
        print(
            "    accepted=%s quality=%.4f delta=%.3f preprocess=%s baseline=%d adaptive=%d threshold=%.3f saturation=%.3f"
            % (
                row.get("accepted"),
                row.get("quality", 0.0),
                row.get("observed_minus_null_total", 0.0),
                cfg["preprocess"],
                cfg["baseline_size"],
                cfg["adaptive_window"],
                cfg["emission_threshold"],
                row.get("mean_saturation", 0.0),
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="CDIP *_xy.nc files")
    parser.add_argument("--channels", nargs="+", default=["z"])
    parser.add_argument("--keep-flags", type=int, nargs="+", default=[2])
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--group-strategy", choices=["first", "balanced"], default="balanced")
    parser.add_argument("--max-groups", type=int, default=6)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--window-step", type=int, default=2048)
    parser.add_argument("--min-clean-samples", type=int, default=None)
    parser.add_argument("--calibration-window-offset", type=int, default=0)
    parser.add_argument("--calibration-windows-per-group", type=int, default=1)
    parser.add_argument("--validation-window-offset", type=int, default=1)
    parser.add_argument("--validation-windows-per-group", type=int, default=1)
    parser.add_argument(
        "--validation-split",
        choices=["next-window", "disjoint-groups"],
        default="next-window",
        help="Heldout strategy: next windows from calibration groups or separate groups",
    )
    parser.add_argument(
        "--validation-groups",
        type=int,
        default=None,
        help="Validation group count for --validation-split disjoint-groups",
    )

    parser.add_argument(
        "--preprocess-grid",
        default="highpass,highpass-dominant-mask",
        help="Comma-separated preprocess modes",
    )
    parser.add_argument("--baselines", default="768,1280")
    parser.add_argument("--adaptive-windows", default="384")
    parser.add_argument("--strides", default="256")
    parser.add_argument("--thresholds", default="2.5,3.0")
    parser.add_argument("--decays", default="0.9")
    parser.add_argument("--min-active-series", default="2")
    parser.add_argument("--highpass-period-minutes", type=float, default=30.0)
    parser.add_argument("--mask-dominant-bins", type=int, default=3)
    parser.add_argument("--mask-bin-radius", type=int, default=1)
    parser.add_argument("--phase-surrogate-seed", type=int, default=20260602)
    parser.add_argument("--null-repeats", type=int, default=12)
    parser.add_argument(
        "--null-modes",
        default="permute",
        help=(
            "Comma-separated null families: permute, shift, block-permute, "
            "phase-surrogate"
        ),
    )
    parser.add_argument(
        "--null-block-size",
        type=int,
        default=8,
        help="Anchor block size for block-permute null mode",
    )
    parser.add_argument(
        "--min-accepted-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of scored windows that must individually pass the "
            "single-window gate; default 0 keeps the CDIP gate batch-first"
        ),
    )
    parser.add_argument(
        "--min-positive-window-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of scored windows with observed total above local null mean",
    )
    parser.add_argument(
        "--min-z-effect",
        type=float,
        default=1.5,
        help="Minimum aggregate z effect for a candidate to be accepted",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument(
        "--matrix-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for persistent cached observer matrices",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=float,
        default=30.0,
        help="Seconds between per-candidate window progress lines when --progress is set",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.max_groups < 1:
        parser.error("--max-groups must be >= 1")
    if args.calibration_windows_per_group < 1 or args.validation_windows_per_group < 1:
        parser.error("window counts per group must be >= 1")
    if args.calibration_window_offset < 0 or args.validation_window_offset < 0:
        parser.error("window offsets must be >= 0")
    if args.null_repeats < 1:
        parser.error("--null-repeats must be >= 1")
    if args.null_block_size < 1:
        parser.error("--null-block-size must be >= 1")
    if args.validation_groups is not None and args.validation_groups < 1:
        parser.error("--validation-groups must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if not 0.0 <= args.min_accepted_fraction <= 1.0:
        parser.error("--min-accepted-fraction must be in [0, 1]")
    if not 0.0 <= args.min_positive_window_fraction <= 1.0:
        parser.error("--min-positive-window-fraction must be in [0, 1]")
    valid_null_modes = {"permute", "shift", "block-permute", "phase-surrogate"}
    for null_mode in _parse_str_list(args.null_modes):
        if null_mode not in valid_null_modes:
            parser.error(f"unknown --null-modes entry: {null_mode}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_cdip_autotune(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
