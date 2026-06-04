"""Batch CDIP multi-buoy observation with shifted null controls.

This script runs the existing agnostic observer over many aligned buoy groups
and windows. It does not add wave-domain features to detection. Research-style
metrics remain post-hoc audit outputs. Use fixed presets for scale runs; do not
tune parameters against local positive windows.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.cdip_observe import (
    CdipRawRecord,
    CdipSeries,
    PREPROCESS_MODES,
    _combination_rows,
    _intersect_time_spans,
    _pearson,
    _raw_valid_time_spans,
    _stable_seed,
    _spearman,
    load_cdip_raw_record,
    posthoc_metrics,
    preprocess_series_values,
)
from spectral_forecast.observation import (
    ObservationPoint,
    ObservationResult,
    ScoreName,
    build_stigmergy,
    observe_series,
)


@dataclass(frozen=True)
class BatchWindow:
    """One aligned group/window to observe."""

    records: tuple[CdipRawRecord, ...]
    start_time: float
    n_samples: int
    target_rate: float
    group_index: int
    window_index: int

    @property
    def platforms(self) -> tuple[str, ...]:
        return tuple(record.platform_id for record in self.records)

    @property
    def source_paths(self) -> tuple[str, ...]:
        return tuple(str(record.path) for record in self.records)


@dataclass(frozen=True)
class _ComboCandidate:
    records: tuple[CdipRawRecord, ...]
    platforms: tuple[str, ...]
    digest: str


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _series_from_records(
    records: Sequence[CdipRawRecord],
    channels: Sequence[str],
    *,
    start_time: float,
    n_samples: int,
    target_rate: float,
    preprocess: str = "none",
    highpass_period_seconds: float = 30.0 * 60.0,
    mask_dominant_bins: int = 3,
    mask_bin_radius: int = 1,
    phase_surrogate_seed: int = 20260602,
) -> list[CdipSeries]:
    grid = start_time + np.arange(n_samples, dtype=np.float64) / target_rate
    series: list[CdipSeries] = []
    for record in records:
        source_index = (grid - record.first_sample_time) * record.sample_rate
        source_x = np.arange(len(record.valid), dtype=np.float64)
        for channel in channels:
            values = record.arrays[channel]
            segment = np.interp(source_index, source_x, values).astype(np.float64)
            seed = _stable_seed(
                phase_surrogate_seed,
                f"{record.platform_id}:{channel}:{start_time:.6f}:{n_samples}",
            )
            segment = preprocess_series_values(
                segment,
                sample_rate=target_rate,
                mode=preprocess,
                highpass_period_seconds=highpass_period_seconds,
                mask_dominant_bins=mask_dominant_bins,
                mask_bin_radius=mask_bin_radius,
                phase_seed=seed,
            )
            series.append(
                CdipSeries(
                    name=f"{record.platform_id}:{channel}",
                    channel=channel,
                    values=segment,
                    sample_rate=target_rate,
                    start_time=start_time,
                    filter_delay=0.0,
                    span_start=0,
                    span_end=n_samples,
                    station_id=record.station_id,
                    platform_id=record.platform_id,
                    platform_name=record.platform_name,
                    source_path=str(record.path),
                )
            )
    return series


def _discover_windows(
    records: Sequence[CdipRawRecord],
    *,
    group_size: int,
    min_clean_samples: int,
    window_samples: int,
    window_step: int,
    max_groups: int,
    max_windows_per_group: int,
    window_offset_per_group: int = 0,
    group_strategy: str = "first",
    progress_label: str | None = None,
    progress_every: float = 0.0,
) -> list[BatchWindow]:
    windows: list[BatchWindow] = []
    if window_offset_per_group < 0:
        raise ValueError("window_offset_per_group must be >= 0")
    group_index = 0
    required_samples = max(min_clean_samples, window_samples)
    usage: Counter[str] = Counter()
    combos = [
        _ComboCandidate(
            records=tuple(combo),
            platforms=tuple(record.platform_id for record in combo),
            digest=hashlib.sha1(
                ",".join(record.platform_id for record in combo).encode("utf-8")
            ).hexdigest(),
        )
        for combo in itertools.combinations(records, group_size)
    ]
    heap: list[tuple[tuple[int, int, str], int, _ComboCandidate]] = []
    started_at = time.time()
    last_progress = 0.0

    def combo_rank(combo: _ComboCandidate) -> tuple[int, int, str]:
        return (
            sum(usage[platform] for platform in combo.platforms),
            max((usage[platform] for platform in combo.platforms), default=0),
            combo.digest,
        )

    if group_strategy == "balanced":
        heap = [(combo_rank(combo), index, combo) for index, combo in enumerate(combos)]
        heapq.heapify(heap)

    while group_index < max_groups:
        if group_strategy == "balanced":
            if not heap:
                break
            while True:
                rank, index, combo_candidate = heapq.heappop(heap)
                current_rank = combo_rank(combo_candidate)
                if current_rank == rank:
                    break
                heapq.heappush(heap, (current_rank, index, combo_candidate))
                if not heap:
                    break
        else:
            if not combos:
                break
            combo_candidate = combos.pop(0)
        combo = combo_candidate.records

        target_rate = min(record.sample_rate for record in combo)
        min_duration = (required_samples - 1) / target_rate
        span_lists = [
            _raw_valid_time_spans(record, min_duration=min_duration)
            for record in combo
        ]
        intervals = _intersect_time_spans(span_lists, min_duration=min_duration)
        if not intervals:
            continue

        start_time, end_time = max(intervals, key=lambda item: item[1] - item[0])
        available = int(np.floor((end_time - start_time) * target_rate)) + 1
        if available < required_samples:
            continue

        selected_windows = 0
        for candidate_index, offset in enumerate(
            range(0, available - window_samples + 1, window_step)
        ):
            if candidate_index < window_offset_per_group:
                continue
            windows.append(
                BatchWindow(
                    records=tuple(combo),
                    start_time=start_time + offset / target_rate,
                    n_samples=window_samples,
                    target_rate=target_rate,
                    group_index=group_index,
                    window_index=candidate_index,
                )
            )
            selected_windows += 1
            if selected_windows >= max_windows_per_group:
                break

        if selected_windows:
            if group_strategy == "balanced":
                usage.update(combo_candidate.platforms)
            group_index += 1
            if progress_every > 0.0:
                now = time.time()
                if now - last_progress >= progress_every:
                    elapsed = max(now - started_at, 1e-9)
                    rate = group_index / elapsed if group_index else 0.0
                    eta = (max_groups - group_index) / rate if rate else 0.0
                    print(
                        "cdip_discover progress label=%s groups=%d/%d "
                        "windows=%d remaining_combos=%d elapsed=%.1fs eta=%.1fs"
                        % (
                            progress_label or "windows",
                            group_index,
                            max_groups,
                            len(windows),
                            len(heap) if group_strategy == "balanced" else len(combos),
                            elapsed,
                            eta,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = now

    return windows


def _select_shard_items(items: Sequence[Any], shard_count: int, shard_index: int) -> list[Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if shard_count == 1:
        return list(items)
    return [item for index, item in enumerate(items) if index % shard_count == shard_index]


def _window_manifest_signature(
    args: argparse.Namespace,
    channels: Sequence[str],
    min_clean_samples: int,
) -> dict[str, Any]:
    return {
        "files": [str(path) for path in args.files],
        "channels": list(channels),
        "keep_flags": list(args.keep_flags),
        "group_size": args.group_size,
        "group_strategy": args.group_strategy,
        "max_groups": args.max_groups,
        "max_windows_per_group": args.max_windows_per_group,
        "window_offset_per_group": getattr(args, "window_offset_per_group", 0),
        "max_total_windows": args.max_total_windows,
        "window_samples": args.window_samples,
        "window_step": args.window_step,
        "min_clean_samples": min_clean_samples,
    }


def _window_to_manifest_row(window: BatchWindow) -> dict[str, Any]:
    return {
        "source_paths": list(window.source_paths),
        "start_time": window.start_time,
        "n_samples": window.n_samples,
        "target_rate": window.target_rate,
        "group_index": window.group_index,
        "window_index": window.window_index,
    }


def _window_from_manifest_row(
    row: dict[str, Any],
    records_by_path: dict[str, CdipRawRecord],
) -> BatchWindow:
    records = []
    missing = []
    for source_path in row["source_paths"]:
        record = records_by_path.get(source_path)
        if record is None:
            missing.append(source_path)
        else:
            records.append(record)
    if missing:
        raise ValueError("Window manifest references missing source paths: " + ", ".join(missing))
    return BatchWindow(
        records=tuple(records),
        start_time=float(row["start_time"]),
        n_samples=int(row["n_samples"]),
        target_rate=float(row["target_rate"]),
        group_index=int(row["group_index"]),
        window_index=int(row["window_index"]),
    )


def _write_window_manifest(
    path: Path,
    *,
    signature: dict[str, Any],
    windows: Sequence[BatchWindow],
) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "windows": [_window_to_manifest_row(window) for window in windows],
    }
    _write_checkpoint(path, payload)


def _normalized_window_manifest_signature(signature: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(signature)
    normalized.setdefault("window_offset_per_group", 0)
    return normalized


def _load_window_manifest(
    path: Path,
    *,
    signature: dict[str, Any],
    records_by_path: dict[str, CdipRawRecord],
) -> list[BatchWindow]:
    payload = json.loads(path.read_text())
    if payload.get("version") != 1:
        raise ValueError("Unsupported window manifest version")
    if _normalized_window_manifest_signature(
        payload.get("signature", {})
    ) != _normalized_window_manifest_signature(signature):
        raise ValueError("Window manifest signature does not match current run arguments")
    return [
        _window_from_manifest_row(row, records_by_path)
        for row in payload.get("windows", [])
    ]


def _shift_observation_result(
    result: ObservationResult,
    *,
    stride: int,
    shift_steps: int,
) -> ObservationResult:
    if not result.points:
        return result
    start = min(point.index for point in result.points)
    span = max(point.index for point in result.points) - start + stride
    shift = (shift_steps * stride) % span
    if shift == 0:
        shift = stride % span
    points = [
        replace(point, index=start + ((point.index - start + shift) % span))
        for point in result.points
    ]
    return replace(result, points=points)


def _permute_observation_result(
    result: ObservationResult,
    *,
    repeat: int,
) -> ObservationResult:
    if len(result.points) <= 1:
        return result
    indexes = [point.index for point in result.points]
    order = sorted(
        range(len(indexes)),
        key=lambda index: hashlib.sha1(
            f"{result.series}:{repeat}:{index}".encode("utf-8")
        ).hexdigest(),
    )
    permuted_indexes = [indexes[index] for index in order]
    if permuted_indexes == indexes:
        shift = 1 + (repeat % (len(indexes) - 1))
        permuted_indexes = indexes[shift:] + indexes[:shift]
    points = [
        replace(point, index=permuted_index)
        for point, permuted_index in zip(result.points, permuted_indexes)
    ]
    return replace(result, points=points)


def _null_stigmergy_summaries(
    results: Sequence[ObservationResult],
    *,
    score: ScoreName,
    emission_threshold: float,
    decay: float,
    stride: int,
    repeats: int,
    mode: str = "shift",
) -> list[float]:
    null_sums = []
    for repeat in range(repeats):
        shifted = []
        for idx, result in enumerate(results):
            if idx == 0:
                shifted.append(result)
            elif mode == "shift":
                shifted.append(
                    _shift_observation_result(
                        result,
                        stride=stride,
                        shift_steps=repeat + idx,
                    )
                )
            elif mode == "permute":
                shifted.append(
                    _permute_observation_result(
                        result,
                        repeat=repeat + idx,
                    )
                )
            else:
                raise ValueError(f"Unknown null mode: {mode}")
        stig = build_stigmergy(
            shifted,
            score=score,
            emission_threshold=emission_threshold,
            decay=decay,
        )
        combos = _combination_rows(stig.points)
        null_sums.append(_multi_platform_emission_sum(combos))
    return null_sums


def _multi_platform_emission_sum(combos: Sequence[dict[str, Any]]) -> float:
    total = 0.0
    for row in combos:
        platforms = [item for item in str(row["active_platforms"]).split(",") if item]
        if len(platforms) >= 2:
            total += float(row["emission_sum"])
    return total


def _metric_records(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    window: int,
) -> list[dict[str, float]]:
    rows = []
    for result in results:
        meta = series_by_name[result.series]
        for point in result.points:
            start = max(0, point.index - window + 1)
            end = min(len(meta.values), point.index + 1)
            metrics = posthoc_metrics(meta.values[start:end], meta.sample_rate).to_dict()
            row = {"score": point.score(score)}
            row.update(metrics)
            rows.append(row)
    return rows


def _correlation_rows(metric_rows: Sequence[dict[str, float]]) -> list[dict[str, Any]]:
    if not metric_rows:
        return []
    scores = [row["score"] for row in metric_rows]
    metrics = [key for key in metric_rows[0] if key != "score"]
    out = []
    for metric in metrics:
        values = [row[metric] for row in metric_rows]
        out.append(
            {
                "metric": metric,
                "n": len(values),
                "pearson": _pearson(scores, values),
                "spearman": _spearman(scores, values),
            }
        )
    return sorted(out, key=lambda row: abs(row["spearman"]), reverse=True)


def _normal_survival_from_z(z_score: float) -> float:
    """One-sided normal survival probability for a z score."""
    if not math.isfinite(z_score):
        return 1.0
    return float(0.5 * math.erfc(z_score / math.sqrt(2.0)))


def _empty_null_window_stats() -> dict[str, float]:
    return {"count": 0.0, "sum": 0.0, "sumsq": 0.0}


def _add_null_window_values(stats: dict[str, float], values: Sequence[float]) -> None:
    if not values:
        return
    array = np.asarray(values, dtype=np.float64)
    stats["count"] = float(stats.get("count", 0.0) + len(array))
    stats["sum"] = float(stats.get("sum", 0.0) + np.sum(array))
    stats["sumsq"] = float(stats.get("sumsq", 0.0) + np.sum(array * array))


def _merge_null_window_stats(
    target: dict[str, float],
    source: dict[str, float],
) -> None:
    target["count"] = float(target.get("count", 0.0) + source.get("count", 0.0))
    target["sum"] = float(target.get("sum", 0.0) + source.get("sum", 0.0))
    target["sumsq"] = float(target.get("sumsq", 0.0) + source.get("sumsq", 0.0))


def _null_window_std_from_stats(stats: dict[str, float]) -> float:
    count = float(stats.get("count", 0.0))
    if count <= 0:
        return 0.0
    mean = float(stats.get("sum", 0.0)) / count
    variance = float(stats.get("sumsq", 0.0)) / count - mean * mean
    return float(math.sqrt(max(variance, 0.0)))


def _null_window_stats_from_values(values: Sequence[float]) -> dict[str, float]:
    stats = _empty_null_window_stats()
    _add_null_window_values(stats, values)
    return stats


def _summary_from_state(
    *,
    records_loaded: int,
    windows_discovered: int,
    all_windows_discovered: int,
    window_rows: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    observed_multi: Sequence[float],
    null_totals: Sequence[float] | np.ndarray,
    null_multi: Sequence[float] | None = None,
    null_window_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    observed_total = float(np.sum(observed_multi)) if observed_multi else 0.0
    null_totals_array = np.asarray(null_totals, dtype=np.float64)
    has_total_null = bool(observed_multi) and len(null_totals_array) > 0
    null_mean_total = float(np.mean(null_totals_array)) if has_total_null else 0.0
    null_total_std = float(np.std(null_totals_array)) if has_total_null else 0.0
    unique_null_totals = (
        np.unique(np.round(null_totals_array, decimals=12)) if has_total_null else np.array([])
    )
    if null_window_stats is None:
        null_window_stats = _null_window_stats_from_values(null_multi or [])
    null_window_std = _null_window_std_from_stats(null_window_stats)
    observed_total_z = (
        (observed_total - null_mean_total) / null_total_std
        if null_total_std > 0
        else 0.0
    )
    null_total_repeats = int(len(null_totals_array)) if has_total_null else 0
    null_total_exceedances = (
        int(np.sum(null_totals_array >= observed_total)) if has_total_null else 0
    )
    null_total_unique_repeats = int(len(unique_null_totals)) if has_total_null else 0
    null_total_unique_exceedances = (
        int(np.sum(unique_null_totals >= observed_total)) if has_total_null else 0
    )
    null_total_empirical_p_floor = 1.0 / (null_total_repeats + 1) if has_total_null else 1.0
    null_total_unique_empirical_p_floor = (
        1.0 / (null_total_unique_repeats + 1) if has_total_null else 1.0
    )
    null_total_p_ge_observed = (
        float((1 + null_total_exceedances) / (null_total_repeats + 1))
        if has_total_null
        else 1.0
    )
    null_total_unique_p_ge_observed = (
        float((1 + null_total_unique_exceedances) / (null_total_unique_repeats + 1))
        if has_total_null
        else 1.0
    )
    observed_total_z_normal_p_one_sided = _normal_survival_from_z(observed_total_z)
    return {
        "records_loaded": records_loaded,
        "groups_discovered": len({tuple(row["group"]) for row in window_rows}),
        "all_windows_discovered": all_windows_discovered,
        "windows_discovered": windows_discovered,
        "windows_run": len(window_rows),
        "windows_skipped": len(skipped),
        "observed_multi_emission_total": observed_total,
        "null_multi_emission_total_estimate": null_mean_total,
        "observed_minus_null_total": observed_total - null_mean_total,
        "null_multi_emission_total_std": null_total_std,
        "observed_total_z": observed_total_z,
        "null_total_p_ge_observed": null_total_p_ge_observed,
        "null_total_empirical_p_ge_observed": null_total_p_ge_observed,
        "null_total_empirical_p_floor": null_total_empirical_p_floor,
        "null_total_exceedances": null_total_exceedances,
        "null_total_repeats": null_total_repeats,
        "null_total_unique_exceedances": null_total_unique_exceedances,
        "null_total_unique_repeats": null_total_unique_repeats,
        "null_total_unique_empirical_p_floor": null_total_unique_empirical_p_floor,
        "null_total_unique_p_ge_observed": null_total_unique_p_ge_observed,
        "observed_total_z_normal_p_one_sided": observed_total_z_normal_p_one_sided,
        "null_window_emission_std": null_window_std,
    }


def _checkpoint_signature(args: argparse.Namespace, channels: Sequence[str]) -> dict[str, Any]:
    return {
        "files": [str(path) for path in args.files],
        "channels": list(channels),
        "keep_flags": list(args.keep_flags),
        "group_size": args.group_size,
        "group_strategy": args.group_strategy,
        "max_groups": args.max_groups,
        "max_windows_per_group": args.max_windows_per_group,
        "window_offset_per_group": getattr(args, "window_offset_per_group", 0),
        "max_total_windows": args.max_total_windows,
        "baseline": args.baseline,
        "adaptive_window": args.adaptive_window,
        "stride": args.stride,
        "window_samples": args.window_samples,
        "window_step": args.window_step,
        "min_clean_samples": args.min_clean_samples,
        "score": args.score,
        "emission_threshold": args.emission_threshold,
        "decay": args.decay,
        "null_repeats": args.null_repeats,
        "null_mode": args.null_mode,
        "preprocess": getattr(args, "preprocess", "none"),
        "highpass_period_minutes": getattr(args, "highpass_period_minutes", 30.0),
        "mask_dominant_bins": getattr(args, "mask_dominant_bins", 3),
        "mask_bin_radius": getattr(args, "mask_bin_radius", 1),
        "phase_surrogate_seed": getattr(args, "phase_surrogate_seed", 20260602),
        "posthoc_window": args.posthoc_window or args.adaptive_window,
        "shard_count": getattr(args, "shard_count", 1),
        "shard_index": getattr(args, "shard_index", 0),
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _load_checkpoint(path: Path, signature: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("signature") != signature:
        raise ValueError("Checkpoint signature does not match current run arguments")
    return payload


def _progress_line(
    *,
    processed: int,
    total: int,
    started_at: float,
    window_rows: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
) -> str:
    elapsed = max(time.time() - started_at, 1e-9)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta = remaining / rate if rate > 0 else 0.0
    observed = float(sum(float(row["observed_multi_emission"]) for row in window_rows))
    return (
        "cdip_batch progress processed=%d/%d windows_run=%d skipped=%d "
        "elapsed=%.1fs rate=%.3f_windows_s eta=%.1fs observed_multi=%.3f"
        % (processed, total, len(window_rows), len(skipped), elapsed, rate, eta, observed)
    )


def _merge_combinations(
    aggregate: dict[str, dict[str, Any]],
    combos: Sequence[dict[str, Any]],
    *,
    group: Sequence[str],
    window_start: str,
    global_window_index: int,
) -> None:
    for row in combos:
        platforms = [item for item in str(row["active_platforms"]).split(",") if item]
        if len(platforms) < 2:
            continue
        key = row["active_platforms"]
        target = aggregate.setdefault(
            key,
            {
                "active_platforms": key,
                "count": 0,
                "emission_sum": 0.0,
                "max_pheromone": 0.0,
                "max_score": 0.0,
                "examples": [],
            },
        )
        target["count"] += int(row["count"])
        target["emission_sum"] += float(row["emission_sum"])
        target["max_pheromone"] = max(target["max_pheromone"], float(row["max_pheromone"]))
        target["max_score"] = max(target["max_score"], float(row["max_score"]))
        if len(target["examples"]) < 5:
            target["examples"].append(
                {
                    "global_window_index": global_window_index,
                    "group": ",".join(group),
                    "window_start": window_start,
                    "series": row["active_series"],
                    "first_index": row["first_index"],
                    "last_index": row["last_index"],
                }
            )


def _merge_aggregate_combinations(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["active_platforms"])
        target = aggregate.setdefault(
            key,
            {
                "active_platforms": key,
                "count": 0,
                "emission_sum": 0.0,
                "max_pheromone": 0.0,
                "max_score": 0.0,
                "examples": [],
            },
        )
        target["count"] += int(row.get("count", 0))
        target["emission_sum"] += float(row.get("emission_sum", 0.0))
        target["max_pheromone"] = max(
            target["max_pheromone"],
            float(row.get("max_pheromone", 0.0)),
        )
        target["max_score"] = max(target["max_score"], float(row.get("max_score", 0.0)))
        target["examples"].extend(row.get("examples", []))

    for row in aggregate.values():
        row["examples"] = sorted(
            row["examples"],
            key=lambda example: (
                example.get("global_window_index", math.inf),
                example.get("window_start", ""),
                example.get("series", ""),
            ),
        )[:5]

    return sorted(
        aggregate.values(),
        key=lambda row: (row["emission_sum"], row["max_pheromone"]),
        reverse=True,
    )


def _merge_state(
    *,
    combo_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, float]],
    window_rows: Sequence[dict[str, Any]],
    observed_multi: Sequence[float],
    null_window_stats: dict[str, float],
    null_totals: Sequence[float] | np.ndarray,
    skipped: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "aggregate_combinations": list(combo_rows),
        "metric_rows": list(metric_rows),
        "window_rows": list(window_rows),
        "observed_multi": [float(value) for value in observed_multi],
        "null_window_stats": {
            "count": float(null_window_stats.get("count", 0.0)),
            "sum": float(null_window_stats.get("sum", 0.0)),
            "sumsq": float(null_window_stats.get("sumsq", 0.0)),
        },
        "null_totals": [float(value) for value in null_totals],
        "skipped": list(skipped),
    }


def _batch_parameters(
    args: argparse.Namespace,
    channels: Sequence[str],
    min_clean_samples: int,
) -> dict[str, Any]:
    return {
        "preset": args.preset,
        "files": [str(path) for path in args.files],
        "channels": list(channels),
        "keep_flags": list(args.keep_flags),
        "group_size": args.group_size,
        "group_strategy": args.group_strategy,
        "max_groups": args.max_groups,
        "max_windows_per_group": args.max_windows_per_group,
        "window_offset_per_group": args.window_offset_per_group,
        "max_total_windows": args.max_total_windows,
        "baseline": args.baseline,
        "adaptive_window": args.adaptive_window,
        "stride": args.stride,
        "window_samples": args.window_samples,
        "window_step": args.window_step,
        "min_clean_samples": min_clean_samples,
        "score": args.score,
        "emission_threshold": args.emission_threshold,
        "decay": args.decay,
        "null_repeats": args.null_repeats,
        "null_mode": args.null_mode,
        "preprocess": args.preprocess,
        "highpass_period_minutes": args.highpass_period_minutes,
        "mask_dominant_bins": args.mask_dominant_bins,
        "mask_bin_radius": args.mask_bin_radius,
        "phase_surrogate_seed": args.phase_surrogate_seed,
        "posthoc_window": args.posthoc_window or args.adaptive_window,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    channels = [channel.lower() for channel in args.channels]
    signature = _checkpoint_signature(args, channels)
    records = [
        load_cdip_raw_record(path, channels, keep_flags=set(args.keep_flags))
        for path in args.files
    ]
    min_clean_samples = args.min_clean_samples or args.baseline + args.adaptive_window + args.stride
    window_manifest_signature = _window_manifest_signature(args, channels, min_clean_samples)
    if args.read_window_manifest is not None:
        records_by_path = {str(record.path): record for record in records}
        windows = _load_window_manifest(
            args.read_window_manifest,
            signature=window_manifest_signature,
            records_by_path=records_by_path,
        )
    else:
        windows = _discover_windows(
            records,
            group_size=args.group_size,
            min_clean_samples=min_clean_samples,
            window_samples=args.window_samples,
            window_step=args.window_step,
            max_groups=args.max_groups,
            max_windows_per_group=args.max_windows_per_group,
            window_offset_per_group=args.window_offset_per_group,
            group_strategy=args.group_strategy,
            progress_label="batch",
            progress_every=args.progress_every,
        )
        if args.max_total_windows > 0:
            windows = windows[: args.max_total_windows]
        if args.write_window_manifest is not None:
            _write_window_manifest(
                args.write_window_manifest,
                signature=window_manifest_signature,
                windows=windows,
            )
    if args.manifest_only:
        return {
            "parameters": _batch_parameters(args, channels, min_clean_samples),
            "summary": {
                "records_loaded": len(records),
                "groups_discovered": len({window.platforms for window in windows}),
                "all_windows_discovered": len(windows),
                "windows_discovered": len(windows),
                "windows_run": 0,
                "windows_skipped": 0,
            },
            "manifest": {
                "read": str(args.read_window_manifest) if args.read_window_manifest else None,
                "written": str(args.write_window_manifest) if args.write_window_manifest else None,
                "signature": window_manifest_signature,
            },
        }
    indexed_windows = list(enumerate(windows))
    all_windows_discovered = len(indexed_windows)
    indexed_windows = _select_shard_items(
        indexed_windows,
        args.shard_count,
        args.shard_index,
    )
    windows = [window for _, window in indexed_windows]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    aggregate_combos: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, float]] = []
    window_rows = []
    observed_multi = []
    null_window_stats = _empty_null_window_stats()
    null_totals = np.zeros(args.null_repeats, dtype=np.float64)
    skipped = []
    start_index = 0

    if args.resume:
        if checkpoint_path is None:
            raise ValueError("--resume requires --checkpoint")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = _load_checkpoint(checkpoint_path, signature)
        aggregate_combos = dict(checkpoint.get("aggregate_combos", {}))
        metric_rows = list(checkpoint.get("metric_rows", []))
        window_rows = list(checkpoint.get("window_rows", []))
        observed_multi = list(checkpoint.get("observed_multi", []))
        if "null_window_stats" in checkpoint:
            null_window_stats = dict(checkpoint["null_window_stats"])
        else:
            null_window_stats = _null_window_stats_from_values(checkpoint.get("null_multi", []))
        null_totals = np.asarray(checkpoint.get("null_totals", []), dtype=np.float64)
        if len(null_totals) != args.null_repeats:
            raise ValueError("Checkpoint null_totals length does not match null_repeats")
        skipped = list(checkpoint.get("skipped", []))
        start_index = int(checkpoint.get("next_window_index", 0))

    def checkpoint_payload(next_window_index: int, *, complete: bool = False) -> dict[str, Any]:
        return {
            "version": 1,
            "signature": signature,
            "complete": complete,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "next_window_index": next_window_index,
            "total_windows": len(windows),
            "aggregate_combos": aggregate_combos,
            "metric_rows": metric_rows,
            "window_rows": window_rows,
            "observed_multi": observed_multi,
            "null_window_stats": null_window_stats,
            "null_totals": null_totals.tolist(),
            "skipped": skipped,
        }

    started_at = time.time()
    last_progress = 0.0
    last_checkpoint = 0.0

    def maybe_report(processed: int, *, force: bool = False) -> None:
        nonlocal last_progress
        if args.progress_every <= 0:
            return
        now = time.time()
        if force or now - last_progress >= args.progress_every:
            print(
                _progress_line(
                    processed=processed,
                    total=len(windows),
                    started_at=started_at,
                    window_rows=window_rows,
                    skipped=skipped,
                ),
                file=sys.stderr,
                flush=True,
            )
            last_progress = now

    def maybe_checkpoint(next_window_index: int, *, force: bool = False, complete: bool = False) -> None:
        nonlocal last_checkpoint
        if checkpoint_path is None:
            return
        now = time.time()
        if force or complete or args.checkpoint_every <= 0 or now - last_checkpoint >= args.checkpoint_every:
            _write_checkpoint(
                checkpoint_path,
                checkpoint_payload(next_window_index, complete=complete),
            )
            last_checkpoint = now

    maybe_report(start_index, force=True)
    for window_index, window in enumerate(windows[start_index:], start=start_index):
        global_window_index = indexed_windows[window_index][0]
        series = _series_from_records(
            window.records,
            channels,
            start_time=window.start_time,
            n_samples=window.n_samples,
            target_rate=window.target_rate,
            preprocess=args.preprocess,
            highpass_period_seconds=args.highpass_period_minutes * 60.0,
            mask_dominant_bins=args.mask_dominant_bins,
            mask_bin_radius=args.mask_bin_radius,
            phase_surrogate_seed=args.phase_surrogate_seed,
        )
        series_by_name = {item.name: item for item in series}
        try:
            results = [
                observe_series(
                    item.values,
                    series=item.name,
                    baseline_size=args.baseline,
                    adaptive_window=args.adaptive_window,
                    stride=args.stride,
                    sample_rate=item.sample_rate,
                )
                for item in series
            ]
        except ValueError as exc:
            skipped.append(
                {
                    "platforms": list(window.platforms),
                    "start": _iso(window.start_time),
                    "global_window_index": global_window_index,
                    "reason": str(exc),
                }
            )
            processed = window_index + 1
            maybe_checkpoint(processed)
            maybe_report(processed)
            continue

        stig = build_stigmergy(
            results,
            score=args.score,
            emission_threshold=args.emission_threshold,
            decay=args.decay,
        )
        combos = _combination_rows(stig.points)
        multi_sum = _multi_platform_emission_sum(combos)
        null_sums = _null_stigmergy_summaries(
            results,
            score=args.score,
            emission_threshold=args.emission_threshold,
            decay=args.decay,
            stride=args.stride,
            repeats=args.null_repeats,
            mode=args.null_mode,
        )
        observed_multi.append(multi_sum)
        _add_null_window_values(null_window_stats, null_sums)
        null_totals[: len(null_sums)] += null_sums
        metric_rows.extend(
            _metric_records(
                results,
                series_by_name,
                score=args.score,
                window=args.posthoc_window or args.adaptive_window,
            )
        )
        _merge_combinations(
            aggregate_combos,
            combos,
            group=window.platforms,
            window_start=_iso(window.start_time),
            global_window_index=global_window_index,
        )

        null_mean = float(np.mean(null_sums)) if null_sums else 0.0
        null_std = float(np.std(null_sums)) if null_sums else 0.0
        window_rows.append(
            {
                "global_window_index": global_window_index,
                "group": list(window.platforms),
                "paths": list(window.source_paths),
                "start": _iso(window.start_time),
                "end": _iso(window.start_time + (window.n_samples - 1) / window.target_rate),
                "sample_rate": window.target_rate,
                "samples": window.n_samples,
                "observed_multi_emission": multi_sum,
                "null_multi_emission_mean": null_mean,
                "null_multi_emission_std": null_std,
                "observed_minus_null": multi_sum - null_mean,
                "top_combinations": combos[: args.top],
            }
        )
        processed = window_index + 1
        maybe_checkpoint(processed)
        maybe_report(processed)

    maybe_checkpoint(len(windows), force=True, complete=True)
    maybe_report(len(windows), force=True)

    combo_rows = sorted(
        aggregate_combos.values(),
        key=lambda row: (row["emission_sum"], row["max_pheromone"]),
        reverse=True,
    )

    report = {
        "parameters": _batch_parameters(args, channels, min_clean_samples),
        "summary": _summary_from_state(
            records_loaded=len(records),
            windows_discovered=len(windows),
            all_windows_discovered=all_windows_discovered,
            window_rows=window_rows,
            skipped=skipped,
            observed_multi=observed_multi,
            null_totals=null_totals,
            null_window_stats=null_window_stats,
        ),
        "top_combinations": combo_rows[: args.top],
        "correlations": _correlation_rows(metric_rows),
        "windows": sorted(
            window_rows,
            key=lambda row: row["observed_minus_null"],
            reverse=True,
        )[: args.top_windows],
        "skipped": skipped[:20],
    }
    if args.include_merge_state or args.shard_count > 1:
        report["merge_state"] = _merge_state(
            combo_rows=combo_rows,
            metric_rows=metric_rows,
            window_rows=window_rows,
            observed_multi=observed_multi,
            null_window_stats=null_window_stats,
            null_totals=null_totals,
            skipped=skipped,
        )
    return report


def merge_reports(
    paths: Sequence[Path],
    *,
    top: int,
    top_windows: int,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one report is required for merge")

    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        if "merge_state" not in report:
            raise ValueError(f"Report is missing merge_state and cannot be merged: {path}")
        reports.append(report)

    first_params = dict(reports[0]["parameters"])
    base_params = {key: value for key, value in first_params.items() if key != "shard_index"}
    shard_indexes = []
    for path, report in zip(paths, reports):
        params = dict(report["parameters"])
        comparable = {key: value for key, value in params.items() if key != "shard_index"}
        if comparable != base_params:
            raise ValueError(f"Report parameters are not merge-compatible: {path}")
        shard_indexes.append(int(params.get("shard_index", 0)))

    shard_count = int(first_params.get("shard_count", len(paths)))
    if len(set(shard_indexes)) != len(shard_indexes):
        raise ValueError("Shard reports contain duplicate shard_index values")
    if shard_count > 1 and sorted(shard_indexes) != list(range(shard_count)):
        raise ValueError(
            "Shard reports must cover every shard index from 0 to shard_count - 1"
        )

    aggregate_input: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    window_rows: list[dict[str, Any]] = []
    observed_multi: list[float] = []
    null_window_stats = _empty_null_window_stats()
    null_totals: np.ndarray | None = None
    skipped: list[dict[str, Any]] = []

    for report in reports:
        state = report["merge_state"]
        aggregate_input.extend(state.get("aggregate_combinations", []))
        metric_rows.extend(state.get("metric_rows", []))
        window_rows.extend(state.get("window_rows", []))
        observed_multi.extend(float(value) for value in state.get("observed_multi", []))
        if "null_window_stats" in state:
            _merge_null_window_stats(null_window_stats, state["null_window_stats"])
        else:
            _merge_null_window_stats(
                null_window_stats,
                _null_window_stats_from_values(state.get("null_multi", [])),
            )
        shard_null_totals = np.asarray(state.get("null_totals", []), dtype=np.float64)
        if null_totals is None:
            null_totals = shard_null_totals
        elif len(null_totals) != len(shard_null_totals):
            raise ValueError("Shard null_totals lengths do not match")
        else:
            null_totals = null_totals + shard_null_totals
        skipped.extend(state.get("skipped", []))

    combo_rows = _merge_aggregate_combinations(aggregate_input)
    summary_windows = [int(report["summary"]["windows_discovered"]) for report in reports]
    windows_discovered = int(sum(summary_windows))
    all_windows_values = {
        int(report["summary"].get("all_windows_discovered", windows_discovered))
        for report in reports
    }
    all_windows_discovered = (
        next(iter(all_windows_values)) if len(all_windows_values) == 1 else windows_discovered
    )
    if len(all_windows_values) == 1 and windows_discovered != all_windows_discovered:
        raise ValueError("Merged shard window counts do not match all_windows_discovered")

    parameters = dict(first_params)
    parameters["shard_index"] = "merged"
    parameters["merged_reports"] = len(paths)

    return {
        "parameters": parameters,
        "summary": _summary_from_state(
            records_loaded=int(reports[0]["summary"]["records_loaded"]),
            windows_discovered=windows_discovered,
            all_windows_discovered=all_windows_discovered,
            window_rows=window_rows,
            skipped=skipped,
            observed_multi=observed_multi,
            null_totals=[] if null_totals is None else null_totals,
            null_window_stats=null_window_stats,
        ),
        "top_combinations": combo_rows[:top],
        "correlations": _correlation_rows(metric_rows),
        "windows": sorted(
            window_rows,
            key=lambda row: row["observed_minus_null"],
            reverse=True,
        )[:top_windows],
        "skipped": skipped[:20],
    }


def _print_text_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("CDIP batch observation")
    print(
        "  records=%d groups=%d windows=%d skipped=%d"
        % (
            summary["records_loaded"],
            summary["groups_discovered"],
            summary["windows_run"],
            summary["windows_skipped"],
        )
    )
    print(
        "  observed_multi=%.3f null_estimate=%.3f delta=%.3f null_total_std=%.3f z=%.3f p_emp_ge=%.6f p_floor=%.6f p_norm=%.3e"
        % (
            summary["observed_multi_emission_total"],
            summary["null_multi_emission_total_estimate"],
            summary["observed_minus_null_total"],
            summary["null_multi_emission_total_std"],
            summary["observed_total_z"],
            summary["null_total_empirical_p_ge_observed"],
            summary["null_total_empirical_p_floor"],
            summary["observed_total_z_normal_p_one_sided"],
        )
    )
    print("  detector_features=agnostic posthoc_metrics=audit_only null=time-shifted-emission-index")
    print(
        "  preset=%s group_strategy=%s tuning=disabled_for_scale_runs"
        % (report["parameters"]["preset"], report["parameters"]["group_strategy"])
    )

    print("\nTop combinations")
    print("%24s %6s %10s %10s %10s" % ("platforms", "count", "emission", "pheromone", "max_score"))
    for row in report["top_combinations"]:
        print(
            "%24s %6d %10.3f %10.3f %10.3f"
            % (
                row["active_platforms"],
                row["count"],
                row["emission_sum"],
                row["max_pheromone"],
                row["max_score"],
            )
        )

    print("\nTop windows vs null")
    print("%26s %24s %10s %10s %10s" % ("group", "start", "observed", "null", "delta"))
    for row in report["windows"]:
        print(
            "%26s %24s %10.3f %10.3f %10.3f"
            % (
                ",".join(row["group"]),
                row["start"].replace("+00:00", "Z"),
                row["observed_multi_emission"],
                row["null_multi_emission_mean"],
                row["observed_minus_null"],
            )
        )

    if report["correlations"]:
        print("\nPost-hoc correlations")
        print("%38s %5s %9s %9s" % ("metric", "n", "pearson", "spearman"))
        for row in report["correlations"][:8]:
            print(
                "%38s %5d %9.3f %9.3f"
                % (row["metric"], row["n"], row["pearson"], row["spearman"])
            )


def _print_manifest_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    manifest = report.get("manifest", {})
    print("CDIP window manifest")
    print(
        "  records=%d groups=%d windows=%d"
        % (
            summary["records_loaded"],
            summary["groups_discovered"],
            summary["windows_discovered"],
        )
    )
    if manifest.get("read"):
        print("  read=%s" % manifest["read"])
    if manifest.get("written"):
        print("  written=%s" % manifest["written"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="*", help="CDIP *_xy.nc files")
    parser.add_argument(
        "--merge-reports",
        type=Path,
        nargs="+",
        default=None,
        help="Merge shard JSON reports produced with merge_state",
    )
    parser.add_argument(
        "--preset",
        choices=["custom", "scale"],
        default="custom",
        help="Use fixed preregistered batch settings for broad expansion",
    )
    parser.add_argument("--channels", nargs="+", default=["z"], help="Channels to observe")
    parser.add_argument("--keep-flags", type=int, nargs="+", default=[2], help="CDIP primary flags to keep")
    parser.add_argument("--group-size", type=int, default=3, help="Buoys per aligned group")
    parser.add_argument(
        "--group-strategy",
        choices=["first", "balanced"],
        default="first",
        help="How to choose aligned buoy groups before applying max-groups",
    )
    parser.add_argument("--max-groups", type=int, default=8, help="Maximum aligned groups to run")
    parser.add_argument("--max-windows-per-group", type=int, default=3, help="Maximum windows per group")
    parser.add_argument(
        "--window-offset-per-group",
        type=int,
        default=0,
        help="Skip this many aligned windows within each group before selecting max-windows-per-group",
    )
    parser.add_argument("--max-total-windows", type=int, default=0, help="Hard cap on total windows; 0 means no cap")
    parser.add_argument("--baseline", type=int, default=1024, help="Frozen baseline samples")
    parser.add_argument("--adaptive-window", type=int, default=512, help="Adaptive window samples")
    parser.add_argument("--stride", type=int, default=512, help="Observation stride")
    parser.add_argument("--window-samples", type=int, default=4096, help="Aligned samples per batch window")
    parser.add_argument("--window-step", type=int, default=2048, help="Sample step between windows")
    parser.add_argument("--min-clean-samples", type=int, default=None, help="Minimum clean samples")
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default="none",
        help="Optional preprocessing applied before observation",
    )
    parser.add_argument(
        "--highpass-period-minutes",
        type=float,
        default=30.0,
        help="Remove periods longer than this when --preprocess=highpass",
    )
    parser.add_argument(
        "--mask-dominant-bins",
        type=int,
        default=3,
        help="Number of strongest non-DC Fourier bins to remove for dominant-mask modes",
    )
    parser.add_argument(
        "--mask-bin-radius",
        type=int,
        default=1,
        help="Neighbor radius around each selected dominant Fourier bin to remove",
    )
    parser.add_argument(
        "--phase-surrogate-seed",
        type=int,
        default=20260602,
        help="Base seed for deterministic phase-randomized preprocessing controls",
    )
    parser.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
        help="Score used for emission",
    )
    parser.add_argument("--emission-threshold", type=float, default=3.0, help="Emission threshold")
    parser.add_argument("--decay", type=float, default=0.9, help="Stigmergy decay")
    parser.add_argument("--null-repeats", type=int, default=5, help="Shifted null repeats per window")
    parser.add_argument(
        "--null-mode",
        choices=["shift", "permute"],
        default="shift",
        help="Null control: circular shift or deterministic within-series timing permutation",
    )
    parser.add_argument("--posthoc-window", type=int, default=None, help="Post-hoc metric window")
    parser.add_argument("--top", type=int, default=10, help="Rows to show")
    parser.add_argument("--top-windows", type=int, default=10, help="Windows to show")
    parser.add_argument("--shard-count", type=int, default=1, help="Total deterministic shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index to run, zero-based")
    parser.add_argument(
        "--include-merge-state",
        action="store_true",
        help="Include full state needed for exact report merging",
    )
    parser.add_argument(
        "--progress-every",
        type=float,
        default=30.0,
        help="Seconds between stderr progress heartbeats; 0 disables",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path for resumable checkpoint JSON")
    parser.add_argument(
        "--write-window-manifest",
        type=Path,
        default=None,
        help="Write discovered aligned windows before shard selection",
    )
    parser.add_argument(
        "--read-window-manifest",
        type=Path,
        default=None,
        help="Read a previously discovered aligned window manifest",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=60.0,
        help="Seconds between checkpoint writes when --checkpoint is set; 0 writes every window",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Discover or read aligned windows, optionally write a manifest, then exit",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args(argv)
    if args.preset == "scale":
        args.group_size = 3
        args.max_groups = max(args.max_groups, 32)
        args.max_windows_per_group = max(args.max_windows_per_group, 8)
        args.baseline = 1024
        args.adaptive_window = 512
        args.stride = 512
        args.window_samples = 4096
        args.window_step = 2048
        args.emission_threshold = 3.0
        args.decay = 0.9
        args.null_repeats = max(args.null_repeats, 10)
        args.group_strategy = "balanced"
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= shard_index < shard_count")
    if args.mask_dominant_bins < 0:
        parser.error("--mask-dominant-bins must be >= 0")
    if args.mask_bin_radius < 0:
        parser.error("--mask-bin-radius must be >= 0")
    if args.window_offset_per_group < 0:
        parser.error("--window-offset-per-group must be >= 0")
    if args.read_window_manifest is not None and args.write_window_manifest is not None:
        parser.error("--read-window-manifest and --write-window-manifest are mutually exclusive")
    if args.merge_reports is None and not args.files:
        parser.error("files are required unless --merge-reports is supplied")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.merge_reports is not None:
        report = merge_reports(args.merge_reports, top=args.top, top_windows=args.top_windows)
    else:
        report = run_batch(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.manifest_only:
        _print_manifest_report(report)
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
