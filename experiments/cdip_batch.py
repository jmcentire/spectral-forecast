"""Batch CDIP multi-buoy observation with shifted null controls.

This script runs the existing agnostic observer over many aligned buoy groups
and windows. It does not add wave-domain features to detection. Research-style
metrics remain post-hoc audit outputs. Use fixed presets for scale runs; do not
tune parameters against local positive windows.
"""

from __future__ import annotations

import argparse
import hashlib
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
    _combination_rows,
    _intersect_time_spans,
    _pearson,
    _raw_valid_time_spans,
    _spearman,
    load_cdip_raw_record,
    posthoc_metrics,
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


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _series_from_records(
    records: Sequence[CdipRawRecord],
    channels: Sequence[str],
    *,
    start_time: float,
    n_samples: int,
    target_rate: float,
) -> list[CdipSeries]:
    grid = start_time + np.arange(n_samples, dtype=np.float64) / target_rate
    series: list[CdipSeries] = []
    for record in records:
        source_index = (grid - record.first_sample_time) * record.sample_rate
        source_x = np.arange(len(record.valid), dtype=np.float64)
        for channel in channels:
            values = record.arrays[channel]
            segment = np.interp(source_index, source_x, values).astype(np.float64)
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
    group_strategy: str = "first",
) -> list[BatchWindow]:
    windows: list[BatchWindow] = []
    group_index = 0
    required_samples = max(min_clean_samples, window_samples)
    usage: Counter[str] = Counter()
    combos = list(itertools.combinations(records, group_size))

    def combo_rank(combo: Sequence[CdipRawRecord]) -> tuple[int, int, str]:
        platforms = tuple(record.platform_id for record in combo)
        digest = hashlib.sha1(",".join(platforms).encode("utf-8")).hexdigest()
        return (
            sum(usage[platform] for platform in platforms),
            max((usage[platform] for platform in platforms), default=0),
            digest,
        )

    while combos and group_index < max_groups:
        if group_strategy == "balanced":
            best_index = min(range(len(combos)), key=lambda index: combo_rank(combos[index]))
            combo = combos.pop(best_index)
        else:
            combo = combos.pop(0)

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

        group_windows = 0
        for offset in range(0, available - window_samples + 1, window_step):
            windows.append(
                BatchWindow(
                    records=tuple(combo),
                    start_time=start_time + offset / target_rate,
                    n_samples=window_samples,
                    target_rate=target_rate,
                    group_index=group_index,
                    window_index=group_windows,
                )
            )
            group_windows += 1
            if group_windows >= max_windows_per_group:
                break

        if group_windows:
            if group_strategy == "balanced":
                usage.update(record.platform_id for record in combo)
            group_index += 1

    return windows


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


def _null_stigmergy_summaries(
    results: Sequence[ObservationResult],
    *,
    score: ScoreName,
    emission_threshold: float,
    decay: float,
    stride: int,
    repeats: int,
) -> list[float]:
    null_sums = []
    for repeat in range(repeats):
        shifted = []
        for idx, result in enumerate(results):
            if idx == 0:
                shifted.append(result)
            else:
                shifted.append(
                    _shift_observation_result(
                        result,
                        stride=stride,
                        shift_steps=repeat + idx,
                    )
                )
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


def _checkpoint_signature(args: argparse.Namespace, channels: Sequence[str]) -> dict[str, Any]:
    return {
        "files": [str(path) for path in args.files],
        "channels": list(channels),
        "keep_flags": list(args.keep_flags),
        "group_size": args.group_size,
        "group_strategy": args.group_strategy,
        "max_groups": args.max_groups,
        "max_windows_per_group": args.max_windows_per_group,
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
        "posthoc_window": args.posthoc_window or args.adaptive_window,
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
                    "group": ",".join(group),
                    "window_start": window_start,
                    "series": row["active_series"],
                    "first_index": row["first_index"],
                    "last_index": row["last_index"],
                }
            )


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    channels = [channel.lower() for channel in args.channels]
    signature = _checkpoint_signature(args, channels)
    records = [
        load_cdip_raw_record(path, channels, keep_flags=set(args.keep_flags))
        for path in args.files
    ]
    min_clean_samples = args.min_clean_samples or args.baseline + args.adaptive_window + args.stride
    windows = _discover_windows(
        records,
        group_size=args.group_size,
        min_clean_samples=min_clean_samples,
        window_samples=args.window_samples,
        window_step=args.window_step,
        max_groups=args.max_groups,
        max_windows_per_group=args.max_windows_per_group,
        group_strategy=args.group_strategy,
    )
    if args.max_total_windows > 0:
        windows = windows[: args.max_total_windows]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    aggregate_combos: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, float]] = []
    window_rows = []
    observed_multi = []
    null_multi = []
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
        null_multi = list(checkpoint.get("null_multi", []))
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
            "null_multi": null_multi,
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
        series = _series_from_records(
            window.records,
            channels,
            start_time=window.start_time,
            n_samples=window.n_samples,
            target_rate=window.target_rate,
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
        )
        observed_multi.append(multi_sum)
        null_multi.extend(null_sums)
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
        )

        null_mean = float(np.mean(null_sums)) if null_sums else 0.0
        null_std = float(np.std(null_sums)) if null_sums else 0.0
        window_rows.append(
            {
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

    observed_total = float(np.sum(observed_multi)) if observed_multi else 0.0
    has_total_null = bool(observed_multi) and len(null_totals) > 0
    null_mean_total = float(np.mean(null_totals)) if has_total_null else 0.0
    null_total_std = float(np.std(null_totals)) if has_total_null else 0.0
    null_window_std = float(np.std(null_multi)) if null_multi else 0.0
    observed_total_z = (
        (observed_total - null_mean_total) / null_total_std
        if null_total_std > 0
        else 0.0
    )
    null_total_repeats = int(len(null_totals)) if has_total_null else 0
    null_total_exceedances = int(np.sum(null_totals >= observed_total)) if has_total_null else 0
    null_total_empirical_p_floor = 1.0 / (null_total_repeats + 1) if has_total_null else 1.0
    null_total_p_ge_observed = (
        float((1 + null_total_exceedances) / (null_total_repeats + 1))
        if has_total_null
        else 1.0
    )
    observed_total_z_normal_p_one_sided = _normal_survival_from_z(observed_total_z)
    combo_rows = sorted(
        aggregate_combos.values(),
        key=lambda row: (row["emission_sum"], row["max_pheromone"]),
        reverse=True,
    )

    return {
        "parameters": {
            "preset": args.preset,
            "files": [str(path) for path in args.files],
            "channels": channels,
            "group_size": args.group_size,
            "group_strategy": args.group_strategy,
            "baseline": args.baseline,
            "adaptive_window": args.adaptive_window,
            "stride": args.stride,
            "window_samples": args.window_samples,
            "window_step": args.window_step,
            "score": args.score,
            "emission_threshold": args.emission_threshold,
            "decay": args.decay,
            "null_repeats": args.null_repeats,
            "posthoc_window": args.posthoc_window or args.adaptive_window,
        },
        "summary": {
            "records_loaded": len(records),
            "groups_discovered": len({tuple(row["group"]) for row in window_rows}),
            "windows_discovered": len(windows),
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
            "observed_total_z_normal_p_one_sided": observed_total_z_normal_p_one_sided,
            "null_window_emission_std": null_window_std,
        },
        "top_combinations": combo_rows[: args.top],
        "correlations": _correlation_rows(metric_rows),
        "windows": sorted(
            window_rows,
            key=lambda row: row["observed_minus_null"],
            reverse=True,
        )[: args.top_windows],
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="CDIP *_xy.nc files")
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
    parser.add_argument("--max-total-windows", type=int, default=0, help="Hard cap on total windows; 0 means no cap")
    parser.add_argument("--baseline", type=int, default=1024, help="Frozen baseline samples")
    parser.add_argument("--adaptive-window", type=int, default=512, help="Adaptive window samples")
    parser.add_argument("--stride", type=int, default=512, help="Observation stride")
    parser.add_argument("--window-samples", type=int, default=4096, help="Aligned samples per batch window")
    parser.add_argument("--window-step", type=int, default=2048, help="Sample step between windows")
    parser.add_argument("--min-clean-samples", type=int, default=None, help="Minimum clean samples")
    parser.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
        help="Score used for emission",
    )
    parser.add_argument("--emission-threshold", type=float, default=3.0, help="Emission threshold")
    parser.add_argument("--decay", type=float, default=0.9, help="Stigmergy decay")
    parser.add_argument("--null-repeats", type=int, default=5, help="Shifted null repeats per window")
    parser.add_argument("--posthoc-window", type=int, default=None, help="Post-hoc metric window")
    parser.add_argument("--top", type=int, default=10, help="Rows to show")
    parser.add_argument("--top-windows", type=int, default=10, help="Windows to show")
    parser.add_argument(
        "--progress-every",
        type=float,
        default=30.0,
        help="Seconds between stderr progress heartbeats; 0 disables",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path for resumable checkpoint JSON")
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=60.0,
        help="Seconds between checkpoint writes when --checkpoint is set; 0 writes every window",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint")
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
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_batch(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
