"""Batch CDIP multi-buoy observation with shifted null controls.

This script runs the existing agnostic observer over many aligned buoy groups
and windows. It does not add wave-domain features to detection. Research-style
metrics remain post-hoc audit outputs. Use fixed presets for scale runs; do not
tune parameters against local positive windows.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
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
) -> list[BatchWindow]:
    windows: list[BatchWindow] = []
    group_index = 0
    required_samples = max(min_clean_samples, window_samples)

    for combo in itertools.combinations(records, group_size):
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
            group_index += 1
            if group_index >= max_groups:
                break

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
    )
    if args.max_total_windows > 0:
        windows = windows[: args.max_total_windows]

    aggregate_combos: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, float]] = []
    window_rows = []
    observed_multi = []
    null_multi = []
    skipped = []

    for window in windows:
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

    observed_total = float(np.sum(observed_multi)) if observed_multi else 0.0
    null_mean_total = float(np.mean(null_multi)) * len(observed_multi) if null_multi else 0.0
    null_std = float(np.std(null_multi)) if null_multi else 0.0
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
            "null_window_emission_std": null_std,
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
        "  observed_multi=%.3f null_estimate=%.3f delta=%.3f"
        % (
            summary["observed_multi_emission_total"],
            summary["null_multi_emission_total_estimate"],
            summary["observed_minus_null_total"],
        )
    )
    print("  detector_features=agnostic posthoc_metrics=audit_only null=time-shifted-emission-index")
    print("  preset=%s tuning=disabled_for_scale_runs" % report["parameters"]["preset"])

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
