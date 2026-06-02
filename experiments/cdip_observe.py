"""Run a bounded spectral+stigmergy observation on a raw CDIP displacement file.

This is experiment tooling. It avoids wave-domain labels and published
rogue-wave predictor features; the optional mesh receives only terms derived
from the spectral observer's own scores and decomposition state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray
from scipy.io import netcdf_file

from spectral_forecast.information import scan_information_readiness
from spectral_forecast.observation import (
    ObservationPoint,
    ObservationResult,
    ScoreName,
    build_stigmergy,
    observe_series,
)


CHANNEL_VARIABLES = {
    "x": "xyzXDisplacement",
    "y": "xyzYDisplacement",
    "z": "xyzZDisplacement",
}


@dataclass(frozen=True)
class CdipSeries:
    """One clean, aligned CDIP displacement segment."""

    name: str
    channel: str
    values: NDArray[np.float64]
    sample_rate: float
    start_time: float
    filter_delay: float
    span_start: int
    span_end: int
    station_id: str
    platform_id: str
    platform_name: str
    source_path: str

    def timestamp_for_index(self, index: int) -> datetime:
        seconds = self.start_time + (self.span_start + index) / self.sample_rate - self.filter_delay
        return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _decode_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar(variable: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(variable.data).reshape(()))
    except Exception:
        return default


def _contiguous_spans(mask: NDArray[np.bool_], min_length: int) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("Expected a 1D validity mask")
    if len(mask) == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    spans = list(zip(changes[0::2], changes[1::2]))
    return [(int(start), int(end)) for start, end in spans if end - start >= min_length]


def _select_span(
    spans: Sequence[tuple[int, int]],
    sample_limit: int | None,
    segment_offset: int,
) -> tuple[int, int]:
    if not spans:
        raise ValueError("No contiguous clean segment satisfies the requested minimum length")

    start, end = max(spans, key=lambda span: span[1] - span[0])
    if segment_offset:
        start += segment_offset
    if start >= end:
        raise ValueError("Segment offset moves past the selected clean segment")
    if sample_limit is not None:
        end = min(end, start + sample_limit)
    if start >= end:
        raise ValueError("Selected segment is empty")
    return int(start), int(end)


def load_cdip_series(
    path: str | Path,
    channels: Sequence[str],
    *,
    keep_flags: set[int],
    min_clean_samples: int,
    sample_limit: int | None,
    segment_offset: int = 0,
) -> list[CdipSeries]:
    """Load aligned clean displacement segments from a CDIP NetCDF file."""

    path = Path(path)
    normalized_channels = [channel.lower() for channel in channels]
    unknown = [channel for channel in normalized_channels if channel not in CHANNEL_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown channels: {', '.join(unknown)}")

    with netcdf_file(path, mmap=False) as nc:
        sample_rate = _scalar(nc.variables["xyzSampleRate"])
        start_time = _scalar(nc.variables["xyzStartTime"])
        filter_delay = _scalar(nc.variables["xyzFilterDelay"])
        flags = np.asarray(nc.variables["xyzFlagPrimary"].data, dtype=np.int16)

        arrays: dict[str, NDArray[np.float64]] = {}
        valid = np.isin(flags, list(keep_flags))
        for channel in normalized_channels:
            variable = CHANNEL_VARIABLES[channel]
            values = np.asarray(nc.variables[variable].data, dtype=np.float64)
            arrays[channel] = values
            valid &= np.isfinite(values)
            valid &= values > -999.0

        spans = _contiguous_spans(valid, min_clean_samples)
        span_start, span_end = _select_span(spans, sample_limit, segment_offset)
        if span_end - span_start < min_clean_samples:
            raise ValueError(
                "Selected segment is shorter than the requested minimum after limiting/offset"
            )

        station_id = _decode_attr(getattr(nc, "cdip_station_id", ""), "unknown")
        platform_id = _decode_attr(getattr(nc, "platform_id", station_id), station_id)
        platform_name = _decode_attr(getattr(nc, "platform_name", platform_id), platform_id)

    out: list[CdipSeries] = []
    for channel, values in arrays.items():
        segment = np.asarray(values[span_start:span_end], dtype=np.float64)
        out.append(
            CdipSeries(
                name=f"{platform_id}:{channel}",
                channel=channel,
                values=segment,
                sample_rate=sample_rate,
                start_time=start_time,
                filter_delay=filter_delay,
                span_start=span_start,
                span_end=span_end,
                station_id=station_id,
                platform_id=platform_id,
                platform_name=platform_name,
                source_path=str(path),
            )
        )
    return out


def _level(value: float) -> str:
    value = abs(float(value))
    if value >= 10.0:
        return "extreme"
    if value >= 5.0:
        return "high"
    if value >= 3.0:
        return "elevated"
    if value >= 1.0:
        return "present"
    return "low"


def _direction(value: float, deadband: float = 1.0) -> str:
    if value >= deadband:
        return "positive"
    if value <= -deadband:
        return "negative"
    return "neutral"


def _safe_token(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def observation_signal_content(point: ObservationPoint, meta: CdipSeries) -> str:
    """Build mesh-facing content solely from detector output and source identity."""

    terms = [
        "spectral-observation",
        f"station-{_safe_token(meta.platform_id)}",
        f"channel-{meta.channel}disp",
        f"series-{_safe_token(point.series)}",
        f"frozen-{_level(point.frozen_score)}",
        f"sliding-{_level(point.sliding_score)}",
        f"drift-{_direction(point.conditional_drift_score)}",
        f"drift-{_level(point.conditional_drift_score)}",
        f"state-{_level(point.state_drift_score)}",
        f"residual-{_direction(point.frozen_residual, deadband=0.0)}",
    ]
    if point.score("max") >= 10.0:
        terms.append("score-extreme")
    elif point.score("max") >= 5.0:
        terms.append("score-high")
    elif point.score("max") >= 3.0:
        terms.append("score-elevated")
    else:
        terms.append("score-low")
    return " ".join(terms)


def observation_metadata(point: ObservationPoint, meta: CdipSeries, score: ScoreName) -> dict[str, Any]:
    timestamp = meta.timestamp_for_index(point.index)
    return {
        "series": point.series,
        "channel": meta.channel,
        "station_id": meta.station_id,
        "platform_id": meta.platform_id,
        "index": point.index,
        "timestamp": timestamp.isoformat(),
        "score_name": score,
        "score": point.score(score),
        "frozen_score": point.frozen_score,
        "sliding_score": point.sliding_score,
        "conditional_drift_score": point.conditional_drift_score,
        "state_drift_score": point.state_drift_score,
        "frozen_residual": point.frozen_residual,
        "sliding_residual": point.sliding_residual,
    }


def _selected_points(
    results: Sequence[ObservationResult],
    *,
    score: ScoreName,
    threshold: float,
    max_signals: int,
) -> list[ObservationPoint]:
    points = [
        point
        for result in results
        for point in result.points
        if np.isfinite(point.score(score)) and point.score(score) > threshold
    ]
    if max_signals > 0 and len(points) > max_signals:
        points = sorted(points, key=lambda point: point.score(score), reverse=True)[:max_signals]
    return sorted(points, key=lambda point: (point.index, point.series))


async def run_stigmergy_mesh(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    threshold: float,
    max_signals: int,
    mesh_src: Path,
    max_workers: int,
) -> dict[str, Any]:
    """Route detector emissions through the sibling Stigmergy mesh."""

    sys.path.insert(0, str(mesh_src))
    from stigmergy.mesh.mesh import Mesh
    from stigmergy.pipeline.processor import AgentRegistry
    from stigmergy.primitives.signal import Signal

    selected = _selected_points(
        results,
        score=score,
        threshold=threshold,
        max_signals=max_signals,
    )

    mesh = Mesh(
        AgentRegistry(),
        dedup_enabled=False,
        max_workers=max_workers,
        worker_capacity=80,
        base_threshold=0.05,
        gap_threshold=0.04,
    )

    seed_count = max(2, len(series_by_name))
    previous = None
    seed_names = sorted(series_by_name) or ["spectral-observation"]
    while len(seed_names) < seed_count:
        seed_names.append(f"seed-{len(seed_names)}")
    for name in seed_names[:seed_count]:
        worker = mesh.spawn_worker(source_name=name, connect_to=previous)
        previous = worker.id

    traces = []
    for point in selected:
        meta = series_by_name[point.series]
        metadata = observation_metadata(point, meta, score)
        signal = Signal(
            content=observation_signal_content(point, meta),
            source="spectral-observer",
            channel=point.series,
            author="spectral-forecast",
            timestamp=meta.timestamp_for_index(point.index),
            metadata=metadata,
        )
        trace = await mesh.ingest(signal)
        familiarity = list(trace.familiarity_scores.values())
        traces.append(
            {
                "series": point.series,
                "index": point.index,
                "timestamp": metadata["timestamp"],
                "score": point.score(score),
                "accepted_workers": len(trace.accepted_workers),
                "hops": trace.total_hops,
                "max_familiarity": max(familiarity) if familiarity else 0.0,
                "content": signal.content,
            }
        )

    workers = []
    for worker in sorted(mesh.workers, key=lambda item: item.context.signal_count, reverse=True):
        workers.append(
            {
                "id": str(worker.id)[:8],
                "label": worker.label,
                "signals": worker.context.signal_count,
                "fullness": worker.fullness,
                "rolling_avg_familiarity": worker.rolling_avg_familiarity,
                "terms": sorted(worker.context.terms)[:20],
            }
        )

    low_familiarity = sorted(traces, key=lambda row: row["max_familiarity"])[:10]
    high_score = sorted(traces, key=lambda row: row["score"], reverse=True)[:10]
    return {
        "signals_routed": len(selected),
        "threshold": threshold,
        "score": score,
        "worker_count": mesh.worker_count,
        "workers": workers,
        "lowest_familiarity": low_familiarity,
        "highest_score": high_score,
    }


def _top_observation_rows(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    top: int,
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for point in result.top(score, top):
            meta = series_by_name[point.series]
            row = point.to_dict()
            row["timestamp"] = meta.timestamp_for_index(point.index).isoformat()
            row["score_name"] = score
            row["score"] = point.score(score)
            rows.append(row)
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _readiness_rows(series: Sequence[CdipSeries], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for item in series:
        scan = scan_information_readiness(
            item.values,
            sample_rate=item.sample_rate,
            min_snr=args.readiness_min_snr,
            min_entropy_deficit=args.readiness_min_entropy_deficit,
            min_usable_bins=args.readiness_min_usable_bins,
            min_size=args.readiness_min_size,
            max_size=args.readiness_max_size,
            step=args.readiness_step,
            stable_windows=args.readiness_stable_windows,
        )
        final = scan.points[-1]
        first = scan.first_ready
        stable = scan.stable_ready
        rows.append(
            {
                "series": item.name,
                "first_ready_n": scan.first_ready_n,
                "first_ready_seconds": first.seconds if first is not None else None,
                "stable_ready_n": scan.stable_ready_n,
                "stable_ready_seconds": stable.seconds if stable is not None else None,
                "final_n": final.n,
                "final_ready": final.ready,
                "final_reason": final.reason,
                "final_readiness_score": final.readiness_score,
                "final_entropy_deficit": final.entropy_deficit,
                "final_peak_surprise": final.peak_surprise,
                "final_peak_p_value": final.peak_p_value,
                "final_peak_period_samples": final.peak_period_samples,
                "scan_points": [point.to_dict() for point in scan.points],
            }
        )
    return rows


def _print_text_report(report: dict[str, Any]) -> None:
    data = report["data"]
    print("CDIP observation")
    print(f"  path={data['path']}")
    print(f"  platform={data['platform_id']} sample_rate={data['sample_rate']:.6g}Hz")
    print(
        "  segment=%d:%d samples=%d flags_kept=%s"
        % (
            data["span_start"],
            data["span_end"],
            data["samples"],
            ",".join(str(flag) for flag in data["flags_kept"]),
        )
    )
    print("  protocol=raw displacement, generic QC, past-only spectral observation")
    print("  mesh_terms=no labels and no published wave predictors")

    if report["readiness"]:
        print("\nInformation readiness")
        print(
            "%16s %8s %8s %8s %8s %10s %10s %10s %s"
            % ("series", "first", "stable", "score", "entropy", "peak", "p_peak", "period", "reason")
        )
        for row in report["readiness"]:
            first = "-" if row["first_ready_n"] is None else str(row["first_ready_n"])
            stable = "-" if row["stable_ready_n"] is None else str(row["stable_ready_n"])
            print(
                "%16s %8s %8s %8.3f %8.4f %10.3f %10.3g %10.1f %s"
                % (
                    row["series"],
                    first,
                    stable,
                    row["final_readiness_score"],
                    row["final_entropy_deficit"],
                    row["final_peak_surprise"],
                    row["final_peak_p_value"],
                    row["final_peak_period_samples"],
                    row["final_reason"],
                )
            )

    print("\nTop observations")
    print(
        "%16s %24s %8s %8s %8s %8s %8s %10s"
        % ("series", "timestamp", "index", "score", "frozen", "sliding", "state", "drift")
    )
    for row in report["top_observations"]:
        print(
            "%16s %24s %8d %8.3f %8.3f %8.3f %8.3f %10.3f"
            % (
                row["series"],
                row["timestamp"].replace("+00:00", "Z"),
                row["index"],
                row["score"],
                row["frozen_score"],
                row["sliding_score"],
                row["state_drift_score"],
                row["conditional_drift_score"],
            )
        )

    if report["stigmergy"]["points"]:
        print("\nStigmergy accumulator")
        print("%8s %10s %10s %8s %10s %s" % ("index", "pheromone", "emission", "series", "max", "active"))
        for row in report["stigmergy"]["top"]:
            print(
                "%8d %10.3f %10.3f %8d %10.3f %s"
                % (
                    row["index"],
                    row["pheromone"],
                    row["emission"],
                    row["active_series_count"],
                    row["max_score"],
                    row["active_series"],
                )
            )

    mesh = report.get("mesh")
    if mesh:
        print("\nStigmergy mesh")
        print(
            "  routed=%d workers=%d score=%s threshold=%.3f"
            % (mesh["signals_routed"], mesh["worker_count"], mesh["score"], mesh["threshold"])
        )
        print("  workers")
        for worker in mesh["workers"]:
            terms = ",".join(worker["terms"][:8])
            print(
                "    %s signals=%d familiarity=%.3f label=%s terms=%s"
                % (
                    worker["id"],
                    worker["signals"],
                    worker["rolling_avg_familiarity"],
                    worker["label"],
                    terms,
                )
            )
        if mesh["lowest_familiarity"]:
            print("  lowest familiarity routed signals")
            for row in mesh["lowest_familiarity"][:5]:
                print(
                    "    %s index=%d score=%.3f familiarity=%.3f terms=%s"
                    % (
                        row["series"],
                        row["index"],
                        row["score"],
                        row["max_familiarity"],
                        row["content"],
                    )
                )


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    min_clean_samples = args.min_clean_samples or args.baseline + args.adaptive_window + args.stride
    series = load_cdip_series(
        args.file,
        args.channels,
        keep_flags=set(args.keep_flags),
        min_clean_samples=min_clean_samples,
        sample_limit=args.sample_limit,
        segment_offset=args.segment_offset,
    )
    series_by_name = {item.name: item for item in series}

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

    stig = build_stigmergy(
        results,
        score=args.score,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
    )

    first = series[0]
    report: dict[str, Any] = {
        "data": {
            "path": str(args.file),
            "platform_id": first.platform_id,
            "platform_name": first.platform_name,
            "station_id": first.station_id,
            "sample_rate": first.sample_rate,
            "span_start": first.span_start,
            "span_end": first.span_end,
            "samples": first.span_end - first.span_start,
            "channels": [item.channel for item in series],
            "flags_kept": sorted(args.keep_flags),
        },
        "parameters": {
            "baseline": args.baseline,
            "adaptive_window": args.adaptive_window,
            "stride": args.stride,
            "score": args.score,
            "emission_threshold": args.emission_threshold,
            "decay": args.decay,
        },
        "readiness": _readiness_rows(series, args),
        "top_observations": _top_observation_rows(
            results,
            series_by_name,
            score=args.score,
            top=args.top,
        ),
        "stigmergy": {
            "points": len(stig.points),
            "top": [point.to_dict() for point in stig.top(args.top)],
        },
    }

    if args.mesh:
        report["mesh"] = await run_stigmergy_mesh(
            results,
            series_by_name,
            score=args.score,
            threshold=args.emission_threshold,
            max_signals=args.mesh_max_signals,
            mesh_src=args.mesh_src,
            max_workers=args.mesh_max_workers,
        )

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Path to CDIP *_xy.nc displacement file")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["z"],
        choices=sorted(CHANNEL_VARIABLES),
        help="Displacement channels to observe",
    )
    parser.add_argument(
        "--keep-flags",
        type=int,
        nargs="+",
        default=[2],
        help="CDIP xyzFlagPrimary values to keep",
    )
    parser.add_argument("--baseline", type=int, default=4096, help="Frozen baseline samples")
    parser.add_argument("--adaptive-window", type=int, default=2048, help="Adaptive window samples")
    parser.add_argument("--stride", type=int, default=512, help="Observation stride")
    parser.add_argument("--sample-limit", type=int, default=32768, help="Limit selected clean samples")
    parser.add_argument("--segment-offset", type=int, default=0, help="Offset into selected clean segment")
    parser.add_argument("--min-clean-samples", type=int, default=None, help="Minimum clean segment length")
    parser.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
        help="Score used for rankings and emissions",
    )
    parser.add_argument("--emission-threshold", type=float, default=3.0, help="Emission threshold")
    parser.add_argument("--decay", type=float, default=0.9, help="Stigmergy accumulator decay")
    parser.add_argument("--top", type=int, default=10, help="Rows to show")
    parser.add_argument("--readiness-min-size", type=int, default=512, help="Minimum prefix for readiness scan")
    parser.add_argument("--readiness-max-size", type=int, default=None, help="Maximum prefix for readiness scan")
    parser.add_argument("--readiness-step", type=int, default=512, help="Prefix step for readiness scan")
    parser.add_argument("--readiness-stable-windows", type=int, default=2, help="Consecutive ready prefixes required")
    parser.add_argument("--readiness-min-snr", type=float, default=2.0, help="Peak surprise threshold")
    parser.add_argument("--readiness-min-usable-bins", type=int, default=16, help="Minimum FFT bins for readiness")
    parser.add_argument(
        "--readiness-min-entropy-deficit",
        type=float,
        default=0.02,
        help="Minimum entropy gap below finite white-noise expectation",
    )
    parser.add_argument("--mesh", action="store_true", help="Route emissions through ../stigmergy mesh")
    parser.add_argument("--mesh-src", type=Path, default=Path("../stigmergy/src"), help="Stigmergy src path")
    parser.add_argument("--mesh-max-signals", type=int, default=160, help="Maximum emissions routed to mesh")
    parser.add_argument("--mesh-max-workers", type=int, default=12, help="Maximum mesh workers")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Report format")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(_main_async(args))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
