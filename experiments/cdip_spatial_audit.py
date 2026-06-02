"""Post-hoc spatial/tide audit for CDIP batch reports.

This does not feed spatial or tide features into detection. It audits merged
batch outputs after the agnostic observer has already produced window scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.io import netcdf_file


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class PlatformLocation:
    platform_id: str
    latitude: float
    longitude: float
    water_depth: float | None
    region: str


@dataclass(frozen=True)
class WindowPoint:
    index: int
    start_epoch: float
    delta: float
    z_delta: float
    latitude: float
    longitude: float
    platforms: tuple[str, ...]
    regions: tuple[str, ...]
    region_label: str


def _decode(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar_variable(nc: Any, name: str) -> float | None:
    if name not in nc.variables:
        return None
    data = np.asarray(nc.variables[name].data)
    if data.size == 0:
        return None
    try:
        return float(data.reshape(-1)[0])
    except Exception:
        return None


def classify_region(latitude: float, longitude: float) -> str:
    """Coarse basin/coastal-regime bins for post-hoc tide/spatial audit."""

    lat = float(latitude)
    lon = float(longitude)
    if 40.0 <= lat <= 49.0 and -90.0 <= lon <= -75.0:
        return "great_lakes"
    if 18.0 <= lat <= 23.0 and -161.0 <= lon <= -154.0:
        return "hawaii"
    if lat <= -5.0 and lon <= -150.0:
        return "south_pacific"
    if lon >= 130.0:
        return "west_pacific"
    if lat >= 45.0 and lon <= -130.0:
        return "north_pacific_alaska"
    if -130.0 <= lon <= -116.0 and lat >= 39.0:
        return "pacific_northwest"
    if -130.0 <= lon <= -116.0 and lat >= 35.0:
        return "central_california"
    if -130.0 <= lon <= -116.0 and lat >= 30.0:
        return "southern_california"
    if 15.0 <= lat <= 22.0 and -70.0 <= lon <= -60.0:
        return "caribbean"
    if -99.0 <= lon < -86.0 and lat < 31.0:
        return "gulf"
    if -86.0 <= lon <= -80.0 and lat < 31.0:
        return "florida_gulf_atlantic"
    if -83.0 <= lon <= -70.0 and lat >= 38.0:
        return "northeast_atlantic"
    if -83.0 <= lon <= -70.0 and lat >= 30.0:
        return "southeast_atlantic"
    return "other"


def _group_region_label(regions: Sequence[str]) -> str:
    unique = sorted(set(regions))
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + "+".join(unique)


def load_platform_location(path: Path) -> PlatformLocation:
    with netcdf_file(str(path), "r", mmap=False) as nc:
        station_id = _decode(getattr(nc, "cdip_station_id", ""), "unknown")
        platform_id = _decode(getattr(nc, "platform_id", station_id), station_id)
        lat = _scalar_variable(nc, "metaDeployLatitude")
        lon = _scalar_variable(nc, "metaDeployLongitude")
        if lat is None:
            lat = float(getattr(nc, "geospatial_lat_min", 0.0))
        if lon is None:
            lon = float(getattr(nc, "geospatial_lon_min", 0.0))
        depth = _scalar_variable(nc, "metaWaterDepth")
    return PlatformLocation(
        platform_id=platform_id,
        latitude=float(lat),
        longitude=float(lon),
        water_depth=depth,
        region=classify_region(float(lat), float(lon)),
    )


def haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    d_phi = math.radians(lat_b - lat_a)
    d_lambda = math.radians(lon_b - lon_a)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _parse_epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _bin_index(value: float, edges: Sequence[float]) -> int | None:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return None


def _bin_label(edges: Sequence[float], index: int, unit: str) -> str:
    high = edges[index + 1]
    if math.isinf(high):
        return f">={edges[index]:g}{unit}"
    return f"{edges[index]:g}-{high:g}{unit}"


def _load_window_points(report: dict[str, Any]) -> list[WindowPoint]:
    locations: dict[str, PlatformLocation] = {}
    deltas = np.asarray(
        [float(row["observed_minus_null"]) for row in report.get("windows", [])],
        dtype=np.float64,
    )
    if len(deltas) == 0:
        return []
    center = float(np.mean(deltas))
    scale = float(np.std(deltas)) or 1.0

    points: list[WindowPoint] = []
    for index, row in enumerate(report.get("windows", [])):
        group = tuple(str(item) for item in row["group"])
        coords = []
        for source_path in row["paths"]:
            path = Path(source_path)
            platform = path.name.split("_", 1)[0]
            location = locations.get(platform)
            if location is None:
                location = load_platform_location(path)
                locations[platform] = location
            coords.append(location)
        if not coords:
            continue
        lat = float(np.mean([item.latitude for item in coords]))
        lon = float(np.mean([item.longitude for item in coords]))
        regions = tuple(location.region for location in coords)
        delta = float(row["observed_minus_null"])
        points.append(
            WindowPoint(
                index=index,
                start_epoch=_parse_epoch(str(row["start"])),
                delta=delta,
                z_delta=(delta - center) / scale,
                latitude=lat,
                longitude=lon,
                platforms=group,
                regions=regions,
                region_label=_group_region_label(regions),
            )
        )
    return points


def geographic_stratification(
    points: Sequence[WindowPoint],
    *,
    high_delta_quantile: float,
) -> dict[str, Any]:
    if not points:
        return {"regions": [], "same_region": {}, "high_delta": {}}

    threshold = float(np.quantile([point.delta for point in points], high_delta_quantile))
    buckets: dict[str, list[WindowPoint]] = {}
    same_region_buckets: dict[str, list[WindowPoint]] = {"same_region": [], "mixed_region": []}

    for point in points:
        buckets.setdefault(point.region_label, []).append(point)
        same_key = "same_region" if len(set(point.regions)) == 1 else "mixed_region"
        same_region_buckets[same_key].append(point)

    def summarize(label: str, values: Sequence[WindowPoint]) -> dict[str, Any]:
        deltas = np.asarray([point.delta for point in values], dtype=np.float64)
        z_values = np.asarray([point.z_delta for point in values], dtype=np.float64)
        return {
            "label": label,
            "windows": len(values),
            "delta_sum": float(np.sum(deltas)) if len(deltas) else 0.0,
            "delta_mean": float(np.mean(deltas)) if len(deltas) else 0.0,
            "z_delta_mean": float(np.mean(z_values)) if len(z_values) else 0.0,
            "positive_fraction": float(np.mean(deltas > 0.0)) if len(deltas) else 0.0,
            "high_delta_windows": int(np.sum(deltas >= threshold)) if len(deltas) else 0,
        }

    region_rows = [summarize(label, values) for label, values in buckets.items()]
    region_rows.sort(key=lambda row: abs(row["delta_sum"]), reverse=True)
    same_rows = {
        label: summarize(label, values)
        for label, values in same_region_buckets.items()
    }
    return {
        "high_delta_threshold": threshold,
        "regions": region_rows,
        "same_region": same_rows,
    }


def spatiotemporal_correlation(
    points: Sequence[WindowPoint],
    *,
    lag_edges_hours: Sequence[float],
    distance_edges_km: Sequence[float],
    max_pairs: int,
) -> list[dict[str, Any]]:
    sums = np.zeros((len(lag_edges_hours) - 1, len(distance_edges_km) - 1), dtype=np.float64)
    counts = np.zeros_like(sums)
    n = len(points)
    total_pairs = n * (n - 1) // 2
    step = max(1, math.ceil(total_pairs / max_pairs)) if max_pairs > 0 else 1
    seen = 0
    sampled = 0
    for i in range(n):
        a = points[i]
        for j in range(i + 1, n):
            if seen % step != 0:
                seen += 1
                continue
            seen += 1
            sampled += 1
            b = points[j]
            lag_hours = abs(b.start_epoch - a.start_epoch) / 3600.0
            distance_km = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
            lag_index = _bin_index(lag_hours, lag_edges_hours)
            distance_index = _bin_index(distance_km, distance_edges_km)
            if lag_index is None or distance_index is None:
                continue
            sums[lag_index, distance_index] += a.z_delta * b.z_delta
            counts[lag_index, distance_index] += 1.0

    rows = []
    for lag_index in range(sums.shape[0]):
        for distance_index in range(sums.shape[1]):
            count = int(counts[lag_index, distance_index])
            rows.append(
                {
                    "lag_bin": _bin_label(lag_edges_hours, lag_index, "h"),
                    "distance_bin": _bin_label(distance_edges_km, distance_index, "km"),
                    "pairs": count,
                    "mean_z_product": float(sums[lag_index, distance_index] / count)
                    if count
                    else 0.0,
                }
            )
    rows.sort(key=lambda row: (row["lag_bin"], row["distance_bin"]))
    return rows


def propagation_candidates(
    points: Sequence[WindowPoint],
    *,
    quantile: float,
    max_lag_hours: float,
    min_lag_hours: float,
    min_distance_km: float,
) -> dict[str, Any]:
    if not points:
        return {"threshold": 0.0, "events": 0, "pairs": 0, "speed_km_h": {}}
    threshold = float(np.quantile([point.delta for point in points], quantile))
    high = [point for point in points if point.delta >= threshold]
    speeds = []
    for i, a in enumerate(high):
        for b in high:
            lag_hours = (b.start_epoch - a.start_epoch) / 3600.0
            if lag_hours < min_lag_hours or lag_hours > max_lag_hours:
                continue
            distance_km = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
            if distance_km < min_distance_km:
                continue
            speeds.append(distance_km / lag_hours)
    if speeds:
        speed_summary = {
            "p10": float(np.quantile(speeds, 0.10)),
            "median": float(np.median(speeds)),
            "p90": float(np.quantile(speeds, 0.90)),
            "max": float(np.max(speeds)),
        }
    else:
        speed_summary = {}
    return {
        "threshold": threshold,
        "events": len(high),
        "pairs": len(speeds),
        "speed_km_h": speed_summary,
    }


def audit_report(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.report.read_text())
    points = _load_window_points(report)
    lag_edges = [float(item) for item in args.lag_edges_hours]
    distance_edges = [float(item) for item in args.distance_edges_km]
    if not math.isinf(distance_edges[-1]):
        distance_edges.append(math.inf)
    rows = spatiotemporal_correlation(
        points,
        lag_edges_hours=lag_edges,
        distance_edges_km=distance_edges,
        max_pairs=args.max_pairs,
    )
    strongest = sorted(rows, key=lambda row: abs(row["mean_z_product"]), reverse=True)[: args.top]
    tide_bins = [
        row
        for row in rows
        if row["lag_bin"] in {"9-15h", "15-24h"}
        and row["pairs"] >= args.min_pairs_for_summary
    ]
    zero_lag_bins = [
        row
        for row in rows
        if row["lag_bin"] == "0-0.75h" and row["pairs"] >= args.min_pairs_for_summary
    ]
    return {
        "parameters": {
            "report": str(args.report),
            "max_pairs": args.max_pairs,
            "lag_edges_hours": lag_edges,
            "distance_edges_km": distance_edges,
            "high_delta_quantile": args.high_delta_quantile,
        },
        "summary": {
            "windows": len(points),
            "time_span_hours": (max(point.start_epoch for point in points) - min(point.start_epoch for point in points))
            / 3600.0
            if points
            else 0.0,
            "delta_mean": float(np.mean([point.delta for point in points])) if points else 0.0,
            "delta_std": float(np.std([point.delta for point in points])) if points else 0.0,
            "strongest_bins": strongest,
            "zero_lag_bins": zero_lag_bins,
            "semidiurnal_lag_bins": tide_bins,
            "geographic_stratification": geographic_stratification(
                points,
                high_delta_quantile=args.high_delta_quantile,
            ),
            "propagation_candidates": propagation_candidates(
                points,
                quantile=args.high_delta_quantile,
                max_lag_hours=args.max_propagation_lag_hours,
                min_lag_hours=args.min_propagation_lag_hours,
                min_distance_km=args.min_propagation_distance_km,
            ),
        },
        "bins": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Merged CDIP batch JSON report")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    parser.add_argument("--max-pairs", type=int, default=2_000_000)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--lag-edges-hours",
        type=float,
        nargs="+",
        default=[0.0, 0.75, 1.5, 3.0, 6.0, 9.0, 15.0, 24.0, 48.0],
    )
    parser.add_argument(
        "--distance-edges-km",
        type=float,
        nargs="+",
        default=[0.0, 50.0, 150.0, 300.0, 600.0, 1200.0],
    )
    parser.add_argument("--high-delta-quantile", type=float, default=0.95)
    parser.add_argument("--min-propagation-lag-hours", type=float, default=0.75)
    parser.add_argument("--max-propagation-lag-hours", type=float, default=24.0)
    parser.add_argument("--min-propagation-distance-km", type=float, default=25.0)
    parser.add_argument("--min-pairs-for-summary", type=int, default=100)
    return parser.parse_args(argv)


def _print_text(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("CDIP spatial/tide audit")
    print(
        "  windows=%d span=%.2fh delta_mean=%.3f delta_std=%.3f"
        % (
            summary["windows"],
            summary["time_span_hours"],
            summary["delta_mean"],
            summary["delta_std"],
        )
    )
    print("\nStrongest lag/distance bins")
    print("%12s %16s %8s %14s" % ("lag", "distance", "pairs", "mean_z_product"))
    for row in summary["strongest_bins"]:
        print(
            "%12s %16s %8d %14.4f"
            % (row["lag_bin"], row["distance_bin"], row["pairs"], row["mean_z_product"])
        )
    prop = summary["propagation_candidates"]
    print("\nHigh-delta propagation candidates")
    print(
        "  threshold=%.3f events=%d pairs=%d speed_km_h=%s"
        % (
            prop["threshold"],
            prop["events"],
            prop["pairs"],
            json.dumps(prop["speed_km_h"], sort_keys=True),
        )
    )
    geo = summary["geographic_stratification"]
    print("\nGeographic stratification")
    same = geo["same_region"]
    for label in ["same_region", "mixed_region"]:
        row = same[label]
        print(
            "  %s windows=%d delta_sum=%.3f delta_mean=%.3f high_delta=%d"
            % (
                label,
                row["windows"],
                row["delta_sum"],
                row["delta_mean"],
                row["high_delta_windows"],
            )
        )
    print("%38s %8s %12s %12s %10s" % ("region", "windows", "delta_sum", "delta_mean", "high_delta"))
    for row in geo["regions"][:10]:
        print(
            "%38s %8d %12.3f %12.3f %10d"
            % (
                row["label"],
                row["windows"],
                row["delta_sum"],
                row["delta_mean"],
                row["high_delta_windows"],
            )
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit_report(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)


if __name__ == "__main__":
    main()
