"""Post-hoc spatial/tide audit for CDIP batch reports.

This does not feed spatial or tide features into detection. It audits merged
batch outputs after the agnostic observer has already produced window scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import combinations
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
    platform_latitudes: tuple[float, ...]
    platform_longitudes: tuple[float, ...]
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


def initial_bearing_degrees(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    phi_a = math.radians(lat_a)
    phi_b = math.radians(lat_b)
    d_lambda = math.radians(lon_b - lon_a)
    y = math.sin(d_lambda) * math.cos(phi_b)
    x = math.cos(phi_a) * math.sin(phi_b) - math.sin(phi_a) * math.cos(phi_b) * math.cos(d_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def bearing_bucket(bearing: float) -> str:
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    index = int(((float(bearing) + 22.5) % 360.0) // 45.0)
    return labels[index]


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
                platform_latitudes=tuple(location.latitude for location in coords),
                platform_longitudes=tuple(location.longitude for location in coords),
                regions=regions,
                region_label=_group_region_label(regions),
            )
        )
    return points


def _platform_locations_from_points(points: Sequence[WindowPoint]) -> dict[str, PlatformLocation]:
    locations: dict[str, PlatformLocation] = {}
    for point in points:
        for platform, lat, lon, region in zip(
            point.platforms,
            point.platform_latitudes,
            point.platform_longitudes,
            point.regions,
        ):
            locations.setdefault(
                platform,
                PlatformLocation(
                    platform_id=platform,
                    latitude=lat,
                    longitude=lon,
                    water_depth=None,
                    region=region,
                ),
            )
    return locations


def _replace_point_geometry(
    point: WindowPoint,
    locations: dict[str, PlatformLocation],
) -> WindowPoint:
    coords = [locations[platform] for platform in point.platforms]
    regions = tuple(location.region for location in coords)
    return replace(
        point,
        latitude=float(np.mean([location.latitude for location in coords])),
        longitude=float(np.mean([location.longitude for location in coords])),
        platform_latitudes=tuple(location.latitude for location in coords),
        platform_longitudes=tuple(location.longitude for location in coords),
        regions=regions,
        region_label=_group_region_label(regions),
    )


def permute_platform_geometry(
    points: Sequence[WindowPoint],
    *,
    rng: np.random.Generator,
) -> list[WindowPoint]:
    """Shuffle platform locations/regions while preserving each window's scores."""

    locations = _platform_locations_from_points(points)
    platforms = sorted(locations)
    shuffled = [locations[platform] for platform in platforms]
    order = rng.permutation(len(shuffled))
    reassigned = {
        platform: PlatformLocation(
            platform_id=platform,
            latitude=shuffled[int(order[index])].latitude,
            longitude=shuffled[int(order[index])].longitude,
            water_depth=shuffled[int(order[index])].water_depth,
            region=shuffled[int(order[index])].region,
        )
        for index, platform in enumerate(platforms)
    }
    return [_replace_point_geometry(point, reassigned) for point in points]


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
    pair_exposure = {
        "same_region_pairs": 0,
        "mixed_region_pairs": 0,
        "same_region_delta": 0.0,
        "mixed_region_delta": 0.0,
    }

    for point in points:
        buckets.setdefault(point.region_label, []).append(point)
        same_key = "same_region" if len(set(point.regions)) == 1 else "mixed_region"
        same_region_buckets[same_key].append(point)
        pairs = list(combinations(point.regions, 2))
        if pairs:
            same_pairs = sum(1 for left, right in pairs if left == right)
            mixed_pairs = len(pairs) - same_pairs
            pair_exposure["same_region_pairs"] += same_pairs
            pair_exposure["mixed_region_pairs"] += mixed_pairs
            pair_exposure["same_region_delta"] += point.delta * same_pairs / len(pairs)
            pair_exposure["mixed_region_delta"] += point.delta * mixed_pairs / len(pairs)

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
    all_regions = sorted({region for point in points for region in point.regions})
    holdouts = []
    for region in all_regions:
        with_region = [point for point in points if region in point.regions]
        without_region = [point for point in points if region not in point.regions]
        holdouts.append(
            {
                "region": region,
                "with_region": summarize("with_region", with_region),
                "without_region": summarize("without_region", without_region),
            }
        )
    holdouts.sort(key=lambda row: abs(row["with_region"]["delta_sum"]), reverse=True)
    same_pairs = pair_exposure["same_region_pairs"]
    mixed_pairs = pair_exposure["mixed_region_pairs"]
    return {
        "high_delta_threshold": threshold,
        "regions": region_rows,
        "same_region": same_rows,
        "pair_exposure": {
            **pair_exposure,
            "same_region_delta_per_pair": pair_exposure["same_region_delta"] / same_pairs
            if same_pairs
            else 0.0,
            "mixed_region_delta_per_pair": pair_exposure["mixed_region_delta"] / mixed_pairs
            if mixed_pairs
            else 0.0,
            "mixed_pair_fraction": mixed_pairs / (same_pairs + mixed_pairs)
            if same_pairs + mixed_pairs
            else 0.0,
        },
        "leave_one_region_out": holdouts,
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


def sampled_spatiotemporal_correlation(
    points: Sequence[WindowPoint],
    *,
    lag_edges_hours: Sequence[float],
    distance_edges_km: Sequence[float],
    pair_count: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    sums = np.zeros((len(lag_edges_hours) - 1, len(distance_edges_km) - 1), dtype=np.float64)
    counts = np.zeros_like(sums)
    n = len(points)
    if n < 2 or pair_count <= 0:
        return _empty_correlation_rows(lag_edges_hours, distance_edges_km, sums, counts)
    total_pairs = n * (n - 1) // 2
    count = min(pair_count, total_pairs)
    left_indexes = rng.integers(0, n - 1, size=count)
    spans = n - left_indexes - 1
    right_indexes = left_indexes + 1 + np.floor(rng.random(size=count) * spans).astype(np.int64)
    for i, j in zip(left_indexes, right_indexes):
        a = points[int(i)]
        b = points[int(j)]
        lag_hours = abs(b.start_epoch - a.start_epoch) / 3600.0
        distance_km = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
        lag_index = _bin_index(lag_hours, lag_edges_hours)
        distance_index = _bin_index(distance_km, distance_edges_km)
        if lag_index is None or distance_index is None:
            continue
        sums[lag_index, distance_index] += a.z_delta * b.z_delta
        counts[lag_index, distance_index] += 1.0
    return _empty_correlation_rows(lag_edges_hours, distance_edges_km, sums, counts)


def _empty_correlation_rows(
    lag_edges_hours: Sequence[float],
    distance_edges_km: Sequence[float],
    sums: np.ndarray,
    counts: np.ndarray,
) -> list[dict[str, Any]]:
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


def _correlation_row(
    rows: Sequence[dict[str, Any]],
    *,
    lag_bin: str,
    distance_bin: str,
) -> dict[str, Any]:
    for row in rows:
        if row["lag_bin"] == lag_bin and row["distance_bin"] == distance_bin:
            return row
    return {"lag_bin": lag_bin, "distance_bin": distance_bin, "pairs": 0, "mean_z_product": 0.0}


def _spatial_metric_values(
    points: Sequence[WindowPoint],
    rows: Sequence[dict[str, Any]],
    *,
    high_delta_quantile: float,
    min_pairs: int,
) -> dict[str, float]:
    valid_rows = [row for row in rows if row["pairs"] >= min_pairs]
    geo = geographic_stratification(points, high_delta_quantile=high_delta_quantile)
    same = geo["same_region"].get("same_region", {})
    mixed = geo["same_region"].get("mixed_region", {})
    near_semidiurnal = _correlation_row(rows, lag_bin="9-15h", distance_bin="0-50km")
    near_zero_lag = _correlation_row(rows, lag_bin="0-0.75h", distance_bin="0-50km")
    near_semidiurnal_value = (
        float(near_semidiurnal["mean_z_product"])
        if int(near_semidiurnal["pairs"]) >= min_pairs
        else 0.0
    )
    near_zero_lag_value = (
        float(near_zero_lag["mean_z_product"])
        if int(near_zero_lag["pairs"]) >= min_pairs
        else 0.0
    )
    return {
        "max_abs_lag_distance_mean_z_product": float(
            max((abs(row["mean_z_product"]) for row in valid_rows), default=0.0)
        ),
        "near_semidiurnal_9_15h_0_50km_mean_z_product": near_semidiurnal_value,
        "near_semidiurnal_9_15h_0_50km_pairs": float(near_semidiurnal["pairs"]),
        "near_zero_lag_0_075h_0_50km_mean_z_product": near_zero_lag_value,
        "near_zero_lag_0_075h_0_50km_pairs": float(near_zero_lag["pairs"]),
        "same_region_delta_sum": float(same.get("delta_sum", 0.0)),
        "same_region_delta_mean": float(same.get("delta_mean", 0.0)),
        "mixed_region_delta_sum": float(mixed.get("delta_sum", 0.0)),
        "mixed_region_delta_mean": float(mixed.get("delta_mean", 0.0)),
    }


def spatial_surrogate_control(
    points: Sequence[WindowPoint],
    *,
    lag_edges_hours: Sequence[float],
    distance_edges_km: Sequence[float],
    high_delta_quantile: float,
    repeats: int,
    pair_count: int,
    seed: int,
    min_pairs: int,
) -> dict[str, Any]:
    if not points or repeats <= 0:
        return {"repeats": 0, "metrics": {}}
    rng = np.random.default_rng(seed)
    observed_rows = sampled_spatiotemporal_correlation(
        points,
        lag_edges_hours=lag_edges_hours,
        distance_edges_km=distance_edges_km,
        pair_count=pair_count,
        rng=rng,
    )
    observed = _spatial_metric_values(
        points,
        observed_rows,
        high_delta_quantile=high_delta_quantile,
        min_pairs=min_pairs,
    )
    null_values: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(repeats):
        surrogate = permute_platform_geometry(points, rng=rng)
        rows = sampled_spatiotemporal_correlation(
            surrogate,
            lag_edges_hours=lag_edges_hours,
            distance_edges_km=distance_edges_km,
            pair_count=pair_count,
            rng=rng,
        )
        values = _spatial_metric_values(
            surrogate,
            rows,
            high_delta_quantile=high_delta_quantile,
            min_pairs=min_pairs,
        )
        for key, value in values.items():
            null_values[key].append(float(value))

    metrics = {}
    for key, values in null_values.items():
        arr = np.asarray(values, dtype=np.float64)
        obs = float(observed[key])
        metrics[key] = {
            "observed": obs,
            "null_mean": float(np.mean(arr)),
            "null_std": float(np.std(arr)),
            "null_p05": float(np.quantile(arr, 0.05)),
            "null_median": float(np.median(arr)),
            "null_p95": float(np.quantile(arr, 0.95)),
            "empirical_p_ge_observed": float((1 + np.sum(arr >= obs)) / (len(arr) + 1)),
            "empirical_p_abs_ge_observed": float(
                (1 + np.sum(np.abs(arr) >= abs(obs))) / (len(arr) + 1)
            ),
        }

    return {
        "seed": seed,
        "repeats": repeats,
        "pair_count": pair_count,
        "min_pairs": min_pairs,
        "description": (
            "Randomly permutes platform coordinates/regions across platform IDs "
            "while preserving each window's time, group membership, and detector delta."
        ),
        "metrics": metrics,
    }


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


def directional_propagation_candidates(
    points: Sequence[WindowPoint],
    *,
    quantile: float,
    max_lag_hours: float,
    min_lag_hours: float,
    min_distance_km: float,
    min_speed_km_h: float,
    max_speed_km_h: float,
) -> dict[str, Any]:
    if not points:
        return {"threshold": 0.0, "events": 0, "pairs": 0, "bounded_speed_pairs": 0}
    threshold = float(np.quantile([point.delta for point in points], quantile))
    high = sorted([point for point in points if point.delta >= threshold], key=lambda point: point.start_epoch)
    speeds = []
    bounded_speeds = []
    bearings = Counter()
    bounded_bearings = Counter()
    corridors = Counter()
    for a in high:
        for b in high:
            lag_hours = (b.start_epoch - a.start_epoch) / 3600.0
            if lag_hours < min_lag_hours or lag_hours > max_lag_hours:
                continue
            distance_km = haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
            if distance_km < min_distance_km:
                continue
            speed = distance_km / lag_hours
            bearing = initial_bearing_degrees(a.latitude, a.longitude, b.latitude, b.longitude)
            bucket = bearing_bucket(bearing)
            speeds.append(speed)
            bearings[bucket] += 1
            if min_speed_km_h <= speed <= max_speed_km_h:
                bounded_speeds.append(speed)
                bounded_bearings[bucket] += 1
                corridors[f"{a.region_label}->{b.region_label}"] += 1

    def summarize(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "p10": float(np.quantile(arr, 0.10)),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(np.max(arr)),
        }

    return {
        "threshold": threshold,
        "events": len(high),
        "pairs": len(speeds),
        "speed_km_h": summarize(speeds),
        "bearing_histogram": dict(sorted(bearings.items())),
        "bounded_speed_range_km_h": [min_speed_km_h, max_speed_km_h],
        "bounded_speed_pairs": len(bounded_speeds),
        "bounded_speed_fraction": float(len(bounded_speeds) / len(speeds)) if speeds else 0.0,
        "bounded_speed_km_h": summarize(bounded_speeds),
        "bounded_bearing_histogram": dict(sorted(bounded_bearings.items())),
        "top_bounded_region_corridors": [
            {"corridor": corridor, "pairs": count}
            for corridor, count in corridors.most_common(10)
        ],
    }


def _directional_metric_values(summary: dict[str, Any]) -> dict[str, float]:
    bearing_counts = list(summary.get("bearing_histogram", {}).values())
    bounded_bearing_counts = list(summary.get("bounded_bearing_histogram", {}).values())
    pairs = float(summary.get("pairs", 0))
    bounded_pairs = float(summary.get("bounded_speed_pairs", 0))
    speed = summary.get("speed_km_h", {})
    bounded_speed = summary.get("bounded_speed_km_h", {})
    return {
        "bounded_speed_fraction": float(summary.get("bounded_speed_fraction", 0.0)),
        "bounded_speed_pairs": bounded_pairs,
        "speed_median_km_h": float(speed.get("median", 0.0)),
        "bounded_speed_median_km_h": float(bounded_speed.get("median", 0.0)),
        "bearing_max_fraction": float(max(bearing_counts) / pairs) if pairs else 0.0,
        "bounded_bearing_max_fraction": float(max(bounded_bearing_counts) / bounded_pairs)
        if bounded_pairs
        else 0.0,
    }


def directional_surrogate_control(
    points: Sequence[WindowPoint],
    *,
    high_delta_quantile: float,
    repeats: int,
    seed: int,
    max_lag_hours: float,
    min_lag_hours: float,
    min_distance_km: float,
    min_speed_km_h: float,
    max_speed_km_h: float,
) -> dict[str, Any]:
    if not points or repeats <= 0:
        return {"repeats": 0, "metrics": {}}
    rng = np.random.default_rng(seed)
    observed_summary = directional_propagation_candidates(
        points,
        quantile=high_delta_quantile,
        max_lag_hours=max_lag_hours,
        min_lag_hours=min_lag_hours,
        min_distance_km=min_distance_km,
        min_speed_km_h=min_speed_km_h,
        max_speed_km_h=max_speed_km_h,
    )
    observed = _directional_metric_values(observed_summary)
    null_values: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(repeats):
        surrogate = permute_platform_geometry(points, rng=rng)
        summary = directional_propagation_candidates(
            surrogate,
            quantile=high_delta_quantile,
            max_lag_hours=max_lag_hours,
            min_lag_hours=min_lag_hours,
            min_distance_km=min_distance_km,
            min_speed_km_h=min_speed_km_h,
            max_speed_km_h=max_speed_km_h,
        )
        values = _directional_metric_values(summary)
        for key, value in values.items():
            null_values[key].append(float(value))
    metrics = {}
    for key, values in null_values.items():
        arr = np.asarray(values, dtype=np.float64)
        obs = float(observed[key])
        metrics[key] = {
            "observed": obs,
            "null_mean": float(np.mean(arr)),
            "null_std": float(np.std(arr)),
            "null_p05": float(np.quantile(arr, 0.05)),
            "null_median": float(np.median(arr)),
            "null_p95": float(np.quantile(arr, 0.95)),
            "empirical_p_ge_observed": float((1 + np.sum(arr >= obs)) / (len(arr) + 1)),
            "empirical_p_abs_ge_observed": float(
                (1 + np.sum(np.abs(arr) >= abs(obs))) / (len(arr) + 1)
            ),
        }
    return {
        "seed": seed,
        "repeats": repeats,
        "description": (
            "Randomly permutes platform geometry and recomputes directional high-delta "
            "candidate metrics while preserving detected window times and deltas."
        ),
        "metrics": metrics,
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
            "spatial_surrogate_repeats": args.spatial_surrogate_repeats,
            "spatial_surrogate_pair_count": args.spatial_surrogate_pair_count,
            "spatial_surrogate_seed": args.spatial_surrogate_seed,
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
            "directional_propagation_candidates": directional_propagation_candidates(
                points,
                quantile=args.high_delta_quantile,
                max_lag_hours=args.max_propagation_lag_hours,
                min_lag_hours=args.min_propagation_lag_hours,
                min_distance_km=args.min_propagation_distance_km,
                min_speed_km_h=args.min_directional_speed_km_h,
                max_speed_km_h=args.max_directional_speed_km_h,
            ),
            "spatial_surrogate_control": spatial_surrogate_control(
                points,
                lag_edges_hours=lag_edges,
                distance_edges_km=distance_edges,
                high_delta_quantile=args.high_delta_quantile,
                repeats=args.spatial_surrogate_repeats,
                pair_count=args.spatial_surrogate_pair_count,
                seed=args.spatial_surrogate_seed,
                min_pairs=args.min_pairs_for_summary,
            ),
            "directional_surrogate_control": directional_surrogate_control(
                points,
                high_delta_quantile=args.high_delta_quantile,
                repeats=args.spatial_surrogate_repeats,
                seed=args.spatial_surrogate_seed + 17,
                max_lag_hours=args.max_propagation_lag_hours,
                min_lag_hours=args.min_propagation_lag_hours,
                min_distance_km=args.min_propagation_distance_km,
                min_speed_km_h=args.min_directional_speed_km_h,
                max_speed_km_h=args.max_directional_speed_km_h,
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
    parser.add_argument("--min-directional-speed-km-h", type=float, default=10.0)
    parser.add_argument("--max-directional-speed-km-h", type=float, default=120.0)
    parser.add_argument("--spatial-surrogate-repeats", type=int, default=0)
    parser.add_argument("--spatial-surrogate-pair-count", type=int, default=250_000)
    parser.add_argument("--spatial-surrogate-seed", type=int, default=20260602)
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
    directional = summary["directional_propagation_candidates"]
    print("\nDirectional high-delta candidates")
    print(
        "  bounded_speed_range=%s bounded_pairs=%d fraction=%.3f bearings=%s"
        % (
            json.dumps(directional["bounded_speed_range_km_h"]),
            directional["bounded_speed_pairs"],
            directional["bounded_speed_fraction"],
            json.dumps(directional["bounded_bearing_histogram"], sort_keys=True),
        )
    )
    surrogate = summary["spatial_surrogate_control"]
    if surrogate["repeats"]:
        print("\nSpatial surrogate control")
        for name, row in surrogate["metrics"].items():
            print(
                "  %s observed=%.4f null_median=%.4f null_p95=%.4f p_abs=%.4f"
                % (
                    name,
                    row["observed"],
                    row["null_median"],
                    row["null_p95"],
                    row["empirical_p_abs_ge_observed"],
                )
            )
        print("\nDirectional surrogate control")
        for name, row in summary["directional_surrogate_control"]["metrics"].items():
            print(
                "  %s observed=%.4f null_median=%.4f null_p95=%.4f p_abs=%.4f"
                % (
                    name,
                    row["observed"],
                    row["null_median"],
                    row["null_p95"],
                    row["empirical_p_abs_ge_observed"],
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
