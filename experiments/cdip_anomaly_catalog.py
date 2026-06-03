"""Build a post-hoc catalog of surfaced CDIP anomaly windows.

The catalog is interpretation-only. It reads an already-produced batch report
and summarizes the windows, platform combinations, regions, and time clusters
that the agnostic observer surfaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cdip_spatial_audit import load_platform_location


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _platform_locations(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = report.get("parameters", {}).get("files", [])
    out: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        location = load_platform_location(path)
        out[location.platform_id] = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "water_depth": location.water_depth,
            "region": location.region,
        }
    return out


def _window_regions(window: dict[str, Any], locations: dict[str, dict[str, Any]]) -> list[str]:
    regions = []
    for platform in window.get("group", []):
        region = locations.get(platform, {}).get("region")
        if region is not None:
            regions.append(str(region))
    return sorted(set(regions))


def _region_label(regions: Sequence[str]) -> str:
    unique = sorted(set(regions))
    if not unique:
        return "unknown"
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + "+".join(unique)


def _top_windows(
    report: dict[str, Any],
    *,
    locations: dict[str, dict[str, Any]],
    top: int,
) -> list[dict[str, Any]]:
    rows = sorted(
        report.get("windows", []),
        key=lambda row: float(row.get("observed_minus_null", 0.0)),
        reverse=True,
    )[:top]
    out = []
    for rank, row in enumerate(rows, start=1):
        regions = _window_regions(row, locations)
        out.append(
            {
                "rank": rank,
                "global_window_index": row.get("global_window_index"),
                "start": row.get("start"),
                "end": row.get("end"),
                "observed_minus_null": row.get("observed_minus_null"),
                "observed_multi_emission": row.get("observed_multi_emission"),
                "null_multi_emission_mean": row.get("null_multi_emission_mean"),
                "group": row.get("group", []),
                "regions": regions,
                "region_label": _region_label(regions),
                "top_combinations": row.get("top_combinations", [])[:5],
            }
        )
    return out


def _recurring_combinations(windows: Sequence[dict[str, Any]], *, top: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    emission_sums: defaultdict[str, float] = defaultdict(float)
    max_scores: defaultdict[str, float] = defaultdict(float)
    example_starts: defaultdict[str, list[str]] = defaultdict(list)
    for window in windows:
        for combo in window.get("top_combinations", []):
            key = str(combo.get("active_platforms", ""))
            if not key:
                continue
            counts[key] += 1
            emission_sums[key] += float(combo.get("emission_sum", 0.0))
            max_scores[key] = max(max_scores[key], float(combo.get("max_score", 0.0)))
            starts = example_starts[key]
            if len(starts) < 5 and window.get("start") is not None:
                starts.append(str(window["start"]))
    rows = []
    for key, count in counts.most_common(top):
        rows.append(
            {
                "active_platforms": key,
                "top_window_count": count,
                "emission_sum": emission_sums[key],
                "max_score": max_scores[key],
                "example_starts": example_starts[key],
            }
        )
    return rows


def _time_clusters(windows: Sequence[dict[str, Any]], *, gap_hours: float) -> list[dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda row: _parse_time(str(row["start"])))
    clusters: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    max_gap = timedelta(hours=gap_hours)
    for row in ordered[1:]:
        previous = _parse_time(str(current[-1]["start"]))
        current_time = _parse_time(str(row["start"]))
        if current_time - previous <= max_gap:
            current.append(row)
        else:
            clusters.append(current)
            current = [row]
    clusters.append(current)

    out = []
    for cluster in clusters:
        platforms = sorted({platform for row in cluster for platform in row.get("group", [])})
        regions = sorted({region for row in cluster for region in row.get("regions", [])})
        out.append(
            {
                "start": cluster[0]["start"],
                "end": cluster[-1]["end"],
                "windows": len(cluster),
                "platforms": platforms,
                "regions": regions,
                "max_observed_minus_null": max(
                    float(row.get("observed_minus_null", 0.0)) for row in cluster
                ),
                "window_indices": [row.get("global_window_index") for row in cluster],
            }
        )
    return sorted(out, key=lambda row: (row["windows"], row["max_observed_minus_null"]), reverse=True)


def _region_summary(windows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    delta_sums: defaultdict[str, float] = defaultdict(float)
    for window in windows:
        label = str(window.get("region_label", "unknown"))
        counts[label] += 1
        delta_sums[label] += float(window.get("observed_minus_null", 0.0))
    return [
        {
            "region_label": label,
            "top_window_count": count,
            "observed_minus_null_sum": delta_sums[label],
        }
        for label, count in counts.most_common()
    ]


def build_catalog(
    report: dict[str, Any],
    *,
    top_windows: int = 100,
    cluster_gap_hours: float = 6.0,
) -> dict[str, Any]:
    locations = _platform_locations(report)
    windows = _top_windows(report, locations=locations, top=top_windows)
    summary = report.get("summary", {})
    return {
        "source": {
            "parameters": report.get("parameters", {}),
            "summary": {
                "windows_run": summary.get("windows_run"),
                "observed_minus_null_total": summary.get("observed_minus_null_total"),
                "observed_total_z": summary.get("observed_total_z"),
                "null_total_unique_repeats": summary.get("null_total_unique_repeats"),
                "null_total_unique_exceedances": summary.get("null_total_unique_exceedances"),
                "null_total_unique_p_ge_observed": summary.get("null_total_unique_p_ge_observed"),
            },
        },
        "interpretation_constraints": {
            "catalog_role": "post-hoc only; these descriptors are not detector inputs",
            "known_controls": [
                "30-minute high-pass made tide/slow forcing unlikely for aggregate signal",
                "dominant local FFT-bin masking reduced but did not erase aggregate signal",
                "spatial/directional surrogate controls did not support a strong propagation/geography claim",
            ],
        },
        "top_windows": windows,
        "recurring_combinations": _recurring_combinations(windows, top=25),
        "time_clusters": _time_clusters(windows, gap_hours=cluster_gap_hours)[:25],
        "region_summary": _region_summary(windows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Merged CDIP batch report JSON")
    parser.add_argument("--top-windows", type=int, default=100)
    parser.add_argument("--cluster-gap-hours", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = json.loads(args.report.read_text())
    catalog = build_catalog(
        report,
        top_windows=args.top_windows,
        cluster_gap_hours=args.cluster_gap_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "top_windows": len(catalog["top_windows"]),
                "time_clusters": len(catalog["time_clusters"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
