"""Derive deterministic CDIP window manifest subsets.

This utility only filters already-discovered windows. It does not change
detector inputs, scores, nulls, or preprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cdip_spatial_audit import load_platform_location


def _window_key(row: dict[str, Any]) -> tuple[tuple[str, ...], float, int, int]:
    return (
        tuple(str(path) for path in row["source_paths"]),
        float(row["start_time"]),
        int(row["group_index"]),
        int(row["window_index"]),
    )


def _parse_time_arg(value: str | None) -> float | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _platform_from_path(path: str) -> str:
    return Path(path).name.split("_", 1)[0]


def _group_region_label(regions: Sequence[str]) -> str:
    unique = sorted(set(regions))
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + "+".join(unique)


def _row_regions(row: dict[str, Any], region_lookup: Mapping[str, str] | None) -> list[str]:
    if region_lookup is None:
        return []
    regions = []
    for raw_path in row.get("source_paths", []):
        path = str(raw_path)
        platform = _platform_from_path(path)
        region = (
            region_lookup.get(path)
            or region_lookup.get(Path(path).name)
            or region_lookup.get(platform)
        )
        if region is not None:
            regions.append(str(region))
    return regions


def _matches_regions(
    row: dict[str, Any],
    *,
    include_regions: set[str],
    region_mode: str,
    region_lookup: Mapping[str, str] | None,
) -> bool:
    if not include_regions:
        return True
    regions = _row_regions(row, region_lookup)
    if not regions:
        return False
    if region_mode == "any":
        return any(region in include_regions for region in regions)
    if region_mode == "all":
        return all(region in include_regions for region in regions)
    if region_mode == "label":
        return _group_region_label(regions) in include_regions
    raise ValueError("region_mode must be any, all, or label")


def build_region_lookup(windows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Load platform regions for manifest source paths."""

    lookup: dict[str, str] = {}
    for row in windows:
        for raw_path in row.get("source_paths", []):
            path = Path(str(raw_path))
            key = str(raw_path)
            if key in lookup:
                continue
            if not path.exists():
                continue
            location = load_platform_location(path)
            lookup[key] = location.region
            lookup[path.name] = location.region
            lookup[location.platform_id] = location.region
    return lookup


def filter_manifest_windows(
    windows: Sequence[dict[str, Any]],
    *,
    group_start: int | None = None,
    group_end: int | None = None,
    parity: str = "all",
    start_time: float | None = None,
    end_time: float | None = None,
    include_regions: Sequence[str] | None = None,
    region_mode: str = "any",
    region_lookup: Mapping[str, str] | None = None,
    max_windows: int = 0,
) -> list[dict[str, Any]]:
    """Filter manifest rows by group, time, and region while preserving order."""

    if parity not in {"all", "even", "odd"}:
        raise ValueError("parity must be all, even, or odd")
    if region_mode not in {"any", "all", "label"}:
        raise ValueError("region_mode must be any, all, or label")
    region_set = {str(region) for region in include_regions or []}
    selected: list[dict[str, Any]] = []
    for row in windows:
        group_index = int(row["group_index"])
        row_start = float(row["start_time"])
        if group_start is not None and group_index < group_start:
            continue
        if group_end is not None and group_index >= group_end:
            continue
        if parity == "even" and group_index % 2 != 0:
            continue
        if parity == "odd" and group_index % 2 != 1:
            continue
        if start_time is not None and row_start < start_time:
            continue
        if end_time is not None and row_start >= end_time:
            continue
        if not _matches_regions(
            row,
            include_regions=region_set,
            region_mode=region_mode,
            region_lookup=region_lookup,
        ):
            continue
        selected.append(dict(row))
        if max_windows > 0 and len(selected) >= max_windows:
            break
    return selected


def subset_manifest(
    manifest: dict[str, Any],
    *,
    group_start: int | None = None,
    group_end: int | None = None,
    parity: str = "all",
    start_time: float | None = None,
    end_time: float | None = None,
    include_regions: Sequence[str] | None = None,
    region_mode: str = "any",
    region_lookup: Mapping[str, str] | None = None,
    max_windows: int = 0,
    label: str = "subset",
) -> dict[str, Any]:
    """Return a manifest-compatible subset with extra split metadata."""

    windows = filter_manifest_windows(
        manifest.get("windows", []),
        group_start=group_start,
        group_end=group_end,
        parity=parity,
        start_time=start_time,
        end_time=end_time,
        include_regions=include_regions,
        region_mode=region_mode,
        region_lookup=region_lookup,
        max_windows=max_windows,
    )
    groups = sorted({int(row["group_index"]) for row in windows})
    return {
        "version": manifest.get("version", 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature": manifest["signature"],
        "split": {
            "label": label,
            "source_created_at": manifest.get("created_at"),
            "source_windows": len(manifest.get("windows", [])),
            "group_start": group_start,
            "group_end": group_end,
            "parity": parity,
            "start_time": start_time,
            "end_time": end_time,
            "include_regions": sorted(str(region) for region in include_regions or []),
            "region_mode": region_mode,
            "max_windows": max_windows,
            "groups": len(groups),
            "windows": len(windows),
        },
        "windows": windows,
    }


def assert_disjoint(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> None:
    overlap = {_window_key(row) for row in left} & {_window_key(row) for row in right}
    if overlap:
        raise ValueError(f"Manifest subsets overlap on {len(overlap)} window(s)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Input CDIP window manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output subset manifest JSON")
    parser.add_argument("--label", default="subset", help="Split label written to manifest metadata")
    parser.add_argument("--group-start", type=int, default=None, help="Inclusive group_index lower bound")
    parser.add_argument("--group-end", type=int, default=None, help="Exclusive group_index upper bound")
    parser.add_argument("--parity", choices=["all", "even", "odd"], default="all")
    parser.add_argument("--start-time", default=None, help="Inclusive UTC ISO window-start lower bound")
    parser.add_argument("--end-time", default=None, help="Exclusive UTC ISO window-start upper bound")
    parser.add_argument(
        "--include-region",
        action="append",
        default=[],
        help="Coarse region to include; repeat for multiple regions",
    )
    parser.add_argument(
        "--region-mode",
        choices=["any", "all", "label"],
        default="any",
        help="How include-region is matched against each group",
    )
    parser.add_argument("--max-windows", type=int, default=0, help="Optional hard cap after filtering")
    parser.add_argument(
        "--assert-disjoint-with",
        type=Path,
        default=None,
        help="Optional manifest that must share no selected windows with the output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    region_lookup = (
        build_region_lookup(manifest.get("windows", [])) if args.include_region else None
    )
    subset = subset_manifest(
        manifest,
        group_start=args.group_start,
        group_end=args.group_end,
        parity=args.parity,
        start_time=_parse_time_arg(args.start_time),
        end_time=_parse_time_arg(args.end_time),
        include_regions=args.include_region,
        region_mode=args.region_mode,
        region_lookup=region_lookup,
        max_windows=args.max_windows,
        label=args.label,
    )
    if args.assert_disjoint_with is not None:
        other = json.loads(args.assert_disjoint_with.read_text())
        assert_disjoint(subset["windows"], other.get("windows", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "groups": subset["split"]["groups"],
                "windows": subset["split"]["windows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
