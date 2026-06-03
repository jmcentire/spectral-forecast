"""Derive deterministic CDIP window manifest subsets.

This utility only filters already-discovered windows. It does not change
detector inputs, scores, nulls, or preprocessing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _window_key(row: dict[str, Any]) -> tuple[tuple[str, ...], float, int, int]:
    return (
        tuple(str(path) for path in row["source_paths"]),
        float(row["start_time"]),
        int(row["group_index"]),
        int(row["window_index"]),
    )


def filter_manifest_windows(
    windows: Sequence[dict[str, Any]],
    *,
    group_start: int | None = None,
    group_end: int | None = None,
    parity: str = "all",
    max_windows: int = 0,
) -> list[dict[str, Any]]:
    """Filter manifest rows by group range/parity while preserving order."""

    if parity not in {"all", "even", "odd"}:
        raise ValueError("parity must be all, even, or odd")
    selected: list[dict[str, Any]] = []
    for row in windows:
        group_index = int(row["group_index"])
        if group_start is not None and group_index < group_start:
            continue
        if group_end is not None and group_index >= group_end:
            continue
        if parity == "even" and group_index % 2 != 0:
            continue
        if parity == "odd" and group_index % 2 != 1:
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
    max_windows: int = 0,
    label: str = "subset",
) -> dict[str, Any]:
    """Return a manifest-compatible subset with extra split metadata."""

    windows = filter_manifest_windows(
        manifest.get("windows", []),
        group_start=group_start,
        group_end=group_end,
        parity=parity,
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
    subset = subset_manifest(
        manifest,
        group_start=args.group_start,
        group_end=args.group_end,
        parity=args.parity,
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
