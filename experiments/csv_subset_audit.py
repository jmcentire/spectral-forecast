"""Audit which exact series subsets carry a frozen CSV observation result."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneObservation,
    build_autotune_observation,
    score_autotune_observation,
)
from spectral_forecast.observation import load_csv_series


def apply_by_fdr(rows: list[dict[str, Any]]) -> None:
    """Apply Benjamini-Yekutieli correction to empirical upper-tail p-values."""

    ordered = sorted(
        (float(row["empirical_p_ge_observed"]), index)
        for index, row in enumerate(rows)
    )
    total = len(ordered)
    harmonic = sum(1.0 / rank for rank in range(1, total + 1))
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank in range(total, 0, -1):
        p_value, index = ordered[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running * harmonic)
    for index, row in enumerate(rows):
        row["fdr_by_q_value"] = adjusted[index]
        row["detected_fdr_by"] = bool(
            adjusted[index] <= 0.05 and float(row["observed_minus_null"]) > 0.0
        )


def replicated_subset_names(segments: dict[str, list[dict[str, Any]]]) -> list[list[str]]:
    detected_sets = [
        {
            tuple(str(name) for name in row["subset"])
            for row in rows
            if row["detected_fdr_by"]
        }
        for rows in segments.values()
    ]
    if not detected_sets:
        return []
    return [list(names) for names in sorted(set.intersection(*detected_sets))]


def audit_segment(
    series: dict[str, np.ndarray],
    config: AutoTuneConfig,
    *,
    subset_size: int,
    sample_rate: float,
    null_mode: str,
    null_block_size: int,
    null_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Score every exact subset against the same frozen observation geometry."""

    names = list(series)
    if subset_size < 2 or subset_size > len(names):
        raise ValueError("subset size must be between 2 and the number of series")
    if config.min_active_series > subset_size:
        raise ValueError("frozen min_active_series cannot exceed subset size")
    observation = build_autotune_observation(series, config, sample_rate=sample_rate)
    rows = []
    for subset_index, indices in enumerate(itertools.combinations(range(len(names)), subset_size)):
        subset_names = [names[index] for index in indices]
        subset_observation = AutoTuneObservation(
            anchors=observation.anchors,
            matrix=observation.matrix[:, indices],
            readiness_score=observation.readiness_score,
            series_count=subset_size,
        )
        score = score_autotune_observation(
            subset_observation,
            config,
            null_repeats=null_repeats,
            seed=seed + subset_index,
            null_mode=null_mode,  # type: ignore[arg-type]
            null_block_size=null_block_size,
        )
        null = score.null_summary
        rows.append(
            {
                "subset": subset_names,
                "observed_total": null.observed_total,
                "null_mean": null.null_mean,
                "observed_minus_null": null.observed_minus_null,
                "z_effect": null.z_effect,
                "null_exceedances": null.null_exceedances,
                "null_repeats": null.null_repeats,
                "empirical_p_ge_observed": null.empirical_p_ge_observed,
                "empirical_p_floor": null.empirical_p_floor,
                "unique_null_totals": null.unique_null_totals,
                "active_windows": null.observed_active_windows,
            }
        )
    apply_by_fdr(rows)
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["detected_fdr_by"]),
            float(row["fdr_by_q_value"]),
            -float(row["observed_minus_null"]),
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--all-numeric", action="store_true")
    parser.add_argument("--columns", nargs="+", default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--baseline", type=int, required=True)
    parser.add_argument("--adaptive-window", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--emission-threshold", type=float, required=True)
    parser.add_argument("--decay", type=float, required=True)
    parser.add_argument("--min-active", type=int, required=True)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument(
        "--null-mode",
        choices=["permute", "shift", "block-permute"],
        default="block-permute",
    )
    parser.add_argument("--null-block-size", type=int, default=8)
    parser.add_argument("--null-repeats", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = AutoTuneConfig(
        baseline_size=args.baseline,
        adaptive_window=args.adaptive_window,
        stride=args.stride,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
        min_active_series=args.min_active,
    )
    config.validate()
    subset_size = args.subset_size or args.min_active
    segment_rows: dict[str, list[dict[str, Any]]] = {}
    segment_metadata = []
    for path_index, path in enumerate(args.files):
        series = load_csv_series(path, columns=args.columns, all_numeric=args.all_numeric)
        start = args.sample_offset
        end = None if args.sample_limit is None else start + args.sample_limit
        series = {name: values[start:end] for name, values in series.items()}
        segment_name = path.stem
        if segment_name in segment_rows:
            segment_name = f"{segment_name}_{path_index}"
        rows = audit_segment(
            series,
            config,
            subset_size=subset_size,
            sample_rate=args.sample_rate,
            null_mode=args.null_mode,
            null_block_size=args.null_block_size,
            null_repeats=args.null_repeats,
            seed=args.seed + 100000 * path_index,
        )
        segment_rows[segment_name] = rows
        segment_metadata.append(
            {
                "name": segment_name,
                "file": str(path),
                "series": list(series),
                "samples": min(len(values) for values in series.values()),
                "tested_subsets": len(rows),
                "detected_subsets": sum(bool(row["detected_fdr_by"]) for row in rows),
            }
        )
    replicated = replicated_subset_names(segment_rows)
    report = {
        "method": {
            "selection": "enumerate exact frozen-config series subsets",
            "null": args.null_mode,
            "multiplicity": "Benjamini-Yekutieli within each segment",
            "replication": "exact subset must survive correction in every supplied segment",
        },
        "parameters": {
            "config": config.to_dict(),
            "subset_size": subset_size,
            "sample_offset": args.sample_offset,
            "sample_limit": args.sample_limit,
            "sample_rate": args.sample_rate,
            "null_mode": args.null_mode,
            "null_block_size": args.null_block_size,
            "null_repeats": args.null_repeats,
        },
        "segments": segment_metadata,
        "segment_results": segment_rows,
        "replicated_subsets": replicated,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CSV subset audit")
    for segment in segment_metadata:
        print(
            "  %s tested=%d detected_by=%d"
            % (segment["name"], segment["tested_subsets"], segment["detected_subsets"])
        )
        for row in segment_rows[str(segment["name"])][: args.top]:
            print(
                "    subset=%s delta=%.3f z=%s p=%.4f q=%.4f detected=%s"
                % (
                    ",".join(row["subset"]),
                    row["observed_minus_null"],
                    "None" if row["z_effect"] is None else "%.2f" % row["z_effect"],
                    row["empirical_p_ge_observed"],
                    row["fdr_by_q_value"],
                    row["detected_fdr_by"],
                )
            )
    print("  replicated exact subsets=%d" % len(replicated))


if __name__ == "__main__":
    main()
