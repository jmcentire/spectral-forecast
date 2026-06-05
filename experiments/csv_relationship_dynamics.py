"""Run frozen relationship-dynamics analysis on numeric CSV segments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from spectral_forecast.autotune import (  # noqa: E402
    AutoTuneConfig,
    build_autotune_observation,
)
from spectral_forecast.observation import load_csv_series  # noqa: E402
from spectral_forecast.relationship_dynamics import (  # noqa: E402
    analyze_relationship_dynamics,
    window_relationship_graphs,
)


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    path: Path
    offset: int
    limit: int


def parse_segment_spec(text: str) -> SegmentSpec:
    """Parse NAME:PATH:OFFSET:LIMIT."""

    parts = text.rsplit(":", 2)
    if len(parts) != 3 or ":" not in parts[0]:
        raise ValueError("segment must be NAME:PATH:OFFSET:LIMIT")
    name, path = parts[0].split(":", 1)
    offset, limit = int(parts[1]), int(parts[2])
    if not name or offset < 0 or limit < 1:
        raise ValueError("segment name, offset, and limit must be valid")
    return SegmentSpec(name=name, path=Path(path), offset=offset, limit=limit)


def detected_keys(result: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row["mode"]), str(row["metric"]))
        for row in result["evidence"]
        if row["detected_fdr_by"]
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        help="Repeat NAME:PATH:OFFSET:LIMIT segment specification",
    )
    parser.add_argument("--columns", nargs="+", default=None)
    parser.add_argument("--all-numeric", action="store_true")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--surface", choices=["raw", "observer"], default="observer")
    parser.add_argument("--baseline", type=int, default=1024)
    parser.add_argument("--adaptive-window", type=int, default=256)
    parser.add_argument("--observer-stride", type=int, default=128)
    parser.add_argument("--emission-threshold", type=float, default=13.629695749397975)
    parser.add_argument("--decay", type=float, default=0.75)
    parser.add_argument("--min-active", type=int, default=3)
    parser.add_argument(
        "--relationship-family",
        choices=["correlation", "spectral_alignment", "lagged_dependence"],
        default="correlation",
    )
    parser.add_argument("--relationship-window", type=int, default=32)
    parser.add_argument("--relationship-stride", type=int, default=2)
    parser.add_argument("--nperseg", type=int, default=64)
    parser.add_argument("--max-lag", type=int, default=16)
    parser.add_argument("--min-segment-windows", type=int, default=4)
    parser.add_argument("--null-repeats", type=int, default=999)
    parser.add_argument(
        "--null-block-size",
        default="auto",
        help="Positive graph-window block length, or 'auto'",
    )
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = [parse_segment_spec(text) for text in args.segment]
    observer_config = AutoTuneConfig(
        baseline_size=args.baseline,
        adaptive_window=args.adaptive_window,
        stride=args.observer_stride,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
        min_active_series=args.min_active,
    )
    observer_config.validate()
    null_block_size = (
        None if args.null_block_size == "auto" else int(args.null_block_size)
    )
    if null_block_size is not None and null_block_size < 1:
        raise ValueError("null block size must be positive or 'auto'")
    segments = []
    result_sets = []
    for index, spec in enumerate(specs):
        series = load_csv_series(
            spec.path,
            columns=args.columns,
            all_numeric=args.all_numeric,
        )
        values = {
            name: column[spec.offset : spec.offset + spec.limit]
            for name, column in series.items()
        }
        lengths = {len(column) for column in values.values()}
        if lengths != {spec.limit}:
            raise ValueError(
                f"segment {spec.name!r} does not contain the requested {spec.limit} rows"
            )
        names = list(values)
        raw_matrix = np.column_stack([values[name] for name in names])
        if not np.all(np.isfinite(raw_matrix)):
            raise ValueError(f"segment {spec.name!r} contains non-finite numeric data")
        if args.surface == "observer":
            observation = build_autotune_observation(
                values,
                observer_config,
                sample_rate=args.sample_rate,
            )
            matrix = observation.matrix
            source_rows = len(raw_matrix)
            surface_rows = len(matrix)
        else:
            matrix = raw_matrix
            source_rows = surface_rows = len(matrix)
        sequence = window_relationship_graphs(
            matrix,
            names,
            family=args.relationship_family,
            window_size=args.relationship_window,
            stride=args.relationship_stride,
            sample_rate=args.sample_rate,
            nperseg=args.nperseg,
            max_lag=args.max_lag,
        )
        result = analyze_relationship_dynamics(
            sequence,
            min_segment_windows=args.min_segment_windows,
            null_repeats=args.null_repeats,
            null_block_size=null_block_size,
            seed=args.seed + 100000 * index,
        ).to_dict()
        result_sets.append(detected_keys(result))
        segments.append(
            {
                "name": spec.name,
                "path": str(spec.path),
                "offset": spec.offset,
                "limit": spec.limit,
                "source_rows": source_rows,
                "surface_rows": surface_rows,
                "result": result,
            }
        )
    replicated = sorted(set.intersection(*result_sets)) if result_sets else []
    return {
        "method": {
            "label_use": "none",
            "surface": args.surface,
            "step_control": (
                "exact-identity CUSUM selects one change point; exact, role, and "
                "motif distances are evaluated at that split against block-null "
                "rescans"
            ),
            "drift_control": (
                "absolute Spearman ordering of distance from the frozen early graph "
                "against block-permuted graph-window order"
            ),
            "multiplicity": "Benjamini-Yekutieli across six mode-metric tests per segment",
        },
        "parameters": {
            "observer_config": observer_config.to_dict(),
            "relationship_family": args.relationship_family,
            "relationship_window": args.relationship_window,
            "relationship_stride": args.relationship_stride,
            "min_segment_windows": args.min_segment_windows,
            "null_repeats": args.null_repeats,
            "null_block_size": args.null_block_size,
            "seed": args.seed,
        },
        "segments": segments,
        "replicated_mode_metrics": [
            {"mode": mode, "metric": metric} for mode, metric in replicated
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("CSV relationship dynamics")
    for segment in report["segments"]:
        detected = [
            row
            for row in segment["result"]["evidence"]
            if row["detected_fdr_by"]
        ]
        print(
            "  %s surface_rows=%d graph_windows=%d detected=%s"
            % (
                segment["name"],
                segment["surface_rows"],
                segment["result"]["graph_windows"],
                ",".join(f"{row['mode']}:{row['metric']}" for row in detected)
                or "none",
            )
        )
        for row in segment["result"]["evidence"]:
            print(
                "    %-5s %-16s observed=%.4f p=%.4f q=%.4f detected=%s"
                % (
                    row["mode"],
                    row["metric"],
                    row["observed"],
                    row["empirical_p_ge_observed"],
                    row["fdr_by_q_value"],
                    row["detected_fdr_by"],
                )
            )
    print(
        "  replicated=%s"
        % (
            ",".join(
                f"{row['mode']}:{row['metric']}"
                for row in report["replicated_mode_metrics"]
            )
            or "none"
        )
    )


if __name__ == "__main__":
    main()
