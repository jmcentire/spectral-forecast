"""Audit recurring relationship dynamics across synthetic PMU events."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from spectral_forecast.relationship_dynamics import (  # noqa: E402
    analyze_relationship_dynamics,
    window_relationship_graphs,
)


def detected_keys(result: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(row["mode"]), str(row["metric"]))
        for row in result["evidence"]
        if row["detected_fdr_by"]
    }


def recurring_keys(
    results: Sequence[dict[str, Any]], minimum_fraction: float
) -> list[tuple[str, str]]:
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be in (0, 1]")
    counts = Counter(key for result in results for key in detected_keys(result))
    required = int(np.ceil(len(results) * minimum_fraction))
    return sorted(key for key, count in counts.items() if count >= required)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", action="append", type=Path, required=True)
    parser.add_argument("--datatype-index", type=int, required=True)
    parser.add_argument("--sensor-limit", type=int, default=24)
    parser.add_argument("--event-limit", type=int, default=0)
    parser.add_argument("--relationship-family", default="correlation")
    parser.add_argument("--relationship-window", type=int, default=64)
    parser.add_argument("--relationship-stride", type=int, default=16)
    parser.add_argument("--min-segment-windows", type=int, default=6)
    parser.add_argument("--null-repeats", type=int, default=999)
    parser.add_argument("--minimum-event-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpora = []
    recurring_sets = []
    for corpus_index, path in enumerate(args.tensor):
        tensor = np.load(path, mmap_mode="r")
        if tensor.ndim != 4:
            raise ValueError(f"expected a four-dimensional PMU tensor: {path}")
        if not 0 <= args.datatype_index < tensor.shape[1]:
            raise ValueError(f"datatype index outside tensor axis: {path}")
        sensor_count = min(args.sensor_limit, tensor.shape[2])
        event_count = tensor.shape[0]
        if args.event_limit > 0:
            event_count = min(event_count, args.event_limit)
        names = tuple(f"pmu_{index:03d}" for index in range(sensor_count))
        events = []
        results = []
        for event_index in range(event_count):
            matrix = np.asarray(
                tensor[event_index, args.datatype_index, :sensor_count, :],
                dtype=np.float64,
            ).T
            sequence = window_relationship_graphs(
                matrix,
                names,
                family=args.relationship_family,
                window_size=args.relationship_window,
                stride=args.relationship_stride,
            )
            result = analyze_relationship_dynamics(
                sequence,
                min_segment_windows=args.min_segment_windows,
                null_repeats=args.null_repeats,
                null_block_size=None,
                seed=args.seed + corpus_index * 100000 + event_index * 1000,
            ).to_dict()
            results.append(result)
            events.append(
                {
                    "event_index": event_index,
                    "detected_mode_metrics": [
                        {"mode": mode, "metric": metric}
                        for mode, metric in sorted(detected_keys(result))
                    ],
                    "result": result,
                }
            )
        recurring = recurring_keys(results, args.minimum_event_fraction)
        recurring_sets.append(set(recurring))
        counts = Counter(key for result in results for key in detected_keys(result))
        corpora.append(
            {
                "path": str(path),
                "tensor_shape": [int(value) for value in tensor.shape],
                "events_analyzed": event_count,
                "sensors": sensor_count,
                "samples_per_event": int(tensor.shape[3]),
                "detected_event_counts": [
                    {
                        "mode": mode,
                        "metric": metric,
                        "events": counts[(mode, metric)],
                        "fraction": counts[(mode, metric)] / event_count,
                    }
                    for mode in ("step", "drift")
                    for metric in ("exact_identity", "structural_role", "global_motif")
                ],
                "recurring_mode_metrics": [
                    {"mode": mode, "metric": metric} for mode, metric in recurring
                ],
                "events": events,
            }
        )
    replicated = (
        sorted(set.intersection(*recurring_sets)) if len(recurring_sets) >= 2 else []
    )
    return {
        "method": {
            "label_use": "none",
            "unit_of_replication": "independent synthetic PMU event",
            "within_event_multiplicity": (
                "Benjamini-Yekutieli across step and drift by exact identity, "
                "structural role, and global motif"
            ),
            "null": "unique block permutations with graph autocorrelation block estimate",
            "recurrence_gate": args.minimum_event_fraction,
        },
        "parameters": {
            "datatype_index": args.datatype_index,
            "sensor_limit": args.sensor_limit,
            "relationship_family": args.relationship_family,
            "relationship_window": args.relationship_window,
            "relationship_stride": args.relationship_stride,
            "min_segment_windows": args.min_segment_windows,
            "null_repeats": args.null_repeats,
            "seed": args.seed,
        },
        "corpora": corpora,
        "replicated_recurring_mode_metrics": [
            {"mode": mode, "metric": metric} for mode, metric in replicated
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("pmuBAGE relationship dynamics")
    for corpus in report["corpora"]:
        counts = ", ".join(
            f"{row['mode']}:{row['metric']}={row['events']}/{corpus['events_analyzed']}"
            for row in corpus["detected_event_counts"]
        )
        recurring = ",".join(
            f"{row['mode']}:{row['metric']}"
            for row in corpus["recurring_mode_metrics"]
        )
        print(f"  {corpus['path']}: {counts}")
        print(f"    recurring={recurring or 'none'}")
    replicated = ",".join(
        f"{row['mode']}:{row['metric']}"
        for row in report["replicated_recurring_mode_metrics"]
    )
    print(f"  replicated recurring={replicated or 'none'}")


if __name__ == "__main__":
    main()
