"""Attribute frozen CHB-MIT relationship steps to channels and labels."""

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

from experiments.chbmit_autotune import (  # noqa: E402
    downsample_and_standardize,
    load_edf_signals,
    parse_chbmit_summary,
)
from spectral_forecast.relationship_dynamics import (  # noqa: E402
    describe_step_change,
    window_relationship_graphs,
)


def seizure_distance(point: float, intervals: Sequence[tuple[float, float]]) -> float | None:
    if not intervals:
        return None
    distances = []
    for start, end in intervals:
        if start <= point <= end:
            return 0.0
        distances.append(min(abs(point - start), abs(point - end)))
    return min(distances)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    parameters = discovery["parameters"]
    attributions = []
    top_three_counts: Counter[str] = Counter()
    for recording in discovery["recordings"]:
        exact = next(
            row
            for row in recording["result"]["evidence"]
            if row["mode"] == "step" and row["metric"] == "exact_identity"
        )
        if not exact["detected_fdr_by"]:
            continue
        path = Path(recording["path"])
        channels = tuple(str(name) for name in recording["channels"])
        signals = load_edf_signals(path, channels)
        series = [
            downsample_and_standardize(
                signal.values,
                source_rate=signal.sample_rate,
                target_rate=float(parameters["target_sample_rate"]),
            )
            for signal in signals
        ]
        sample_count = min(len(values) for values in series)
        duration_limit = float(parameters["max_duration_seconds"])
        if duration_limit > 0.0:
            sample_count = min(
                sample_count,
                int(duration_limit * float(parameters["target_sample_rate"])),
            )
        matrix = np.column_stack([values[:sample_count] for values in series])
        sequence = window_relationship_graphs(
            matrix,
            channels,
            family=str(parameters["relationship_family"]),
            window_size=int(parameters["relationship_window"]),
            stride=int(parameters["relationship_stride"]),
            sample_rate=float(parameters["target_sample_rate"]),
        )
        description = describe_step_change(
            sequence,
            int(exact["attributes"]["best_split_graph_index"]),
            top_k=args.top_k,
        )
        top_three_counts.update(
            row["entity"] for row in description["top_changed_entities"][:3]
        )
        summary_path = path.parent / f"{path.parent.name}-summary.txt"
        labels = parse_chbmit_summary(summary_path).get(path.name)
        intervals = labels.seizures if labels is not None else ()
        split_seconds = description["split_anchor"] / float(
            parameters["target_sample_rate"]
        )
        attributions.append(
            {
                "path": str(path),
                "entities": list(sequence.entities),
                "split_seconds": split_seconds,
                "seizures": [list(interval) for interval in intervals],
                "nearest_seizure_distance_seconds": seizure_distance(
                    split_seconds, intervals
                ),
                "description": description,
            }
        )
    return {
        "method": {
            "discovery_labels_used": False,
            "attribution_labels_used_post_hoc": True,
            "split_locations": "frozen from discovery artifact",
        },
        "discovery": str(args.discovery),
        "detected_recordings": len(attributions),
        "top_three_channel_counts": [
            {"channel": channel, "recordings": count}
            for channel, count in top_three_counts.most_common()
        ],
        "attributions": attributions,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"detected recordings={report['detected_recordings']}")
    for row in report["attributions"]:
        channels = ",".join(
            item["entity"] for item in row["description"]["top_changed_entities"][:3]
        )
        print(
            f"  {row['path']}: split={row['split_seconds']:.1f}s "
            f"nearest_seizure={row['nearest_seizure_distance_seconds']} "
            f"top_channels={channels} "
            f"top3_fraction={row['description']['top_three_entity_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
