"""Audit label-free relationship dynamics in CHB-MIT EEG recordings."""

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
    DEFAULT_CHANNELS,
    downsample_and_standardize,
    load_edf_signals,
)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", action="append", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--target-sample-rate", type=float, default=32.0)
    parser.add_argument("--max-duration-seconds", type=float, default=0.0)
    parser.add_argument("--relationship-family", default="correlation")
    parser.add_argument("--relationship-window", type=int, default=1024)
    parser.add_argument("--relationship-stride", type=int, default=512)
    parser.add_argument("--min-segment-windows", type=int, default=12)
    parser.add_argument("--null-repeats", type=int, default=999)
    parser.add_argument("--minimum-recording-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    recordings = []
    counts: Counter[tuple[str, str]] = Counter()
    for recording_index, path in enumerate(args.edf):
        signals = load_edf_signals(path, args.channels)
        names = tuple(signal.label for signal in signals)
        series = [
            downsample_and_standardize(
                signal.values,
                source_rate=signal.sample_rate,
                target_rate=args.target_sample_rate,
            )
            for signal in signals
        ]
        sample_count = min(len(values) for values in series)
        if args.max_duration_seconds > 0.0:
            sample_count = min(
                sample_count,
                int(args.max_duration_seconds * args.target_sample_rate),
            )
        matrix = np.column_stack([values[:sample_count] for values in series])
        sequence = window_relationship_graphs(
            matrix,
            names,
            family=args.relationship_family,
            window_size=args.relationship_window,
            stride=args.relationship_stride,
            sample_rate=args.target_sample_rate,
        )
        result = analyze_relationship_dynamics(
            sequence,
            min_segment_windows=args.min_segment_windows,
            null_repeats=args.null_repeats,
            null_block_size=None,
            seed=args.seed + recording_index * 100000,
        ).to_dict()
        detected = sorted(detected_keys(result))
        counts.update(detected)
        recordings.append(
            {
                "path": str(path),
                "channels": list(names),
                "samples": sample_count,
                "duration_seconds": sample_count / args.target_sample_rate,
                "detected_mode_metrics": [
                    {"mode": mode, "metric": metric} for mode, metric in detected
                ],
                "result": result,
            }
        )
    required = int(np.ceil(len(recordings) * args.minimum_recording_fraction))
    recurring = sorted(key for key, count in counts.items() if count >= required)
    return {
        "method": {
            "label_use": "none",
            "surface": "downsampled robust-standardized raw EEG channels",
            "unit_of_replication": "recording from a different subject",
            "within_recording_multiplicity": (
                "Benjamini-Yekutieli across step and drift by exact identity, "
                "structural role, and global motif"
            ),
            "null": "unique block permutations with graph autocorrelation block estimate",
            "recurrence_gate": args.minimum_recording_fraction,
            "required_recordings": required,
        },
        "parameters": {
            "target_sample_rate": args.target_sample_rate,
            "max_duration_seconds": args.max_duration_seconds,
            "relationship_family": args.relationship_family,
            "relationship_window": args.relationship_window,
            "relationship_stride": args.relationship_stride,
            "min_segment_windows": args.min_segment_windows,
            "null_repeats": args.null_repeats,
            "seed": args.seed,
        },
        "recordings": recordings,
        "detected_recording_counts": [
            {
                "mode": mode,
                "metric": metric,
                "recordings": counts[(mode, metric)],
                "fraction": counts[(mode, metric)] / len(recordings),
            }
            for mode in ("step", "drift")
            for metric in ("exact_identity", "structural_role", "global_motif")
        ],
        "recurring_mode_metrics": [
            {"mode": mode, "metric": metric} for mode, metric in recurring
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
    print("CHB-MIT relationship dynamics")
    for recording in report["recordings"]:
        detected = ",".join(
            f"{row['mode']}:{row['metric']}"
            for row in recording["detected_mode_metrics"]
        )
        result = recording["result"]
        print(
            f"  {recording['path']}: graphs={result['graph_windows']} "
            f"block={result['null_block_size']} nulls={result['null_repeats']} "
            f"detected={detected or 'none'}"
        )
    recurring = ",".join(
        f"{row['mode']}:{row['metric']}" for row in report["recurring_mode_metrics"]
    )
    print(f"  recurring={recurring or 'none'}")


if __name__ == "__main__":
    main()
