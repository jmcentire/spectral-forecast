"""Calibrate directional-quality power for an observed matrix geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_forecast.directional_resolution import directional_resolution_calibration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--series-count", type=int, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--null-repeats", type=int, default=24)
    parser.add_argument("--active-z-threshold", type=float, default=1.5)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--aggregate-quantile", type=float, default=0.9)
    parser.add_argument("--null-block-size", type=int, default=4)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--min-z-effect", type=float, default=2.0)
    parser.add_argument("--min-detection-rate", type=float, default=0.8)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = directional_resolution_calibration(
        n=args.observations,
        series_count=args.series_count,
        trials=args.trials,
        null_repeats=args.null_repeats,
        seed=args.seed,
        active_z_threshold=args.active_z_threshold,
        max_lag=args.max_lag,
        aggregate_quantile=args.aggregate_quantile,
        null_block_size=args.null_block_size,
        significance_level=args.significance_level,
        min_z_effect=args.min_z_effect,
        min_detection_rate=args.min_detection_rate,
        max_false_positive_rate=args.max_false_positive_rate,
    ).to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
