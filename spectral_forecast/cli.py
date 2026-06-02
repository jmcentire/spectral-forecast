"""CLI entry point for spectral-forecast."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from spectral_forecast.benchmark import load_csv_dataset, run_benchmark
from spectral_forecast.forecast import SpectralForecaster
from spectral_forecast.observation import (
    build_stigmergy,
    load_csv_series,
    observe_series,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="spectral-forecast",
        description="Three-layer analytical time series forecasting",
    )
    subparsers = parser.add_subparsers(dest="command")

    # forecast command
    p_forecast = subparsers.add_parser("forecast", help="Forecast from CSV data")
    p_forecast.add_argument("file", help="Path to CSV file")
    p_forecast.add_argument("--column", default="OT", help="Column to forecast (default: OT)")
    p_forecast.add_argument("--horizon", type=int, default=96, help="Steps to forecast (default: 96)")
    p_forecast.add_argument("--context", type=int, default=None, help="Context window (default: all data)")
    p_forecast.add_argument("--describe", action="store_true", help="Print decomposition details")

    # benchmark command
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark on dataset")
    p_bench.add_argument("file", help="Path to CSV dataset")
    p_bench.add_argument("--column", default="OT", help="Column to benchmark")
    p_bench.add_argument("--context", type=int, default=512, help="Context window size")
    p_bench.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
        help="Prediction lengths to test",
    )
    p_bench.add_argument("--name", default=None, help="Dataset name for report")

    # decompose command — just show what the model finds
    p_decomp = subparsers.add_parser("decompose", help="Decompose signal and show components")
    p_decomp.add_argument("file", help="Path to CSV file")
    p_decomp.add_argument("--column", default="OT", help="Column to decompose")

    # observe command — anomaly/drift/stigmergy inspection
    p_observe = subparsers.add_parser(
        "observe",
        help="Observe frozen/adaptive anomaly, drift, and stigmergy signals",
    )
    p_observe.add_argument("file", help="Path to CSV file")
    p_observe.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Columns to observe (default: OT if present, otherwise last column)",
    )
    p_observe.add_argument(
        "--all-numeric",
        action="store_true",
        help="Observe every fully numeric column",
    )
    p_observe.add_argument("--baseline", type=int, default=512, help="Frozen baseline size")
    p_observe.add_argument(
        "--adaptive-window",
        type=int,
        default=256,
        help="Past window size for adaptive scoring",
    )
    p_observe.add_argument("--stride", type=int, default=16, help="Observation stride")
    p_observe.add_argument("--sample-rate", type=float, default=1.0, help="Samples per time unit")
    p_observe.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
        help="Score used for ranking and stigmergy emission",
    )
    p_observe.add_argument(
        "--emission-threshold",
        type=float,
        default=3.0,
        help="Score above which a series emits stigmergic evidence",
    )
    p_observe.add_argument(
        "--decay",
        type=float,
        default=0.9,
        help="Stigmergy pheromone decay in [0, 1)",
    )
    p_observe.add_argument("--top", type=int, default=10, help="Rows to show per ranking")
    p_observe.add_argument(
        "--format",
        choices=["text", "jsonl"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "forecast":
        data = load_csv_dataset(args.file, column=args.column)
        if args.context is not None:
            data = data[-args.context :]

        forecaster = SpectralForecaster()
        result = forecaster.fit_forecast(data, args.horizon)

        if args.describe:
            print(result.describe())
            print()

        print(f"Point forecast ({args.horizon} steps):")
        for i, (pt, lo, hi) in enumerate(
            zip(result.point_forecast, result.lower_bound, result.upper_bound)
        ):
            print(f"  t+{i + 1:3d}: {pt:12.4f}  [{lo:12.4f}, {hi:12.4f}]")

    elif args.command == "benchmark":
        data = load_csv_dataset(args.file, column=args.column)
        name = args.name or args.file
        result = run_benchmark(
            data,
            prediction_lengths=args.horizons,
            context_length=args.context,
            dataset_name=name,
            column=args.column,
        )
        print(result.summary())

    elif args.command == "decompose":
        data = load_csv_dataset(args.file, column=args.column)
        forecaster = SpectralForecaster()
        forecaster.fit(data)
        # Use forecast(0) just to get the decomposition description
        result = forecaster.forecast(1)
        print(result.describe())
        print(f"\nResidual stats: mean={np.mean(forecaster._residual):.6f} "
              f"std={result.noise_std:.6f}")

    elif args.command == "observe":
        series_map = load_csv_series(
            args.file,
            columns=args.columns,
            all_numeric=args.all_numeric,
        )
        results = [
            observe_series(
                values,
                series=name,
                baseline_size=args.baseline,
                adaptive_window=args.adaptive_window,
                stride=args.stride,
                sample_rate=args.sample_rate,
            )
            for name, values in series_map.items()
        ]
        stigmergy = build_stigmergy(
            results,
            score=args.score,
            emission_threshold=args.emission_threshold,
            decay=args.decay,
        )

        if args.format == "jsonl":
            for result in results:
                for point in result.points:
                    row = point.to_dict()
                    row["kind"] = "observation"
                    print(json.dumps(row, sort_keys=True))
            for point in stigmergy.points:
                row = point.to_dict()
                row["kind"] = "stigmergy"
                print(json.dumps(row, sort_keys=True))
            return

        print("Observation")
        print(
            "  score=%s baseline=%d adaptive_window=%d stride=%d sample_rate=%.6g"
            % (args.score, args.baseline, args.adaptive_window, args.stride, args.sample_rate)
        )
        print("  scoring uses residual surprise, frozen/adaptive disagreement, and decomposition-state drift")
        for result in results:
            print("\nSeries: %s (%d points)" % (result.series, len(result.points)))
            print(
                "%8s %10s %10s %10s %10s %10s %12s"
                % ("index", "score", "frozen", "sliding", "drift", "state", "actual")
            )
            for point in result.top(args.score, args.top):
                print(
                    "%8d %10.3f %10.3f %10.3f %10.3f %10.3f %12.4f"
                    % (
                        point.index,
                        point.score(args.score),
                        point.frozen_score,
                        point.sliding_score,
                        point.conditional_drift_score,
                        point.state_drift_score,
                        point.actual,
                    )
                )

        if len(results) > 1:
            print("\nStigmergy")
            print(
                "%8s %10s %10s %8s %10s %s"
                % ("index", "pheromone", "emission", "series", "max_score", "active")
            )
            for point in stigmergy.top(args.top):
                print(
                    "%8d %10.3f %10.3f %8d %10.3f %s"
                    % (
                        point.index,
                        point.pheromone,
                        point.emission,
                        point.active_series_count,
                        point.max_score,
                        ",".join(point.active_series),
                    )
                )


if __name__ == "__main__":
    main()
