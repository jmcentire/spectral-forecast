"""CLI entry point for spectral-forecast."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from spectral_forecast.benchmark import load_csv_dataset, run_benchmark
from spectral_forecast.forecast import SpectralForecaster


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


if __name__ == "__main__":
    main()
