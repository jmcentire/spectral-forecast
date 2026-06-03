"""Run label-free autotuning on numeric CSV time-series columns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_forecast.autotune import AutoTuneConfig, tune_observation
from spectral_forecast.observation import load_csv_series


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_configs(args: argparse.Namespace) -> list[AutoTuneConfig] | None:
    if not any(
        [
            args.baselines,
            args.adaptive_windows,
            args.strides,
            args.thresholds,
            args.decays,
            args.min_active_series,
        ]
    ):
        return None

    baselines = _parse_int_list(args.baselines) if args.baselines else [args.baseline]
    adaptive_windows = (
        _parse_int_list(args.adaptive_windows) if args.adaptive_windows else [args.adaptive_window]
    )
    strides = _parse_int_list(args.strides) if args.strides else [args.stride]
    thresholds = _parse_float_list(args.thresholds) if args.thresholds else [args.emission_threshold]
    decays = _parse_float_list(args.decays) if args.decays else [args.decay]
    min_active = (
        _parse_int_list(args.min_active_series)
        if args.min_active_series
        else [args.min_active]
    )

    configs = []
    for baseline in baselines:
        for adaptive in adaptive_windows:
            for stride in strides:
                for threshold in thresholds:
                    for decay in decays:
                        for active in min_active:
                            configs.append(
                                AutoTuneConfig(
                                    baseline_size=baseline,
                                    adaptive_window=adaptive,
                                    stride=stride,
                                    emission_threshold=threshold,
                                    decay=decay,
                                    min_active_series=active,
                                )
                            )
    return configs


def print_report(report: dict[str, object], *, top: int) -> None:
    best = report.get("best")
    print("Autotune observation")
    if not best:
        print("  no valid configs")
        return
    best = dict(best)  # type: ignore[arg-type]
    config = best["config"]
    null = best["null_summary"]
    print("  best_quality=%.4f" % best["quality"])
    print("  accepted=%s" % best["accepted"])
    print(
        "  config baseline=%d adaptive=%d stride=%d threshold=%.3f decay=%.3f min_active=%d"
        % (
            config["baseline_size"],
            config["adaptive_window"],
            config["stride"],
            config["emission_threshold"],
            config["decay"],
            config["min_active_series"],
        )
    )
    print(
        "  null observed=%.3f null_mean=%.3f z=%s p_ge=%.4f exceed=%d/%d unique=%d"
        % (
            null["observed_total"],
            null["null_mean"],
            "None" if null["z_effect"] is None else "%.2f" % null["z_effect"],
            null["empirical_p_ge_observed"],
            null["null_exceedances"],
            null["null_repeats"],
            null["unique_null_totals"],
        )
    )
    print(
        "  terms readiness=%.3f lift=%.3f stability=%.3f compression=%.3f residual=%.3f saturation=%.3f fragility=%.3f"
        % (
            best["readiness_score"],
            best["null_lift_score"],
            best["stability_score"],
            best["compression_score"],
            best["residual_activity_score"],
            best["saturation_penalty"],
            best["fragility_penalty"],
        )
    )
    scores = list(report.get("scores", []))[:top]  # type: ignore[arg-type]
    if len(scores) > 1:
        print("  top configs")
        for row in scores:
            cfg = row["config"]
            print(
                "    quality=%.4f baseline=%d adaptive=%d threshold=%.3f min_active=%d saturation=%.3f"
                % (
                    row["quality"],
                    cfg["baseline_size"],
                    cfg["adaptive_window"],
                    cfg["emission_threshold"],
                    cfg["min_active_series"],
                    row["saturation_penalty"],
                )
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="CSV file")
    parser.add_argument("--columns", nargs="+", default=None, help="Numeric columns to tune over")
    parser.add_argument("--all-numeric", action="store_true", help="Use every numeric column")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit samples per series")
    parser.add_argument("--sample-offset", type=int, default=0, help="Offset before sample limiting")
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")

    parser.add_argument("--baseline", type=int, default=128)
    parser.add_argument("--adaptive-window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--emission-threshold", type=float, default=3.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--min-active", type=int, default=2)

    parser.add_argument("--baselines", default=None, help="Comma-separated baseline grid")
    parser.add_argument("--adaptive-windows", default=None, help="Comma-separated adaptive window grid")
    parser.add_argument("--strides", default=None, help="Comma-separated stride grid")
    parser.add_argument("--thresholds", default=None, help="Comma-separated threshold grid")
    parser.add_argument("--decays", default=None, help="Comma-separated decay grid")
    parser.add_argument("--min-active-series", default=None, help="Comma-separated min-active grid")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    series = load_csv_series(args.file, columns=args.columns, all_numeric=args.all_numeric)
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be >= 0")
    if args.sample_limit is not None and args.sample_limit < 1:
        raise ValueError("--sample-limit must be >= 1")
    if args.sample_offset or args.sample_limit is not None:
        start = args.sample_offset
        end = None if args.sample_limit is None else start + args.sample_limit
        series = {name: values[start:end] for name, values in series.items()}
    configs = build_configs(args)
    result = tune_observation(
        series,
        configs=configs,
        sample_rate=args.sample_rate,
        null_repeats=args.null_repeats,
        seed=args.seed,
    )
    report = result.to_dict()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, top=args.top)


if __name__ == "__main__":
    main()
