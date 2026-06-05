"""Run label-free autotuning on numeric CSV time-series columns."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_forecast.autotune import (
    AutoTuneConfig,
    build_autotune_observation,
    default_autotune_configs,
    tune_observation,
)
from spectral_forecast.observation import load_csv_series

DEFAULT_BASELINE = 128
DEFAULT_ADAPTIVE_WINDOW = 64
DEFAULT_STRIDE = 16
DEFAULT_EMISSION_THRESHOLD = 3.0
DEFAULT_DECAY = 0.9
DEFAULT_MIN_ACTIVE = 2


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_configs(args: argparse.Namespace) -> list[AutoTuneConfig] | None:
    if not any(
        [
            args.single_config,
            args.baselines,
            args.adaptive_windows,
            args.strides,
            args.thresholds,
            args.decays,
            args.min_active_series,
            args.baseline != DEFAULT_BASELINE,
            args.adaptive_window != DEFAULT_ADAPTIVE_WINDOW,
            args.stride != DEFAULT_STRIDE,
            args.emission_threshold != DEFAULT_EMISSION_THRESHOLD,
            args.decay != DEFAULT_DECAY,
            args.min_active != DEFAULT_MIN_ACTIVE,
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


def derive_quantile_thresholds(
    series: dict[str, np.ndarray],
    configs: Sequence[AutoTuneConfig],
    *,
    quantiles: Sequence[float],
    sample_rate: float,
) -> list[float]:
    """Derive fixed numeric thresholds from calibration-only score quantiles."""

    if not quantiles or any(not 0.0 < quantile < 1.0 for quantile in quantiles):
        raise ValueError("threshold quantiles must all be in (0, 1)")
    geometries: dict[tuple[object, ...], AutoTuneConfig] = {}
    for config in configs:
        key = (
            config.baseline_size,
            config.adaptive_window,
            config.stride,
            config.score,
        )
        geometries.setdefault(key, config)
    values = []
    for config in geometries.values():
        observation = build_autotune_observation(series, config, sample_rate=sample_rate)
        finite = observation.matrix[np.isfinite(observation.matrix)]
        if len(finite):
            values.append(finite)
    if not values:
        raise ValueError("no finite observer scores available for threshold quantiles")
    pooled = np.concatenate(values)
    return sorted(set(float(value) for value in np.quantile(pooled, quantiles)))


def expand_threshold_grid(
    configs: Sequence[AutoTuneConfig],
    thresholds: Sequence[float],
) -> list[AutoTuneConfig]:
    """Replace candidate thresholds and remove duplicate full configurations."""

    expanded: dict[tuple[object, ...], AutoTuneConfig] = {}
    for config in configs:
        for threshold in thresholds:
            candidate = replace(config, emission_threshold=float(threshold))
            key = tuple(candidate.to_dict().values())
            expanded.setdefault(key, candidate)
    return list(expanded.values())


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
    parser.add_argument(
        "--null-mode",
        choices=["permute", "shift", "block-permute"],
        default="permute",
    )
    parser.add_argument("--null-block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")

    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Score exactly the singular configuration flags instead of the default grid",
    )
    parser.add_argument("--baseline", type=int, default=DEFAULT_BASELINE)
    parser.add_argument("--adaptive-window", type=int, default=DEFAULT_ADAPTIVE_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--emission-threshold", type=float, default=DEFAULT_EMISSION_THRESHOLD)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--min-active", type=int, default=DEFAULT_MIN_ACTIVE)

    parser.add_argument("--baselines", default=None, help="Comma-separated baseline grid")
    parser.add_argument("--adaptive-windows", default=None, help="Comma-separated adaptive window grid")
    parser.add_argument("--strides", default=None, help="Comma-separated stride grid")
    parser.add_argument("--thresholds", default=None, help="Comma-separated threshold grid")
    parser.add_argument(
        "--threshold-quantiles",
        default=None,
        help="Comma-separated calibration score quantiles used as a label-free threshold grid",
    )
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
    derived_thresholds = None
    if args.threshold_quantiles:
        if args.thresholds:
            raise ValueError("--threshold-quantiles and --thresholds are mutually exclusive")
        if configs is None:
            common_length = min(len(values) for values in series.values())
            configs = default_autotune_configs(common_length)
        derived_thresholds = derive_quantile_thresholds(
            series,
            configs,
            quantiles=_parse_float_list(args.threshold_quantiles),
            sample_rate=args.sample_rate,
        )
        configs = expand_threshold_grid(configs, derived_thresholds)
    result = tune_observation(
        series,
        configs=configs,
        sample_rate=args.sample_rate,
        null_repeats=args.null_repeats,
        seed=args.seed,
        null_mode=args.null_mode,
        null_block_size=args.null_block_size,
    )
    report = result.to_dict()
    report["parameters"] = {
        "file": str(args.file),
        "columns": args.columns,
        "all_numeric": args.all_numeric,
        "sample_rate": args.sample_rate,
        "sample_offset": args.sample_offset,
        "sample_limit": args.sample_limit,
        "null_mode": args.null_mode,
        "null_block_size": args.null_block_size,
        "null_repeats": args.null_repeats,
        "threshold_quantiles": (
            _parse_float_list(args.threshold_quantiles) if args.threshold_quantiles else None
        ),
        "derived_thresholds": derived_thresholds,
        "candidate_count": len(configs) if configs is not None else None,
        "single_config": args.single_config,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, top=args.top)


if __name__ == "__main__":
    main()
