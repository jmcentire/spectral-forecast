"""Validate label-free autotune on bounded CDIP aligned buoy windows.

This script uses CDIP as a calibration domain, not as a target-label source. It
chooses observation settings from raw aligned buoy windows using the generic
autotune objective, then freezes the selected settings and evaluates the next
aligned windows as heldout validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.cdip_batch import (
    BatchWindow,
    _discover_windows,
    _series_from_records,
)
from experiments.cdip_observe import PREPROCESS_MODES, load_cdip_raw_record
from spectral_forecast.autotune import AutoTuneConfig, AutoTuneScore, score_autotune_config


@dataclass(frozen=True)
class CdipAutoTuneCandidate:
    """One CDIP candidate: preprocessing plus core autotune settings."""

    preprocess: str
    config: AutoTuneConfig

    def to_dict(self) -> dict[str, object]:
        row = {"preprocess": self.preprocess}
        row.update(self.config.to_dict())
        return row


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_candidates(args: argparse.Namespace) -> list[CdipAutoTuneCandidate]:
    candidates = []
    for preprocess in _parse_str_list(args.preprocess_grid):
        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f"Unknown preprocess mode: {preprocess}")
        for baseline in _parse_int_list(args.baselines):
            for adaptive in _parse_int_list(args.adaptive_windows):
                for stride in _parse_int_list(args.strides):
                    for threshold in _parse_float_list(args.thresholds):
                        for decay in _parse_float_list(args.decays):
                            for min_active in _parse_int_list(args.min_active_series):
                                candidates.append(
                                    CdipAutoTuneCandidate(
                                        preprocess=preprocess,
                                        config=AutoTuneConfig(
                                            baseline_size=baseline,
                                            adaptive_window=adaptive,
                                            stride=stride,
                                            emission_threshold=threshold,
                                            decay=decay,
                                            min_active_series=min_active,
                                        ),
                                    )
                                )
    return candidates


def _window_identity(window: BatchWindow) -> dict[str, object]:
    return {
        "group": list(window.platforms),
        "paths": list(window.source_paths),
        "start": _iso(window.start_time),
        "end": _iso(window.start_time + (window.n_samples - 1) / window.target_rate),
        "sample_rate": window.target_rate,
        "samples": window.n_samples,
        "group_index": window.group_index,
        "window_index": window.window_index,
    }


def _score_window(
    window: BatchWindow,
    candidate: CdipAutoTuneCandidate,
    *,
    channels: Sequence[str],
    highpass_period_minutes: float,
    mask_dominant_bins: int,
    mask_bin_radius: int,
    phase_surrogate_seed: int,
    null_repeats: int,
    seed: int,
) -> AutoTuneScore:
    series = _series_from_records(
        window.records,
        channels,
        start_time=window.start_time,
        n_samples=window.n_samples,
        target_rate=window.target_rate,
        preprocess=candidate.preprocess,
        highpass_period_seconds=highpass_period_minutes * 60.0,
        mask_dominant_bins=mask_dominant_bins,
        mask_bin_radius=mask_bin_radius,
        phase_surrogate_seed=phase_surrogate_seed,
    )
    values = {item.name: item.values for item in series}
    return score_autotune_config(
        values,
        candidate.config,
        sample_rate=window.target_rate,
        null_repeats=null_repeats,
        seed=seed,
    )


def _aggregate_scores(
    candidate: CdipAutoTuneCandidate,
    scores: Sequence[AutoTuneScore],
    *,
    skipped: Sequence[dict[str, object]],
    min_accepted_fraction: float,
    min_z_effect: float,
) -> dict[str, Any]:
    if not scores:
        return {
            "candidate": candidate.to_dict(),
            "accepted": False,
            "windows_scored": 0,
            "windows_skipped": len(skipped),
            "quality": float("-inf"),
            "skipped": list(skipped)[:5],
        }

    observed = float(sum(score.null_summary.observed_total for score in scores))
    null_mean = float(sum(score.null_summary.null_mean for score in scores))
    variance = float(sum(score.null_summary.null_std**2 for score in scores))
    z_effect = (observed - null_mean) / math.sqrt(variance) if variance > 0 else None
    accepted_windows = sum(1 for score in scores if score.accepted)
    accepted_fraction = accepted_windows / len(scores)
    mean_quality = float(np.mean([score.quality for score in scores]))
    mean_saturation = float(np.mean([score.saturation_penalty for score in scores]))
    mean_lift = float(np.mean([score.null_lift_score for score in scores]))
    mean_readiness = float(np.mean([score.readiness_score for score in scores]))
    quality = mean_quality + 0.10 * accepted_fraction
    aggregate_z = z_effect if z_effect is not None else 0.0
    accepted = (
        quality > 0.0
        and observed > null_mean
        and accepted_fraction >= min_accepted_fraction
        and aggregate_z >= min_z_effect
        and mean_saturation < 1.0
        and mean_lift > 0.0
    )

    return {
        "candidate": candidate.to_dict(),
        "accepted": accepted,
        "windows_scored": len(scores),
        "windows_skipped": len(skipped),
        "accepted_windows": accepted_windows,
        "accepted_fraction": accepted_fraction,
        "quality": quality,
        "mean_quality": mean_quality,
        "mean_readiness": mean_readiness,
        "mean_null_lift": mean_lift,
        "mean_saturation": mean_saturation,
        "mean_stability": float(np.mean([score.stability_score for score in scores])),
        "mean_fragility": float(np.mean([score.fragility_penalty for score in scores])),
        "observed_total": observed,
        "null_mean_total": null_mean,
        "observed_minus_null_total": observed - null_mean,
        "z_effect": z_effect,
        "min_accepted_fraction": min_accepted_fraction,
        "min_z_effect": min_z_effect,
        "null_repeats_per_window": scores[0].null_summary.null_repeats,
        "null_exceedance_sum": int(sum(score.null_summary.null_exceedances for score in scores)),
        "null_repeat_sum": int(sum(score.null_summary.null_repeats for score in scores)),
        "window_scores": [
            {
                "quality": score.quality,
                "accepted": score.accepted,
                "observed": score.null_summary.observed_total,
                "null_mean": score.null_summary.null_mean,
                "z_effect": score.null_summary.z_effect,
                "exceedances": score.null_summary.null_exceedances,
                "repeats": score.null_summary.null_repeats,
                "saturation": score.saturation_penalty,
                "null_lift": score.null_lift_score,
            }
            for score in scores
        ],
        "skipped": list(skipped)[:5],
    }


def score_candidate_windows(
    windows: Sequence[BatchWindow],
    candidate: CdipAutoTuneCandidate,
    *,
    channels: Sequence[str],
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    scores: list[AutoTuneScore] = []
    skipped: list[dict[str, object]] = []
    for index, window in enumerate(windows):
        try:
            scores.append(
                _score_window(
                    window,
                    candidate,
                    channels=channels,
                    highpass_period_minutes=args.highpass_period_minutes,
                    mask_dominant_bins=args.mask_dominant_bins,
                    mask_bin_radius=args.mask_bin_radius,
                    phase_surrogate_seed=args.phase_surrogate_seed,
                    null_repeats=args.null_repeats,
                    seed=args.seed + seed_offset + index,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report invalid candidate/window pairs.
            skipped.append(
                {
                    "window": _window_identity(window),
                    "reason": str(exc),
                }
            )
    return _aggregate_scores(
        candidate,
        scores,
        skipped=skipped,
        min_accepted_fraction=args.min_accepted_fraction,
        min_z_effect=args.min_z_effect,
    )


def discover_cdip_windows(
    args: argparse.Namespace,
    *,
    channels: Sequence[str],
    window_offset_per_group: int,
    max_windows_per_group: int,
) -> list[BatchWindow]:
    records = [
        load_cdip_raw_record(path, channels, keep_flags=set(args.keep_flags))
        for path in args.files
    ]
    min_clean_samples = args.min_clean_samples or (
        max(_parse_int_list(args.baselines))
        + max(_parse_int_list(args.adaptive_windows))
        + max(_parse_int_list(args.strides))
    )
    return _discover_windows(
        records,
        group_size=args.group_size,
        min_clean_samples=min_clean_samples,
        window_samples=args.window_samples,
        window_step=args.window_step,
        max_groups=args.max_groups,
        max_windows_per_group=max_windows_per_group,
        window_offset_per_group=window_offset_per_group,
        group_strategy=args.group_strategy,
    )


def run_cdip_autotune(args: argparse.Namespace) -> dict[str, Any]:
    channels = [channel.lower() for channel in args.channels]
    candidates = build_candidates(args)
    calibration_windows = discover_cdip_windows(
        args,
        channels=channels,
        window_offset_per_group=args.calibration_window_offset,
        max_windows_per_group=args.calibration_windows_per_group,
    )
    started = time.time()
    candidate_reports = []
    for index, candidate in enumerate(candidates):
        if args.progress:
            print(
                "cdip_autotune candidate %d/%d preprocess=%s config=%s"
                % (index + 1, len(candidates), candidate.preprocess, candidate.config.to_dict()),
                file=sys.stderr,
                flush=True,
            )
        candidate_reports.append(
            score_candidate_windows(
                calibration_windows,
                candidate,
                channels=channels,
                args=args,
                seed_offset=1000 * index,
            )
        )

    candidate_reports = sorted(
        candidate_reports,
        key=lambda row: (
            bool(row.get("accepted", False)),
            float(row.get("observed_minus_null_total", 0.0)) > 0.0,
            float(row.get("quality", float("-inf"))),
            float(row.get("observed_minus_null_total", float("-inf"))),
        ),
        reverse=True,
    )
    best_report = candidate_reports[0] if candidate_reports else None
    best_candidate = None
    validation_report = None
    validation_windows: list[BatchWindow] = []
    if best_report is not None:
        candidate_row = best_report["candidate"]
        best_candidate = CdipAutoTuneCandidate(
            preprocess=str(candidate_row["preprocess"]),
            config=AutoTuneConfig(
                baseline_size=int(candidate_row["baseline_size"]),
                adaptive_window=int(candidate_row["adaptive_window"]),
                stride=int(candidate_row["stride"]),
                score=str(candidate_row["score"]),  # type: ignore[arg-type]
                emission_threshold=float(candidate_row["emission_threshold"]),
                decay=float(candidate_row["decay"]),
                min_active_series=int(candidate_row["min_active_series"]),
            ),
        )
        validation_windows = discover_cdip_windows(
            args,
            channels=channels,
            window_offset_per_group=args.validation_window_offset,
            max_windows_per_group=args.validation_windows_per_group,
        )
        validation_report = score_candidate_windows(
            validation_windows,
            best_candidate,
            channels=channels,
            args=args,
            seed_offset=900000,
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "method": {
            "label_use": "none",
            "selection": "rank candidate settings by label-free aggregate autotune quality on calibration windows",
            "validation": "freeze selected candidate and score heldout windows",
        },
        "parameters": {
            "files": [str(path) for path in args.files],
            "channels": channels,
            "keep_flags": list(args.keep_flags),
            "group_size": args.group_size,
            "group_strategy": args.group_strategy,
            "max_groups": args.max_groups,
            "window_samples": args.window_samples,
            "window_step": args.window_step,
            "calibration_window_offset": args.calibration_window_offset,
            "calibration_windows_per_group": args.calibration_windows_per_group,
            "validation_window_offset": args.validation_window_offset,
            "validation_windows_per_group": args.validation_windows_per_group,
            "candidate_count": len(candidates),
            "null_repeats": args.null_repeats,
            "min_accepted_fraction": args.min_accepted_fraction,
            "min_z_effect": args.min_z_effect,
            "preprocess_grid": _parse_str_list(args.preprocess_grid),
            "baselines": _parse_int_list(args.baselines),
            "adaptive_windows": _parse_int_list(args.adaptive_windows),
            "strides": _parse_int_list(args.strides),
            "thresholds": _parse_float_list(args.thresholds),
            "decays": _parse_float_list(args.decays),
            "min_active_series": _parse_int_list(args.min_active_series),
        },
        "calibration": {
            "windows": [_window_identity(window) for window in calibration_windows],
            "best": best_report,
            "candidates": candidate_reports[: args.top_candidates],
        },
        "validation": {
            "windows": [_window_identity(window) for window in validation_windows],
            "selected_candidate": best_candidate.to_dict() if best_candidate else None,
            "result": validation_report,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print("CDIP autotune calibration")
    print(
        "  files=%d candidates=%d calibration_windows=%d validation_windows=%d elapsed=%.1fs"
        % (
            len(report["parameters"]["files"]),
            report["parameters"]["candidate_count"],
            len(report["calibration"]["windows"]),
            len(report["validation"]["windows"]),
            report["elapsed_seconds"],
        )
    )
    best = report["calibration"]["best"]
    if not best:
        print("  no valid candidates")
        return
    candidate = best["candidate"]
    print(
        "  best accepted=%s quality=%.4f delta=%.3f z=%s windows=%d/%d"
        % (
            best["accepted"],
            best["quality"],
            best.get("observed_minus_null_total", 0.0),
            "None" if best.get("z_effect") is None else "%.2f" % best["z_effect"],
            best.get("accepted_windows", 0),
            best.get("windows_scored", 0),
        )
    )
    print(
        "  config preprocess=%s baseline=%d adaptive=%d stride=%d threshold=%.3f decay=%.3f min_active=%d"
        % (
            candidate["preprocess"],
            candidate["baseline_size"],
            candidate["adaptive_window"],
            candidate["stride"],
            candidate["emission_threshold"],
            candidate["decay"],
            candidate["min_active_series"],
        )
    )
    validation = report["validation"]["result"]
    if validation:
        print(
            "  validation accepted=%s quality=%.4f delta=%.3f z=%s windows=%d/%d"
            % (
                validation["accepted"],
                validation["quality"],
                validation.get("observed_minus_null_total", 0.0),
                "None" if validation.get("z_effect") is None else "%.2f" % validation["z_effect"],
                validation.get("accepted_windows", 0),
                validation.get("windows_scored", 0),
            )
        )
    print("  top candidates")
    for row in report["calibration"]["candidates"]:
        cfg = row["candidate"]
        print(
            "    accepted=%s quality=%.4f delta=%.3f preprocess=%s baseline=%d adaptive=%d threshold=%.3f saturation=%.3f"
            % (
                row.get("accepted"),
                row.get("quality", 0.0),
                row.get("observed_minus_null_total", 0.0),
                cfg["preprocess"],
                cfg["baseline_size"],
                cfg["adaptive_window"],
                cfg["emission_threshold"],
                row.get("mean_saturation", 0.0),
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="CDIP *_xy.nc files")
    parser.add_argument("--channels", nargs="+", default=["z"])
    parser.add_argument("--keep-flags", type=int, nargs="+", default=[2])
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--group-strategy", choices=["first", "balanced"], default="balanced")
    parser.add_argument("--max-groups", type=int, default=6)
    parser.add_argument("--window-samples", type=int, default=4096)
    parser.add_argument("--window-step", type=int, default=2048)
    parser.add_argument("--min-clean-samples", type=int, default=None)
    parser.add_argument("--calibration-window-offset", type=int, default=0)
    parser.add_argument("--calibration-windows-per-group", type=int, default=1)
    parser.add_argument("--validation-window-offset", type=int, default=1)
    parser.add_argument("--validation-windows-per-group", type=int, default=1)

    parser.add_argument(
        "--preprocess-grid",
        default="highpass,highpass-dominant-mask",
        help="Comma-separated preprocess modes",
    )
    parser.add_argument("--baselines", default="768,1280")
    parser.add_argument("--adaptive-windows", default="384")
    parser.add_argument("--strides", default="256")
    parser.add_argument("--thresholds", default="2.5,3.0")
    parser.add_argument("--decays", default="0.9")
    parser.add_argument("--min-active-series", default="2")
    parser.add_argument("--highpass-period-minutes", type=float, default=30.0)
    parser.add_argument("--mask-dominant-bins", type=int, default=3)
    parser.add_argument("--mask-bin-radius", type=int, default=1)
    parser.add_argument("--phase-surrogate-seed", type=int, default=20260602)
    parser.add_argument("--null-repeats", type=int, default=12)
    parser.add_argument(
        "--min-accepted-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of scored windows that must be individually accepted",
    )
    parser.add_argument(
        "--min-z-effect",
        type=float,
        default=1.5,
        help="Minimum aggregate z effect for a candidate to be accepted",
    )
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.max_groups < 1:
        parser.error("--max-groups must be >= 1")
    if args.calibration_windows_per_group < 1 or args.validation_windows_per_group < 1:
        parser.error("window counts per group must be >= 1")
    if args.calibration_window_offset < 0 or args.validation_window_offset < 0:
        parser.error("window offsets must be >= 0")
    if args.null_repeats < 1:
        parser.error("--null-repeats must be >= 1")
    if not 0.0 <= args.min_accepted_fraction <= 1.0:
        parser.error("--min-accepted-fraction must be in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_cdip_autotune(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
