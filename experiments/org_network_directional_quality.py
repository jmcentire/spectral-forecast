"""Diagnose which directional structures organize temporal network surfaces.

This experiment complements the positive-coactivation stigmergy objective. It
tests simultaneous coactivation, mutual exclusion, lagged succession, and
nonzero-lag phase offset against an independent block-permutation null. The
statistics are label-free and describe organization, not meaning or utility.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.org_network_autotune import (
    DATASETS,
    build_selected_network,
    read_edges,
    read_labels,
    read_simplicial_edges,
    remap_edges_by_order,
    resolve_dataset_files,
    slice_series,
)
from spectral_forecast.directional import DirectionalQuality, directional_quality


def _synthetic_cases(seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = 768
    count = 8

    noise = {f"n{i}": rng.normal(0.0, 1.0, n) for i in range(count)}

    coactivation = {
        name: values.copy()
        for name, values in noise.items()
    }
    for start in range(24, 744, 72):
        for values in coactivation.values():
            values[start : start + 6] += 6.0

    exclusion = {f"e{i}": np.zeros(n, dtype=np.float64) for i in range(count)}
    for index in range(n):
        exclusion[f"e{int(rng.integers(0, count))}"][index] = 8.0

    succession = {f"s{i}": np.zeros(n, dtype=np.float64) for i in range(count)}
    for start in range(12, n - 24, 32):
        for channel in range(count):
            succession[f"s{channel}"][start + 2 * channel] = 8.0

    t = np.arange(n, dtype=np.float64)
    phase_offset = {
        f"p{i}": np.sin(2.0 * np.pi * t / 48.0 + i * np.pi / 8.0)
        + rng.normal(0.0, 0.05, n)
        for i in range(count)
    }
    return {
        "independent_noise": noise,
        "coactivation": coactivation,
        "exclusion": exclusion,
        "lagged_succession": succession,
        "phase_offset": phase_offset,
    }


def run_synthetic_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Check mechanism identification against known-answer synthetic surfaces."""

    reports: dict[str, Any] = {}
    expected = {
        "independent_noise": None,
        "coactivation": "coactivation",
        "exclusion": "exclusion",
        "lagged_succession": "lagged_succession",
        "phase_offset": "phase_offset",
    }
    for index, (name, series) in enumerate(_synthetic_cases(args.seed).items()):
        quality = directional_quality(
            series,
            null_repeats=args.synthetic_null_repeats,
            seed=args.seed + 1000 + index,
            active_z_threshold=args.active_z_threshold,
            max_lag=args.max_lag,
            aggregate_quantile=args.aggregate_quantile,
            null_block_size=args.null_block_size,
            max_series=args.max_series,
            significance_level=args.significance_level,
            min_z_effect=args.min_z_effect,
        )
        expected_mechanism = expected[name]
        passed = (
            quality.verdict != "structured"
            if expected_mechanism is None
            else quality.strongest_mechanism == expected_mechanism
            and quality.metrics[expected_mechanism].detected
        )
        reports[name] = {
            "expected_strongest_mechanism": expected_mechanism,
            "passed": passed,
            "quality": quality.to_dict(),
        }
    return {
        "passed": all(bool(report["passed"]) for report in reports.values()),
        "cases": reports,
    }


def _quality_for_segment(
    series: Mapping[str, np.ndarray],
    *,
    args: argparse.Namespace,
    seed_offset: int,
) -> DirectionalQuality:
    return directional_quality(
        series,
        null_repeats=args.null_repeats,
        seed=args.seed + seed_offset,
        active_z_threshold=args.active_z_threshold,
        max_lag=args.max_lag,
        aggregate_quantile=args.aggregate_quantile,
        null_block_size=args.null_block_size,
        max_series=args.max_series,
        significance_level=args.significance_level,
        min_z_effect=args.min_z_effect,
    )


def _replication_summary(
    calibration: DirectionalQuality,
    validation: DirectionalQuality,
) -> dict[str, Any]:
    stable = sorted(set(calibration.detected_mechanisms) & set(validation.detected_mechanisms))
    same_strongest = bool(
        calibration.strongest_mechanism is not None
        and calibration.strongest_mechanism == validation.strongest_mechanism
    )
    if stable:
        grade = "replicated_directional_structure"
    elif calibration.detected_mechanisms or validation.detected_mechanisms:
        grade = "segment_specific_structure"
    else:
        grade = "no_detected_directional_structure"
    return {
        "grade": grade,
        "stable_detected_mechanisms": stable,
        "same_strongest_mechanism": same_strongest,
        "calibration_strongest": calibration.strongest_mechanism,
        "validation_strongest": validation.strongest_mechanism,
        "conservative_quality_score": (
            min(calibration.quality_score, validation.quality_score)
            if stable
            else 0.0
        ),
    }


def compare_candidate_to_controls(
    candidate: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare replicated candidate mechanisms with matched order controls.

    The delta advantage is descriptive. Confirming that advantage would require
    a second-level null over candidate/control assignments.
    """

    if not controls:
        raise ValueError("at least one matched control report is required")
    candidate_stable = set(candidate["replication"]["stable_detected_mechanisms"])
    control_stable = {
        mechanism
        for control in controls
        for mechanism in control["replication"]["stable_detected_mechanisms"]
    }
    metric_names = set(candidate["calibration"]["metrics"])
    rows: dict[str, Any] = {}
    for name in sorted(metric_names):
        candidate_deltas = [
            float(candidate[segment]["metrics"][name]["observed_minus_null"])
            for segment in ("calibration", "validation")
        ]
        candidate_z = [
            candidate[segment]["metrics"][name]["z_effect"]
            for segment in ("calibration", "validation")
        ]
        control_min_deltas = [
            min(
                float(control[segment]["metrics"][name]["observed_minus_null"])
                for segment in ("calibration", "validation")
            )
            for control in controls
        ]
        control_min_z = [
            min(
                float(control[segment]["metrics"][name]["z_effect"])
                if control[segment]["metrics"][name]["z_effect"] is not None
                else float("-inf")
                for segment in ("calibration", "validation")
            )
            for control in controls
        ]
        candidate_min_delta = min(candidate_deltas)
        strongest_control_min_delta = max(control_min_deltas)
        candidate_min_z = min(
            float(value) if value is not None else float("-inf")
            for value in candidate_z
        )
        strongest_control_min_z = max(control_min_z)
        rows[name] = {
            "candidate_replicated": name in candidate_stable,
            "control_replicated": name in control_stable,
            "selective_to_candidate": name in candidate_stable and name not in control_stable,
            "candidate_min_delta": candidate_min_delta,
            "strongest_control_min_delta": strongest_control_min_delta,
            "descriptive_delta_advantage": candidate_min_delta - strongest_control_min_delta,
            "candidate_min_z_effect": candidate_min_z,
            "strongest_control_min_z_effect": strongest_control_min_z,
        }

    selective = sorted(
        name for name, row in rows.items() if row["selective_to_candidate"]
    )
    shared_stronger = sorted(
        name
        for name, row in rows.items()
        if row["candidate_replicated"]
        and row["control_replicated"]
        and row["descriptive_delta_advantage"] > 0.0
    )
    if selective:
        grade = "selective_replicated_structure"
    elif shared_stronger:
        grade = "replicated_structure_stronger_than_control"
    elif candidate_stable:
        grade = "replicated_but_not_control_separated"
    else:
        grade = "no_replicated_candidate_structure"
    return {
        "grade": grade,
        "candidate_stable_mechanisms": sorted(candidate_stable),
        "control_stable_mechanisms": sorted(control_stable),
        "selective_candidate_mechanisms": selective,
        "shared_but_descriptively_stronger_mechanisms": shared_stronger,
        "control_gap_confirmed_by_second_level_null": False,
        "metrics": rows,
    }


def run_org_network_directional_quality(args: argparse.Namespace) -> dict[str, Any]:
    args = argparse.Namespace(**vars(args))
    edge_path, label_path, dataset_metadata = resolve_dataset_files(args)
    data_format = str(dataset_metadata.get("data_format", args.data_format))
    if data_format == "edges":
        edges = read_edges(
            edge_path,
            source_col=args.source_col,
            target_col=args.target_col,
            time_col=args.time_col,
            max_edges=args.max_edges,
        )
    elif data_format == "simplices":
        edges = read_simplicial_edges(
            edge_path,
            prefix=args.simplex_prefix,
            max_edges=args.max_edges,
        )
    else:
        raise ValueError(f"unknown data format: {data_format}")
    labels = read_labels(label_path)
    requested_bin_seconds = float(args.bin_seconds)
    if args.event_order != "real-time":
        edges = remap_edges_by_order(
            edges,
            order=args.event_order,
            labels=labels,
            directed=args.directed,
            seed=args.seed,
        )
        args.bin_seconds = float(args.event_bin_size)

    network = build_selected_network(edges, labels, args=args)
    n_bins = int(network.metadata["bin_count"])
    split = int(n_bins * args.validation_start_fraction)
    if split < 16 or n_bins - split < 16:
        raise ValueError("calibration and validation segments must each contain at least 16 bins")

    calibration = _quality_for_segment(
        slice_series(network.series, start=0, end=split),
        args=args,
        seed_offset=10_000,
    )
    validation = _quality_for_segment(
        slice_series(network.series, start=split, end=n_bins),
        args=args,
        seed_offset=20_000,
    )
    synthetic = None if args.skip_synthetic_validation else run_synthetic_validation(args)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "dataset": {
            **dataset_metadata,
            "edge_path": str(edge_path),
            "label_path": str(label_path) if label_path else None,
            "data_format": data_format,
        },
        "analysis_projection": {
            "surface_mode": network.metadata["surface_mode"],
            "event_order": args.event_order,
            "requested_bin_seconds": requested_bin_seconds,
            "analysis_bin_seconds": float(args.bin_seconds),
            "coordinate_kind": "canonical_time" if args.event_order == "real-time" else "event_rank",
        },
        "series_metadata": network.metadata,
        "calibration_bins": split,
        "validation_bins": n_bins - split,
        "synthetic_validation": synthetic,
        "calibration": calibration.to_dict(),
        "validation": validation.to_dict(),
        "replication": _replication_summary(calibration, validation),
    }


def print_report(report: Mapping[str, Any]) -> None:
    dataset = report["dataset"]
    projection = report["analysis_projection"]
    replication = report["replication"]
    print(
        "dataset=%s surface=%s order=%s bins=%s+%s series=%s"
        % (
            dataset.get("dataset", "custom"),
            projection["surface_mode"],
            projection["event_order"],
            report["calibration_bins"],
            report["validation_bins"],
            report["series_metadata"]["series_count"],
        )
    )
    synthetic = report.get("synthetic_validation")
    if synthetic is not None:
        print(f"synthetic_validation_passed={synthetic['passed']}")
    for segment in ("calibration", "validation"):
        quality = report[segment]
        print(
            "%s verdict=%s strongest=%s z=%s detected=%s"
            % (
                segment,
                quality["verdict"],
                quality["strongest_mechanism"],
                quality["strongest_z_effect"],
                ",".join(quality["detected_mechanisms"]) or "none",
            )
        )
        for name, evidence in quality["metrics"].items():
            print(
                "  %-18s observed=% .6f delta=% .6f z=%s p_ge=%.4f detected=%s"
                % (
                    name,
                    evidence["observed"],
                    evidence["observed_minus_null"],
                    evidence["z_effect"],
                    evidence["empirical_p_ge_observed"],
                    evidence["detected"],
                )
            )
    print(
        "replication=%s stable=%s same_strongest=%s quality=%.3f"
        % (
            replication["grade"],
            ",".join(replication["stable_detected_mechanisms"]) or "none",
            replication["same_strongest_mechanism"],
            replication["conservative_quality_score"],
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/org_networks"))
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--data-format", choices=["edges", "simplices"], default="edges")
    parser.add_argument("--simplex-prefix", default="email-Enron")
    parser.add_argument("--source-col", type=int, default=0)
    parser.add_argument("--target-col", type=int, default=1)
    parser.add_argument("--time-col", type=int, default=2)
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--max-bins", type=int, default=0)
    parser.add_argument("--bin-seconds", type=float, default=86400.0)
    parser.add_argument("--event-bin-size", type=int, default=512)
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--no-global-series", action="store_true")
    parser.add_argument("--no-node-series", action="store_true")
    parser.add_argument("--no-group-series", action="store_true")
    parser.add_argument("--no-relation-series", action="store_true")
    parser.add_argument("--relation-only", action="store_true")
    parser.add_argument(
        "--surface-mode",
        choices=["full", "relation", "graph", "group"],
        default="full",
    )
    parser.add_argument(
        "--event-order",
        choices=[
            "real-time",
            "timestamp",
            "timestamp_bucket_partial_order",
            "first_seen_pair",
            "degree_descending",
            "group_then_time",
            "stable_id_or_alphabetic",
            "hash_order",
            "random_order",
            "silly_proxy_order",
        ],
        default="real-time",
    )
    parser.add_argument("--transform", choices=["none", "log", "robust", "log-robust"], default="log-robust")
    parser.add_argument("--validation-start-fraction", type=float, default=0.5)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--synthetic-null-repeats", type=int, default=50)
    parser.add_argument("--null-block-size", type=int, default=8)
    parser.add_argument("--active-z-threshold", type=float, default=1.5)
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--aggregate-quantile", type=float, default=0.9)
    parser.add_argument("--max-series", type=int, default=64)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--min-z-effect", type=float, default=2.0)
    parser.add_argument("--skip-synthetic-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if not 0.0 < args.validation_start_fraction < 1.0:
        raise ValueError("--validation-start-fraction must be in (0, 1)")
    if args.event_bin_size <= 0:
        raise ValueError("--event-bin-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_org_network_directional_quality(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
