"""Apply directional-quality diagnostics to frozen spectral-observer scores.

The input is a completed organizational feature-surface directional report.
This runner reconstructs the same surface, applies one frozen spectral-observer
geometry, and asks which structural mechanisms survive, appear, or disappear
in the observer score matrix.
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
    build_selected_network,
    read_edges,
    read_labels,
    read_simplicial_edges,
    remap_edges_by_order,
    resolve_dataset_files,
    slice_series,
)
from spectral_forecast.autotune import (
    AutoTuneConfig,
    build_autotune_observation,
    score_autotune_observation,
)
from spectral_forecast.directional import DirectionalQuality, directional_quality
from spectral_forecast.directional_resolution import directional_resolution_calibration


def _namespace_from_surface_report(report: Mapping[str, Any]) -> argparse.Namespace:
    payload = dict(report["args"])
    for name in ("data_dir", "edges", "labels"):
        if payload.get(name) is not None:
            payload[name] = Path(payload[name])
    payload["output"] = None
    return argparse.Namespace(**payload)


def build_network_from_surface_report(
    report: Mapping[str, Any],
    *,
    order_override: str | None = None,
    seed_override: int | None = None,
):
    """Reconstruct a candidate or control network surface from its report."""

    args = _namespace_from_surface_report(report)
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
    order = order_override or args.event_order
    if order != "real-time":
        edges = remap_edges_by_order(
            edges,
            order=order,
            labels=labels,
            directed=args.directed,
            seed=args.seed if seed_override is None else seed_override,
        )
        args.bin_seconds = float(args.event_bin_size)
    args.event_order = order
    network = build_selected_network(edges, labels, args=args)
    return network, args, dataset_metadata, edge_path, label_path


def observer_config(args: argparse.Namespace) -> AutoTuneConfig:
    config = AutoTuneConfig(
        baseline_size=args.baseline_size,
        adaptive_window=args.adaptive_window,
        stride=args.observer_stride,
        score=args.score,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
        min_active_series=args.min_active_series,
    )
    config.validate()
    return config


def observer_directional_segment(
    series: Mapping[str, np.ndarray],
    *,
    input_feature_names: Sequence[str],
    score_feature_names: Sequence[str] | None = None,
    config: AutoTuneConfig,
    args: argparse.Namespace,
    seed_offset: int,
    run_resolution_calibration: bool = True,
) -> dict[str, Any]:
    """Build and diagnose one frozen observer score matrix."""

    missing = [name for name in input_feature_names if name not in series]
    if missing:
        return {
            "status": "missing_input_features",
            "missing_input_features": missing[:20],
            "input_feature_count": len(input_feature_names),
        }
    selected = {
        name: np.asarray(series[name], dtype=np.float64)
        for name in input_feature_names
    }
    input_bins = min(len(values) for values in selected.values())
    if input_bins <= config.baseline_size:
        return {
            "status": "insufficient_input_bins",
            "input_bins": input_bins,
            "baseline_size": config.baseline_size,
            "input_feature_count": len(selected),
            "input_feature_names": list(selected),
        }

    observation = build_autotune_observation(
        selected,
        config,
        sample_rate=args.sample_rate,
    )
    effective_aggregate_block_size = min(
        args.null_block_size,
        max(1, len(observation.anchors) // 2),
    )
    aggregate_score = score_autotune_observation(
        observation,
        config,
        null_repeats=args.null_repeats,
        seed=args.seed + seed_offset + 777,
        null_mode="block-permute",
        null_block_size=effective_aggregate_block_size,
    )
    aggregate = {
        "coherence_direction": (
            "surplus"
            if aggregate_score.null_summary.observed_minus_null > 0
            else "deficit"
            if aggregate_score.null_summary.observed_minus_null < 0
            else "none"
        ),
        "observed_minus_null_total": aggregate_score.null_summary.observed_minus_null,
        "z_effect": aggregate_score.null_summary.z_effect,
        "empirical_p_ge_observed": aggregate_score.null_summary.empirical_p_ge_observed,
        "empirical_p_le_observed": aggregate_score.null_summary.empirical_p_le_observed,
        "empirical_p_two_sided": aggregate_score.null_summary.empirical_p_two_sided,
        "null_block_size": effective_aggregate_block_size,
        "emission_threshold": config.emission_threshold,
        "decay": config.decay,
        "min_active_series": config.min_active_series,
    }
    score_series = {
        name: np.asarray(observation.matrix[:, index], dtype=np.float64)
        for index, name in enumerate(selected)
    }
    if score_feature_names is not None:
        missing_scores = [name for name in score_feature_names if name not in score_series]
        if missing_scores:
            return {
                "status": "missing_observer_score_features",
                "missing_observer_score_features": missing_scores[:20],
                "input_feature_count": len(selected),
                "input_feature_names": list(selected),
            }
        score_series = {
            name: score_series[name]
            for name in score_feature_names
        }
    if len(observation.anchors) < args.min_observer_anchors:
        resolution = (
            directional_resolution_calibration(
                n=len(observation.anchors),
                series_count=len(score_series),
                trials=args.resolution_trials,
                null_repeats=args.resolution_null_repeats,
                seed=args.seed + seed_offset + 999,
                active_z_threshold=args.active_z_threshold,
                max_lag=args.max_lag,
                aggregate_quantile=args.aggregate_quantile,
                null_block_size=args.null_block_size,
                significance_level=args.significance_level,
                min_z_effect=args.min_z_effect,
                min_detection_rate=args.resolution_min_detection_rate,
                max_false_positive_rate=args.resolution_max_false_positive_rate,
            ).to_dict()
            if run_resolution_calibration
            else None
        )
        return {
            "status": "insufficient_observer_anchors",
            "input_bins": input_bins,
            "observer_anchor_count": len(observation.anchors),
            "required_observer_anchors": args.min_observer_anchors,
            "observer_anchor_first": observation.anchors[0] if observation.anchors else None,
            "observer_anchor_last": observation.anchors[-1] if observation.anchors else None,
            "input_feature_count": len(selected),
            "input_feature_names": list(selected),
            "positive_coactivation_aggregate": aggregate,
            "resolution_calibration": resolution,
        }

    quality = directional_quality(
        score_series,
        null_repeats=args.null_repeats,
        seed=args.seed + seed_offset,
        active_z_threshold=args.active_z_threshold,
        activation_mode="positive",
        max_lag=args.max_lag,
        aggregate_quantile=args.aggregate_quantile,
        null_block_size=args.null_block_size,
        max_series=(
            len(score_feature_names)
            if score_feature_names is not None
            else args.max_score_series
        ),
        significance_level=args.significance_level,
        min_z_effect=args.min_z_effect,
    )
    resolution = (
        directional_resolution_calibration(
            n=len(observation.anchors),
            series_count=quality.usable_series_count,
            trials=args.resolution_trials,
            null_repeats=args.resolution_null_repeats,
            seed=args.seed + seed_offset + 999,
            active_z_threshold=args.active_z_threshold,
            max_lag=args.max_lag,
            aggregate_quantile=args.aggregate_quantile,
            null_block_size=args.null_block_size,
            significance_level=args.significance_level,
            min_z_effect=args.min_z_effect,
            min_detection_rate=args.resolution_min_detection_rate,
            max_false_positive_rate=args.resolution_max_false_positive_rate,
        ).to_dict()
        if run_resolution_calibration
        else None
    )
    return {
        **quality.to_dict(),
        "status": "ok",
        "input_bins": input_bins,
        "observer_anchor_count": len(observation.anchors),
        "observer_anchor_first": observation.anchors[0],
        "observer_anchor_last": observation.anchors[-1],
        "input_feature_count": len(selected),
        "input_feature_names": list(selected),
        "observer_readiness_score": observation.readiness_score,
        "positive_coactivation_aggregate": aggregate,
        "resolution_calibration": resolution,
    }


def observer_replication_summary(
    calibration: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = {
        "calibration": calibration["status"],
        "validation": validation["status"],
    }
    if any(status != "ok" for status in statuses.values()):
        return {
            "grade": "insufficient_observer_resolution",
            "segment_statuses": statuses,
            "stable_detected_mechanisms": [],
            "same_strongest_mechanism": False,
            "conservative_quality_score": 0.0,
        }
    stable = sorted(
        set(calibration["detected_mechanisms"])
        & set(validation["detected_mechanisms"])
    )
    calibration_resolution = calibration.get("resolution_calibration") or {}
    validation_resolution = validation.get("resolution_calibration") or {}
    supported = sorted(
        set(calibration_resolution.get("supported_mechanisms", []))
        & set(validation_resolution.get("supported_mechanisms", []))
    )
    return {
        "grade": (
            "replicated_observer_directional_structure"
            if stable
            else "no_replicated_observer_directional_structure"
        ),
        "segment_statuses": statuses,
        "stable_detected_mechanisms": stable,
        "resolution_supported_mechanisms": supported,
        "same_strongest_mechanism": bool(
            calibration["strongest_mechanism"] is not None
            and calibration["strongest_mechanism"] == validation["strongest_mechanism"]
        ),
        "calibration_strongest": calibration["strongest_mechanism"],
        "validation_strongest": validation["strongest_mechanism"],
        "conservative_quality_score": (
            min(float(calibration["quality_score"]), float(validation["quality_score"]))
            if stable
            else 0.0
        ),
    }


def layer_comparison(
    surface_report: Mapping[str, Any],
    replication: Mapping[str, Any],
) -> dict[str, Any]:
    surface = set(surface_report["replication"]["stable_detected_mechanisms"])
    observer = set(replication["stable_detected_mechanisms"])
    supported = set(replication.get("resolution_supported_mechanisms", []))
    not_observed = surface - observer
    return {
        "surface_stable_mechanisms": sorted(surface),
        "observer_stable_mechanisms": sorted(observer),
        "survived_surface_to_observer": sorted(surface & observer),
        "appeared_at_observer": sorted(observer - surface),
        "not_observed_at_observer": sorted(not_observed),
        "interpretable_absences": sorted(not_observed & supported),
        "underpowered_or_unresolved_absences": sorted(not_observed - supported),
        "observer_resolution_sufficient": replication["grade"] != "insufficient_observer_resolution",
    }


def _reference_deficit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for segment in ("calibration", "validation"):
        source = report.get(segment)
        if source is None:
            rows[segment] = None
            continue
        if isinstance(source, Mapping) and source.get("best") is not None:
            source = source["best"]
        rows[segment] = {
            "coherence_direction": source.get("coherence_direction"),
            "observed_minus_null_total": source.get("observed_minus_null_total"),
            "z_effect": source.get("z_effect"),
            "empirical_p_le_observed": source.get("null_total_empirical_p_le_observed"),
            "anchors": (
                source.get("null_mode_scores", [{}])[0]
                .get("score", {})
                .get("null_summary", {})
                .get("anchors")
            ),
        }
    return {"path": str(path), "segments": rows}


def run_observer_directional_quality(
    surface_report: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    network, surface_args, dataset_metadata, edge_path, label_path = build_network_from_surface_report(
        surface_report
    )
    config = observer_config(args)
    n_bins = int(network.metadata["bin_count"])
    split = int(n_bins * surface_args.validation_start_fraction)
    segments: dict[str, dict[str, Any]] = {}
    for index, (name, start, end) in enumerate(
        (("calibration", 0, split), ("validation", split, n_bins))
    ):
        if args.feature_selection == "surface-selected":
            input_feature_names = list(surface_report[name]["selected_series"])
        else:
            input_feature_names = list(network.series)
        segments[name] = observer_directional_segment(
            slice_series(network.series, start=start, end=end),
            input_feature_names=input_feature_names,
            config=config,
            args=args,
            seed_offset=10_000 * (index + 1),
        )
    replication = observer_replication_summary(
        segments["calibration"],
        segments["validation"],
    )
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
        },
        "analysis_projection": surface_report["analysis_projection"],
        "series_metadata": network.metadata,
        "calibration_bins": split,
        "validation_bins": n_bins - split,
        "observer_config": config.to_dict(),
        "reference_positive_coactivation_result": _reference_deficit(args.reference_autotune_report),
        "calibration": segments["calibration"],
        "validation": segments["validation"],
        "replication": replication,
        "layer_comparison": layer_comparison(surface_report, replication),
    }


def print_report(report: Mapping[str, Any]) -> None:
    print(
        "dataset=%s surface=%s order=%s observer=%s/%s/%s feature_selection=%s"
        % (
            report["dataset"].get("dataset", "custom"),
            report["analysis_projection"]["surface_mode"],
            report["analysis_projection"]["event_order"],
            report["observer_config"]["baseline_size"],
            report["observer_config"]["adaptive_window"],
            report["observer_config"]["stride"],
            report["args"]["feature_selection"],
        )
    )
    for name in ("calibration", "validation"):
        segment = report[name]
        if segment["status"] != "ok":
            print(
                "%s status=%s anchors=%s input_bins=%s"
                % (
                    name,
                    segment["status"],
                    segment.get("observer_anchor_count"),
                    segment.get("input_bins"),
                )
            )
            aggregate = segment.get("positive_coactivation_aggregate")
            if aggregate is not None:
                print(
                    "  positive_aggregate direction=%s delta=% .6f z=%s p_le=%.4f"
                    % (
                        aggregate["coherence_direction"],
                        aggregate["observed_minus_null_total"],
                        aggregate["z_effect"],
                        aggregate["empirical_p_le_observed"],
                    )
                )
            continue
        print(
            "%s anchors=%s verdict=%s strongest=%s detected=%s"
            % (
                name,
                segment["observer_anchor_count"],
                segment["verdict"],
                segment["strongest_mechanism"],
                ",".join(segment["detected_mechanisms"]) or "none",
            )
        )
        aggregate = segment["positive_coactivation_aggregate"]
        print(
            "  positive_aggregate direction=%s delta=% .6f z=%s p_le=%.4f"
            % (
                aggregate["coherence_direction"],
                aggregate["observed_minus_null_total"],
                aggregate["z_effect"],
                aggregate["empirical_p_le_observed"],
            )
        )
        for metric, evidence in segment["metrics"].items():
            print(
                "  %-18s delta=% .6f z=%s p_ge=%.4f detected=%s"
                % (
                    metric,
                    evidence["observed_minus_null"],
                    evidence["z_effect"],
                    evidence["empirical_p_ge_observed"],
                    evidence["detected"],
                )
            )
    comparison = report["layer_comparison"]
    print(
        "replication=%s survived=%s appeared=%s not_observed=%s"
        % (
            report["replication"]["grade"],
            ",".join(comparison["survived_surface_to_observer"]) or "none",
            ",".join(comparison["appeared_at_observer"]) or "none",
            ",".join(comparison["not_observed_at_observer"]) or "none",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-report", type=Path, required=True)
    parser.add_argument("--reference-autotune-report", type=Path, default=None)
    parser.add_argument("--baseline-size", type=int, required=True)
    parser.add_argument("--adaptive-window", type=int, required=True)
    parser.add_argument("--observer-stride", type=int, required=True)
    parser.add_argument("--score", choices=["frozen", "sliding", "drift", "max"], default="max")
    parser.add_argument("--emission-threshold", type=float, default=1.0)
    parser.add_argument("--decay", type=float, default=0.75)
    parser.add_argument("--min-active-series", type=int, default=1)
    parser.add_argument(
        "--feature-selection",
        choices=["surface-selected", "all"],
        default="surface-selected",
    )
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--min-observer-anchors", type=int, default=16)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--null-block-size", type=int, default=4)
    parser.add_argument("--active-z-threshold", type=float, default=1.5)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--aggregate-quantile", type=float, default=0.9)
    parser.add_argument("--max-score-series", type=int, default=64)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--min-z-effect", type=float, default=2.0)
    parser.add_argument("--resolution-trials", type=int, default=20)
    parser.add_argument("--resolution-null-repeats", type=int, default=24)
    parser.add_argument("--resolution-min-detection-rate", type=float, default=0.8)
    parser.add_argument("--resolution-max-false-positive-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    surface_report = json.loads(args.surface_report.read_text(encoding="utf-8"))
    report = run_observer_directional_quality(surface_report, args=args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
