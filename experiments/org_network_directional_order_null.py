"""Test candidate directional structure against repeated random event orders."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from spectral_forecast.directional import directional_quality


def _namespace_from_report(report: Mapping[str, Any]) -> argparse.Namespace:
    payload = dict(report["args"])
    for name in ("data_dir", "edges", "labels"):
        if payload.get(name) is not None:
            payload[name] = Path(payload[name])
    payload["output"] = None
    return argparse.Namespace(**payload)


def _candidate_min_deltas(report: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: min(
            float(report[segment]["metrics"][name]["observed_minus_null"])
            for segment in ("calibration", "validation")
        )
        for name in report["calibration"]["metrics"]
    }


def summarize_order_null(
    candidate_report: Mapping[str, Any],
    control_min_deltas: Mapping[str, Sequence[float]],
    *,
    significance_level: float,
) -> dict[str, Any]:
    candidate_deltas = _candidate_min_deltas(candidate_report)
    stable = set(candidate_report["replication"]["stable_detected_mechanisms"])
    rows: dict[str, Any] = {}
    confirmed: list[str] = []
    for name, candidate_value in candidate_deltas.items():
        null = np.asarray(control_min_deltas[name], dtype=np.float64)
        mean = float(np.mean(null))
        std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
        exceedances = int(np.sum(null >= candidate_value))
        p_ge = float((exceedances + 1) / (len(null) + 1))
        z_effect = (candidate_value - mean) / std if std > 1e-12 else None
        order_sensitive = bool(
            name in stable
            and candidate_value > mean
            and p_ge <= significance_level
        )
        if order_sensitive:
            confirmed.append(name)
        rows[name] = {
            "candidate_replicated": name in stable,
            "candidate_min_delta": candidate_value,
            "random_order_mean_min_delta": mean,
            "random_order_std_min_delta": std,
            "candidate_minus_random_order_mean": candidate_value - mean,
            "z_effect_size": z_effect,
            "random_order_exceedances": exceedances,
            "empirical_p_ge_candidate": p_ge,
            "empirical_p_floor": float(1.0 / (len(null) + 1)),
            "unique_random_order_values": int(len(np.unique(np.round(null, decimals=12)))),
            "confirmed_order_sensitive": order_sensitive,
        }
    return {
        "grade": (
            "confirmed_order_sensitive_structure"
            if confirmed
            else "no_confirmed_order_sensitive_structure"
        ),
        "confirmed_order_sensitive_mechanisms": sorted(confirmed),
        "candidate_replicated_mechanisms": sorted(stable),
        "metrics": rows,
    }


def run_order_null(
    candidate_report: Mapping[str, Any],
    *,
    repeats: int,
    inner_null_repeats: int,
    seed: int,
    progress_every: float,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if inner_null_repeats < 1:
        raise ValueError("inner_null_repeats must be >= 1")
    args = _namespace_from_report(candidate_report)
    edge_path, label_path, dataset_metadata = resolve_dataset_files(args)
    data_format = str(dataset_metadata.get("data_format", args.data_format))
    if data_format == "edges":
        canonical_edges = read_edges(
            edge_path,
            source_col=args.source_col,
            target_col=args.target_col,
            time_col=args.time_col,
            max_edges=args.max_edges,
        )
    elif data_format == "simplices":
        canonical_edges = read_simplicial_edges(
            edge_path,
            prefix=args.simplex_prefix,
            max_edges=args.max_edges,
        )
    else:
        raise ValueError(f"unknown data format: {data_format}")
    labels = read_labels(label_path)

    metric_names = list(candidate_report["calibration"]["metrics"])
    controls: dict[str, list[float]] = {name: [] for name in metric_names}
    started = time.time()
    last_progress = 0.0
    for repeat in range(repeats):
        control_edges = remap_edges_by_order(
            canonical_edges,
            order="random_order",
            labels=labels,
            directed=args.directed,
            seed=seed + repeat,
        )
        args.bin_seconds = float(args.event_bin_size)
        network = build_selected_network(control_edges, labels, args=args)
        n_bins = int(network.metadata["bin_count"])
        split = int(n_bins * args.validation_start_fraction)
        segment_results = []
        for segment_index, (segment_name, start, end) in enumerate(
            (("calibration", 0, split), ("validation", split, n_bins))
        ):
            segment_series = slice_series(network.series, start=start, end=end)
            selected_names = list(candidate_report[segment_name]["selected_series"])
            missing = [name for name in selected_names if name not in segment_series]
            if missing:
                raise ValueError(
                    "random-order control dropped candidate-selected series: "
                    + ", ".join(missing[:5])
                )
            segment_results.append(
                directional_quality(
                    {
                        name: segment_series[name]
                        for name in selected_names
                    },
                    null_repeats=inner_null_repeats,
                    seed=seed + 100_000 * (repeat + 1) + segment_index,
                    active_z_threshold=args.active_z_threshold,
                    max_lag=args.max_lag,
                    aggregate_quantile=args.aggregate_quantile,
                    null_block_size=args.null_block_size,
                    max_series=len(selected_names),
                    significance_level=args.significance_level,
                    min_z_effect=args.min_z_effect,
                )
            )
        for name in metric_names:
            controls[name].append(
                min(
                    float(result.metrics[name].observed_minus_null)
                    for result in segment_results
                )
            )
        now = time.time()
        if progress_every > 0 and now - last_progress >= progress_every:
            elapsed = max(now - started, 1e-9)
            done = repeat + 1
            eta = (repeats - done) / (done / elapsed)
            print(
                "directional_order_null progress=%d/%d elapsed=%.1fs eta=%.1fs"
                % (done, repeats, elapsed, eta),
                file=sys.stderr,
                flush=True,
            )
            last_progress = now

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": candidate_report["dataset"],
        "candidate_projection": candidate_report["analysis_projection"],
        "order_null": {
            "mode": "random_order",
            "repeats": repeats,
            "inner_null_repeats": inner_null_repeats,
            "seed": seed,
            "feature_selection": "frozen_candidate_segment_series",
            "comparison_statistic": "minimum segment observed-minus-block-null delta",
        },
        "summary": summarize_order_null(
            candidate_report,
            controls,
            significance_level=float(args.significance_level),
        ),
        "random_order_min_deltas": controls,
        "elapsed_seconds": time.time() - started,
    }


def print_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print(
        "dataset=%s grade=%s confirmed=%s repeats=%s elapsed=%.1fs"
        % (
            report["dataset"].get("dataset", "custom"),
            summary["grade"],
            ",".join(summary["confirmed_order_sensitive_mechanisms"]) or "none",
            report["order_null"]["repeats"],
            report["elapsed_seconds"],
        )
    )
    for name, row in summary["metrics"].items():
        print(
            "  %-18s candidate_delta=% .6f random_mean=% .6f z=%s p_ge=%.4f confirmed=%s"
            % (
                name,
                row["candidate_min_delta"],
                row["random_order_mean_min_delta"],
                row["z_effect_size"],
                row["empirical_p_ge_candidate"],
                row["confirmed_order_sensitive"],
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--inner-null-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--progress-every", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    report = run_order_null(
        candidate_report,
        repeats=args.repeats,
        inner_null_repeats=args.inner_null_repeats,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
