"""Test observer-score directional structure against repeated random orders."""

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

from experiments.org_network_autotune import slice_series
from experiments.org_network_directional_order_null import summarize_order_null
from experiments.org_network_observer_directional_quality import (
    _namespace_from_surface_report,
    build_network_from_surface_report,
    observer_config,
    observer_directional_segment,
)


def run_observer_order_null(
    candidate_report: Mapping[str, Any],
    surface_report: Mapping[str, Any],
    *,
    repeats: int,
    inner_null_repeats: int,
    seed: int,
    progress_every: float,
    checkpoint: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if candidate_report["replication"]["grade"] == "insufficient_observer_resolution":
        raise ValueError("candidate observer report has insufficient resolution")
    args = argparse.Namespace(**candidate_report["args"])
    args.null_repeats = inner_null_repeats
    config = observer_config(args)
    surface_args = _namespace_from_surface_report(surface_report)
    metric_names = list(candidate_report["calibration"]["metrics"])
    controls: dict[str, list[float]] = {name: [] for name in metric_names}
    start_repeat = 0
    if resume:
        if checkpoint is None or not checkpoint.exists():
            raise ValueError("--resume requires an existing --checkpoint")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("metric_names") != metric_names:
            raise ValueError("checkpoint metric names do not match candidate report")
        if int(saved.get("target_repeats", -1)) != repeats:
            raise ValueError("checkpoint repeat target does not match --repeats")
        if int(saved.get("inner_null_repeats", -1)) != inner_null_repeats:
            raise ValueError("checkpoint inner-null count does not match")
        controls = {
            name: [float(value) for value in saved["controls"][name]]
            for name in metric_names
        }
        start_repeat = int(saved["next_repeat"])
    started = time.time()
    last_progress = 0.0

    for repeat in range(start_repeat, repeats):
        network, _, _, _, _ = build_network_from_surface_report(
            surface_report,
            order_override="random_order",
            seed_override=seed + repeat,
        )
        n_bins = int(network.metadata["bin_count"])
        split = int(n_bins * surface_args.validation_start_fraction)
        segment_results: list[dict[str, Any]] = []
        for segment_index, (segment_name, start, end) in enumerate(
            (("calibration", 0, split), ("validation", split, n_bins))
        ):
            result = observer_directional_segment(
                slice_series(network.series, start=start, end=end),
                input_feature_names=candidate_report[segment_name]["input_feature_names"],
                score_feature_names=candidate_report[segment_name]["selected_series"],
                config=config,
                args=args,
                seed_offset=100_000 * (repeat + 1) + segment_index,
                run_resolution_calibration=False,
            )
            if result["status"] != "ok":
                raise ValueError(
                    f"random-order observer control failed for {segment_name}: {result['status']}"
                )
            segment_results.append(result)
        for name in metric_names:
            controls[name].append(
                min(
                    float(result["metrics"][name]["observed_minus_null"])
                    for result in segment_results
                )
            )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "metric_names": metric_names,
                        "target_repeats": repeats,
                        "inner_null_repeats": inner_null_repeats,
                        "seed": seed,
                        "next_repeat": repeat + 1,
                        "controls": controls,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            tmp.replace(checkpoint)
        now = time.time()
        if progress_every > 0 and now - last_progress >= progress_every:
            elapsed = max(now - started, 1e-9)
            done = repeat + 1
            eta = (repeats - done) / (done / elapsed)
            print(
                "observer_directional_order_null progress=%d/%d elapsed=%.1fs eta=%.1fs"
                % (done, repeats, elapsed, eta),
                file=sys.stderr,
                flush=True,
            )
            last_progress = now

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": candidate_report["dataset"],
        "candidate_projection": candidate_report["analysis_projection"],
        "observer_config": candidate_report["observer_config"],
        "order_null": {
            "mode": "random_order",
            "repeats": repeats,
            "inner_null_repeats": inner_null_repeats,
            "seed": seed,
            "resumed_from_repeat": start_repeat,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "input_feature_selection": "frozen_candidate_segment_features",
            "score_feature_selection": "frozen_candidate_segment_scores",
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
        "dataset=%s observer=%s/%s/%s grade=%s confirmed=%s repeats=%s elapsed=%.1fs"
        % (
            report["dataset"].get("dataset", "custom"),
            report["observer_config"]["baseline_size"],
            report["observer_config"]["adaptive_window"],
            report["observer_config"]["stride"],
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
    parser.add_argument("--surface-report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--inner-null-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--progress-every", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    surface_report = json.loads(args.surface_report.read_text(encoding="utf-8"))
    report = run_observer_order_null(
        candidate_report,
        surface_report,
        repeats=args.repeats,
        inner_null_repeats=args.inner_null_repeats,
        seed=args.seed,
        progress_every=args.progress_every,
        checkpoint=args.checkpoint,
        resume=args.resume,
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
