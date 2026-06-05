"""Benchmark established multivariate change-point methods without event labels.

The methods are frozen before real-data evaluation. Labels are used only after
detection to score known synthetic transitions and the fixed pmuBAGE event onset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:  # noqa: E402
    from changeforest import Control, changeforest
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "changeforest is required; install spectral-forecast[research]"
    ) from exc


METHODS = ("random_forest", "knn")
VIEWS = ("raw", "ar1_residual")
SCENARIOS = (
    "stationary_iid",
    "stationary_ar1",
    "mean_shift",
    "covariance_shift",
    "tail_shift",
)


def robust_standardize(matrix: np.ndarray) -> np.ndarray:
    """Scale columns without using transition labels or event timing."""

    values = np.asarray(matrix, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    std = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, std)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - median) / scale


def transform_view(matrix: np.ndarray, view: str) -> np.ndarray:
    """Apply a frozen dependence treatment before change-point analysis."""

    values = robust_standardize(matrix)
    if view == "raw":
        return values
    if view != "ar1_residual":
        raise ValueError(f"unsupported view: {view}")
    lagged = values[:-1]
    current = values[1:]
    denominator = np.sum(lagged * lagged, axis=0)
    phi = np.divide(
        np.sum(lagged * current, axis=0),
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    phi = np.clip(phi, -0.99, 0.99)
    return robust_standardize(current - lagged * phi)


def run_changeforest(
    matrix: np.ndarray,
    *,
    method: str,
    view: str,
    seed: int,
    permutations: int,
    estimators: int,
    jobs: int,
) -> dict[str, Any]:
    """Run one frozen changeforest binary-segmentation analysis."""

    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    values = transform_view(matrix, view)
    control = Control(
        model_selection_alpha=0.05,
        model_selection_n_permutations=permutations,
        seed=seed,
        random_forest_n_estimators=estimators,
        random_forest_n_jobs=jobs,
    )
    result = changeforest(
        values,
        method=method,
        segmentation_type="bs",
        control=control,
    )
    split_points = [int(value) for value in result.split_points()]
    return {
        "method": method,
        "view": view,
        "detected": bool(result.is_significant),
        "root_best_split": int(result.best_split),
        "root_p_value": float(result.p_value),
        "root_gain": float(result.max_gain),
        "split_points": split_points,
    }


def synthetic_matrix(
    scenario: str,
    *,
    samples: int,
    dimensions: int,
    seed: int,
) -> tuple[np.ndarray, int | None]:
    """Generate a known-answer multivariate change or stationary control."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if samples < 80 or dimensions < 4:
        raise ValueError("synthetic benchmark requires >=80 samples and >=4 dimensions")
    rng = np.random.default_rng(seed)
    split = samples // 2
    if scenario == "stationary_iid":
        return rng.normal(size=(samples, dimensions)), None
    if scenario == "stationary_ar1":
        innovations = rng.normal(size=(samples, dimensions))
        matrix = np.zeros_like(innovations)
        for index in range(1, samples):
            matrix[index] = 0.85 * matrix[index - 1] + innovations[index]
        return matrix, None
    if scenario == "mean_shift":
        matrix = rng.normal(size=(samples, dimensions))
        matrix[split:] += 0.65
        return matrix, split
    if scenario == "tail_shift":
        left = rng.normal(size=(split, dimensions))
        right = rng.standard_t(df=3, size=(samples - split, dimensions)) / math.sqrt(3.0)
        return np.vstack((left, right)), split

    matrix = np.empty((samples, dimensions), dtype=np.float64)
    left_factors = rng.normal(size=(split, dimensions // 2))
    for column in range(dimensions):
        matrix[:split, column] = (
            left_factors[:, column // 2]
            + rng.normal(0.0, 0.35, split)
        )
    right_factors = rng.normal(size=(samples - split, 2))
    for column in range(dimensions):
        matrix[split:, column] = (
            0.75 * right_factors[:, 0]
            + (0.75 if column % 2 == 0 else -0.75) * right_factors[:, 1]
            + rng.normal(0.0, 0.35, samples - split)
        )
    return matrix, split


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a 95% Wilson interval for a binomial rate."""

    if total < 1:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_trials(
    rows: Sequence[dict[str, Any]],
    *,
    expected_split: int | None,
    tolerance: int,
) -> dict[str, Any]:
    """Summarize detection and localization without hiding missed trials."""

    detected = [row for row in rows if row["detected"]]
    summary: dict[str, Any] = {
        "trials": len(rows),
        "detections": len(detected),
        "detection_rate": len(detected) / len(rows) if rows else 0.0,
        "detection_rate_wilson95": wilson_interval(len(detected), len(rows)),
    }
    if expected_split is None:
        summary["median_split_points"] = (
            float(np.median([len(row["split_points"]) for row in rows]))
            if rows
            else 0.0
        )
        return summary
    errors = []
    root_errors = []
    for row in detected:
        points = row["split_points"]
        if points:
            errors.append(min(abs(point - expected_split) for point in points))
        root_errors.append(abs(row["root_best_split"] - expected_split))
    localized = sum(error <= tolerance for error in errors)
    root_localized = sum(error <= tolerance for error in root_errors)
    summary.update(
        {
            "expected_split": expected_split,
            "tolerance": tolerance,
            "localized_trials": localized,
            "localized_rate_all_trials": localized / len(rows) if rows else 0.0,
            "localized_rate_wilson95": wilson_interval(localized, len(rows)),
            "median_absolute_error_detected": (
                float(np.median(errors)) if errors else None
            ),
            "root_localized_trials": root_localized,
            "root_localized_rate_all_trials": (
                root_localized / len(rows) if rows else 0.0
            ),
            "root_localized_rate_wilson95": wilson_interval(
                root_localized, len(rows)
            ),
            "root_median_absolute_error_detected": (
                float(np.median(root_errors)) if root_errors else None
            ),
            "median_split_points": (
                float(np.median([len(row["split_points"]) for row in rows]))
                if rows
                else 0.0
            ),
        }
    )
    return summary


def heldout_gain_gate(
    event_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    expected_split: int,
    tolerance: int,
    quantile: float,
) -> dict[str, Any]:
    """Calibrate gain on early files and score a frozen gate on later files."""

    if not 0.0 < quantile <= 1.0:
        raise ValueError("gain quantile must be in (0, 1]")
    files = sorted(
        {str(row["file"]) for row in baseline_rows},
        key=lambda value: int(Path(value).stem.rsplit("_", 1)[-1]),
    )
    split = max(1, len(files) // 2)
    calibration_files = set(files[:split])
    validation_files = set(files[split:])
    calibration = [
        row for row in baseline_rows if row["file"] in calibration_files
    ]
    validation_baseline = [
        row for row in baseline_rows if row["file"] in validation_files
    ]
    validation_events = [
        row for row in event_rows if row["file"] in validation_files
    ]
    if not calibration or not validation_baseline or not validation_events:
        raise ValueError("gain gate requires non-empty calibration and validation files")
    threshold = float(
        np.quantile(
            [row["root_gain"] for row in calibration],
            quantile,
            method="higher",
        )
    )

    def gated(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "detected": bool(row["root_gain"] > threshold),
                "split_points": [row["root_best_split"]],
            }
            for row in rows
        ]

    return {
        "quantile": quantile,
        "threshold": threshold,
        "calibration_files": len(calibration_files),
        "calibration_baselines": len(calibration),
        "validation_files": len(validation_files),
        "validation_events": summarize_trials(
            gated(validation_events),
            expected_split=expected_split,
            tolerance=tolerance,
        ),
        "validation_pre_event": summarize_trials(
            gated(validation_baseline), expected_split=None, tolerance=0
        ),
    }


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    summaries = []
    tolerance = max(5, int(round(args.synthetic_samples * 0.05)))
    for scenario_index, scenario in enumerate(SCENARIOS):
        for method_index, method in enumerate(METHODS):
            for view_index, view in enumerate(VIEWS):
                trials = []
                expected_split = None
                for replicate in range(args.synthetic_repeats):
                    matrix, expected_split = synthetic_matrix(
                        scenario,
                        samples=args.synthetic_samples,
                        dimensions=args.synthetic_dimensions,
                        seed=args.seed + scenario_index * 100000 + replicate,
                    )
                    scored_split = (
                        expected_split - 1
                        if expected_split is not None and view == "ar1_residual"
                        else expected_split
                    )
                    result = run_changeforest(
                        matrix,
                        method=method,
                        view=view,
                        seed=(
                            args.seed
                            + method_index * 1000000
                            + view_index * 500000
                            + scenario_index * 10000
                            + replicate
                        ),
                        permutations=args.permutations,
                        estimators=args.estimators,
                        jobs=args.jobs,
                    )
                    result.update({"scenario": scenario, "replicate": replicate})
                    trials.append(result)
                    rows.append(result)
                summaries.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "view": view,
                        **summarize_trials(
                            trials,
                            expected_split=scored_split,
                            tolerance=tolerance,
                        ),
                    }
                )
    result: dict[str, Any] = {"summaries": summaries}
    if args.include_trials:
        result["trials"] = rows
    return result


def pmubage_files(root: Path, kind: str, limit: int) -> list[Path]:
    paths = sorted((root / kind).glob(f"{kind}_*.npy"), key=lambda path: int(path.stem.split("_")[-1]))
    return paths[:limit] if limit > 0 else paths


def checkpoint_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": 1,
        "pmubage_root": str(args.pmubage_root.resolve()),
        "pmu_file_limit": args.pmu_file_limit,
        "pmu_events_per_file": args.pmu_events_per_file,
        "pmu_sensor_limit": args.pmu_sensor_limit,
        "pmu_event_onset": args.pmu_event_onset,
        "permutations": args.permutations,
        "estimators": args.estimators,
        "seed": args.seed,
    }


def load_checkpoint(args: argparse.Namespace) -> dict[tuple[str, str, int, str, str, str], dict[str, Any]]:
    if args.checkpoint is None or not args.checkpoint.exists():
        return {}
    rows: dict[tuple[str, str, int, str, str, str], dict[str, Any]] = {}
    with args.checkpoint.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("signature") != checkpoint_signature(args):
            raise ValueError("checkpoint signature does not match current arguments")
        for line in handle:
            row = json.loads(line)
            key = (
                row["kind"],
                row["file"],
                int(row["event_index"]),
                row["method"],
                row["view"],
                row["scope"],
            )
            rows[key] = row["result"]
    return rows


def initialize_checkpoint(args: argparse.Namespace) -> None:
    if args.checkpoint is None or args.checkpoint.exists():
        return
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.write_text(
        json.dumps({"signature": checkpoint_signature(args)}) + "\n",
        encoding="utf-8",
    )


def append_checkpoint(
    args: argparse.Namespace,
    *,
    kind: str,
    path: Path,
    event_index: int,
    method: str,
    view: str,
    scope: str,
    result: dict[str, Any],
) -> None:
    if args.checkpoint is None:
        return
    row = {
        "kind": kind,
        "file": str(path),
        "event_index": event_index,
        "method": method,
        "view": view,
        "scope": scope,
        "result": result,
    }
    with args.checkpoint.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_pmubage(args: argparse.Namespace) -> dict[str, Any]:
    initialize_checkpoint(args)
    cached = load_checkpoint(args)
    resumed_rows = len(cached)
    computed_rows = 0
    corpora = []
    for kind, datatype_index in (("frequency", 3), ("voltage", 2)):
        keys = [(method, view) for method in METHODS for view in VIEWS]
        event_rows = {key: [] for key in keys}
        baseline_rows = {key: [] for key in keys}
        paths = pmubage_files(args.pmubage_root, kind, args.pmu_file_limit)
        event_counter = 0
        for file_index, path in enumerate(paths):
            tensor = np.load(path, mmap_mode="r")
            event_limit = tensor.shape[0]
            if args.pmu_events_per_file > 0:
                event_limit = min(event_limit, args.pmu_events_per_file)
            for event_index in range(event_limit):
                event = np.asarray(
                    tensor[event_index, datatype_index, : args.pmu_sensor_limit, :],
                    dtype=np.float64,
                ).T
                for method_index, method in enumerate(METHODS):
                    for view_index, view in enumerate(VIEWS):
                        base_seed = (
                            args.seed
                            + (0 if kind == "frequency" else 10000000)
                            + method_index * 1000000
                            + view_index * 500000
                            + file_index * 1000
                            + event_index
                        )
                        full_key = (
                            kind,
                            str(path),
                            event_index,
                            method,
                            view,
                            "full_event",
                        )
                        full = cached.get(full_key)
                        if full is None:
                            full = run_changeforest(
                                event,
                                method=method,
                                view=view,
                                seed=base_seed,
                                permutations=args.permutations,
                                estimators=args.estimators,
                                jobs=args.jobs,
                            )
                            full.update({"file": str(path), "event_index": event_index})
                            append_checkpoint(
                                args,
                                kind=kind,
                                path=path,
                                event_index=event_index,
                                method=method,
                                view=view,
                                scope="full_event",
                                result=full,
                            )
                            computed_rows += 1
                        event_rows[(method, view)].append(full)
                        baseline_key = (
                            kind,
                            str(path),
                            event_index,
                            method,
                            view,
                            "pre_event",
                        )
                        baseline = cached.get(baseline_key)
                        if baseline is None:
                            baseline = run_changeforest(
                                event[: args.pmu_event_onset],
                                method=method,
                                view=view,
                                seed=base_seed + 50000000,
                                permutations=args.permutations,
                                estimators=args.estimators,
                                jobs=args.jobs,
                            )
                            baseline.update({"file": str(path), "event_index": event_index})
                            append_checkpoint(
                                args,
                                kind=kind,
                                path=path,
                                event_index=event_index,
                                method=method,
                                view=view,
                                scope="pre_event",
                                result=baseline,
                            )
                            computed_rows += 1
                        baseline_rows[(method, view)].append(baseline)
                event_counter += 1
                if event_counter % args.progress_every == 0:
                    print(
                        f"pmu progress kind={kind} events={event_counter} "
                        f"computed_rows={computed_rows} resumed_rows={resumed_rows}",
                        flush=True,
                    )
        methods = []
        for method, view in keys:
            expected_split = (
                args.pmu_event_onset - 1
                if view == "ar1_residual"
                else args.pmu_event_onset
            )
            method_result = {
                "method": method,
                "view": view,
                "full_event": summarize_trials(
                    event_rows[(method, view)],
                    expected_split=expected_split,
                    tolerance=args.pmu_tolerance,
                ),
                "pre_event_negative": summarize_trials(
                    baseline_rows[(method, view)], expected_split=None, tolerance=0
                ),
                "heldout_gain_gate": heldout_gain_gate(
                    event_rows[(method, view)],
                    baseline_rows[(method, view)],
                    expected_split=expected_split,
                    tolerance=args.pmu_tolerance,
                    quantile=args.gain_quantile,
                ),
            }
            if args.include_trials:
                method_result["event_trials"] = event_rows[(method, view)]
                method_result["pre_event_trials"] = baseline_rows[(method, view)]
            methods.append(method_result)
        corpora.append(
            {
                "kind": kind,
                "datatype_index": datatype_index,
                "files": len(paths),
                "events": event_counter,
                "methods": methods,
            }
        )
    return {
        "corpora": corpora,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "resumed_rows": resumed_rows,
        "computed_rows": computed_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    return {
        "method": {
            "label_use": "none during detection; known answers used only for scoring",
            "algorithms": [
                "changeforest random-forest classifier two-sample gain with binary segmentation",
                "changeforest k-nearest-neighbor two-sample gain with binary segmentation",
            ],
            "views": [
                "raw observations (diagnostic only when serial dependence is present)",
                "per-channel AR(1) residuals estimated without event timing",
            ],
            "calibration": "package model-selection permutation test at alpha 0.05; no real-data tuning",
            "pmubage_ground_truth": "600 samples per event with fixed event start at sample 300",
            "gain_gate_status": (
                "post-hoc design after package p-values failed on autocorrelated "
                "baselines; threshold uses only first-half-file pre-event gains and "
                "is frozen on second-half files"
            ),
        },
        "parameters": {
            "seed": args.seed,
            "permutations": args.permutations,
            "estimators": args.estimators,
            "synthetic_repeats": args.synthetic_repeats,
            "synthetic_samples": args.synthetic_samples,
            "synthetic_dimensions": args.synthetic_dimensions,
            "pmu_sensor_limit": args.pmu_sensor_limit,
            "pmu_event_onset": args.pmu_event_onset,
            "pmu_tolerance": args.pmu_tolerance,
            "gain_quantile": args.gain_quantile,
        },
        "synthetic": run_synthetic(args),
        "pmubage": run_pmubage(args),
        "elapsed_seconds": time.time() - started,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmubage-root", type=Path, default=Path("data/pmuBAGE/data"))
    parser.add_argument("--pmu-file-limit", type=int, default=0)
    parser.add_argument("--pmu-events-per-file", type=int, default=0)
    parser.add_argument("--pmu-sensor-limit", type=int, default=100)
    parser.add_argument("--pmu-event-onset", type=int, default=300)
    parser.add_argument("--pmu-tolerance", type=int, default=15)
    parser.add_argument("--gain-quantile", type=float, default=0.99)
    parser.add_argument("--synthetic-repeats", type=int, default=100)
    parser.add_argument("--synthetic-samples", type=int, default=400)
    parser.add_argument("--synthetic-dimensions", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--include-trials", action="store_true")
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Established change-point benchmark")
    for row in report["synthetic"]["summaries"]:
        print(
            "  synthetic %-18s %-13s %-12s detect=%5.1f%% root=%5.1f%% any=%5.1f%%"
            % (
                row["scenario"],
                row["method"],
                row["view"],
                100.0 * row["detection_rate"],
                100.0 * row.get("root_localized_rate_all_trials", 0.0),
                100.0 * row.get("localized_rate_all_trials", 0.0),
            )
        )
    for corpus in report["pmubage"]["corpora"]:
        for row in corpus["methods"]:
            print(
                "  pmu %-9s %-13s %-12s events=%d detect=%5.1f%% root=%5.1f%% any=%5.1f%% pre-FP=%5.1f%%"
                % (
                    corpus["kind"],
                    row["method"],
                    row["view"],
                    corpus["events"],
                    100.0 * row["full_event"]["detection_rate"],
                    100.0 * row["full_event"]["root_localized_rate_all_trials"],
                    100.0 * row["full_event"]["localized_rate_all_trials"],
                    100.0 * row["pre_event_negative"]["detection_rate"],
                )
            )
            gate = row["heldout_gain_gate"]
            print(
                "    heldout gain q=%.3f threshold=%.3f detect=%5.1f%% root=%5.1f%% pre-FP=%5.1f%%"
                % (
                    gate["quantile"],
                    gate["threshold"],
                    100.0 * gate["validation_events"]["detection_rate"],
                    100.0
                    * gate["validation_events"]["root_localized_rate_all_trials"],
                    100.0 * gate["validation_pre_event"]["detection_rate"],
                )
            )
    print(f"  elapsed={report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
