"""Run a bounded spectral+stigmergy observation on synthetic pmuBAGE tensors.

pmuBAGE is useful as an adapter smoke test for grid-like multi-sensor data. It
is synthetic, so results from this script should not be treated as validation on
real grid disturbances.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray

from experiments.chbmit_observe import _multi_channel_emissions, _score_matrix, permutation_null_summary
from spectral_forecast.observation import ScoreName, build_stigmergy, observe_series


DATATYPES = ("real_power", "reactive_power", "voltage_magnitude", "frequency")


def robust_standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    y = np.asarray(values, dtype=np.float64)
    center = float(np.median(y))
    mad = float(np.median(np.abs(y - center)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(y))
    if scale < 1e-9:
        scale = 1.0
    return ((y - center) / scale).astype(np.float64)


def coherent_rows(
    results: Sequence[Any],
    *,
    score: ScoreName,
    threshold: float,
    min_active_sensors: int,
    sample_rate: float,
    top: int,
) -> list[dict[str, Any]]:
    anchors, matrix = _score_matrix(results, score)
    sensors = [result.series for result in results]
    emissions = _multi_channel_emissions(
        matrix,
        threshold=threshold,
        min_active_channels=min_active_sensors,
    )
    rows = []
    for row_index, emission in sorted(
        enumerate(emissions),
        key=lambda item: item[1],
        reverse=True,
    )[:top]:
        anchor = anchors[row_index]
        scores = matrix[row_index]
        active = [
            {"sensor": name, "score": float(value)}
            for name, value in zip(sensors, scores, strict=True)
            if value > threshold
        ]
        rows.append(
            {
                "index": anchor,
                "seconds": float(anchor) / sample_rate,
                "emission": float(emission),
                "active_sensor_count": len(active),
                "active_sensors": active,
                "max_score": float(np.max(scores)) if len(scores) else 0.0,
                "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            }
        )
    return rows


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    tensor = np.load(args.tensor, mmap_mode="r")
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D pmuBAGE tensor, got shape {tensor.shape}")
    if args.event_index < 0 or args.event_index >= tensor.shape[0]:
        raise ValueError(f"event index {args.event_index} outside tensor event axis {tensor.shape[0]}")
    if args.datatype_index < 0 or args.datatype_index >= tensor.shape[1]:
        raise ValueError(f"datatype index {args.datatype_index} outside tensor datatype axis {tensor.shape[1]}")

    sensor_count = min(args.sensor_limit, tensor.shape[2])
    values = np.asarray(
        tensor[args.event_index, args.datatype_index, :sensor_count, :],
        dtype=np.float64,
    )
    series = [robust_standardize(values[index]) for index in range(sensor_count)]
    results = [
        observe_series(
            row,
            series=f"pmu_{index:03d}",
            baseline_size=args.baseline,
            adaptive_window=args.adaptive_window,
            stride=args.stride,
            sample_rate=args.sample_rate,
        )
        for index, row in enumerate(series)
    ]
    stig = build_stigmergy(
        results,
        score=args.score,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
    )
    null = permutation_null_summary(
        results,
        score=args.score,
        threshold=args.emission_threshold,
        min_active_channels=args.min_active_sensors,
        repeats=args.null_repeats,
        seed=args.seed,
    )

    top = coherent_rows(
        results,
        score=args.score,
        threshold=args.emission_threshold,
        min_active_sensors=args.min_active_sensors,
        sample_rate=args.sample_rate,
        top=args.top,
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "pmuBAGE synthetic PMU events",
            "source": "https://github.com/NanpengYu/pmuBAGE",
            "validation_boundary": "synthetic adapter smoke test, not real-grid validation",
        },
        "tensor": {
            "path": str(args.tensor),
            "shape": list(tensor.shape),
            "event_index": args.event_index,
            "datatype_index": args.datatype_index,
            "datatype": DATATYPES[args.datatype_index] if args.datatype_index < len(DATATYPES) else "unknown",
            "sensor_count": sensor_count,
        },
        "parameters": {
            "sample_rate": args.sample_rate,
            "baseline": args.baseline,
            "adaptive_window": args.adaptive_window,
            "stride": args.stride,
            "score": args.score,
            "emission_threshold": args.emission_threshold,
            "decay": args.decay,
            "min_active_sensors": args.min_active_sensors,
            "null_repeats": args.null_repeats,
            "seed": args.seed,
        },
        "summary": {
            "points_per_sensor": len(results[0].points) if results else 0,
            "stigmergy_points": len(stig.points),
            "permutation_null": null,
            "coherent_top": top,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    tensor = report["tensor"]
    null = report["summary"]["permutation_null"]
    print("pmuBAGE spectral+stigmergy smoke")
    print(
        "  tensor=%s event=%d datatype=%s sensors=%d"
        % (
            Path(tensor["path"]).name,
            tensor["event_index"],
            tensor["datatype"],
            tensor["sensor_count"],
        )
    )
    print(
        "  observed=%.3f null_mean=%.3f z=%s p_ge=%.4f exceed=%d/%d active_windows=%d"
        % (
            null["observed_total"],
            null["null_mean"],
            "None" if null["z_effect"] is None else f"{null['z_effect']:.2f}",
            null["empirical_p_ge_observed"],
            null["null_exceedances"],
            null["null_repeats"],
            null["observed_active_windows"],
        )
    )
    for row in report["summary"]["coherent_top"][:5]:
        active = ",".join(item["sensor"] for item in row["active_sensors"][:6])
        print(
            "    t=%5.2fs emission=%8.3f active=%d %s"
            % (row["seconds"], row["emission"], row["active_sensor_count"], active)
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", type=Path, help="pmuBAGE .npy tensor")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--datatype-index", type=int, default=3, help="PQVF axis index, default frequency")
    parser.add_argument("--sensor-limit", type=int, default=24)
    parser.add_argument("--sample-rate", type=float, default=30.0, help="600 samples over 20 seconds")
    parser.add_argument("--baseline", type=int, default=128)
    parser.add_argument("--adaptive-window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
    )
    parser.add_argument("--emission-threshold", type=float, default=3.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--min-active-sensors", type=int, default=3)
    parser.add_argument("--null-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.baseline <= args.adaptive_window:
        parser.error("--baseline must be greater than --adaptive-window")
    if args.sensor_limit < 1:
        parser.error("--sensor-limit must be >= 1")
    if args.min_active_sensors < 1:
        parser.error("--min-active-sensors must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_experiment(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
