"""Test whether CDIP observer aggregate lift depends on original buoy grouping.

The trajectory-regroup null preserves every complete observer-score trajectory
and the shared anchor-position profile, then rebuilds groups from unrelated
trajectories. It attacks the shared-observer-template failure mode left open by
within-window timing permutation.
"""

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

from experiments.cdip_autotune import _cache_digest, _load_observation_npz
from spectral_forecast.autotune import AutoTuneConfig
from spectral_forecast.relationship_nulls import grouped_matrix_null_summary


def _parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _nearby_float_values(value: float, *, radius: int) -> list[float]:
    values = [float(value)]
    lower = float(value)
    upper = float(value)
    for _ in range(radius):
        lower = float(np.nextafter(lower, -np.inf))
        upper = float(np.nextafter(upper, np.inf))
        values.extend((lower, upper))
    return values


def _candidate_config(row: Mapping[str, Any]) -> AutoTuneConfig:
    return AutoTuneConfig(
        baseline_size=int(row["baseline_size"]),
        adaptive_window=int(row["adaptive_window"]),
        stride=int(row["stride"]),
        score=str(row["score"]),  # type: ignore[arg-type]
        emission_threshold=float(row["emission_threshold"]),
        decay=float(row["decay"]),
        min_active_series=int(row["min_active_series"]),
    )


def _cache_path_for_window(
    window: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    preprocess: str,
    cache_dir: Path,
    phase_surrogate_seed: int,
    ulp_radius: int,
) -> Path | None:
    parsed_start = datetime.fromisoformat(str(window["start"])).timestamp()
    for start_time in _nearby_float_values(parsed_start, radius=ulp_radius):
        key = {
            "version": 1,
            "window": {
                "paths": list(window["paths"]),
                "start_time": start_time,
                "n_samples": int(window["samples"]),
                "target_rate": float(window["sample_rate"]),
            },
            "candidate": {
                "preprocess": preprocess,
                "baseline_size": int(candidate["baseline_size"]),
                "adaptive_window": int(candidate["adaptive_window"]),
                "stride": int(candidate["stride"]),
                "score": str(candidate["score"]),
            },
            "phase_surrogate_seed": phase_surrogate_seed,
        }
        path = cache_dir / f"{_cache_digest(key)}.npz"
        if path.exists():
            return path
    return None


def load_segment_matrices(
    report: Mapping[str, Any],
    *,
    segment: str,
    preprocess: str,
    cache_dir: Path,
    phase_surrogate_seed: int,
    ulp_radius: int,
    progress_every: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    candidate = report["validation"]["selected_candidate"]
    matrices: list[np.ndarray] = []
    missing: list[dict[str, Any]] = []
    shape_counts: dict[str, int] = {}
    windows = report[segment]["windows"]
    for index, window in enumerate(windows):
        path = _cache_path_for_window(
            window,
            candidate,
            preprocess=preprocess,
            cache_dir=cache_dir,
            phase_surrogate_seed=phase_surrogate_seed,
            ulp_radius=ulp_radius,
        )
        if path is None:
            missing.append(
                {
                    "group_index": window.get("group_index"),
                    "window_index": window.get("window_index"),
                    "start": window.get("start"),
                }
            )
        else:
            matrix = np.asarray(_load_observation_npz(path).matrix, dtype=np.float64)
            matrices.append(matrix)
            shape = "x".join(str(value) for value in matrix.shape)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(
                "cdip_group_null load preprocess=%s segment=%s windows=%d/%d missing=%d"
                % (preprocess, segment, index + 1, len(windows), len(missing)),
                file=sys.stderr,
                flush=True,
            )
    return matrices, {
        "requested_windows": len(windows),
        "loaded_windows": len(matrices),
        "missing_windows": len(missing),
        "missing_examples": missing[:10],
        "shape_counts": dict(sorted(shape_counts.items())),
    }


def _null_comparison(
    matrices: Sequence[np.ndarray],
    *,
    config: AutoTuneConfig,
    null_repeats: int,
    seed: int,
) -> dict[str, Any]:
    anchor = grouped_matrix_null_summary(
        matrices,
        threshold=config.emission_threshold,
        min_active_series=config.min_active_series,
        null_mode="anchor-permute",
        null_repeats=null_repeats,
        seed=seed,
    )
    regroup = grouped_matrix_null_summary(
        matrices,
        threshold=config.emission_threshold,
        min_active_series=config.min_active_series,
        null_mode="trajectory-regroup",
        null_repeats=null_repeats,
        seed=seed + 1,
    )
    anchor_delta = anchor.observed_minus_null
    regroup_delta = regroup.observed_minus_null
    return {
        "anchor_permute": anchor.to_dict(),
        "trajectory_regroup": regroup.to_dict(),
        "group_specific_fraction_of_anchor_permute_surplus": (
            regroup_delta / anchor_delta if anchor_delta > 0 else None
        ),
        "diagnosis": (
            "group_specific_structure"
            if regroup_delta > 0 and regroup.empirical_p_ge_observed <= 0.05
            else "shared_anchor_template_or_group_independent_distribution"
            if anchor_delta > 0 and anchor.empirical_p_ge_observed <= 0.05
            else "no_anchor_alignment_surplus"
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_group_null(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.autotune_report.read_text(encoding="utf-8"))
    candidate = source["validation"]["selected_candidate"]
    config = _candidate_config(candidate)
    cache_dir = args.cache_dir or Path(source["matrix_cache"]["directory"])
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(args.autotune_report),
        "cache_dir": str(cache_dir),
        "method": {
            "selection": "frozen candidate from source report",
            "anchor_permute": (
                "preserve each trajectory score distribution; independently destroy "
                "anchor alignment"
            ),
            "trajectory_regroup": (
                "preserve every complete trajectory and shared anchor-position profile; "
                "destroy original buoy group membership"
            ),
        },
        "candidate": dict(candidate),
        "null_repeats": args.null_repeats,
        "phase_surrogate_seed": args.phase_surrogate_seed,
        "preprocess": {},
    }
    for preprocess_index, preprocess in enumerate(_parse_str_list(args.preprocesses)):
        preprocess_result: dict[str, Any] = {}
        for segment_index, segment in enumerate(("calibration", "validation")):
            started = time.time()
            print(
                f"cdip_group_null start preprocess={preprocess} segment={segment}",
                file=sys.stderr,
                flush=True,
            )
            matrices, load_summary = load_segment_matrices(
                source,
                segment=segment,
                preprocess=preprocess,
                cache_dir=cache_dir,
                phase_surrogate_seed=args.phase_surrogate_seed,
                ulp_radius=args.ulp_radius,
                progress_every=args.progress_every,
            )
            if load_summary["missing_windows"] > 0:
                raise ValueError(
                    "cache lookup missed %d/%d windows for %s/%s"
                    % (
                        load_summary["missing_windows"],
                        load_summary["requested_windows"],
                        preprocess,
                        segment,
                    )
                )
            shapes = {tuple(matrix.shape) for matrix in matrices}
            if len(shapes) != 1:
                raise ValueError(f"mixed matrix shapes for {preprocess}/{segment}: {shapes}")
            comparison = _null_comparison(
                matrices,
                config=config,
                null_repeats=args.null_repeats,
                seed=args.seed + 10_000 * preprocess_index + 100 * segment_index,
            )
            preprocess_result[segment] = {
                "load": load_summary,
                "matrix_shape": list(matrices[0].shape),
                "comparison": comparison,
                "elapsed_seconds": time.time() - started,
            }
            report["preprocess"][preprocess] = preprocess_result
            if args.checkpoint is not None:
                _write_json(args.checkpoint, report)
            print(
                "cdip_group_null done preprocess=%s segment=%s diagnosis=%s elapsed=%.1fs"
                % (
                    preprocess,
                    segment,
                    comparison["diagnosis"],
                    preprocess_result[segment]["elapsed_seconds"],
                ),
                file=sys.stderr,
                flush=True,
            )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autotune-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--preprocesses",
        default="none,highpass,highpass-dominant-mask",
    )
    parser.add_argument("--null-repeats", type=int, default=200)
    parser.add_argument("--phase-surrogate-seed", type=int, default=20260602)
    parser.add_argument("--ulp-radius", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_group_null(args)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
