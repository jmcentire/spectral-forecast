"""Test whether CHB-MIT observer lift depends on channels sharing a recording.

The trajectory-regroup null preserves every complete channel score trajectory
and shared anchor-position profile, then rebuilds synthetic recordings from
channels drawn across different files. Seizure labels are not read or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.chbmit_autotune import load_eeg_file
from spectral_forecast.autotune import AutoTuneConfig, build_autotune_observation
from spectral_forecast.relationship_nulls import grouped_matrix_null_summary


def _cache_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


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


def _build_file_matrix(
    payload: tuple[str, tuple[str, ...], float, dict[str, Any], str | None],
) -> tuple[str, np.ndarray, bool]:
    path_text, channels, target_sample_rate, candidate, cache_dir_text = payload
    path = Path(path_text)
    cache_path = None
    if cache_dir_text is not None:
        cache_dir = Path(cache_dir_text)
        key = {
            "version": 1,
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "channels": list(channels),
            "target_sample_rate": target_sample_rate,
            "candidate": candidate,
        }
        cache_path = cache_dir / f"{_cache_digest(key)}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                return path.name, np.asarray(saved["matrix"], dtype=np.float64), True
    config = _candidate_config(candidate)
    eeg_file = load_eeg_file(
        path,
        channels=channels,
        summary_labels={},
        target_sample_rate=target_sample_rate,
    )
    observation = build_autotune_observation(
        eeg_file.series,
        config,
        sample_rate=eeg_file.sample_rate,
    )
    matrix = np.asarray(observation.matrix, dtype=np.float64)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, matrix=matrix)
        tmp.replace(cache_path)
    return path.name, matrix, False


def _dominant_shape_matrices(
    rows: Sequence[tuple[str, np.ndarray]],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not rows:
        raise ValueError("no observer matrices were built")
    shape_counts = Counter(tuple(matrix.shape) for _, matrix in rows)
    dominant_shape, dominant_count = shape_counts.most_common(1)[0]
    included = [matrix for _, matrix in rows if tuple(matrix.shape) == dominant_shape]
    excluded = [
        {"file": name, "shape": list(matrix.shape)}
        for name, matrix in rows
        if tuple(matrix.shape) != dominant_shape
    ]
    return included, {
        "dominant_shape": list(dominant_shape),
        "dominant_shape_files": dominant_count,
        "shape_counts": {
            "x".join(str(value) for value in shape): count
            for shape, count in sorted(shape_counts.items())
        },
        "excluded_non_dominant_shape": excluded,
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
    slot_regroup = grouped_matrix_null_summary(
        matrices,
        threshold=config.emission_threshold,
        min_active_series=config.min_active_series,
        null_mode="trajectory-regroup-within-slot",
        null_repeats=null_repeats,
        seed=seed + 2,
    )
    return {
        "anchor_permute": anchor.to_dict(),
        "trajectory_regroup": regroup.to_dict(),
        "trajectory_regroup_within_channel": slot_regroup.to_dict(),
        "group_specific_fraction_of_anchor_permute_surplus": (
            slot_regroup.observed_minus_null / anchor.observed_minus_null
            if anchor.observed_minus_null > 0
            else None
        ),
        "diagnosis": (
            "recording_specific_channel_structure"
            if slot_regroup.observed_minus_null > 0
            and slot_regroup.empirical_p_ge_observed <= 0.05
            else "shared_anchor_template_or_recording_independent_distribution"
            if anchor.observed_minus_null > 0 and anchor.empirical_p_ge_observed <= 0.05
            else "no_anchor_alignment_surplus"
        ),
    }


def assess_subject(
    path: Path,
    *,
    null_repeats: int,
    seed: int,
    progress: bool,
    workers: int,
    matrix_cache_dir: Path | None,
) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    candidate = source["validation"]["selected_candidate"]
    if candidate is None:
        raise ValueError(f"{path} has no selected validation candidate")
    config = _candidate_config(candidate)
    calibration_names = set(source["calibration"]["files"])
    validation_names = set(source["validation"]["files"])
    rows: dict[str, list[tuple[str, np.ndarray]]] = {
        "calibration": [],
        "validation": [],
    }
    started = time.time()
    payloads = [
        (
            str(eeg_path_text),
            tuple(source["parameters"]["channels"]),
            float(source["parameters"]["target_sample_rate"]),
            dict(candidate),
            str(matrix_cache_dir) if matrix_cache_dir is not None else None,
        )
        for eeg_path_text in source["parameters"]["files"]
    ]
    if workers == 1:
        built = map(_build_file_matrix, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        built = executor.map(_build_file_matrix, payloads)
    cache_hits = 0
    try:
        for index, (file_name, matrix, cache_hit) in enumerate(built):
            cache_hits += int(cache_hit)
            segment = (
                "calibration"
                if file_name in calibration_names
                else "validation"
                if file_name in validation_names
                else None
            )
            if segment is None:
                raise ValueError(f"{file_name} is absent from report file splits")
            rows[segment].append((file_name, matrix))
            if progress:
                print(
                    "chbmit_group_null subject=%s files=%d/%d segment=%s anchors=%d cache_hit=%s"
                    % (
                        source["dataset"]["summary_path"],
                        index + 1,
                        len(payloads),
                        segment,
                        matrix.shape[0],
                        cache_hit,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if workers != 1:
            executor.shutdown()

    segments: dict[str, Any] = {}
    for segment_index, segment in enumerate(("calibration", "validation")):
        matrices, shape_summary = _dominant_shape_matrices(rows[segment])
        comparison = _null_comparison(
            matrices,
            config=config,
            null_repeats=null_repeats,
            seed=seed + 100 * segment_index,
        )
        segments[segment] = {
            "files_requested": len(rows[segment]),
            "files_assessed": len(matrices),
            "shape": shape_summary,
            "comparison": comparison,
        }
    return {
        "source_report": str(path),
        "subject": source["dataset"]["summary_path"],
        "label_use": "none",
        "candidate": dict(candidate),
        "matrix_cache": {
            "directory": str(matrix_cache_dir) if matrix_cache_dir is not None else None,
            "hits": cache_hits,
            "misses": len(payloads) - cache_hits,
        },
        "segments": segments,
        "elapsed_seconds": time.time() - started,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autotune-report", type=Path, action="append", required=True)
    parser.add_argument("--null-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "selection": "frozen candidate from each source report",
            "label_use": "none",
            "trajectory_regroup": (
                "preserve complete channel score trajectories and shared anchor profile; "
                "destroy original recording membership"
            ),
            "trajectory_regroup_within_channel": (
                "preserve complete trajectories and electrode/channel identity; destroy "
                "only shared recording membership"
            ),
            "shape_handling": (
                "assess the dominant observer-matrix shape in each split; report excluded "
                "short files"
            ),
        },
        "null_repeats": args.null_repeats,
        "subjects": [],
    }
    for index, path in enumerate(args.autotune_report):
        subject = assess_subject(
            path,
            null_repeats=args.null_repeats,
            seed=args.seed + 10_000 * index,
            progress=args.progress,
            workers=args.workers,
            matrix_cache_dir=args.matrix_cache_dir,
        )
        report["subjects"].append(subject)
        if args.checkpoint is not None:
            _write_json(args.checkpoint, report)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
