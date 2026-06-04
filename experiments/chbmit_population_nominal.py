"""Cross-fit a population spectral nominal and evaluate held-out CHB-MIT brains.

For each outer fold, two subjects develop a label-free spectral-shape geometry
and robust channel-level population nominal. The third subject is then scored
without contributing to candidate selection or the nominal. Seizure labels are
used only after scoring for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray

from experiments.chbmit_observe import (
    DEFAULT_CHANNELS,
    _point_phase,
    downsample_and_standardize,
    load_edf_signals,
    parse_chbmit_summary,
)
from spectral_forecast.population import (
    fit_population_nominal,
    population_deviance_matrix,
    population_feature_deviance_tensor,
)
from spectral_forecast.relationship_nulls import (
    grouped_emission_total,
    grouped_matrix_null_summary,
)


@dataclass(frozen=True)
class FeatureGeometry:
    """Label-free population spectral-feature geometry."""

    window_samples: int
    stride: int
    spectral_bins: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PopulationCandidate:
    """Population nominal plus stigmergic relationship settings."""

    geometry: FeatureGeometry
    feature_quantile: float
    emission_threshold: float
    min_active_channels: int

    def to_dict(self) -> dict[str, object]:
        return {
            **self.geometry.to_dict(),
            "feature_quantile": self.feature_quantile,
            "emission_threshold": self.emission_threshold,
            "min_active_channels": self.min_active_channels,
        }


@dataclass(frozen=True)
class FeatureFile:
    """One recording represented as generic channel-feature observations."""

    subject: str
    path: str
    anchors: NDArray[np.int64]
    tensor: NDArray[np.float64]
    entity_names: tuple[str, ...]
    feature_names: tuple[str, ...]


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _spectral_shape_tensor(
    series: Mapping[str, NDArray[np.floating]],
    *,
    window_samples: int,
    stride: int,
    spectral_bins: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], tuple[str, ...]]:
    if window_samples < 16:
        raise ValueError("window_samples must be >= 16")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if spectral_bins < 2:
        raise ValueError("spectral_bins must be >= 2")
    names = tuple(series)
    n = min(len(values) for values in series.values())
    anchors = np.arange(window_samples, n + 1, stride, dtype=np.int64)
    if len(anchors) < 8:
        raise ValueError("too few population feature windows")
    feature_names = tuple(
        [f"log_power_fraction_{index}" for index in range(spectral_bins)]
        + [
            "spectral_entropy",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_peak_fraction",
            "spectral_flatness",
            "lag1_autocorrelation",
        ]
    )
    tensor = np.empty((len(anchors), len(names), len(feature_names)), dtype=np.float64)
    taper = np.hanning(window_samples)
    for channel_index, name in enumerate(names):
        values = np.asarray(series[name][:n], dtype=np.float64)
        for row_index, anchor in enumerate(anchors):
            window = values[int(anchor) - window_samples : int(anchor)]
            centered = window - float(np.mean(window))
            tapered = centered * taper
            power = np.abs(np.fft.rfft(tapered))[1:] ** 2
            if len(power) < spectral_bins:
                raise ValueError("spectral_bins exceeds usable frequency bins")
            total = float(np.sum(power))
            if total <= 1e-18:
                fractions = np.full(spectral_bins, 1.0 / spectral_bins)
                full_probs = np.full(len(power), 1.0 / len(power))
            else:
                full_probs = power / total
                fractions = np.asarray(
                    [np.sum(part) for part in np.array_split(full_probs, spectral_bins)],
                    dtype=np.float64,
                )
            frequencies = np.linspace(0.0, 1.0, len(full_probs), dtype=np.float64)
            centroid = float(np.sum(frequencies * full_probs))
            bandwidth = float(
                np.sqrt(np.sum(((frequencies - centroid) ** 2) * full_probs))
            )
            entropy = -float(np.sum(full_probs * np.log(full_probs + 1e-18)))
            entropy /= math.log(len(full_probs)) if len(full_probs) > 1 else 1.0
            flatness = float(
                np.exp(np.mean(np.log(power + 1e-18))) / max(np.mean(power), 1e-18)
            )
            ac1 = (
                float(np.corrcoef(centered[:-1], centered[1:])[0, 1])
                if np.std(centered[:-1]) > 1e-12 and np.std(centered[1:]) > 1e-12
                else 0.0
            )
            tensor[row_index, channel_index] = np.concatenate(
                [
                    np.log(fractions + 1e-18),
                    np.asarray(
                        [
                            entropy,
                            centroid,
                            bandwidth,
                            float(np.max(full_probs)),
                            flatness,
                            ac1 if np.isfinite(ac1) else 0.0,
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
    return anchors, tensor, feature_names


def _feature_cache_payload(
    path: Path,
    *,
    subject: str,
    channels: Sequence[str],
    target_sample_rate: float,
    geometry: FeatureGeometry,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "version": 1,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "subject": subject,
        "channels": list(channels),
        "target_sample_rate": target_sample_rate,
        "geometry": geometry.to_dict(),
    }


def _build_feature_file(
    payload: tuple[str, str, tuple[str, ...], float, dict[str, int], str],
) -> tuple[str, str, bool]:
    subject, path_text, channels, target_sample_rate, geometry_row, cache_dir_text = payload
    path = Path(path_text)
    geometry = FeatureGeometry(**geometry_row)
    cache_dir = Path(cache_dir_text)
    cache_payload = _feature_cache_payload(
        path,
        subject=subject,
        channels=channels,
        target_sample_rate=target_sample_rate,
        geometry=geometry,
    )
    cache_path = cache_dir / f"{_digest(cache_payload)}.npz"
    if cache_path.exists():
        return subject, str(cache_path), True
    raw = load_edf_signals(path, channels)
    series = {
        signal.label: downsample_and_standardize(
            signal.values,
            source_rate=signal.sample_rate,
            target_rate=target_sample_rate,
        )
        for signal in raw
    }
    anchors, tensor, feature_names = _spectral_shape_tensor(
        series,
        window_samples=geometry.window_samples,
        stride=geometry.stride,
        spectral_bins=geometry.spectral_bins,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            subject=np.asarray([subject]),
            path=np.asarray([str(path)]),
            anchors=anchors,
            tensor=tensor,
            entity_names=np.asarray(tuple(series)),
            feature_names=np.asarray(feature_names),
        )
    tmp.replace(cache_path)
    return subject, str(cache_path), False


def _load_feature_file(path: Path) -> FeatureFile:
    with np.load(path, allow_pickle=False) as saved:
        return FeatureFile(
            subject=str(saved["subject"][0]),
            path=str(saved["path"][0]),
            anchors=np.asarray(saved["anchors"], dtype=np.int64),
            tensor=np.asarray(saved["tensor"], dtype=np.float64),
            entity_names=tuple(str(value) for value in saved["entity_names"]),
            feature_names=tuple(str(value) for value in saved["feature_names"]),
        )


def _paths_for_subject(data_root: Path, subject: str, max_files: int) -> list[Path]:
    paths = sorted((data_root / subject).glob(f"{subject}_*.edf"))
    return paths[:max_files] if max_files > 0 else paths


def _file_split(files: Sequence[FeatureFile]) -> tuple[list[FeatureFile], list[FeatureFile]]:
    ordered = sorted(files, key=lambda item: item.path)
    return ordered[::2], ordered[1::2]


def _dominant_shape(matrices: Sequence[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    counts: dict[tuple[int, int], int] = {}
    for matrix in matrices:
        counts[tuple(matrix.shape)] = counts.get(tuple(matrix.shape), 0) + 1
    shape = max(counts, key=counts.get)
    return [matrix for matrix in matrices if tuple(matrix.shape) == shape]


def _emission_rows(
    matrix: NDArray[np.float64],
    *,
    threshold: float,
    min_active_channels: int,
) -> NDArray[np.float64]:
    excess = np.maximum(matrix - threshold, 0.0)
    active = np.sum(excess > 0.0, axis=1)
    emissions = np.sum(excess, axis=1)
    emissions[active < min_active_channels] = 0.0
    return emissions.astype(np.float64)


def _score_files(
    files: Sequence[FeatureFile],
    *,
    nominal: Any,
    candidate: PopulationCandidate,
) -> list[NDArray[np.float64]]:
    return [
        population_deviance_matrix(
            item.tensor,
            nominal,
            feature_quantile=candidate.feature_quantile,
        )
        for item in files
    ]


def _candidate_quality(
    matrices: Sequence[NDArray[np.float64]],
    *,
    candidate: PopulationCandidate,
    null_repeats: int,
    seed: int,
) -> dict[str, Any]:
    same_shape = _dominant_shape(matrices)
    summary = grouped_matrix_null_summary(
        same_shape,
        threshold=candidate.emission_threshold,
        min_active_series=candidate.min_active_channels,
        null_mode="trajectory-regroup-within-slot",
        null_repeats=null_repeats,
        seed=seed,
    )
    positive_files = 0
    active_rows = 0
    total_rows = 0
    active_cells = 0
    total_cells = 0
    for matrix in matrices:
        emissions = _emission_rows(
            matrix,
            threshold=candidate.emission_threshold,
            min_active_channels=candidate.min_active_channels,
        )
        positive_files += int(np.any(emissions > 0.0))
        active_rows += int(np.sum(emissions > 0.0))
        total_rows += len(emissions)
        active_cells += int(np.sum(matrix > candidate.emission_threshold))
        total_cells += matrix.size
    file_support = positive_files / max(len(matrices), 1)
    row_support = active_rows / max(total_rows, 1)
    saturation = active_cells / max(total_cells, 1)
    z = float(summary.z_effect) if summary.z_effect is not None else float("-inf")
    accepted = bool(
        summary.observed_minus_null > 0.0
        and summary.empirical_p_ge_observed <= 0.05
        and z >= 2.0
        and file_support >= 0.5
        and saturation < 0.5
    )
    quality = (
        (z if np.isfinite(z) else -100.0)
        + file_support
        + min(row_support, 0.5)
        - 4.0 * saturation
    )
    return {
        "candidate": candidate.to_dict(),
        "accepted": accepted,
        "quality": quality,
        "group_null": summary.to_dict(),
        "files_scored": len(matrices),
        "same_shape_files": len(same_shape),
        "file_support": file_support,
        "row_support": row_support,
        "saturation": saturation,
    }


def _phase_attribution(
    files: Sequence[FeatureFile],
    matrices: Sequence[NDArray[np.float64]],
    *,
    data_root: Path,
    candidate: PopulationCandidate,
    sample_rate: float,
    preictal_seconds: float,
    postictal_seconds: float,
) -> dict[str, Any]:
    phase_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    for item, matrix in zip(files, matrices, strict=True):
        summary = parse_chbmit_summary(
            data_root / item.subject / f"{item.subject}-summary.txt"
        )
        seizures = summary.get(Path(item.path).name)
        intervals = seizures.seizures if seizures is not None else ()
        emissions = _emission_rows(
            matrix,
            threshold=candidate.emission_threshold,
            min_active_channels=candidate.min_active_channels,
        )
        for anchor, emission in zip(item.anchors, emissions, strict=True):
            phase = _point_phase(
                float(anchor) / sample_rate,
                intervals,
                preictal_seconds=preictal_seconds,
                postictal_seconds=postictal_seconds,
            )
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            positive_counts[phase] = positive_counts.get(phase, 0) + int(emission > 0.0)
    return {
        "label_use": "posthoc attribution only",
        "phase_counts": dict(sorted(phase_counts.items())),
        "positive_phase_counts": dict(sorted(positive_counts.items())),
        "positive_rates": {
            phase: positive_counts.get(phase, 0) / count
            for phase, count in sorted(phase_counts.items())
        },
    }


def _heldout_summary(
    files: Sequence[FeatureFile],
    matrices: Sequence[NDArray[np.float64]],
    *,
    nominal: Any,
    candidate: PopulationCandidate,
    null_repeats: int,
    seed: int,
) -> dict[str, Any]:
    same_shape = _dominant_shape(matrices)
    anchor = grouped_matrix_null_summary(
        same_shape,
        threshold=candidate.emission_threshold,
        min_active_series=candidate.min_active_channels,
        null_mode="anchor-permute",
        null_repeats=null_repeats,
        seed=seed,
    )
    regroup = grouped_matrix_null_summary(
        same_shape,
        threshold=candidate.emission_threshold,
        min_active_series=candidate.min_active_channels,
        null_mode="trajectory-regroup-within-slot",
        null_repeats=null_repeats,
        seed=seed + 1,
    )
    row_emissions = np.concatenate(
        [
            _emission_rows(
                matrix,
                threshold=candidate.emission_threshold,
                min_active_channels=candidate.min_active_channels,
            )
            for matrix in matrices
        ]
    )
    cell_scores = np.concatenate([matrix.reshape(-1) for matrix in matrices])
    entity_scores = np.concatenate(matrices, axis=0)
    feature_z = np.concatenate(
        [population_feature_deviance_tensor(item.tensor, nominal) for item in files],
        axis=0,
    )
    channel_attribution = {
        name: {
            "median_score": float(np.median(entity_scores[:, index])),
            "q95_score": float(np.quantile(entity_scores[:, index], 0.95)),
            "positive_fraction": float(
                np.mean(entity_scores[:, index] > candidate.emission_threshold)
            ),
        }
        for index, name in enumerate(files[0].entity_names)
    }
    feature_rows = []
    for index, name in enumerate(files[0].feature_names):
        values = feature_z[:, :, index].reshape(-1)
        feature_rows.append(
            {
                "feature": name,
                "median_abs_z": float(np.median(values)),
                "q95_abs_z": float(np.quantile(values, 0.95)),
                "mean_abs_z": float(np.mean(values)),
            }
        )
    feature_rows.sort(key=lambda row: (row["q95_abs_z"], row["mean_abs_z"]), reverse=True)
    observed_total, active_rows = grouped_emission_total(
        same_shape,
        threshold=candidate.emission_threshold,
        min_active_series=candidate.min_active_channels,
    )
    return {
        "files": len(files),
        "same_shape_files": len(same_shape),
        "matrix_shapes": sorted({f"{matrix.shape[0]}x{matrix.shape[1]}" for matrix in matrices}),
        "observed_total_same_shape": observed_total,
        "active_rows_same_shape": active_rows,
        "row_emission_median": float(np.median(row_emissions)),
        "row_emission_q95": float(np.quantile(row_emissions, 0.95)),
        "positive_row_fraction": float(np.mean(row_emissions > 0.0)),
        "cell_score_median": float(np.median(cell_scores)),
        "cell_score_q95": float(np.quantile(cell_scores, 0.95)),
        "channel_attribution": channel_attribution,
        "top_feature_attribution": feature_rows[:10],
        "anchor_permute": anchor.to_dict(),
        "channel_preserving_recording_regroup": regroup.to_dict(),
        "recording_specific_structure": bool(
            regroup.observed_minus_null > 0.0
            and regroup.empirical_p_ge_observed <= 0.05
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_population_nominal(args: argparse.Namespace) -> dict[str, Any]:
    subjects = list(args.subjects)
    geometries = [
        FeatureGeometry(
            window_samples=int(round(seconds * args.target_sample_rate)),
            stride=int(round(stride_seconds * args.target_sample_rate)),
            spectral_bins=bins,
        )
        for seconds in _parse_float_list(args.window_seconds)
        for stride_seconds in _parse_float_list(args.stride_seconds)
        for bins in _parse_int_list(args.spectral_bins)
    ]
    candidates = [
        PopulationCandidate(
            geometry=geometry,
            feature_quantile=quantile,
            emission_threshold=threshold,
            min_active_channels=min_active,
        )
        for geometry in geometries
        for quantile in _parse_float_list(args.feature_quantiles)
        for threshold in _parse_float_list(args.thresholds)
        for min_active in _parse_int_list(args.min_active_channels)
    ]
    subject_paths = {
        subject: _paths_for_subject(args.data_root, subject, args.max_files_per_subject)
        for subject in subjects
    }
    payloads = [
        (
            subject,
            str(path),
            tuple(args.channels),
            args.target_sample_rate,
            geometry.to_dict(),
            str(args.feature_cache_dir),
        )
        for subject, paths in subject_paths.items()
        for path in paths
        for geometry in geometries
    ]
    started = time.time()
    cache_hits = 0
    feature_files: dict[tuple[str, tuple[int, int, int]], list[FeatureFile]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (subject, cache_path, cache_hit) in enumerate(
            executor.map(_build_feature_file, payloads)
        ):
            cache_hits += int(cache_hit)
            item = _load_feature_file(Path(cache_path))
            # Map by source payload geometry rather than inferred shape.
            payload_geometry = geometries[index % len(geometries)]
            feature_files.setdefault(
                (
                    subject,
                    (
                        payload_geometry.window_samples,
                        payload_geometry.stride,
                        payload_geometry.spectral_bins,
                    ),
                ),
                [],
            ).append(item)
            if args.progress and (index + 1) % 25 == 0:
                print(
                    "chbmit_population features=%d/%d cache_hits=%d elapsed=%.1fs"
                    % (index + 1, len(payloads), cache_hits, time.time() - started),
                    file=sys.stderr,
                    flush=True,
                )

    folds = []
    candidate_reports_by_heldout: dict[str, dict[str, dict[str, Any]]] = {}
    for fold_index, heldout in enumerate(subjects):
        training = [subject for subject in subjects if subject != heldout]
        candidate_reports = []
        for candidate_index, candidate in enumerate(candidates):
            key = (
                candidate.geometry.window_samples,
                candidate.geometry.stride,
                candidate.geometry.spectral_bins,
            )
            calibration: list[FeatureFile] = []
            validation: list[FeatureFile] = []
            for subject in training:
                left, right = _file_split(feature_files[(subject, key)])
                calibration.extend(left)
                validation.extend(right)
            first = calibration[0]
            nominal = fit_population_nominal(
                [item.tensor for item in calibration],
                entity_names=first.entity_names,
                feature_names=first.feature_names,
            )
            validation_matrices = _score_files(
                validation,
                nominal=nominal,
                candidate=candidate,
            )
            candidate_reports.append(
                _candidate_quality(
                    validation_matrices,
                    candidate=candidate,
                    null_repeats=args.selection_null_repeats,
                    seed=args.seed + 100_000 * fold_index + candidate_index,
                )
            )
        candidate_reports.sort(
            key=lambda row: (
                bool(row["accepted"]),
                float(row["quality"]),
                float(row["group_null"]["observed_minus_null"]),
            ),
            reverse=True,
        )
        candidate_reports_by_heldout[heldout] = {
            json.dumps(row["candidate"], sort_keys=True): row
            for row in candidate_reports
        }
        best = candidate_reports[0]
        best_row = best["candidate"]
        selected = PopulationCandidate(
            geometry=FeatureGeometry(
                window_samples=int(best_row["window_samples"]),
                stride=int(best_row["stride"]),
                spectral_bins=int(best_row["spectral_bins"]),
            ),
            feature_quantile=float(best_row["feature_quantile"]),
            emission_threshold=float(best_row["emission_threshold"]),
            min_active_channels=int(best_row["min_active_channels"]),
        )
        key = (
            selected.geometry.window_samples,
            selected.geometry.stride,
            selected.geometry.spectral_bins,
        )
        training_files = [
            item
            for subject in training
            for item in feature_files[(subject, key)]
        ]
        heldout_files = feature_files[(heldout, key)]
        first = training_files[0]
        nominal = fit_population_nominal(
            [item.tensor for item in training_files],
            entity_names=first.entity_names,
            feature_names=first.feature_names,
        )
        heldout_matrices = _score_files(
            heldout_files,
            nominal=nominal,
            candidate=selected,
        )
        heldout_summary = _heldout_summary(
            heldout_files,
            heldout_matrices,
            nominal=nominal,
            candidate=selected,
            null_repeats=args.heldout_null_repeats,
            seed=args.seed + 1_000_000 * (fold_index + 1),
        )
        heldout_summary["phase_attribution"] = _phase_attribution(
            heldout_files,
            heldout_matrices,
            data_root=args.data_root,
            candidate=selected,
            sample_rate=args.target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        folds.append(
            {
                "heldout_subject": heldout,
                "training_subjects": training,
                "selection": {
                    "label_use": "none",
                    "best": best,
                    "top_candidates": candidate_reports[: args.top_candidates],
                    "candidate_count": len(candidate_reports),
                },
                "selected_candidate": selected.to_dict(),
                "population_nominal": nominal.to_dict(),
                "heldout": heldout_summary,
            }
        )
        if args.checkpoint is not None:
            _write_json(
                args.checkpoint,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "complete_folds": folds,
                },
            )

    shared_candidate_reports = []
    for candidate in candidates:
        key = json.dumps(candidate.to_dict(), sort_keys=True)
        rows = [candidate_reports_by_heldout[subject][key] for subject in subjects]
        qualities = [float(row["quality"]) for row in rows]
        shared_candidate_reports.append(
            {
                "candidate": candidate.to_dict(),
                "accepted_folds": sum(int(row["accepted"]) for row in rows),
                "minimum_quality": min(qualities),
                "median_quality": float(np.median(qualities)),
                "minimum_group_delta": min(
                    float(row["group_null"]["observed_minus_null"])
                    for row in rows
                ),
                "maximum_group_p": max(
                    float(row["group_null"]["empirical_p_ge_observed"])
                    for row in rows
                ),
                "minimum_file_support": min(float(row["file_support"]) for row in rows),
                "maximum_saturation": max(float(row["saturation"]) for row in rows),
                "folds": {
                    subject: {
                        "accepted": row["accepted"],
                        "quality": row["quality"],
                        "group_delta": row["group_null"]["observed_minus_null"],
                        "group_z_effect": row["group_null"]["z_effect"],
                        "group_empirical_p": row["group_null"]["empirical_p_ge_observed"],
                        "file_support": row["file_support"],
                        "row_support": row["row_support"],
                        "saturation": row["saturation"],
                    }
                    for subject, row in zip(subjects, rows, strict=True)
                },
            }
        )
    shared_candidate_reports.sort(
        key=lambda row: (
            int(row["accepted_folds"]),
            float(row["minimum_quality"]),
            float(row["median_quality"]),
            float(row["minimum_group_delta"]),
        ),
        reverse=True,
    )
    shared_best = shared_candidate_reports[0]
    shared_row = shared_best["candidate"]
    shared_candidate = PopulationCandidate(
        geometry=FeatureGeometry(
            window_samples=int(shared_row["window_samples"]),
            stride=int(shared_row["stride"]),
            spectral_bins=int(shared_row["spectral_bins"]),
        ),
        feature_quantile=float(shared_row["feature_quantile"]),
        emission_threshold=float(shared_row["emission_threshold"]),
        min_active_channels=int(shared_row["min_active_channels"]),
    )
    shared_key = (
        shared_candidate.geometry.window_samples,
        shared_candidate.geometry.stride,
        shared_candidate.geometry.spectral_bins,
    )
    shared_subject_runs = []
    for fold_index, heldout in enumerate(subjects):
        training = [subject for subject in subjects if subject != heldout]
        training_files = [
            item
            for subject in training
            for item in feature_files[(subject, shared_key)]
        ]
        heldout_files = feature_files[(heldout, shared_key)]
        first = training_files[0]
        nominal = fit_population_nominal(
            [item.tensor for item in training_files],
            entity_names=first.entity_names,
            feature_names=first.feature_names,
        )
        heldout_matrices = _score_files(
            heldout_files,
            nominal=nominal,
            candidate=shared_candidate,
        )
        heldout_summary = _heldout_summary(
            heldout_files,
            heldout_matrices,
            nominal=nominal,
            candidate=shared_candidate,
            null_repeats=args.heldout_null_repeats,
            seed=args.seed + 10_000_000 + 1_000_000 * (fold_index + 1),
        )
        heldout_summary["phase_attribution"] = _phase_attribution(
            heldout_files,
            heldout_matrices,
            data_root=args.data_root,
            candidate=shared_candidate,
            sample_rate=args.target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        shared_subject_runs.append(
            {
                "heldout_subject": heldout,
                "nominal_training_subjects": training,
                "population_nominal": nominal.to_dict(),
                "heldout": heldout_summary,
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "method": {
            "split": "outer leave-one-subject-out",
            "selection": (
                "candidate geometry selected without labels on aggregate training-subject "
                "calibration/validation files"
            ),
            "nominal": (
                "robust channel-feature population nominal fit on all training-subject "
                "recordings after selection"
            ),
            "heldout": "heldout subject contributes to neither selection nor nominal",
            "nuisance_normalization": (
                "per-recording robust location/scale removal before scale-free spectral "
                "shape extraction"
            ),
            "limits": (
                "detects population spectral-shape/relationship difference; does not assess "
                "absolute voltage amplitude or establish medical abnormality"
            ),
        },
        "parameters": {
            "subjects": subjects,
            "channels": list(args.channels),
            "target_sample_rate": args.target_sample_rate,
            "max_files_per_subject": args.max_files_per_subject,
            "feature_geometries": [geometry.to_dict() for geometry in geometries],
            "candidate_count_per_fold": len(candidates),
            "selection_null_repeats": args.selection_null_repeats,
            "heldout_null_repeats": args.heldout_null_repeats,
        },
        "feature_cache": {
            "directory": str(args.feature_cache_dir),
            "requests": len(payloads),
            "hits": cache_hits,
            "misses": len(payloads) - cache_hits,
        },
        "folds": folds,
        "shared_population_instrument": {
            "selection": {
                "label_use": "none",
                "method": (
                    "choose one common candidate by accepted-fold count, then "
                    "conservative minimum quality across all cross-fitted development folds"
                ),
                "scope": (
                    "aggregate-developed common instrument; a new unseen subject is still "
                    "required for strict external generalization"
                ),
                "best": shared_best,
                "top_candidates": shared_candidate_reports[: args.top_candidates],
            },
            "selected_candidate": shared_candidate.to_dict(),
            "subject_runs": shared_subject_runs,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit"))
    parser.add_argument("--subjects", nargs="+", default=["chb01", "chb02", "chb03"])
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--target-sample-rate", type=float, default=16.0)
    parser.add_argument("--max-files-per-subject", type=int, default=0)
    parser.add_argument("--window-seconds", default="16,32,64")
    parser.add_argument("--stride-seconds", default="16")
    parser.add_argument("--spectral-bins", default="8,16")
    parser.add_argument("--feature-quantiles", default="0.9,1.0")
    parser.add_argument("--thresholds", default="2.5,3.0")
    parser.add_argument("--min-active-channels", default="2,3")
    parser.add_argument("--selection-null-repeats", type=int, default=100)
    parser.add_argument("--heldout-null-repeats", type=int, default=1000)
    parser.add_argument("--preictal-minutes", type=float, default=10.0)
    parser.add_argument("--postictal-minutes", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("/tmp/spectral-forecast-chbmit-population-cache"),
    )
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_population_nominal(args)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
