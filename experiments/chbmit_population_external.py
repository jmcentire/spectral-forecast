"""Evaluate a frozen CHB-MIT population instrument on genuinely unseen brains.

The source report supplies the already-selected common candidate. Training
subjects fit one frozen population nominal. External subjects contribute to
neither candidate selection nor the population nominal.

Each external subject is also split into self-calibration and self-validation
recordings in two ways. The same frozen candidate then compares:

* population deviance: persistently unlike the training population;
* balanced self deviance: changing relative to interleaved recordings;
* ordered self deviance: later recordings changing relative to an earlier prefix.

Seizure labels are introduced only after both structure surfaces are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray

from experiments.chbmit_population_nominal import (
    FeatureFile,
    FeatureGeometry,
    PopulationCandidate,
    _build_feature_file,
    _emission_rows,
    _file_split,
    _heldout_summary,
    _load_feature_file,
    _paths_for_subject,
    _phase_attribution,
    _score_files,
    _write_json,
)
from spectral_forecast.population import (
    fit_population_nominal,
    population_feature_deviance_tensor,
    population_signed_deviance_tensor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_from_source(report: Mapping[str, Any]) -> PopulationCandidate:
    row = report["shared_population_instrument"]["selected_candidate"]
    return PopulationCandidate(
        geometry=FeatureGeometry(
            window_samples=int(row["window_samples"]),
            stride=int(row["stride"]),
            spectral_bins=int(row["spectral_bins"]),
        ),
        feature_quantile=float(row["feature_quantile"]),
        emission_threshold=float(row["emission_threshold"]),
        min_active_channels=int(row["min_active_channels"]),
    )


def _axis_comparison(
    files: Sequence[FeatureFile],
    population_matrices: Sequence[NDArray[np.float64]],
    self_matrices: Sequence[NDArray[np.float64]],
    *,
    candidate: PopulationCandidate,
) -> dict[str, Any]:
    population_rows = []
    self_rows = []
    file_rows = []
    for item, population_matrix, self_matrix in zip(
        files,
        population_matrices,
        self_matrices,
        strict=True,
    ):
        population_emissions = _emission_rows(
            population_matrix,
            threshold=candidate.emission_threshold,
            min_active_channels=candidate.min_active_channels,
        )
        self_emissions = _emission_rows(
            self_matrix,
            threshold=candidate.emission_threshold,
            min_active_channels=candidate.min_active_channels,
        )
        population_rows.append(population_emissions)
        self_rows.append(self_emissions)
        population_mask = population_emissions > 0.0
        self_mask = self_emissions > 0.0
        file_rows.append(
            {
                "file": item.path,
                "rows": len(population_emissions),
                "population_positive_fraction": float(np.mean(population_mask)),
                "self_positive_fraction": float(np.mean(self_mask)),
                "both_positive_fraction": float(np.mean(population_mask & self_mask)),
                "population_only_fraction": float(
                    np.mean(population_mask & ~self_mask)
                ),
                "self_only_fraction": float(np.mean(self_mask & ~population_mask)),
            }
        )

    population = np.concatenate(population_rows)
    self_values = np.concatenate(self_rows)
    population_mask = population > 0.0
    self_mask = self_values > 0.0
    union = population_mask | self_mask
    correlation = (
        float(np.corrcoef(population, self_values)[0, 1])
        if np.std(population) > 1e-12 and np.std(self_values) > 1e-12
        else None
    )
    file_rows.sort(
        key=lambda row: (
            row["population_only_fraction"] + row["self_only_fraction"],
            row["population_positive_fraction"] + row["self_positive_fraction"],
        ),
        reverse=True,
    )
    return {
        "rows": len(population),
        "population_positive_fraction": float(np.mean(population_mask)),
        "self_positive_fraction": float(np.mean(self_mask)),
        "both_positive_fraction": float(np.mean(population_mask & self_mask)),
        "population_only_fraction": float(np.mean(population_mask & ~self_mask)),
        "self_only_fraction": float(np.mean(self_mask & ~population_mask)),
        "neither_fraction": float(np.mean(~union)),
        "positive_jaccard": (
            float(np.sum(population_mask & self_mask) / np.sum(union))
            if np.any(union)
            else None
        ),
        "emission_correlation": correlation,
        "top_disagreement_files": file_rows[:10],
    }


def _ordered_population_profile(
    files: Sequence[FeatureFile],
    matrices: Sequence[NDArray[np.float64]],
    *,
    candidate: PopulationCandidate,
) -> dict[str, Any]:
    rows = []
    for index, (item, matrix) in enumerate(zip(files, matrices, strict=True)):
        emissions = _emission_rows(
            matrix,
            threshold=candidate.emission_threshold,
            min_active_channels=candidate.min_active_channels,
        )
        rows.append(
            {
                "index": index,
                "file": item.path,
                "anchors": len(emissions),
                "positive_fraction": float(np.mean(emissions > 0.0)),
                "emission_mean": float(np.mean(emissions)),
                "emission_q95": float(np.quantile(emissions, 0.95)),
                "cell_score_median": float(np.median(matrix)),
                "cell_score_q95": float(np.quantile(matrix, 0.95)),
            }
        )
    fractions = np.asarray([row["positive_fraction"] for row in rows], dtype=np.float64)
    midpoint = len(rows) // 2
    steps = np.diff(fractions)
    largest_step_index = int(np.argmax(np.abs(steps))) if len(steps) else None
    return {
        "order": "sorted recording filename",
        "files": rows,
        "positive_fraction_slope_per_file": (
            float(np.polyfit(np.arange(len(fractions)), fractions, 1)[0])
            if len(fractions) >= 2
            else None
        ),
        "first_half_positive_fraction_mean": (
            float(np.mean(fractions[:midpoint])) if midpoint else None
        ),
        "second_half_positive_fraction_mean": (
            float(np.mean(fractions[midpoint:])) if midpoint < len(fractions) else None
        ),
        "second_minus_first_half": (
            float(np.mean(fractions[midpoint:]) - np.mean(fractions[:midpoint]))
            if midpoint and midpoint < len(fractions)
            else None
        ),
        "largest_adjacent_step": (
            {
                "from_file": rows[largest_step_index]["file"],
                "to_file": rows[largest_step_index + 1]["file"],
                "delta": float(steps[largest_step_index]),
            }
            if largest_step_index is not None
            else None
        ),
    }


def _population_fingerprint(
    files: Sequence[FeatureFile],
    *,
    nominal: Any,
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    signed_z = np.concatenate(
        [population_signed_deviance_tensor(item.tensor, nominal) for item in files],
        axis=0,
    )
    absolute_z = np.concatenate(
        [population_feature_deviance_tensor(item.tensor, nominal) for item in files],
        axis=0,
    )
    median_signed = np.median(signed_z, axis=0)
    q95_absolute = np.quantile(absolute_z, 0.95, axis=0)
    rows = [
        {
            "channel": channel,
            "feature": feature,
            "median_signed_z": float(median_signed[channel_index, feature_index]),
            "q95_abs_z": float(q95_absolute[channel_index, feature_index]),
        }
        for channel_index, channel in enumerate(files[0].entity_names)
        for feature_index, feature in enumerate(files[0].feature_names)
    ]
    rows.sort(
        key=lambda row: (abs(row["median_signed_z"]), row["q95_abs_z"]),
        reverse=True,
    )
    return (
        {
            "definition": (
                "median signed population deviation for each frozen channel-feature cell"
            ),
            "dimensions": {
                "channels": list(files[0].entity_names),
                "features": list(files[0].feature_names),
            },
            "rms_median_signed_z": float(np.sqrt(np.mean(median_signed**2))),
            "q95_abs_median_signed_z": float(
                np.quantile(np.abs(median_signed), 0.95)
            ),
            "top_channel_features": rows[:20],
        },
        median_signed.reshape(-1).astype(np.float64),
    )


def _pairwise_fingerprint_relations(
    fingerprints: Mapping[str, NDArray[np.float64]],
) -> list[dict[str, Any]]:
    subjects = sorted(fingerprints)
    rows = []
    for left_index, left in enumerate(subjects):
        for right in subjects[left_index + 1 :]:
            left_values = fingerprints[left]
            right_values = fingerprints[right]
            denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
            rows.append(
                {
                    "subject_a": left,
                    "subject_b": right,
                    "rms_difference": float(
                        np.sqrt(np.mean((left_values - right_values) ** 2))
                    ),
                    "cosine_similarity": (
                        float(
                            np.clip(
                                np.dot(left_values, right_values) / denominator,
                                -1.0,
                                1.0,
                            )
                        )
                        if denominator > 1e-18
                        else None
                    ),
                }
            )
    rows.sort(key=lambda row: row["rms_difference"])
    return rows


def _compact_structure_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    regroup = summary["channel_preserving_recording_regroup"]
    return {
        "positive_row_fraction": summary["positive_row_fraction"],
        "cell_score_median": summary["cell_score_median"],
        "cell_score_q95": summary["cell_score_q95"],
        "recording_regroup_delta": regroup["observed_minus_null"],
        "recording_regroup_z_effect": regroup["z_effect"],
        "recording_regroup_p": regroup["empirical_p_ge_observed"],
        "recording_specific_structure": summary["recording_specific_structure"],
    }


def _prefix_split(
    files: Sequence[FeatureFile],
) -> tuple[list[FeatureFile], list[FeatureFile]]:
    ordered = sorted(files, key=lambda item: item.path)
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def _feature_files(
    *,
    data_root: Path,
    subjects: Sequence[str],
    channels: Sequence[str],
    target_sample_rate: float,
    candidate: PopulationCandidate,
    max_files_per_subject: int,
    feature_cache_dir: Path,
    workers: int,
    progress: bool,
) -> tuple[dict[str, list[FeatureFile]], dict[str, Any]]:
    subject_paths = {
        subject: _paths_for_subject(data_root, subject, max_files_per_subject)
        for subject in subjects
    }
    empty = [subject for subject, paths in subject_paths.items() if not paths]
    if empty:
        raise ValueError(f"subjects contain no EDF files: {', '.join(empty)}")
    payloads = [
        (
            subject,
            str(path),
            tuple(channels),
            target_sample_rate,
            candidate.geometry.to_dict(),
            str(feature_cache_dir),
        )
        for subject, paths in subject_paths.items()
        for path in paths
    ]
    started = time.time()
    hits = 0
    files: dict[str, list[FeatureFile]] = {subject: [] for subject in subjects}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (subject, cache_path, cache_hit) in enumerate(
            executor.map(_build_feature_file, payloads)
        ):
            hits += int(cache_hit)
            files[subject].append(_load_feature_file(Path(cache_path)))
            if progress and ((index + 1) % 10 == 0 or index + 1 == len(payloads)):
                print(
                    "chbmit_external features=%d/%d cache_hits=%d elapsed=%.1fs"
                    % (index + 1, len(payloads), hits, time.time() - started),
                    file=sys.stderr,
                    flush=True,
                )
    for subject in files:
        files[subject].sort(key=lambda item: item.path)
    return files, {
        "directory": str(feature_cache_dir),
        "requests": len(payloads),
        "hits": hits,
        "misses": len(payloads) - hits,
    }


def run_external_population(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    candidate = _candidate_from_source(source)
    source_parameters = source["parameters"]
    channels = tuple(source_parameters["channels"])
    target_sample_rate = float(source_parameters["target_sample_rate"])
    training_subjects = list(args.training_subjects)
    external_subjects = list(args.external_subjects)
    overlap = sorted(set(training_subjects) & set(external_subjects))
    if overlap:
        raise ValueError(f"training and external subjects overlap: {', '.join(overlap)}")

    all_subjects = training_subjects + external_subjects
    feature_files, cache_summary = _feature_files(
        data_root=args.data_root,
        subjects=all_subjects,
        channels=channels,
        target_sample_rate=target_sample_rate,
        candidate=candidate,
        max_files_per_subject=args.max_files_per_subject,
        feature_cache_dir=args.feature_cache_dir,
        workers=args.workers,
        progress=args.progress,
    )
    training_files = [
        item for subject in training_subjects for item in feature_files[subject]
    ]
    first = training_files[0]
    population_nominal = fit_population_nominal(
        [item.tensor for item in training_files],
        entity_names=first.entity_names,
        feature_names=first.feature_names,
    )
    omission_nominals = {
        omitted: fit_population_nominal(
            [
                item.tensor
                for training_subject in training_subjects
                if training_subject != omitted
                for item in feature_files[training_subject]
            ],
            entity_names=first.entity_names,
            feature_names=first.feature_names,
        )
        for omitted in training_subjects
    }

    subject_runs = []
    fingerprints: dict[str, NDArray[np.float64]] = {}
    for subject_index, subject in enumerate(external_subjects):
        files = feature_files[subject]
        alternating_calibration, alternating_validation = _file_split(files)
        prefix_calibration, later_validation = _prefix_split(files)
        if not alternating_calibration or not alternating_validation:
            raise ValueError(f"{subject} needs at least two recordings for self split")
        alternating_first = alternating_calibration[0]
        alternating_nominal = fit_population_nominal(
            [item.tensor for item in alternating_calibration],
            entity_names=alternating_first.entity_names,
            feature_names=alternating_first.feature_names,
        )
        prefix_first = prefix_calibration[0]
        prefix_nominal = fit_population_nominal(
            [item.tensor for item in prefix_calibration],
            entity_names=prefix_first.entity_names,
            feature_names=prefix_first.feature_names,
        )

        population_all = _score_files(
            files,
            nominal=population_nominal,
            candidate=candidate,
        )
        population_alternating_validation = _score_files(
            alternating_validation,
            nominal=population_nominal,
            candidate=candidate,
        )
        population_alternating_calibration = _score_files(
            alternating_calibration,
            nominal=population_nominal,
            candidate=candidate,
        )
        alternating_self_validation = _score_files(
            alternating_validation,
            nominal=alternating_nominal,
            candidate=candidate,
        )
        population_later_validation = _score_files(
            later_validation,
            nominal=population_nominal,
            candidate=candidate,
        )
        prefix_self_later_validation = _score_files(
            later_validation,
            nominal=prefix_nominal,
            candidate=candidate,
        )
        seed = args.seed + 1_000_000 * (subject_index + 1)
        population_all_summary = _heldout_summary(
            files,
            population_all,
            nominal=population_nominal,
            candidate=candidate,
            null_repeats=args.null_repeats,
            seed=seed,
        )
        population_alternating_summary = _heldout_summary(
            alternating_validation,
            population_alternating_validation,
            nominal=population_nominal,
            candidate=candidate,
            null_repeats=args.null_repeats,
            seed=seed + 100_000,
        )
        population_alternating_calibration_summary = _heldout_summary(
            alternating_calibration,
            population_alternating_calibration,
            nominal=population_nominal,
            candidate=candidate,
            null_repeats=args.null_repeats,
            seed=seed + 150_000,
        )
        alternating_self_summary = _heldout_summary(
            alternating_validation,
            alternating_self_validation,
            nominal=alternating_nominal,
            candidate=candidate,
            null_repeats=args.null_repeats,
            seed=seed + 200_000,
        )
        population_later_summary = _heldout_summary(
            later_validation,
            population_later_validation,
            nominal=population_nominal,
            candidate=candidate,
            null_repeats=args.null_repeats,
            seed=seed + 300_000,
        )
        prefix_self_summary = _heldout_summary(
            later_validation,
            prefix_self_later_validation,
            nominal=prefix_nominal,
            candidate=candidate,
            null_repeats=args.null_repeats,
            seed=seed + 400_000,
        )

        population_all_summary["phase_attribution"] = _phase_attribution(
            files,
            population_all,
            data_root=args.data_root,
            candidate=candidate,
            sample_rate=target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        population_alternating_summary["phase_attribution"] = _phase_attribution(
            alternating_validation,
            population_alternating_validation,
            data_root=args.data_root,
            candidate=candidate,
            sample_rate=target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        alternating_self_summary["phase_attribution"] = _phase_attribution(
            alternating_validation,
            alternating_self_validation,
            data_root=args.data_root,
            candidate=candidate,
            sample_rate=target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        population_later_summary["phase_attribution"] = _phase_attribution(
            later_validation,
            population_later_validation,
            data_root=args.data_root,
            candidate=candidate,
            sample_rate=target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        prefix_self_summary["phase_attribution"] = _phase_attribution(
            later_validation,
            prefix_self_later_validation,
            data_root=args.data_root,
            candidate=candidate,
            sample_rate=target_sample_rate,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
        )
        fingerprint_report, fingerprint_vector = _population_fingerprint(
            files,
            nominal=population_nominal,
        )
        fingerprints[subject] = fingerprint_vector
        _, alternating_calibration_fingerprint = _population_fingerprint(
            alternating_calibration,
            nominal=population_nominal,
        )
        _, alternating_validation_fingerprint = _population_fingerprint(
            alternating_validation,
            nominal=population_nominal,
        )
        split_fingerprint_relation = _pairwise_fingerprint_relations(
            {
                "alternating_calibration": alternating_calibration_fingerprint,
                "alternating_validation": alternating_validation_fingerprint,
            }
        )[0]
        sensitivity_rows = []
        for omitted_index, (omitted, omission_nominal) in enumerate(
            omission_nominals.items()
        ):
            omission_matrices = _score_files(
                files,
                nominal=omission_nominal,
                candidate=candidate,
            )
            omission_summary = _heldout_summary(
                files,
                omission_matrices,
                nominal=omission_nominal,
                candidate=candidate,
                null_repeats=args.null_repeats,
                seed=seed + 500_000 + 100_000 * omitted_index,
            )
            _, omission_fingerprint = _population_fingerprint(
                files,
                nominal=omission_nominal,
            )
            relation = _pairwise_fingerprint_relations(
                {
                    "all_training_subjects": fingerprint_vector,
                    f"omit_{omitted}": omission_fingerprint,
                }
            )[0]
            sensitivity_rows.append(
                {
                    "omitted_training_subject": omitted,
                    "nominal_training_subjects": [
                        value for value in training_subjects if value != omitted
                    ],
                    "positive_row_fraction": omission_summary[
                        "positive_row_fraction"
                    ],
                    "cell_score_median": omission_summary["cell_score_median"],
                    "cell_score_q95": omission_summary["cell_score_q95"],
                    "recording_regroup_delta": omission_summary[
                        "channel_preserving_recording_regroup"
                    ]["observed_minus_null"],
                    "recording_regroup_z_effect": omission_summary[
                        "channel_preserving_recording_regroup"
                    ]["z_effect"],
                    "recording_regroup_p": omission_summary[
                        "channel_preserving_recording_regroup"
                    ]["empirical_p_ge_observed"],
                    "fingerprint_relation_to_all_training_nominal": {
                        "rms_difference": relation["rms_difference"],
                        "cosine_similarity": relation["cosine_similarity"],
                    },
                }
            )
        sensitivity_positive = [
            float(row["positive_row_fraction"]) for row in sensitivity_rows
        ]
        subject_runs.append(
            {
                "subject": subject,
                "files": len(files),
                "input_files": [item.path for item in files],
                "alternating_self_split": {
                    "method": "sorted alternating recordings",
                    "calibration_files": [
                        item.path for item in alternating_calibration
                    ],
                    "validation_files": [item.path for item in alternating_validation],
                },
                "prefix_self_split": {
                    "method": "sorted earlier half to later half",
                    "calibration_files": [item.path for item in prefix_calibration],
                    "validation_files": [item.path for item in later_validation],
                },
                "alternating_self_nominal": alternating_nominal.to_dict(),
                "prefix_self_nominal": prefix_nominal.to_dict(),
                "population_all_files": population_all_summary,
                "ordered_population_profile": _ordered_population_profile(
                    files,
                    population_all,
                    candidate=candidate,
                ),
                "population_fingerprint": fingerprint_report,
                "population_alternating_split_replication": {
                    "purpose": (
                        "test whether the subject's population-deviation geometry repeats "
                        "across disjoint sequence-balanced recording sets"
                    ),
                    "calibration_half": _compact_structure_summary(
                        population_alternating_calibration_summary
                    ),
                    "validation_half": _compact_structure_summary(
                        population_alternating_summary
                    ),
                    "signed_fingerprint_relation": {
                        "rms_difference": split_fingerprint_relation["rms_difference"],
                        "cosine_similarity": split_fingerprint_relation[
                            "cosine_similarity"
                        ],
                    },
                },
                "population_reference_sensitivity": {
                    "method": (
                        "same frozen candidate; refit nominal after omitting each original "
                        "training subject"
                    ),
                    "all_training_positive_row_fraction": population_all_summary[
                        "positive_row_fraction"
                    ],
                    "omission_positive_fraction_min": min(sensitivity_positive),
                    "omission_positive_fraction_max": max(sensitivity_positive),
                    "omission_positive_fraction_span": (
                        max(sensitivity_positive) - min(sensitivity_positive)
                    ),
                    "omission_runs": sensitivity_rows,
                },
                "alternating_validation": {
                    "population": population_alternating_summary,
                    "self": alternating_self_summary,
                    "axis_comparison": _axis_comparison(
                        alternating_validation,
                        population_alternating_validation,
                        alternating_self_validation,
                        candidate=candidate,
                    ),
                },
                "later_validation": {
                    "population": population_later_summary,
                    "self": prefix_self_summary,
                    "axis_comparison": _axis_comparison(
                        later_validation,
                        population_later_validation,
                        prefix_self_later_validation,
                        candidate=candidate,
                    ),
                },
            }
        )
        if args.checkpoint is not None:
            _write_json(
                args.checkpoint,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_report": str(args.source_report),
                    "completed_subject_runs": subject_runs,
                },
            )
        if args.progress:
            print(
                "chbmit_external subject=%s complete=%d/%d elapsed=%.1fs"
                % (
                    subject,
                    subject_index + 1,
                    len(external_subjects),
                    time.time() - started,
                ),
                file=sys.stderr,
                flush=True,
            )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "method": {
            "core_goal": "label-free hidden-structure discovery, not prediction",
            "candidate": "frozen exactly from source report before external data acquisition",
            "population_nominal": "fit only on training-subject recordings",
            "external_subjects": (
                "contribute to neither candidate selection nor population nominal"
            ),
            "self_nominal": (
                "two views: alternating recordings for coverage-balanced self change, and "
                "earlier-prefix to later-recording comparison for order-sensitive drift"
            ),
            "labels": "posthoc attribution only after both structure surfaces are frozen",
        },
        "source_report": {
            "path": str(args.source_report),
            "sha256": _sha256(args.source_report),
            "selected_candidate": candidate.to_dict(),
        },
        "parameters": {
            "data_root": str(args.data_root),
            "training_subjects": training_subjects,
            "external_subjects": external_subjects,
            "training_files": [item.path for item in training_files],
            "channels": list(channels),
            "target_sample_rate": target_sample_rate,
            "max_files_per_subject": args.max_files_per_subject,
            "null_repeats": args.null_repeats,
        },
        "feature_cache": cache_summary,
        "population_nominal": population_nominal.to_dict(),
        "external_subject_relationships": _pairwise_fingerprint_relations(fingerprints),
        "subject_runs": subject_runs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path(
            "experiments/results/2026-06-04-chbmit-population-nominal.json"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/chbmit"))
    parser.add_argument(
        "--training-subjects",
        nargs="+",
        default=["chb01", "chb02", "chb03"],
    )
    parser.add_argument(
        "--external-subjects",
        nargs="+",
        default=["chb04", "chb05", "chb06"],
    )
    parser.add_argument("--max-files-per-subject", type=int, default=14)
    parser.add_argument("--null-repeats", type=int, default=1000)
    parser.add_argument("--preictal-minutes", type=float, default=10.0)
    parser.add_argument("--postictal-minutes", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path("/tmp/spectral-forecast-chbmit-population-cache"),
    )
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_external_population(args)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
