"""Fresh label-free autotune pass for CHB-MIT EEG files.

Seizure labels are not used for candidate selection. They are parsed only after
the selected candidate has surfaced coherent cross-channel windows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
    FileSeizures,
    _nearest_seizure_distance,
    _point_phase,
    downsample_and_standardize,
    load_edf_signals,
    parse_chbmit_summary,
)
from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneObservation,
    AutoTuneScore,
    build_autotune_observation,
    null_lift_score,
    observation_null_totals_for_config,
    score_autotune_observation,
    summarize_observed_vs_null_totals,
)


@dataclass(frozen=True)
class EegFile:
    """One loaded CHB-MIT EDF file, with labels held for post-hoc attribution."""

    path: Path
    series: dict[str, NDArray[np.float64]]
    sample_rate: float
    seizures: tuple[tuple[float, float], ...]

    @property
    def duration_seconds(self) -> float:
        if not self.series:
            return 0.0
        return float(min(len(values) for values in self.series.values()) / self.sample_rate)


@dataclass(frozen=True)
class EegFileScore:
    file_name: str
    score: AutoTuneScore
    null_totals: list[float]


class ObservationCache:
    """In-memory observer matrix cache keyed by file and matrix settings."""

    def __init__(self) -> None:
        self.entries: dict[tuple[object, ...], AutoTuneObservation] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple[object, ...]) -> AutoTuneObservation | None:
        observation = self.entries.get(key)
        if observation is not None:
            self.hits += 1
            return observation
        self.misses += 1
        return None

    def put(self, key: tuple[object, ...], observation: AutoTuneObservation) -> None:
        self.entries[key] = observation

    def to_dict(self) -> dict[str, int]:
        return {"entries": len(self.entries), "hits": self.hits, "misses": self.misses}


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_configs(args: argparse.Namespace) -> list[AutoTuneConfig]:
    configs = []
    for baseline in _parse_int_list(args.baselines):
        for adaptive in _parse_int_list(args.adaptive_windows):
            for stride in _parse_int_list(args.strides):
                for threshold in _parse_float_list(args.thresholds):
                    for decay in _parse_float_list(args.decays):
                        for min_active in _parse_int_list(args.min_active_channels):
                            config = AutoTuneConfig(
                                baseline_size=baseline,
                                adaptive_window=adaptive,
                                stride=stride,
                                emission_threshold=threshold,
                                decay=decay,
                                min_active_series=min_active,
                            )
                            try:
                                config.validate()
                            except ValueError:
                                continue
                            configs.append(config)
    if not configs:
        raise ValueError("no valid CHB-MIT autotune configs for requested grid")
    return configs


def load_eeg_file(
    path: Path,
    *,
    channels: Sequence[str],
    summary_labels: Mapping[str, FileSeizures],
    target_sample_rate: float,
) -> EegFile:
    raw_signals = load_edf_signals(path, channels)
    series = {
        signal.label: downsample_and_standardize(
            signal.values,
            source_rate=signal.sample_rate,
            target_rate=target_sample_rate,
        )
        for signal in raw_signals
    }
    labels = summary_labels.get(path.name, FileSeizures(path.name, ()))
    return EegFile(
        path=path,
        series=series,
        sample_rate=target_sample_rate,
        seizures=labels.seizures,
    )


def split_files(files: Sequence[EegFile], *, split: str, validation_count: int) -> tuple[list[EegFile], list[EegFile]]:
    ordered = list(files)
    if split == "none":
        return ordered, []
    if split == "alternate":
        return ordered[::2], ordered[1::2]
    if split == "last":
        if validation_count < 1:
            raise ValueError("--validation-count must be >= 1 for --validation-split last")
        if validation_count >= len(ordered):
            raise ValueError("--validation-count must be smaller than file count")
        return ordered[:-validation_count], ordered[-validation_count:]
    raise ValueError(f"unknown validation split: {split}")


def observation_for_file(
    eeg_file: EegFile,
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
) -> AutoTuneObservation:
    key = (
        str(eeg_file.path),
        config.baseline_size,
        config.adaptive_window,
        config.stride,
        config.score,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    observation = build_autotune_observation(
        eeg_file.series,
        config,
        sample_rate=eeg_file.sample_rate,
    )
    cache.put(key, observation)
    return observation


def score_file(
    eeg_file: EegFile,
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    null_repeats: int,
    seed: int,
    null_mode: str,
    null_block_size: int,
) -> EegFileScore:
    observation = observation_for_file(eeg_file, config, cache=cache)
    null_totals = observation_null_totals_for_config(
        observation,
        config,
        null_repeats=null_repeats,
        seed=seed,
        null_mode=null_mode,  # type: ignore[arg-type]
        null_block_size=null_block_size,
        total_kind="emission",
    )
    return EegFileScore(
        file_name=eeg_file.path.name,
        score=score_autotune_observation(
            observation,
            config,
            null_totals=null_totals,
            total_kind="emission",
        ),
        null_totals=null_totals,
    )


def aggregate_file_scores(
    config: AutoTuneConfig,
    scores: Sequence[EegFileScore],
    *,
    skipped: Sequence[dict[str, object]],
    min_positive_file_fraction: float,
    min_z_effect: float,
) -> dict[str, Any]:
    if not scores:
        return {
            "candidate": config.to_dict(),
            "accepted": False,
            "quality": float("-inf"),
            "files_scored": 0,
            "files_skipped": len(skipped),
            "skipped": list(skipped)[:10],
        }

    observed = float(sum(item.score.null_summary.observed_total for item in scores))
    active_windows = int(sum(item.score.null_summary.observed_active_windows for item in scores))
    null_repeats = min(len(item.null_totals) for item in scores)
    aggregate_null_totals = [
        float(sum(item.null_totals[index] for item in scores))
        for index in range(null_repeats)
    ]
    aggregate_summary = summarize_observed_vs_null_totals(
        anchors=int(sum(item.score.null_summary.anchors for item in scores)),
        observed_total=observed,
        observed_active_windows=active_windows,
        null_totals=aggregate_null_totals,
    )
    aggregate_lift = null_lift_score(aggregate_summary)
    positive_files = sum(1 for item in scores if item.score.null_summary.observed_minus_null > 0.0)
    positive_file_fraction = positive_files / len(scores)
    mean_quality = float(np.mean([item.score.quality for item in scores]))
    mean_saturation = float(np.mean([item.score.saturation_penalty for item in scores]))
    mean_fragility = float(np.mean([item.score.fragility_penalty for item in scores]))
    mean_stability = float(np.mean([item.score.stability_score for item in scores]))
    quality = mean_quality + 0.25 * aggregate_lift + 0.05 * positive_file_fraction
    z_effect = aggregate_summary.z_effect if aggregate_summary.z_effect is not None else 0.0
    accepted = (
        quality > 0.0
        and aggregate_summary.observed_minus_null > 0.0
        and positive_file_fraction >= min_positive_file_fraction
        and z_effect >= min_z_effect
        and mean_saturation < 1.0
        and aggregate_lift > 0.0
    )
    return {
        "candidate": config.to_dict(),
        "accepted": accepted,
        "files_scored": len(scores),
        "files_skipped": len(skipped),
        "positive_files": positive_files,
        "positive_file_fraction": positive_file_fraction,
        "quality": quality,
        "mean_quality": mean_quality,
        "mean_saturation": mean_saturation,
        "mean_fragility": mean_fragility,
        "mean_stability": mean_stability,
        "aggregate_null_lift": aggregate_lift,
        "observed_total": aggregate_summary.observed_total,
        "observed_active_windows": aggregate_summary.observed_active_windows,
        "null_mean_total": aggregate_summary.null_mean,
        "null_std_total": aggregate_summary.null_std,
        "observed_minus_null_total": aggregate_summary.observed_minus_null,
        "z_effect": aggregate_summary.z_effect,
        "null_total_repeats": aggregate_summary.null_repeats,
        "null_total_exceedances": aggregate_summary.null_exceedances,
        "null_total_empirical_p_ge_observed": aggregate_summary.empirical_p_ge_observed,
        "null_total_empirical_p_floor": aggregate_summary.empirical_p_floor,
        "null_total_unique_repeats": aggregate_summary.unique_null_totals,
        "file_scores": [
            {
                "file": file_score.file_name,
                "observed": file_score.score.null_summary.observed_total,
                "null_mean": file_score.score.null_summary.null_mean,
                "delta": file_score.score.null_summary.observed_minus_null,
                "z_effect": file_score.score.null_summary.z_effect,
                "exceedances": file_score.score.null_summary.null_exceedances,
            }
            for file_score in scores
        ],
        "skipped": list(skipped)[:10],
    }


def score_files_for_null_mode(
    files: Sequence[EegFile],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    args: argparse.Namespace,
    null_mode: str,
    seed_offset: int,
) -> dict[str, Any]:
    started = time.time()
    last_progress = 0.0
    scores: list[EegFileScore] = []
    skipped: list[dict[str, object]] = []
    for index, eeg_file in enumerate(files):
        try:
            scores.append(
                score_file(
                    eeg_file,
                    config,
                    cache=cache,
                    null_repeats=args.null_repeats,
                    seed=args.seed + seed_offset + index,
                    null_mode=null_mode,
                    null_block_size=args.null_block_size,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report invalid file/candidate pairs.
            skipped.append({"file": str(eeg_file.path), "reason": str(exc)})

        if args.progress and args.progress_every > 0:
            now = time.time()
            if now - last_progress >= args.progress_every:
                processed = index + 1
                elapsed = max(now - started, 1e-9)
                rate = processed / elapsed
                eta = (len(files) - processed) / rate if rate > 0 else 0.0
                print(
                    (
                        "chbmit_autotune progress baseline=%d adaptive=%d stride=%d "
                        "threshold=%.3f min_active=%d null_mode=%s files=%d/%d "
                        "skipped=%d elapsed=%.1fs eta=%.1fs cache_hits=%d cache_misses=%d"
                    )
                    % (
                        config.baseline_size,
                        config.adaptive_window,
                        config.stride,
                        config.emission_threshold,
                        config.min_active_series,
                        null_mode,
                        processed,
                        len(files),
                        len(skipped),
                        elapsed,
                        eta,
                        cache.hits,
                        cache.misses,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now

    report = aggregate_file_scores(
        config,
        scores,
        skipped=skipped,
        min_positive_file_fraction=args.min_positive_file_fraction,
        min_z_effect=args.min_z_effect,
    )
    report["null_mode"] = null_mode
    return report


def combine_null_mode_reports(
    config: AutoTuneConfig,
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    def number(row: dict[str, Any], key: str, default: float) -> float:
        value = row.get(key, default)
        return float(value) if value is not None else default

    worst = min(
        reports,
        key=lambda row: (
            bool(row.get("accepted", False)),
            number(row, "quality", float("-inf")),
            number(row, "observed_minus_null_total", float("-inf")),
        ),
    )
    return {
        "candidate": config.to_dict(),
        "accepted": all(bool(row.get("accepted", False)) for row in reports),
        "files_scored": min(int(row.get("files_scored", 0)) for row in reports),
        "files_skipped": sum(int(row.get("files_skipped", 0)) for row in reports),
        "positive_file_fraction": min(number(row, "positive_file_fraction", 0.0) for row in reports),
        "quality": min(number(row, "quality", float("-inf")) for row in reports),
        "mean_quality": min(number(row, "mean_quality", float("-inf")) for row in reports),
        "mean_saturation": max(number(row, "mean_saturation", 1.0) for row in reports),
        "mean_fragility": max(number(row, "mean_fragility", 1.0) for row in reports),
        "mean_stability": min(number(row, "mean_stability", 0.0) for row in reports),
        "aggregate_null_lift": min(number(row, "aggregate_null_lift", 0.0) for row in reports),
        "observed_total": worst.get("observed_total", 0.0),
        "null_mean_total": worst.get("null_mean_total", 0.0),
        "null_std_total": worst.get("null_std_total", 0.0),
        "observed_minus_null_total": min(
            number(row, "observed_minus_null_total", float("-inf")) for row in reports
        ),
        "z_effect": min(number(row, "z_effect", float("-inf")) for row in reports),
        "null_total_repeats": worst.get("null_total_repeats"),
        "null_total_exceedances": worst.get("null_total_exceedances"),
        "null_total_empirical_p_ge_observed": worst.get("null_total_empirical_p_ge_observed"),
        "null_total_empirical_p_floor": worst.get("null_total_empirical_p_floor"),
        "null_total_unique_repeats": worst.get("null_total_unique_repeats"),
        "worst_null_mode": worst.get("null_mode"),
        "null_modes": [str(row.get("null_mode")) for row in reports],
        "null_mode_reports": list(reports),
    }


def score_candidate(
    files: Sequence[EegFile],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    reports = [
        score_files_for_null_mode(
            files,
            config,
            cache=cache,
            args=args,
            null_mode=null_mode,
            seed_offset=seed_offset + 100000 * mode_index,
        )
        for mode_index, null_mode in enumerate(_parse_str_list(args.null_modes))
    ]
    return combine_null_mode_reports(config, reports)


def _emissions(matrix: NDArray[np.float64], config: AutoTuneConfig) -> NDArray[np.float64]:
    excess = np.maximum(matrix - config.emission_threshold, 0.0)
    active = np.sum(excess > 0.0, axis=1)
    emissions = np.sum(excess, axis=1)
    emissions[active < config.min_active_series] = 0.0
    return emissions.astype(np.float64)


def phase_report(
    eeg_file: EegFile,
    observation: AutoTuneObservation,
    config: AutoTuneConfig,
    *,
    preictal_seconds: float,
    postictal_seconds: float,
    top: int,
) -> dict[str, Any]:
    emissions = _emissions(observation.matrix, config)
    phase_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    top_counts: dict[str, int] = {}
    rows = []
    ordered = np.argsort(emissions)[::-1]
    positive = set(int(index) for index in np.flatnonzero(emissions > 0.0))
    top_indices = [int(index) for index in ordered[:top] if emissions[int(index)] > 0.0]
    for row_index, anchor in enumerate(observation.anchors):
        seconds = float(anchor) / eeg_file.sample_rate
        phase = _point_phase(
            seconds,
            eeg_file.seizures,
            preictal_seconds=preictal_seconds,
            postictal_seconds=postictal_seconds,
        )
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if row_index in positive:
            positive_counts[phase] = positive_counts.get(phase, 0) + 1
    for row_index in top_indices:
        anchor = observation.anchors[row_index]
        seconds = float(anchor) / eeg_file.sample_rate
        phase = _point_phase(
            seconds,
            eeg_file.seizures,
            preictal_seconds=preictal_seconds,
            postictal_seconds=postictal_seconds,
        )
        top_counts[phase] = top_counts.get(phase, 0) + 1
        scores = observation.matrix[row_index]
        active = [
            {"series": name, "score": float(value)}
            for name, value in zip(eeg_file.series.keys(), scores, strict=True)
            if value > config.emission_threshold
        ]
        rows.append(
            {
                "index": int(anchor),
                "seconds": seconds,
                "phase": phase,
                "nearest_seizure_distance_seconds": _nearest_seizure_distance(
                    seconds,
                    eeg_file.seizures,
                ),
                "emission": float(emissions[row_index]),
                "active_series_count": len(active),
                "active_series": active,
                "max_score": float(np.max(scores)) if len(scores) else 0.0,
                "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            }
        )

    phases = sorted(set(phase_counts) | set(positive_counts) | set(top_counts))
    rates = {
        phase: {
            "anchors": phase_counts.get(phase, 0),
            "positive": positive_counts.get(phase, 0),
            "top": top_counts.get(phase, 0),
            "positive_rate": (
                positive_counts.get(phase, 0) / phase_counts[phase]
                if phase_counts.get(phase, 0)
                else 0.0
            ),
        }
        for phase in phases
    }
    return {
        "file": eeg_file.path.name,
        "path": str(eeg_file.path),
        "duration_seconds": eeg_file.duration_seconds,
        "seizures": [{"start": start, "end": end} for start, end in eeg_file.seizures],
        "anchors": len(observation.anchors),
        "positive_windows": len(positive),
        "phase_counts": dict(sorted(phase_counts.items())),
        "positive_phase_counts": dict(sorted(positive_counts.items())),
        "top_phase_counts": dict(sorted(top_counts.items())),
        "phase_rates": rates,
        "top_rows": rows,
    }


def selected_phase_reports(
    files: Sequence[EegFile],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return [
        phase_report(
            eeg_file,
            observation_for_file(eeg_file, config, cache=cache),
            config,
            preictal_seconds=args.preictal_minutes * 60.0,
            postictal_seconds=args.postictal_minutes * 60.0,
            top=args.top_per_file,
        )
        for eeg_file in files
    ]


def aggregate_phase_counts(reports: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in reports:
        for phase, count in report.get(key, {}).items():
            counts[phase] = counts.get(phase, 0) + int(count)
    return dict(sorted(counts.items()))


def _checkpoint_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "files": [str(path) for path in args.files],
        "summary": str(args.summary),
        "channels": list(args.channels),
        "target_sample_rate": args.target_sample_rate,
        "validation_split": args.validation_split,
        "validation_count": args.validation_count,
        "baselines": _parse_int_list(args.baselines),
        "adaptive_windows": _parse_int_list(args.adaptive_windows),
        "strides": _parse_int_list(args.strides),
        "thresholds": _parse_float_list(args.thresholds),
        "decays": _parse_float_list(args.decays),
        "min_active_channels": _parse_int_list(args.min_active_channels),
        "null_modes": _parse_str_list(args.null_modes),
        "null_repeats": args.null_repeats,
        "null_block_size": args.null_block_size,
        "min_positive_file_fraction": args.min_positive_file_fraction,
        "min_z_effect": args.min_z_effect,
        "seed": args.seed,
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(path: Path, signature: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise ValueError("Checkpoint signature does not match current CHB-MIT autotune arguments")
    return payload


def run_chbmit_autotune(args: argparse.Namespace) -> dict[str, Any]:
    summary_labels = parse_chbmit_summary(args.summary)
    started = time.time()
    eeg_files = [
        load_eeg_file(
            path,
            channels=args.channels,
            summary_labels=summary_labels,
            target_sample_rate=args.target_sample_rate,
        )
        for path in args.files
    ]
    calibration_files, validation_files = split_files(
        eeg_files,
        split=args.validation_split,
        validation_count=args.validation_count,
    )
    configs = build_configs(args)
    cache = ObservationCache()
    signature = _checkpoint_signature(args)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    candidate_reports: list[dict[str, Any]] = []
    start_candidate_index = 0
    last_checkpoint = 0.0

    if args.resume:
        if checkpoint_path is None:
            raise ValueError("--resume requires --checkpoint")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = _load_checkpoint(checkpoint_path, signature)
        candidate_reports = list(checkpoint.get("candidate_reports", []))
        start_candidate_index = int(checkpoint.get("next_candidate_index", len(candidate_reports)))
        if start_candidate_index != len(candidate_reports):
            raise ValueError("Checkpoint candidate index does not match saved report count")

    def checkpoint_payload(next_candidate_index: int, *, complete: bool = False) -> dict[str, Any]:
        return {
            "version": 1,
            "signature": signature,
            "complete": complete,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "next_candidate_index": next_candidate_index,
            "total_candidates": len(configs),
            "candidate_reports": candidate_reports,
            "cache": cache.to_dict(),
        }

    def maybe_checkpoint(next_candidate_index: int, *, force: bool = False, complete: bool = False) -> None:
        nonlocal last_checkpoint
        if checkpoint_path is None:
            return
        now = time.time()
        if force or complete or args.checkpoint_every <= 0 or now - last_checkpoint >= args.checkpoint_every:
            _write_checkpoint(
                checkpoint_path,
                checkpoint_payload(next_candidate_index, complete=complete),
            )
            last_checkpoint = now

    maybe_checkpoint(start_candidate_index, force=True)
    for index, config in enumerate(configs[start_candidate_index:], start=start_candidate_index):
        if args.progress:
            print(
                "chbmit_autotune candidate %d/%d config=%s"
                % (index + 1, len(configs), config.to_dict()),
                file=sys.stderr,
                flush=True,
            )
        candidate_reports.append(
            score_candidate(
                calibration_files,
                config,
                cache=cache,
                args=args,
                seed_offset=1000 * index,
            )
        )
        maybe_checkpoint(index + 1)
    maybe_checkpoint(len(configs), force=True, complete=True)
    candidate_reports = sorted(
        candidate_reports,
        key=lambda row: (
            bool(row.get("accepted", False)),
            float(row.get("quality", float("-inf"))),
            float(row.get("observed_minus_null_total", float("-inf"))),
        ),
        reverse=True,
    )
    best = candidate_reports[0] if candidate_reports else None
    selected_config = None
    validation = None
    calibration_phase = []
    validation_phase = []
    if best is not None:
        row = best["candidate"]
        selected_config = AutoTuneConfig(
            baseline_size=int(row["baseline_size"]),
            adaptive_window=int(row["adaptive_window"]),
            stride=int(row["stride"]),
            score=str(row["score"]),  # type: ignore[arg-type]
            emission_threshold=float(row["emission_threshold"]),
            decay=float(row["decay"]),
            min_active_series=int(row["min_active_series"]),
        )
        if validation_files:
            validation = score_candidate(
                validation_files,
                selected_config,
                cache=cache,
                args=args,
                seed_offset=900000,
            )
        calibration_phase = selected_phase_reports(
            calibration_files,
            selected_config,
            cache=cache,
            args=args,
        )
        validation_phase = selected_phase_reports(
            validation_files,
            selected_config,
            cache=cache,
            args=args,
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "dataset": {
            "name": "CHB-MIT Scalp EEG Database",
            "summary_path": str(args.summary),
            "labels_used": "posthoc attribution only",
        },
        "method": {
            "label_use": "none during selection",
            "selection": "rank candidate settings by label-free aggregate cross-channel emission lift on calibration files",
            "validation": "freeze selected candidate and score disjoint heldout files",
        },
        "parameters": {
            "files": [str(path) for path in args.files],
            "channels": list(args.channels),
            "target_sample_rate": args.target_sample_rate,
            "validation_split": args.validation_split,
            "validation_count": args.validation_count,
            "candidate_count": len(configs),
            "baselines": _parse_int_list(args.baselines),
            "adaptive_windows": _parse_int_list(args.adaptive_windows),
            "strides": _parse_int_list(args.strides),
            "thresholds": _parse_float_list(args.thresholds),
            "decays": _parse_float_list(args.decays),
            "min_active_channels": _parse_int_list(args.min_active_channels),
            "null_modes": _parse_str_list(args.null_modes),
            "null_repeats": args.null_repeats,
            "null_block_size": args.null_block_size,
            "min_positive_file_fraction": args.min_positive_file_fraction,
            "min_z_effect": args.min_z_effect,
            "preictal_minutes": args.preictal_minutes,
            "postictal_minutes": args.postictal_minutes,
        },
        "cache": cache.to_dict(),
        "calibration": {
            "files": [item.path.name for item in calibration_files],
            "best": best,
            "candidates": candidate_reports[: args.top_candidates],
            "phase_reports": calibration_phase,
            "phase_summary": {
                "anchor_phase_counts": aggregate_phase_counts(calibration_phase, "phase_counts"),
                "positive_phase_counts": aggregate_phase_counts(calibration_phase, "positive_phase_counts"),
                "top_phase_counts": aggregate_phase_counts(calibration_phase, "top_phase_counts"),
            },
        },
        "validation": {
            "files": [item.path.name for item in validation_files],
            "selected_candidate": selected_config.to_dict() if selected_config else None,
            "result": validation,
            "phase_reports": validation_phase,
            "phase_summary": {
                "anchor_phase_counts": aggregate_phase_counts(validation_phase, "phase_counts"),
                "positive_phase_counts": aggregate_phase_counts(validation_phase, "positive_phase_counts"),
                "top_phase_counts": aggregate_phase_counts(validation_phase, "top_phase_counts"),
            },
        },
    }


def print_report(report: dict[str, Any]) -> None:
    params = report["parameters"]
    best = report["calibration"]["best"]
    print("CHB-MIT autotune")
    print("  labels=posthoc_only")
    print(
        "  files=%d calibration=%d validation=%d candidates=%d null_modes=%s elapsed=%.1fs"
        % (
            len(params["files"]),
            len(report["calibration"]["files"]),
            len(report["validation"]["files"]),
            params["candidate_count"],
            ",".join(params["null_modes"]),
            report["elapsed_seconds"],
        )
    )
    print("  cache entries=%d hits=%d misses=%d" % (
        report["cache"]["entries"],
        report["cache"]["hits"],
        report["cache"]["misses"],
    ))
    if not best:
        print("  no valid candidates")
        return
    cfg = best["candidate"]
    print(
        "  best accepted=%s quality=%.4f delta=%.3f z=%s files=%d positive=%.3f"
        % (
            best["accepted"],
            best["quality"],
            best["observed_minus_null_total"],
            "None" if best["z_effect"] is None else "%.2f" % best["z_effect"],
            best["files_scored"],
            best["positive_file_fraction"],
        )
    )
    print(
        "  best null repeats=%s exceedances=%s p_ge=%s p_floor=%s unique=%s worst=%s"
        % (
            best["null_total_repeats"],
            best["null_total_exceedances"],
            best["null_total_empirical_p_ge_observed"],
            best["null_total_empirical_p_floor"],
            best["null_total_unique_repeats"],
            best["worst_null_mode"],
        )
    )
    print(
        "  config baseline=%d adaptive=%d stride=%d threshold=%.3f decay=%.3f min_active=%d"
        % (
            cfg["baseline_size"],
            cfg["adaptive_window"],
            cfg["stride"],
            cfg["emission_threshold"],
            cfg["decay"],
            cfg["min_active_series"],
        )
    )
    validation = report["validation"]["result"]
    if validation:
        print(
            "  validation accepted=%s quality=%.4f delta=%.3f z=%s files=%d positive=%.3f"
            % (
                validation["accepted"],
                validation["quality"],
                validation["observed_minus_null_total"],
                "None" if validation["z_effect"] is None else "%.2f" % validation["z_effect"],
                validation["files_scored"],
                validation["positive_file_fraction"],
            )
        )
        print(
            "  validation null repeats=%s exceedances=%s p_ge=%s p_floor=%s unique=%s worst=%s"
            % (
                validation["null_total_repeats"],
                validation["null_total_exceedances"],
                validation["null_total_empirical_p_ge_observed"],
                validation["null_total_empirical_p_floor"],
                validation["null_total_unique_repeats"],
                validation["worst_null_mode"],
            )
        )
    print("  calibration top phases=%s" % report["calibration"]["phase_summary"]["top_phase_counts"])
    print("  validation top phases=%s" % report["validation"]["phase_summary"]["top_phase_counts"])
    print("  top candidates")
    for row in report["calibration"]["candidates"]:
        cfg = row["candidate"]
        print(
            "    accepted=%s quality=%.4f delta=%.3f z=%s baseline=%d adaptive=%d threshold=%.3f min_active=%d worst=%s"
            % (
                row["accepted"],
                row["quality"],
                row["observed_minus_null_total"],
                "None" if row["z_effect"] is None else "%.2f" % row["z_effect"],
                cfg["baseline_size"],
                cfg["adaptive_window"],
                cfg["emission_threshold"],
                cfg["min_active_series"],
                row["worst_null_mode"],
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="CHB-MIT EDF files")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--target-sample-rate", type=float, default=16.0)
    parser.add_argument("--validation-split", choices=["alternate", "last", "none"], default="alternate")
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--baselines", default="512,1024")
    parser.add_argument("--adaptive-windows", default="256,512")
    parser.add_argument("--strides", default="128,256")
    parser.add_argument("--thresholds", default="2.5,3.0")
    parser.add_argument("--decays", default="0.9")
    parser.add_argument("--min-active-channels", default="2,3")
    parser.add_argument("--null-modes", default="permute")
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--null-block-size", type=int, default=8)
    parser.add_argument("--min-positive-file-fraction", type=float, default=0.5)
    parser.add_argument("--min-z-effect", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top-candidates", type=int, default=8)
    parser.add_argument("--top-per-file", type=int, default=12)
    parser.add_argument("--preictal-minutes", type=float, default=10.0)
    parser.add_argument("--postictal-minutes", type=float, default=10.0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=float, default=30.0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=float,
        default=60.0,
        help="Seconds between checkpoint writes when --checkpoint is set; 0 writes after every candidate",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.null_repeats < 1:
        parser.error("--null-repeats must be >= 1")
    if args.null_block_size < 1:
        parser.error("--null-block-size must be >= 1")
    if not 0.0 <= args.min_positive_file_fraction <= 1.0:
        parser.error("--min-positive-file-fraction must be in [0, 1]")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    valid_nulls = {"permute", "shift", "block-permute"}
    for null_mode in _parse_str_list(args.null_modes):
        if null_mode not in valid_nulls:
            parser.error(f"unknown --null-modes entry: {null_mode}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_chbmit_autotune(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
