"""Fresh label-free autotune pass for synthetic pmuBAGE PMU tensors.

pmuBAGE is synthetic. Use this as an infrastructure-shaped adapter/control
domain, not as validation on real utility disturbance data.
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

from experiments.pmubage_observe import DATATYPES, robust_standardize
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
class PmuEvent:
    event_id: str
    tensor_path: Path
    tensor_shape: tuple[int, ...]
    event_index: int
    datatype_index: int
    datatype: str
    series: dict[str, NDArray[np.float64]]
    sample_rate: float


@dataclass(frozen=True)
class PmuEventScore:
    event_id: str
    score: AutoTuneScore
    null_totals: list[float]


class ObservationCache:
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
                        for min_active in _parse_int_list(args.min_active_sensors):
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
        raise ValueError("no valid pmuBAGE autotune configs for requested grid")
    return configs


def selected_event_indices(count: int, spec: str) -> list[int]:
    if spec == "all":
        return list(range(count))
    indices = _parse_int_list(spec)
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise ValueError(f"event indices outside tensor event axis {count}: {invalid}")
    return indices


def load_pmu_events(
    paths: Sequence[Path],
    *,
    event_indices: str,
    datatype_index: int,
    sensor_limit: int,
    sample_rate: float,
    max_events: int,
) -> list[PmuEvent]:
    events: list[PmuEvent] = []
    for path in paths:
        tensor = np.load(path, mmap_mode="r")
        if tensor.ndim != 4:
            raise ValueError(f"Expected a 4D pmuBAGE tensor, got {path}: {tensor.shape}")
        if datatype_index < 0 or datatype_index >= tensor.shape[1]:
            raise ValueError(
                f"datatype index {datatype_index} outside {path} datatype axis {tensor.shape[1]}"
            )
        sensor_count = min(sensor_limit, tensor.shape[2])
        datatype = DATATYPES[datatype_index] if datatype_index < len(DATATYPES) else "unknown"
        for event_index in selected_event_indices(tensor.shape[0], event_indices):
            values = np.asarray(
                tensor[event_index, datatype_index, :sensor_count, :],
                dtype=np.float64,
            )
            series = {
                f"pmu_{index:03d}": robust_standardize(values[index])
                for index in range(sensor_count)
            }
            events.append(
                PmuEvent(
                    event_id=f"{path.stem}:event{event_index}:datatype{datatype_index}",
                    tensor_path=path,
                    tensor_shape=tuple(int(item) for item in tensor.shape),
                    event_index=event_index,
                    datatype_index=datatype_index,
                    datatype=datatype,
                    series=series,
                    sample_rate=sample_rate,
                )
            )
            if max_events > 0 and len(events) >= max_events:
                return events
    return events


def split_events(
    events: Sequence[PmuEvent],
    *,
    split: str,
    validation_count: int,
) -> tuple[list[PmuEvent], list[PmuEvent]]:
    ordered = list(events)
    if split == "none":
        return ordered, []
    if split == "alternate":
        return ordered[::2], ordered[1::2]
    if split == "last":
        if validation_count < 1:
            raise ValueError("--validation-count must be >= 1 for --validation-split last")
        if validation_count >= len(ordered):
            raise ValueError("--validation-count must be smaller than event count")
        return ordered[:-validation_count], ordered[-validation_count:]
    raise ValueError(f"unknown validation split: {split}")


def observation_for_event(
    event: PmuEvent,
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
) -> AutoTuneObservation:
    key = (
        event.event_id,
        config.baseline_size,
        config.adaptive_window,
        config.stride,
        config.score,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    observation = build_autotune_observation(event.series, config, sample_rate=event.sample_rate)
    cache.put(key, observation)
    return observation


def score_event(
    event: PmuEvent,
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    null_repeats: int,
    seed: int,
    null_mode: str,
    null_block_size: int,
) -> PmuEventScore:
    observation = observation_for_event(event, config, cache=cache)
    null_totals = observation_null_totals_for_config(
        observation,
        config,
        null_repeats=null_repeats,
        seed=seed,
        null_mode=null_mode,  # type: ignore[arg-type]
        null_block_size=null_block_size,
        total_kind="emission",
    )
    return PmuEventScore(
        event_id=event.event_id,
        score=score_autotune_observation(
            observation,
            config,
            null_totals=null_totals,
            total_kind="emission",
        ),
        null_totals=null_totals,
    )


def aggregate_event_scores(
    config: AutoTuneConfig,
    scores: Sequence[PmuEventScore],
    *,
    skipped: Sequence[dict[str, object]],
    min_positive_event_fraction: float,
    min_z_effect: float,
) -> dict[str, Any]:
    if not scores:
        return {
            "candidate": config.to_dict(),
            "accepted": False,
            "quality": float("-inf"),
            "events_scored": 0,
            "events_skipped": len(skipped),
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
    positive_events = sum(1 for item in scores if item.score.null_summary.observed_minus_null > 0.0)
    positive_event_fraction = positive_events / len(scores)
    mean_quality = float(np.mean([item.score.quality for item in scores]))
    mean_saturation = float(np.mean([item.score.saturation_penalty for item in scores]))
    mean_fragility = float(np.mean([item.score.fragility_penalty for item in scores]))
    mean_stability = float(np.mean([item.score.stability_score for item in scores]))
    quality = mean_quality + 0.25 * aggregate_lift + 0.05 * positive_event_fraction
    z_effect = aggregate_summary.z_effect if aggregate_summary.z_effect is not None else 0.0
    accepted = (
        quality > 0.0
        and aggregate_summary.observed_minus_null > 0.0
        and positive_event_fraction >= min_positive_event_fraction
        and z_effect >= min_z_effect
        and mean_saturation < 1.0
        and aggregate_lift > 0.0
    )
    return {
        "candidate": config.to_dict(),
        "accepted": accepted,
        "events_scored": len(scores),
        "events_skipped": len(skipped),
        "positive_events": positive_events,
        "positive_event_fraction": positive_event_fraction,
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
        "event_scores": [
            {
                "event": event_score.event_id,
                "observed": event_score.score.null_summary.observed_total,
                "null_mean": event_score.score.null_summary.null_mean,
                "delta": event_score.score.null_summary.observed_minus_null,
                "z_effect": event_score.score.null_summary.z_effect,
                "exceedances": event_score.score.null_summary.null_exceedances,
                "active_windows": event_score.score.null_summary.observed_active_windows,
                "saturation": event_score.score.saturation_penalty,
            }
            for event_score in scores
        ],
        "skipped": list(skipped)[:10],
    }


def score_events_for_null_mode(
    events: Sequence[PmuEvent],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    args: argparse.Namespace,
    null_mode: str,
    seed_offset: int,
) -> dict[str, Any]:
    started = time.time()
    last_progress = 0.0
    scores: list[PmuEventScore] = []
    skipped: list[dict[str, object]] = []
    for index, event in enumerate(events):
        try:
            scores.append(
                score_event(
                    event,
                    config,
                    cache=cache,
                    null_repeats=args.null_repeats,
                    seed=args.seed + seed_offset + index,
                    null_mode=null_mode,
                    null_block_size=args.null_block_size,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report invalid event/candidate pairs.
            skipped.append({"event": event.event_id, "reason": str(exc)})

        if args.progress and args.progress_every > 0:
            now = time.time()
            if now - last_progress >= args.progress_every:
                processed = index + 1
                elapsed = max(now - started, 1e-9)
                rate = processed / elapsed
                eta = (len(events) - processed) / rate if rate > 0 else 0.0
                print(
                    (
                        "pmubage_autotune progress baseline=%d adaptive=%d stride=%d "
                        "threshold=%.3f min_active=%d null_mode=%s events=%d/%d "
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
                        len(events),
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

    report = aggregate_event_scores(
        config,
        scores,
        skipped=skipped,
        min_positive_event_fraction=args.min_positive_event_fraction,
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
        "events_scored": min(int(row.get("events_scored", 0)) for row in reports),
        "events_skipped": sum(int(row.get("events_skipped", 0)) for row in reports),
        "positive_event_fraction": min(number(row, "positive_event_fraction", 0.0) for row in reports),
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
    events: Sequence[PmuEvent],
    config: AutoTuneConfig,
    *,
    cache: ObservationCache,
    args: argparse.Namespace,
    seed_offset: int,
) -> dict[str, Any]:
    reports = [
        score_events_for_null_mode(
            events,
            config,
            cache=cache,
            args=args,
            null_mode=null_mode,
            seed_offset=seed_offset + 100000 * mode_index,
        )
        for mode_index, null_mode in enumerate(_parse_str_list(args.null_modes))
    ]
    return combine_null_mode_reports(config, reports)


def event_metadata(events: Sequence[PmuEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event.event_id,
            "tensor": str(event.tensor_path),
            "tensor_shape": list(event.tensor_shape),
            "event_index": event.event_index,
            "datatype_index": event.datatype_index,
            "datatype": event.datatype,
            "sensor_count": len(event.series),
            "samples": min(len(values) for values in event.series.values()) if event.series else 0,
        }
        for event in events
    ]


def _checkpoint_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tensors": [str(path) for path in args.tensors],
        "event_indices": args.event_indices,
        "datatype_index": args.datatype_index,
        "sensor_limit": args.sensor_limit,
        "sample_rate": args.sample_rate,
        "max_events": args.max_events,
        "validation_split": args.validation_split,
        "validation_count": args.validation_count,
        "baselines": _parse_int_list(args.baselines),
        "adaptive_windows": _parse_int_list(args.adaptive_windows),
        "strides": _parse_int_list(args.strides),
        "thresholds": _parse_float_list(args.thresholds),
        "decays": _parse_float_list(args.decays),
        "min_active_sensors": _parse_int_list(args.min_active_sensors),
        "null_modes": _parse_str_list(args.null_modes),
        "null_repeats": args.null_repeats,
        "null_block_size": args.null_block_size,
        "min_positive_event_fraction": args.min_positive_event_fraction,
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
        raise ValueError("Checkpoint signature does not match current pmuBAGE autotune arguments")
    return payload


def run_pmubage_autotune(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    events = load_pmu_events(
        args.tensors,
        event_indices=args.event_indices,
        datatype_index=args.datatype_index,
        sensor_limit=args.sensor_limit,
        sample_rate=args.sample_rate,
        max_events=args.max_events,
    )
    calibration_events, validation_events = split_events(
        events,
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
                "pmubage_autotune candidate %d/%d config=%s"
                % (index + 1, len(configs), config.to_dict()),
                file=sys.stderr,
                flush=True,
            )
        candidate_reports.append(
            score_candidate(
                calibration_events,
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
        if validation_events:
            validation = score_candidate(
                validation_events,
                selected_config,
                cache=cache,
                args=args,
                seed_offset=900000,
            )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "dataset": {
            "name": "pmuBAGE synthetic PMU events",
            "source": "https://github.com/NanpengYu/pmuBAGE",
            "validation_boundary": "synthetic adapter/control case, not real-grid validation",
        },
        "method": {
            "label_use": "none",
            "selection": "rank candidate settings by aggregate cross-PMU emission lift on calibration events",
            "validation": "freeze selected candidate and score disjoint heldout events",
        },
        "parameters": {
            "tensors": [str(path) for path in args.tensors],
            "event_indices": args.event_indices,
            "datatype_index": args.datatype_index,
            "sensor_limit": args.sensor_limit,
            "sample_rate": args.sample_rate,
            "max_events": args.max_events,
            "validation_split": args.validation_split,
            "validation_count": args.validation_count,
            "candidate_count": len(configs),
            "baselines": _parse_int_list(args.baselines),
            "adaptive_windows": _parse_int_list(args.adaptive_windows),
            "strides": _parse_int_list(args.strides),
            "thresholds": _parse_float_list(args.thresholds),
            "decays": _parse_float_list(args.decays),
            "min_active_sensors": _parse_int_list(args.min_active_sensors),
            "null_modes": _parse_str_list(args.null_modes),
            "null_repeats": args.null_repeats,
            "null_block_size": args.null_block_size,
            "min_positive_event_fraction": args.min_positive_event_fraction,
            "min_z_effect": args.min_z_effect,
        },
        "cache": cache.to_dict(),
        "calibration": {
            "events": event_metadata(calibration_events),
            "best": best,
            "candidates": candidate_reports[: args.top_candidates],
        },
        "validation": {
            "events": event_metadata(validation_events),
            "selected_candidate": selected_config.to_dict() if selected_config else None,
            "result": validation,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    params = report["parameters"]
    best = report["calibration"]["best"]
    print("pmuBAGE autotune")
    print(
        "  events=%d calibration=%d validation=%d candidates=%d datatype=%s null_modes=%s elapsed=%.1fs"
        % (
            len(report["calibration"]["events"]) + len(report["validation"]["events"]),
            len(report["calibration"]["events"]),
            len(report["validation"]["events"]),
            params["candidate_count"],
            DATATYPES[params["datatype_index"]] if params["datatype_index"] < len(DATATYPES) else "unknown",
            ",".join(params["null_modes"]),
            report["elapsed_seconds"],
        )
    )
    print(
        "  cache entries=%d hits=%d misses=%d"
        % (report["cache"]["entries"], report["cache"]["hits"], report["cache"]["misses"])
    )
    if not best:
        print("  no valid candidates")
        return
    cfg = best["candidate"]
    print(
        "  best accepted=%s quality=%.4f delta=%.3f z=%s events=%d positive=%.3f saturation=%.3f"
        % (
            best["accepted"],
            best["quality"],
            best["observed_minus_null_total"],
            "None" if best["z_effect"] is None else "%.2f" % best["z_effect"],
            best["events_scored"],
            best["positive_event_fraction"],
            best["mean_saturation"],
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
            "  validation accepted=%s quality=%.4f delta=%.3f z=%s events=%d positive=%.3f saturation=%.3f"
            % (
                validation["accepted"],
                validation["quality"],
                validation["observed_minus_null_total"],
                "None" if validation["z_effect"] is None else "%.2f" % validation["z_effect"],
                validation["events_scored"],
                validation["positive_event_fraction"],
                validation["mean_saturation"],
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
    print("  top candidates")
    for row in report["calibration"]["candidates"]:
        cfg = row["candidate"]
        print(
            "    accepted=%s quality=%.4f delta=%.3f z=%s baseline=%d adaptive=%d threshold=%.3f min_active=%d saturation=%.3f worst=%s"
            % (
                row["accepted"],
                row["quality"],
                row["observed_minus_null_total"],
                "None" if row["z_effect"] is None else "%.2f" % row["z_effect"],
                cfg["baseline_size"],
                cfg["adaptive_window"],
                cfg["emission_threshold"],
                cfg["min_active_series"],
                row["mean_saturation"],
                row["worst_null_mode"],
            )
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensors", type=Path, nargs="+", help="pmuBAGE .npy tensors")
    parser.add_argument("--event-indices", default="all", help="'all' or comma-separated event indices")
    parser.add_argument("--datatype-index", type=int, default=3)
    parser.add_argument("--sensor-limit", type=int, default=24)
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--max-events", type=int, default=0, help="0 means all selected events")
    parser.add_argument("--validation-split", choices=["alternate", "last", "none"], default="alternate")
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--baselines", default="128,256")
    parser.add_argument("--adaptive-windows", default="64,128")
    parser.add_argument("--strides", default="16,32")
    parser.add_argument("--thresholds", default="2.5,3.0,3.5")
    parser.add_argument("--decays", default="0.9")
    parser.add_argument("--min-active-sensors", default="3,6,12")
    parser.add_argument("--null-modes", default="permute")
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--null-block-size", type=int, default=8)
    parser.add_argument("--min-positive-event-fraction", type=float, default=0.5)
    parser.add_argument("--min-z-effect", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--top-candidates", type=int, default=8)
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
    if args.sensor_limit < 1:
        parser.error("--sensor-limit must be >= 1")
    if args.max_events < 0:
        parser.error("--max-events must be >= 0")
    if args.null_repeats < 1:
        parser.error("--null-repeats must be >= 1")
    if args.null_block_size < 1:
        parser.error("--null-block-size must be >= 1")
    if not 0.0 <= args.min_positive_event_fraction <= 1.0:
        parser.error("--min-positive-event-fraction must be in [0, 1]")
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
    report = run_pmubage_autotune(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
