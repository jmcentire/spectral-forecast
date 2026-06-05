"""Discover explicit relationship hypotheses in canonical aligned CDIP waveforms.

The runner uses a prior aligned-window report only as a reproducible source of
paths and time ranges. It does not use prior observer scores or selected
anomaly settings. Each relationship family keeps its identity, attributes,
source mapping, family-specific null, and exact-geometry known-answer gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import netcdf_file

from experiments.cdip_observe import CHANNEL_VARIABLES, _decode_attr, _scalar
from spectral_forecast.relationships import (
    calibrate_relationship_geometry,
    discover_relationships,
    normalized_spectral_profile,
    residualize_relationship_view,
    spectral_profile_alignment,
)


DEFAULT_SOURCE_REPORT = Path(
    "experiments/results/"
    "2026-06-03-cdip-autotune-big-ocean-naive-512-512-heldout-shift-permute50.json"
)

ENVELOPE_PROFILE_FRACTIONS = {
    "raw_profile": 0.0,
    "envelope_attenuation_0.25": 0.25,
    "envelope_attenuation_0.50": 0.50,
    "envelope_attenuation_0.75": 0.75,
    "envelope_attenuation_1.00": 1.00,
}
SIGNED_ENVELOPE_PROFILE = "signed_envelope_residual"
SPECTRAL_PROFILE_KEYS = tuple(ENVELOPE_PROFILE_FRACTIONS) + (
    SIGNED_ENVELOPE_PROFILE,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rng_for(seed: int, *parts: object) -> np.random.Generator:
    """Return a stable null stream that is independent of hypothesis ordering."""

    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    derived = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return np.random.default_rng(derived)


def _window_key(segment: str, window: Mapping[str, Any]) -> str:
    return "%s:%s:%s:%s" % (
        segment,
        window["group_index"],
        window["window_index"],
        window["start"],
    )


def _resume_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that affect calibrated per-window discoveries."""

    payload = dict(signature)
    payload.pop("corpus_null_repeats", None)
    payload.pop("hypothesis_decay", None)
    payload.pop("replication_separation_hours", None)
    payload.pop("block_size", None)
    payload.pop("calibration_report", None)
    payload["spectral_profiles_only"] = bool(
        payload.get("spectral_profiles_only", False)
    )
    return payload


def _select_windows(
    source: Mapping[str, Any],
    *,
    groups_per_segment: int,
    windows_per_group: int,
    window_index_step: int,
) -> list[tuple[str, dict[str, Any]]]:
    selected = []
    for segment in ("calibration", "validation"):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for window in source[segment]["windows"]:
            grouped[int(window["group_index"])].append(dict(window))
        for group_index in sorted(grouped)[:groups_per_segment]:
            windows = sorted(grouped[group_index], key=lambda row: int(row["window_index"]))
            spaced = windows[::window_index_step][:windows_per_group]
            selected.extend((segment, window) for window in spaced)
    return selected


def _load_canonical_window(
    window: Mapping[str, Any],
    *,
    channel: str,
    keep_flags: set[int],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    variable = CHANNEL_VARIABLES[channel]
    n_samples = int(window["samples"])
    target_rate = float(window["sample_rate"])
    start_time = datetime.fromisoformat(str(window["start"])).timestamp()
    grid = start_time + np.arange(n_samples, dtype=np.float64) / target_rate
    columns = []
    entities = []
    sources = []

    for source_path in window["paths"]:
        path = Path(source_path)
        with netcdf_file(path, mmap=True) as nc:
            sample_rate = _scalar(nc.variables["xyzSampleRate"])
            first_sample_time = (
                _scalar(nc.variables["xyzStartTime"])
                - _scalar(nc.variables["xyzFilterDelay"])
            )
            source_index = (grid - first_sample_time) * sample_rate
            lower = max(0, int(np.floor(np.min(source_index))) - 1)
            upper = min(
                len(nc.variables[variable].data),
                int(np.ceil(np.max(source_index))) + 2,
            )
            if upper - lower < 2:
                raise ValueError(f"{path} has insufficient source samples for window")
            values = np.array(
                nc.variables[variable].data[lower:upper],
                dtype=np.float64,
                copy=True,
            )
            flags = np.array(
                nc.variables["xyzFlagPrimary"].data[lower:upper],
                dtype=np.int16,
                copy=True,
            )
            local = source_index - lower
            floor = np.clip(np.floor(local).astype(np.int64), 0, len(values) - 1)
            ceil = np.clip(np.ceil(local).astype(np.int64), 0, len(values) - 1)
            nearest = np.clip(np.rint(local).astype(np.int64), 0, len(values) - 1)
            effectively_integer = np.abs(local - nearest) <= 1e-5
            floor = np.where(effectively_integer, nearest, floor)
            ceil = np.where(effectively_integer, nearest, ceil)
            valid = (
                np.isin(flags[floor], list(keep_flags))
                & np.isin(flags[ceil], list(keep_flags))
                & np.isfinite(values[floor])
                & np.isfinite(values[ceil])
                & (values[floor] > -999.0)
                & (values[ceil] > -999.0)
            )
            if not np.all(valid):
                raise ValueError(
                    f"{path} contains invalid source samples in selected aligned window"
                )
            segment = np.interp(
                local,
                np.arange(len(values), dtype=np.float64),
                values,
            ).astype(np.float64)
            station_id = _decode_attr(getattr(nc, "cdip_station_id", ""), "unknown")
            platform_id = _decode_attr(
                getattr(nc, "platform_id", station_id),
                station_id,
            )
            platform_name = _decode_attr(
                getattr(nc, "platform_name", platform_id),
                platform_id,
            )
        columns.append(segment)
        entities.append(platform_id)
        sources.append(
            {
                "entity": platform_id,
                "station_id": station_id,
                "platform_name": platform_name,
                "path": str(path),
                "source_sample_rate": sample_rate,
            }
        )
    return np.column_stack(columns), entities, {
        "sources": sources,
        "target_sample_rate": target_rate,
        "samples": n_samples,
        "start": window["start"],
        "end": window["end"],
    }


def _parse_layers(text: str) -> list[str]:
    layers = [value.strip() for value in text.split(",") if value.strip()]
    valid = {"raw"}
    for kind in ("common", "dominant"):
        for fraction in ("0.25", "0.5", "0.75", "1.0"):
            valid.add(f"{kind}:{fraction}")
    unknown = [layer for layer in layers if layer not in valid]
    if unknown:
        raise ValueError(f"unknown relationship layers: {', '.join(unknown)}")
    if "raw" not in layers:
        layers.insert(0, "raw")
    return layers


def _layer_view(matrix: np.ndarray, layer: str) -> tuple[np.ndarray, dict[str, Any]]:
    if layer == "raw":
        return matrix.copy(), {
            "parent": None,
            "common_mode_fraction": 0.0,
            "dominant_spectral_fraction": 0.0,
        }
    kind, fraction_text = layer.split(":", 1)
    fraction = float(fraction_text)
    if kind == "common":
        return residualize_relationship_view(
            matrix,
            common_mode_fraction=fraction,
        ), {
            "parent": "raw",
            "common_mode_fraction": fraction,
            "dominant_spectral_fraction": 0.0,
        }
    if kind == "dominant":
        return residualize_relationship_view(
            matrix,
            dominant_spectral_fraction=fraction,
        ), {
            "parent": "raw",
            "common_mode_fraction": 0.0,
            "dominant_spectral_fraction": fraction,
        }
    raise ValueError(f"unknown layer: {layer}")


def _evidence_deposit(
    row: Mapping[str, Any],
    *,
    layer: str,
    supported: set[tuple[str, str]],
) -> float:
    if (layer, str(row["family"])) not in supported or not bool(row["detected"]):
        return 0.0
    z_effect = row.get("z_effect")
    z = max(0.0, float(z_effect)) if z_effect is not None else 0.0
    return float(z * (1.0 - float(row["empirical_p_ge_observed"])))


def _null_summary(observed: float, null_values: Sequence[float]) -> dict[str, Any]:
    null = np.asarray(null_values, dtype=np.float64)
    if len(null) == 0:
        return {
            "observed": float(observed),
            "null_repeats": 0,
            "null_unique_values": 0,
            "null_mean": None,
            "null_std": None,
            "observed_minus_null": None,
            "z_effect": None,
            "null_exceedances": None,
            "empirical_p_ge_observed": None,
            "empirical_p_floor": None,
            "detected": False,
        }
    mean = float(np.mean(null))
    std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    delta = float(observed - mean)
    z_effect = delta / std if std > 1e-12 else None
    exceedances = int(np.sum(null >= observed))
    p_ge = float((exceedances + 1) / (len(null) + 1))
    return {
        "observed": float(observed),
        "null_repeats": len(null),
        "null_unique_values": len(np.unique(np.round(null, decimals=12))),
        "null_mean": mean,
        "null_std": std,
        "observed_minus_null": delta,
        "z_effect": z_effect,
        "null_exceedances": exceedances,
        "empirical_p_ge_observed": p_ge,
        "empirical_p_floor": float(1.0 / (len(null) + 1)),
        "detected": bool(
            delta > 0.0
            and p_ge <= 0.05
            and z_effect is not None
            and z_effect >= 1.5
        ),
    }


def _add_envelope_profiles(records: Sequence[dict[str, Any]]) -> None:
    """Attach the preregistered corpus-envelope attenuation ladder in place."""

    powers = np.stack([row["power"] for row in records])
    log_power = np.log(np.maximum(powers, 1e-12))
    baseline = np.median(log_power, axis=0)
    mad = 1.4826 * np.median(np.abs(log_power - baseline[None, :]), axis=0)
    mad = np.where(mad > 1e-6, mad, 1.0)
    for index, row in enumerate(records):
        for profile_key, fraction in ENVELOPE_PROFILE_FRACTIONS.items():
            if fraction == 0.0:
                profile = powers[index]
            else:
                profile = np.exp(log_power[index] - fraction * baseline)
                profile = profile / np.sum(profile)
            row[profile_key] = profile
        row[SIGNED_ENVELOPE_PROFILE] = (
            log_power[index] - baseline
        ) / mad


def _apply_pair_fdr(pair_summaries: Sequence[dict[str, Any]]) -> None:
    """Apply BH jointly across every supported pair, view, and profile tried."""

    for control in ("domain_regroup", "temporal_regroup"):
        eligible = []
        for index, row in enumerate(pair_summaries):
            for profile_key in SPECTRAL_PROFILE_KEYS:
                summary = row[profile_key][control]
                summary["fdr_q_value"] = None
                summary["detected_fdr"] = False
                p_value = summary.get("empirical_p_ge_observed")
                if (
                    row["spectral_alignment_calibrated_supported"]
                    and p_value is not None
                ):
                    eligible.append((float(p_value), index, profile_key))
        eligible.sort(key=lambda item: item[0])
        adjusted: dict[tuple[int, str], float] = {}
        running = 1.0
        total = len(eligible)
        for rank in range(total, 0, -1):
            p_value, index, profile_key = eligible[rank - 1]
            running = min(running, p_value * total / rank)
            adjusted[(index, profile_key)] = float(min(1.0, running))
        for (index, profile_key), q_value in adjusted.items():
            summary = pair_summaries[index][profile_key][control]
            z_effect = summary.get("z_effect")
            summary["fdr_q_value"] = q_value
            summary["detected_fdr"] = bool(
                q_value <= 0.05
                and float(summary["observed_minus_null"]) > 0.0
                and z_effect is not None
                and float(z_effect) >= 1.5
            )


def _classify_pair_summary(row: dict[str, Any]) -> None:
    domain_fractions = [
        fraction
        for profile_key, fraction in ENVELOPE_PROFILE_FRACTIONS.items()
        if row[profile_key]["domain_regroup"]["detected_fdr"]
    ]
    temporal_fractions = [
        fraction
        for profile_key, fraction in ENVELOPE_PROFILE_FRACTIONS.items()
        if row[profile_key]["temporal_regroup"]["detected_fdr"]
    ]
    concurrent_fractions = sorted(set(domain_fractions) & set(temporal_fractions))
    signed_domain = bool(
        row[SIGNED_ENVELOPE_PROFILE]["domain_regroup"]["detected_fdr"]
    )
    signed_temporal = bool(
        row[SIGNED_ENVELOPE_PROFILE]["temporal_regroup"]["detected_fdr"]
    )
    row["attenuation_survival"] = {
        "domain_fractions": domain_fractions,
        "temporal_fractions": temporal_fractions,
        "concurrent_fractions": concurrent_fractions,
        "deepest_domain_fraction": max(domain_fractions) if domain_fractions else None,
        "deepest_concurrent_fraction": (
            max(concurrent_fractions) if concurrent_fractions else None
        ),
        "signed_residual_domain": signed_domain,
        "signed_residual_temporal": signed_temporal,
        "signed_residual_concurrent": signed_domain and signed_temporal,
    }
    deepest = row["attenuation_survival"]["deepest_domain_fraction"]
    concurrent = bool(concurrent_fractions)
    if signed_domain:
        base = "signed_residual"
        concurrent = signed_temporal
    elif deepest == 1.0:
        base = "fully_envelope_attenuated"
    elif deepest is not None and deepest > 0.0:
        base = "partially_envelope_attenuated"
    elif deepest == 0.0:
        base = "ordinary_spectral_shape"
    elif temporal_fractions or signed_temporal:
        row["classification"] = "concurrence_without_pair_specificity"
        return
    else:
        row["classification"] = "generic_or_unresolved"
        return
    row["classification"] = (
        f"{base}_concurrent_pair" if concurrent else f"{base}_pair_specific"
    )


def _mean_group_alignment(
    records: Sequence[Mapping[str, Any]],
    *,
    profile_key: str,
) -> float:
    scores = []
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (str(row["segment"]), int(row["window_index"]), int(row["group_index"]))
        ].append(row)
    for group in grouped.values():
        for left, right in combinations(group, 2):
            if left["entity"] == right["entity"]:
                continue
            scores.append(
                spectral_profile_alignment(
                    np.asarray(left[profile_key], dtype=np.float64),
                    np.asarray(right[profile_key], dtype=np.float64),
                )
            )
    return float(np.mean(scores)) if scores else 0.0


def _spectral_specificity(
    windows: Sequence[Mapping[str, Any]],
    *,
    supported_by_layer: Mapping[str, Sequence[str]],
    null_repeats: int,
    seed: int,
    replication_separation_seconds: float,
    progress: bool = False,
) -> dict[str, Any]:
    """Test spectral hypotheses while preserving the obvious domain spectrum."""

    def report(message: str) -> None:
        if progress:
            print(f"cdip_relationship corpus={message}", file=sys.stderr, flush=True)

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        for layer in window["layers"]:
            for profile in layer.get("spectral_profiles", []):
                by_layer[str(layer["layer"])].append(
                    {
                        "segment": str(window["segment"]),
                        "group_index": int(window["group_index"]),
                        "window_index": int(window["window_index"]),
                        "start": str(window["source"]["start"]),
                        "entity": str(profile["entity"]),
                        "power": np.asarray(profile["power"], dtype=np.float64),
                    }
                )

    layer_summaries = []
    temporal_pairs = []
    pair_summaries = []
    for layer_name, records in sorted(by_layer.items()):
        if not records:
            continue
        report(f"layer={layer_name} phase=envelope_profiles records={len(records)}")
        _add_envelope_profiles(records)
        domain_observed = {
            profile_key: _mean_group_alignment(records, profile_key=profile_key)
            for profile_key in SPECTRAL_PROFILE_KEYS
        }
        strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            strata[(row["segment"], row["window_index"])].append(row)
        domain_null = {profile_key: [] for profile_key in SPECTRAL_PROFILE_KEYS}
        domain_rng = _rng_for(seed, layer_name, "layer_domain_regroup")
        for _ in range(null_repeats):
            regrouped = []
            for stratum in strata.values():
                group_sizes = [
                    len([row for row in stratum if row["group_index"] == group])
                    for group in sorted({row["group_index"] for row in stratum})
                ]
                shuffled = [
                    stratum[int(index)]
                    for index in domain_rng.permutation(len(stratum))
                ]
                offset = 0
                for synthetic_group, size in enumerate(group_sizes):
                    for row in shuffled[offset : offset + size]:
                        regrouped.append({**row, "group_index": synthetic_group})
                    offset += size
            for profile_key in domain_null:
                domain_null[profile_key].append(
                    _mean_group_alignment(
                        regrouped,
                        profile_key=profile_key,
                    )
                )
        report(f"layer={layer_name} phase=layer_domain_regroup complete")

        temporal_groups: dict[tuple[str, int], dict[str, dict[int, dict[str, Any]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        for row in records:
            temporal_groups[(row["segment"], row["group_index"])][row["entity"]][
                row["window_index"]
            ] = row

        def temporal_score(
            *,
            profile_key: str,
            randomize: bool,
            score_rng: np.random.Generator | None = None,
        ) -> float:
            scores = []
            for entity_rows in temporal_groups.values():
                entities = sorted(entity_rows)
                if len(entities) < 2:
                    continue
                common_windows = sorted(
                    set.intersection(
                        *(set(entity_rows[entity]) for entity in entities)
                    )
                )
                if len(common_windows) < 2:
                    continue
                assigned = {entities[0]: common_windows}
                for entity in entities[1:]:
                    assigned[entity] = (
                        [
                            common_windows[int(index)]
                            for index in score_rng.permutation(len(common_windows))
                        ]
                        if randomize
                        else common_windows
                    )
                for left_entity, right_entity in combinations(entities, 2):
                    for position in range(len(common_windows)):
                        left = entity_rows[left_entity][assigned[left_entity][position]]
                        right = entity_rows[right_entity][assigned[right_entity][position]]
                        scores.append(
                            spectral_profile_alignment(
                                np.asarray(left[profile_key], dtype=np.float64),
                                np.asarray(right[profile_key], dtype=np.float64),
                            )
                        )
            return float(np.mean(scores)) if scores else 0.0

        temporal_observed = {
            profile_key: temporal_score(
                profile_key=profile_key,
                randomize=False,
            )
            for profile_key in SPECTRAL_PROFILE_KEYS
        }
        temporal_null = {profile_key: [] for profile_key in SPECTRAL_PROFILE_KEYS}
        for profile_key in temporal_null:
            temporal_rng = _rng_for(
                seed,
                layer_name,
                profile_key,
                "layer_temporal_regroup",
            )
            for _ in range(null_repeats):
                temporal_null[profile_key].append(
                    temporal_score(
                        profile_key=profile_key,
                        randomize=True,
                        score_rng=temporal_rng,
                    )
                )
        report(f"layer={layer_name} phase=layer_temporal_regroup complete")

        for (segment, group_index), entity_rows in temporal_groups.items():
            entities = sorted(entity_rows)
            for left_entity, right_entity in combinations(entities, 2):
                common_windows = sorted(
                    set(entity_rows[left_entity]) & set(entity_rows[right_entity])
                )
                if len(common_windows) < 2:
                    continue
                identity = tuple(range(len(common_windows)))
                candidate_permutations = list(permutations(range(len(common_windows))))
                pair_row: dict[str, Any] = {
                    "layer": layer_name,
                    "segment": segment,
                    "group_index": group_index,
                    "entities": [left_entity, right_entity],
                    "window_indices": common_windows,
                }
                for profile_key in SPECTRAL_PROFILE_KEYS:
                    scores = []
                    for permutation in candidate_permutations:
                        scores.append(
                            float(
                                np.mean(
                                    [
                                        spectral_profile_alignment(
                                            np.asarray(
                                                entity_rows[left_entity][window][profile_key],
                                                dtype=np.float64,
                                            ),
                                            np.asarray(
                                                entity_rows[right_entity][
                                                    common_windows[permutation[position]]
                                                ][profile_key],
                                                dtype=np.float64,
                                            ),
                                        )
                                        for position, window in enumerate(common_windows)
                                    ]
                                )
                            )
                        )
                    observed_index = candidate_permutations.index(identity)
                    observed = scores[observed_index]
                    nonconcurrent = [
                        score
                        for index, score in enumerate(scores)
                        if index != observed_index
                    ]
                    pair_row[profile_key] = {
                        **_null_summary(observed, nonconcurrent),
                        "identity_is_best": bool(observed >= max(scores) - 1e-12),
                        "exact_permutations": len(scores),
                    }
                temporal_pairs.append(pair_row)

        occurrence_groups: dict[
            tuple[str, int, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in records:
            occurrence_groups[
                (row["segment"], row["group_index"], row["window_index"])
            ].append(row)
        pair_occurrences: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for (segment, group_index, window_index), group in occurrence_groups.items():
            for left, right in combinations(sorted(group, key=lambda row: row["entity"]), 2):
                if left["entity"] == right["entity"]:
                    continue
                occurrence = {
                    "segment": segment,
                    "group_index": group_index,
                    "window_index": window_index,
                    "start": left["start"],
                    "left": left,
                    "right": right,
                }
                for profile_key in SPECTRAL_PROFILE_KEYS:
                    occurrence[profile_key] = spectral_profile_alignment(
                        left[profile_key],
                        right[profile_key],
                    )
                pair_occurrences[(left["entity"], right["entity"])].append(occurrence)
        all_pair_scores = {
            profile_key: [
                (pair, float(occurrence[profile_key]))
                for pair, occurrences in pair_occurrences.items()
                for occurrence in occurrences
            ]
            for profile_key in SPECTRAL_PROFILE_KEYS
        }
        repeated_pairs = [
            pair for pair, occurrences in pair_occurrences.items() if len(occurrences) >= 2
        ]
        processed_pairs = 0
        for pair, occurrences in pair_occurrences.items():
            if len(occurrences) < 2:
                continue
            timestamps = sorted(
                datetime.fromisoformat(str(row["start"])).timestamp()
                for row in occurrences
            )
            time_clusters = []
            for timestamp in timestamps:
                if (
                    not time_clusters
                    or timestamp - time_clusters[-1][-1]
                    >= replication_separation_seconds
                ):
                    time_clusters.append([timestamp])
                else:
                    time_clusters[-1].append(timestamp)
            pair_summary: dict[str, Any] = {
                "layer": layer_name,
                "entities": list(pair),
                "occurrences": len(occurrences),
                "segments": sorted({row["segment"] for row in occurrences}),
                "replicated_across_segments": (
                    len({row["segment"] for row in occurrences}) > 1
                ),
                "independent_time_clusters": len(time_clusters),
                "replicated_across_distinct_times": len(time_clusters) > 1,
                "replication_separation_seconds": replication_separation_seconds,
                "maximum_time_separation_seconds": (
                    float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
                ),
                "source_windows": [
                    {
                        "segment": row["segment"],
                        "group_index": row["group_index"],
                        "window_index": row["window_index"],
                        "start": row["start"],
                    }
                    for row in occurrences
                ],
            }
            for profile_key in SPECTRAL_PROFILE_KEYS:
                observed = float(
                    np.mean([row[profile_key] for row in occurrences])
                )
                other_scores = [
                    score
                    for candidate_pair, score in all_pair_scores[profile_key]
                    if candidate_pair != pair
                ]
                domain_rng = _rng_for(
                    seed,
                    layer_name,
                    pair[0],
                    pair[1],
                    profile_key,
                    "pair_domain_regroup",
                )
                pair_domain_null = np.mean(
                    domain_rng.choice(
                        np.asarray(other_scores, dtype=np.float64),
                        size=(null_repeats, len(occurrences)),
                        replace=True,
                    ),
                    axis=1,
                ).tolist()
                grouped_occurrences: dict[
                    tuple[str, int], list[dict[str, Any]]
                ] = defaultdict(list)
                for row in occurrences:
                    grouped_occurrences[(row["segment"], row["group_index"])].append(row)
                eligible = [
                    sorted(rows, key=lambda row: row["window_index"])
                    for rows in grouped_occurrences.values()
                    if len(rows) >= 2
                ]
                temporal_observed_scores = [
                    float(row[profile_key])
                    for rows in eligible
                    for row in rows
                ]
                pair_temporal_null = []
                temporal_null_method = None
                temporal_permutation_space = 0
                if temporal_observed_scores:
                    temporal_rng = _rng_for(
                        seed,
                        layer_name,
                        pair[0],
                        pair[1],
                        profile_key,
                        "pair_temporal_regroup",
                    )
                    group_permutation_sums = []
                    for rows in eligible:
                        sums = []
                        for order in permutations(range(len(rows))):
                            sums.append(
                                float(
                                    sum(
                                        spectral_profile_alignment(
                                            row["left"][profile_key],
                                            rows[int(order[index])]["right"][profile_key],
                                        )
                                        for index, row in enumerate(rows)
                                    )
                                )
                            )
                        group_permutation_sums.append(np.asarray(sums, dtype=np.float64))
                    temporal_permutation_space = int(
                        np.prod(
                            [len(values) for values in group_permutation_sums],
                            dtype=object,
                        )
                    )
                    total_observations = len(temporal_observed_scores)
                    if temporal_permutation_space <= null_repeats:
                        pair_temporal_null = [
                            float(sum(values) / total_observations)
                            for values in product(*group_permutation_sums)
                        ]
                        temporal_null_method = "exact_permutation_space"
                    else:
                        totals = np.zeros(null_repeats, dtype=np.float64)
                        for values in group_permutation_sums:
                            totals += values[
                                temporal_rng.integers(len(values), size=null_repeats)
                            ]
                        pair_temporal_null = (totals / total_observations).tolist()
                        temporal_null_method = "sampled_permutation_space"
                temporal_summary = _null_summary(
                    (
                        float(np.mean(temporal_observed_scores))
                        if temporal_observed_scores
                        else 0.0
                    ),
                    pair_temporal_null,
                )
                temporal_summary["null_generation"] = temporal_null_method
                temporal_summary["permutation_space_size"] = (
                    temporal_permutation_space
                    if temporal_observed_scores
                    else 0
                )
                pair_summary[profile_key] = {
                    "domain_regroup": _null_summary(observed, pair_domain_null),
                    "temporal_regroup": temporal_summary,
                }
            pair_summary["spectral_alignment_calibrated_supported"] = (
                "spectral_alignment" in set(supported_by_layer.get(layer_name, ()))
            )
            pair_summaries.append(pair_summary)
            processed_pairs += 1
            if processed_pairs % 50 == 0 or processed_pairs == len(repeated_pairs):
                report(
                    f"layer={layer_name} phase=pair_controls "
                    f"pairs={processed_pairs}/{len(repeated_pairs)}"
                )

        layer_summaries.append(
            {
                "layer": layer_name,
                "spectral_alignment_calibrated_supported": (
                    "spectral_alignment" in set(supported_by_layer.get(layer_name, ()))
                ),
                "domain_regroup": {
                    profile_key: _null_summary(
                        domain_observed[profile_key],
                        domain_null[profile_key],
                    )
                    for profile_key in domain_observed
                },
                "temporal_regroup": {
                    profile_key: _null_summary(
                        temporal_observed[profile_key],
                        temporal_null[profile_key],
                    )
                    for profile_key in temporal_observed
                },
                "profiles": len(records),
            }
        )
        report(f"layer={layer_name} phase=complete")

    temporal_pairs.sort(
        key=lambda row: (
            row[SIGNED_ENVELOPE_PROFILE]["identity_is_best"],
            row[SIGNED_ENVELOPE_PROFILE]["observed_minus_null"] or float("-inf"),
        ),
        reverse=True,
    )
    _apply_pair_fdr(pair_summaries)
    for row in pair_summaries:
        _classify_pair_summary(row)
    classification_rank = {
        "signed_residual_concurrent_pair": 9,
        "signed_residual_pair_specific": 8,
        "fully_envelope_attenuated_concurrent_pair": 7,
        "fully_envelope_attenuated_pair_specific": 6,
        "partially_envelope_attenuated_concurrent_pair": 5,
        "partially_envelope_attenuated_pair_specific": 4,
        "ordinary_spectral_shape_concurrent_pair": 3,
        "ordinary_spectral_shape_pair_specific": 2,
        "concurrence_without_pair_specificity": 1,
        "generic_or_unresolved": 0,
    }
    pair_summaries.sort(
        key=lambda row: (
            classification_rank[row["classification"]],
            row["replicated_across_distinct_times"],
            row["raw_profile"]["domain_regroup"]["z_effect"]
            if row["raw_profile"]["domain_regroup"]["z_effect"] is not None
            else float("-inf"),
        ),
        reverse=True,
    )
    return {
        "method": {
            "domain_regroup": (
                "preserves every spectral profile and view stratum while "
                "randomizing candidate group membership"
            ),
            "temporal_regroup": (
                "preserves entity identity, group membership, every complete "
                "spectral profile, and the generic domain envelope while "
                "randomizing only within-group window concurrence"
            ),
            "signed_envelope_residual": (
                "robust log-spectrum residual after subtracting the corpus median "
                "and scaling each frequency bin by corpus MAD"
            ),
            "envelope_attenuation_ladder": {
                "fractions": list(ENVELOPE_PROFILE_FRACTIONS.values()),
                "definition": (
                    "normalize(exp(log_power - fraction * corpus_median_log_power)); "
                    "fraction zero is the ordinary spectrum and fraction one is the "
                    "positive ratio-to-envelope endpoint"
                ),
                "preregistered": True,
            },
            "pair_multiplicity": (
                "Benjamini-Hochberg FDR correction is applied jointly across every "
                "supported pair, waveform view, and tried attenuation profile, "
                "separately for the domain-regroup and temporal-regroup controls"
            ),
        },
        "layer_summary": layer_summaries,
        "pair_summary": pair_summaries,
        "top_temporal_pairs": temporal_pairs[:50],
    }


def _aggregate(
    windows: Sequence[Mapping[str, Any]],
    *,
    supported_by_layer: Mapping[str, Sequence[str]],
    hypothesis_decay: float,
    corpus_null_repeats: int,
    seed: int,
    replication_separation_seconds: float,
    progress: bool = False,
) -> dict[str, Any]:
    supported = {
        (str(layer), str(family))
        for layer, families in supported_by_layer.items()
        for family in families
    }
    family_rows: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    hypothesis_rows: dict[tuple[str, str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    detection_index: dict[tuple[str, str, str, tuple[str, ...]], bool] = {}
    top_source_rows = []

    for window in windows:
        segment = str(window["segment"])
        for layer in window["layers"]:
            layer_name = str(layer["layer"])
            for row in layer["discovery"]["evidence"]:
                family = str(row["family"])
                entities = tuple(str(value) for value in row["entities"])
                family_rows[(segment, layer_name, family)].append(row)
                hypothesis_rows[(layer_name, family, entities)].append(
                    {
                        "segment": segment,
                        "window_key": window["window_key"],
                        "start": window["source"]["start"],
                        "group_index": window["group_index"],
                        "window_index": window["window_index"],
                        "deposit": _evidence_deposit(
                            row,
                            layer=layer_name,
                            supported=supported,
                        ),
                        "evidence": row,
                        "source": window["source"],
                    }
                )
                detection_index[(segment, layer_name, family, entities)] = bool(
                    detection_index.get(
                        (segment, layer_name, family, entities),
                        False,
                    )
                    or (row["detected"] and (layer_name, family) in supported)
                )
                deposit = _evidence_deposit(
                    row,
                    layer=layer_name,
                    supported=supported,
                )
                if deposit > 0.0:
                    top_source_rows.append(
                        {
                            "deposit": deposit,
                            "segment": segment,
                            "layer": layer_name,
                            "family": family,
                            "entities": list(entities),
                            "evidence": row,
                            "source": window["source"],
                            "group_index": window["group_index"],
                            "window_index": window["window_index"],
                        }
                    )

    family_summary = []
    for (segment, layer, family), rows in sorted(family_rows.items()):
        detected = [
            row
            for row in rows
            if row["detected"] and (layer, family) in supported
        ]
        z_values = [
            float(row["z_effect"])
            for row in rows
            if row.get("z_effect") is not None
        ]
        family_summary.append(
            {
                "segment": segment,
                "layer": layer,
                "family": family,
                "calibrated_supported": (layer, family) in supported,
                "tests": len(rows),
                "detected": len(detected),
                "detected_fraction": len(detected) / len(rows),
                "median_observed_minus_null": float(
                    np.median([float(row["observed_minus_null"]) for row in rows])
                ),
                "median_z_effect": (
                    float(np.median(z_values)) if z_values else None
                ),
            }
        )

    hypotheses = []
    for (layer, family, entities), rows in sorted(hypothesis_rows.items()):
        pheromone = 0.0
        maximum = 0.0
        for row in sorted(rows, key=lambda value: (value["start"], value["window_key"])):
            pheromone = hypothesis_decay * pheromone + float(row["deposit"])
            maximum = max(maximum, pheromone)
        segments = sorted({str(row["segment"]) for row in rows if row["deposit"] > 0.0})
        hypotheses.append(
            {
                "layer": layer,
                "family": family,
                "entities": list(entities),
                "windows": len(rows),
                "detected_windows": int(sum(row["deposit"] > 0.0 for row in rows)),
                "detected_segments": segments,
                "replicated_across_segments": len(segments) > 1,
                "final_pheromone": pheromone,
                "max_pheromone": maximum,
                "total_deposit": float(sum(row["deposit"] for row in rows)),
            }
        )
    hypotheses.sort(
        key=lambda row: (row["replicated_across_segments"], row["max_pheromone"]),
        reverse=True,
    )

    replication = []
    layers = sorted({key[1] for key in family_rows})
    families = sorted({key[2] for key in family_rows})
    for layer in layers:
        for family in families:
            calibration = next(
                (
                    row
                    for row in family_summary
                    if row["segment"] == "calibration"
                    and row["layer"] == layer
                    and row["family"] == family
                ),
                None,
            )
            validation = next(
                (
                    row
                    for row in family_summary
                    if row["segment"] == "validation"
                    and row["layer"] == layer
                    and row["family"] == family
                ),
                None,
            )
            if calibration is not None and validation is not None:
                replication.append(
                    {
                        "layer": layer,
                        "family": family,
                        "calibrated_supported": (layer, family) in supported,
                        "calibration_detected_fraction": calibration["detected_fraction"],
                        "validation_detected_fraction": validation["detected_fraction"],
                        "minimum_segment_detected_fraction": min(
                            calibration["detected_fraction"],
                            validation["detected_fraction"],
                        ),
                    }
                )
    replication.sort(
        key=lambda row: (
            row["calibrated_supported"],
            row["minimum_segment_detected_fraction"],
        ),
        reverse=True,
    )

    novelty = []
    nonraw_layers = sorted({key[1] for key in detection_index if key[1] != "raw"})
    for layer in nonraw_layers:
        for family in families:
            keys = {
                (segment, entities)
                for segment, candidate_layer, candidate_family, entities in detection_index
                if candidate_layer in {"raw", layer} and candidate_family == family
            }
            residual_only = 0
            lost_from_raw = 0
            retained = 0
            for segment, entities in keys:
                raw = detection_index.get((segment, "raw", family, entities), False)
                residual = detection_index.get((segment, layer, family, entities), False)
                residual_only += int(residual and not raw)
                lost_from_raw += int(raw and not residual)
                retained += int(raw and residual)
            novelty.append(
                {
                    "layer": layer,
                    "family": family,
                    "residual_only_hypotheses": residual_only,
                    "lost_from_raw": lost_from_raw,
                    "retained_from_raw": retained,
                }
            )

    top_source_rows.sort(key=lambda row: row["deposit"], reverse=True)
    return {
        "family_segment_summary": family_summary,
        "family_replication": replication,
        "hypothesis_field": hypotheses,
        "residual_layer_novelty": novelty,
        "top_source_mappings": top_source_rows[:50],
        "spectral_specificity": _spectral_specificity(
            windows,
            supported_by_layer=supported_by_layer,
            null_repeats=corpus_null_repeats,
            seed=seed,
            replication_separation_seconds=replication_separation_seconds,
            progress=progress,
        ),
    }


def run_relationship_discovery(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    layers = _parse_layers(args.layers)
    selected = _select_windows(
        source,
        groups_per_segment=args.groups_per_segment,
        windows_per_group=args.windows_per_group,
        window_index_step=args.window_index_step,
    )
    if not selected:
        raise ValueError("source report produced no selected windows")
    first_window = selected[0][1]
    signature = {
        "source_report": str(args.source_report),
        "calibration_report": (
            str(args.calibration_report) if args.calibration_report is not None else None
        ),
        "channel": args.channel,
        "layers": layers,
        "groups_per_segment": args.groups_per_segment,
        "windows_per_group": args.windows_per_group,
        "window_index_step": args.window_index_step,
        "null_repeats": args.null_repeats,
        "corpus_null_repeats": args.corpus_null_repeats,
        "calibration_trials": args.calibration_trials,
        "calibration_null_repeats": args.calibration_null_repeats,
        "min_detection_rate": args.min_detection_rate,
        "max_false_positive_rate": args.max_false_positive_rate,
        "nperseg": args.nperseg,
        "max_lag": args.max_lag,
        "hypothesis_decay": args.hypothesis_decay,
        "replication_separation_hours": args.replication_separation_hours,
        "spectral_profiles_only": args.spectral_profiles_only,
        "seed": args.seed,
    }
    completed: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    calibrations: dict[str, dict[str, Any]] = {}
    if args.calibration_report is not None:
        calibration_source = json.loads(
            args.calibration_report.read_text(encoding="utf-8")
        )
        available = calibration_source.get("calibration", {}).get("by_layer", {})
        for layer in layers:
            if layer not in available:
                raise ValueError(
                    f"calibration report does not contain requested layer: {layer}"
                )
            payload = dict(available[layer])
            expected = {
                "n": int(first_window["samples"]),
                "series_count": len(first_window["paths"]),
                "trials": args.calibration_trials,
                "null_repeats": args.calibration_null_repeats,
                "min_detection_rate": args.min_detection_rate,
                "max_false_positive_rate": args.max_false_positive_rate,
            }
            mismatches = {
                key: (payload.get(key), value)
                for key, value in expected.items()
                if payload.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"calibration report geometry/settings mismatch for {layer}: "
                    f"{mismatches}"
                )
            calibrations[layer] = payload
    if args.resume:
        if args.checkpoint is None or not args.checkpoint.exists():
            raise ValueError("--resume requires an existing --checkpoint")
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if _resume_signature(checkpoint.get("signature", {})) != _resume_signature(
            signature
        ):
            raise ValueError("checkpoint signature does not match current run")
        completed = list(checkpoint.get("windows", []))
        for window in completed:
            for layer in window.get("layers", []):
                layer.get("discovery", {}).pop("block_size", None)
        completed_keys = {str(row["window_key"]) for row in completed}
        calibrations.update(
            {
                str(layer): dict(payload)
                for layer, payload in checkpoint.get("calibrations", {}).items()
            }
        )

    def write_checkpoint(*, complete: bool) -> None:
        if args.checkpoint is None:
            return
        _write_json(
            args.checkpoint,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "complete": complete,
                "signature": signature,
                "calibrations": calibrations,
                "windows": completed,
            },
        )

    write_checkpoint(complete=False)
    for layer_index, layer in enumerate(layers):
        if layer in calibrations:
            continue
        calibration = calibrate_relationship_geometry(
            n=int(first_window["samples"]),
            series_count=len(first_window["paths"]),
            sample_rate=float(first_window["sample_rate"]),
            trials=args.calibration_trials,
            null_repeats=args.calibration_null_repeats,
            seed=args.seed + 1_000_000 * layer_index,
            nperseg=args.nperseg,
            max_lag=args.max_lag,
            min_detection_rate=args.min_detection_rate,
            max_false_positive_rate=args.max_false_positive_rate,
            layer=layer,
            matrix_transform=lambda matrix, selected_layer=layer: _layer_view(
                matrix,
                selected_layer,
            )[0],
        )
        calibrations[layer] = calibration.to_dict()
        write_checkpoint(complete=False)
        if args.progress and (
            len(completed) == 1
            or len(completed) % 25 == 0
            or len(completed) == len(selected)
        ):
            print(
                "cdip_relationship calibration=%d/%d layer=%s supported=%s elapsed=%.1fs"
                % (
                    len(calibrations),
                    len(layers),
                    layer,
                    ",".join(calibration.supported_families) or "none",
                    time.time() - started,
                ),
                file=sys.stderr,
                flush=True,
            )

    supported_by_layer = {
        layer: list(calibrations[layer]["supported_families"])
        for layer in layers
    }
    for selection_index, (segment, window) in enumerate(selected):
        key = _window_key(segment, window)
        if key in completed_keys:
            continue
        matrix, entities, source_mapping = _load_canonical_window(
            window,
            channel=args.channel,
            keep_flags=set(args.keep_flags),
        )
        layer_rows = []
        for layer_index, layer in enumerate(layers):
            view, residualization = _layer_view(matrix, layer)
            if args.spectral_profiles_only:
                discovery_payload = {
                    "layer": layer,
                    "evidence": [],
                    "spectral_profiles_only": True,
                }
            else:
                discovery_payload = discover_relationships(
                    view,
                    entities,
                    sample_rate=float(window["sample_rate"]),
                    layer=layer,
                    null_repeats=args.null_repeats,
                    seed=args.seed + 100_000 * selection_index + 1000 * layer_index,
                    nperseg=args.nperseg,
                    max_lag=args.max_lag,
                ).to_dict()
            profiles = []
            frequencies = None
            for column, entity in enumerate(entities):
                profile_frequencies, power = normalized_spectral_profile(
                    view[:, column],
                    sample_rate=float(window["sample_rate"]),
                    nperseg=args.nperseg,
                )
                if frequencies is None:
                    frequencies = profile_frequencies
                profiles.append(
                    {
                        "entity": entity,
                        "power": [float(value) for value in power],
                    }
                )
            layer_rows.append(
                {
                    "layer": layer,
                    "residualization": residualization,
                    "discovery": discovery_payload,
                    "spectral_frequencies": [
                        float(value)
                        for value in (
                            frequencies
                            if frequencies is not None
                            else np.asarray([], dtype=np.float64)
                        )
                    ],
                    "spectral_profiles": profiles,
                }
            )
        completed.append(
            {
                "window_key": key,
                "segment": segment,
                "group_index": window["group_index"],
                "window_index": window["window_index"],
                "entities": entities,
                "source": source_mapping,
                "layers": layer_rows,
            }
        )
        completed_keys.add(key)
        write_checkpoint(complete=False)
        if args.progress:
            print(
                "cdip_relationship windows=%d/%d segment=%s group=%s index=%s elapsed=%.1fs"
                % (
                    len(completed),
                    len(selected),
                    segment,
                    window["group_index"],
                    window["window_index"],
                    time.time() - started,
                ),
                file=sys.stderr,
                flush=True,
            )
    write_checkpoint(complete=True)
    aggregate = _aggregate(
        completed,
        supported_by_layer=supported_by_layer,
        hypothesis_decay=args.hypothesis_decay,
        corpus_null_repeats=args.corpus_null_repeats,
        seed=args.seed + 50_000_000,
        replication_separation_seconds=args.replication_separation_hours * 3600.0,
        progress=args.progress,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "method": {
            "label_use": "none",
            "source_report_use": (
                "aligned raw-window paths and timestamps only; prior observer scores "
                "and anomaly candidate settings are ignored"
            ),
            "canonical_surface": "aligned raw CDIP displacement waveform",
            "relationship_families": [
                "common_mode",
                "spectral_alignment",
                "phase_coherence",
                "lagged_dependence",
                "relation_change",
            ],
            "residualization": (
                "non-destructive branches; every layer retains raw source back-mapping"
            ),
            "stigmergy": (
                "decayed evidence deposits accumulate on explicit "
                "(layer, family, entities) hypotheses"
            ),
            "corpus_controls": (
                "spectral domain-regroup and within-group temporal-regroup controls "
                "preserve the obvious spectral envelope before testing specificity"
            ),
            "spectral_profiles_only": args.spectral_profiles_only,
        },
        "signature": signature,
        "calibration": {
            "by_layer": calibrations,
            "supported_hypotheses": supported_by_layer,
        },
        "selected_windows": len(selected),
        "completed_windows": len(completed),
        "aggregate": aggregate,
        "windows": completed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=None,
        help="Reuse compatible per-layer exact-geometry calibration from a prior run",
    )
    parser.add_argument("--channel", choices=sorted(CHANNEL_VARIABLES), default="z")
    parser.add_argument("--keep-flags", type=int, nargs="+", default=[2])
    parser.add_argument("--groups-per-segment", type=int, default=4)
    parser.add_argument("--windows-per-group", type=int, default=3)
    parser.add_argument("--window-index-step", type=int, default=2)
    parser.add_argument(
        "--layers",
        default="raw,common:0.5,common:1.0,dominant:0.5,dominant:1.0",
    )
    parser.add_argument("--null-repeats", type=int, default=19)
    parser.add_argument("--corpus-null-repeats", type=int, default=999)
    parser.add_argument("--calibration-trials", type=int, default=8)
    parser.add_argument("--calibration-null-repeats", type=int, default=19)
    parser.add_argument("--min-detection-rate", type=float, default=0.75)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.125)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--max-lag", type=int, default=64)
    parser.add_argument("--hypothesis-decay", type=float, default=0.8)
    parser.add_argument(
        "--replication-separation-hours",
        type=float,
        default=24.0,
        help="Minimum time gap for an independent repeated pair observation",
    )
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--spectral-profiles-only",
        action="store_true",
        help=(
            "Skip expensive per-window relationship families while retaining "
            "calibrated layers, canonical source mappings, and corpus spectral tests"
        ),
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.groups_per_segment < 1 or args.windows_per_group < 1:
        parser.error("group and window counts must be positive")
    if args.window_index_step < 1:
        parser.error("--window-index-step must be positive")
    if (
        args.null_repeats < 1
        or args.corpus_null_repeats < 1
        or args.calibration_null_repeats < 1
    ):
        parser.error("null repeat counts must be positive")
    if not 0.0 <= args.hypothesis_decay < 1.0:
        parser.error("--hypothesis-decay must be in [0, 1)")
    if args.replication_separation_hours <= 0.0:
        parser.error("--replication-separation-hours must be positive")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_relationship_discovery(args)
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
