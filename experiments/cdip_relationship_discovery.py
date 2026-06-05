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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import netcdf_file
from scipy.stats import rankdata

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


def _null_summary(
    observed: float,
    null_values: Sequence[float],
    *,
    exact: bool = False,
) -> dict[str, Any]:
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
            "exact_null": exact,
            "detected": False,
        }
    mean = float(np.mean(null))
    std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    delta = float(observed - mean)
    z_effect = delta / std if std > 1e-12 else None
    exceedances = int(np.sum(null >= observed))
    p_ge = float(
        exceedances / len(null)
        if exact
        else (exceedances + 1) / (len(null) + 1)
    )
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
        "empirical_p_floor": float(1.0 / (len(null) if exact else len(null) + 1)),
        "exact_null": exact,
        "detected": bool(
            delta > 0.0
            and p_ge <= 0.05
            and z_effect is not None
            and z_effect >= 1.5
        ),
    }


def _add_envelope_profiles(
    records: Sequence[dict[str, Any]],
    *,
    stratum_key: str | None = None,
) -> None:
    """Attach the envelope ladder, optionally normalized within known strata."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get(stratum_key, "all")) if stratum_key else "all"].append(row)
    for stratum_records in grouped.values():
        powers = np.stack([row["power"] for row in stratum_records])
        log_power = np.log(np.maximum(powers, 1e-12))
        baseline = np.median(log_power, axis=0)
        mad = 1.4826 * np.median(np.abs(log_power - baseline[None, :]), axis=0)
        mad = np.where(mad > 1e-6, mad, 1.0)
        for index, row in enumerate(stratum_records):
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


def _apply_pair_fdr(
    pair_summaries: Sequence[dict[str, Any]],
    *,
    selected_method: str = "bh",
) -> None:
    """Apply BH and BY jointly across every supported pair/view/profile tried."""

    if selected_method not in {"bh", "by"}:
        raise ValueError(f"unknown pair FDR method: {selected_method}")
    for control in ("domain_regroup", "temporal_regroup"):
        eligible = []
        for index, row in enumerate(pair_summaries):
            for profile_key in SPECTRAL_PROFILE_KEYS:
                summary = row[profile_key][control]
                summary["fdr_q_value"] = None
                summary["detected_fdr"] = False
                summary["fdr_bh_q_value"] = None
                summary["detected_fdr_bh"] = False
                summary["fdr_by_q_value"] = None
                summary["detected_fdr_by"] = False
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
        harmonic = float(sum(1.0 / rank for rank in range(1, total + 1)))
        for (index, profile_key), bh_q_value in adjusted.items():
            summary = pair_summaries[index][profile_key][control]
            z_effect = summary.get("z_effect")

            def detected(q_value: float) -> bool:
                return bool(
                    q_value <= 0.05
                    and float(summary["observed_minus_null"]) > 0.0
                    and z_effect is not None
                    and float(z_effect) >= 1.5
                )

            by_q_value = float(min(1.0, bh_q_value * harmonic))
            summary["fdr_bh_q_value"] = bh_q_value
            summary["detected_fdr_bh"] = detected(bh_q_value)
            summary["fdr_by_q_value"] = by_q_value
            summary["detected_fdr_by"] = detected(by_q_value)
            selected_q = bh_q_value if selected_method == "bh" else by_q_value
            summary["fdr_q_value"] = selected_q
            summary["detected_fdr"] = bool(
                selected_q <= 0.05
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


def _apply_identity_persistence_fdr(
    summaries: Sequence[dict[str, Any]],
    *,
    selected_method: str,
) -> None:
    """Correct aggregate identity-persistence hypotheses jointly."""

    if selected_method not in {"bh", "by"}:
        raise ValueError(f"unknown identity FDR method: {selected_method}")
    eligible = [
        (float(row["empirical_p_ge_observed"]), index)
        for index, row in enumerate(summaries)
        if row.get("empirical_p_ge_observed") is not None
    ]
    eligible.sort(key=lambda item: item[0])
    adjusted: dict[int, float] = {}
    running = 1.0
    total = len(eligible)
    for rank in range(total, 0, -1):
        p_value, index = eligible[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[index] = float(min(1.0, running))
    harmonic = float(sum(1.0 / rank for rank in range(1, total + 1)))
    for index, row in enumerate(summaries):
        row["fdr_bh_q_value"] = None
        row["detected_fdr_bh"] = False
        row["fdr_by_q_value"] = None
        row["detected_fdr_by"] = False
        row["fdr_q_value"] = None
        row["detected_fdr"] = False
        if index not in adjusted:
            continue
        bh_q_value = adjusted[index]
        by_q_value = float(min(1.0, bh_q_value * harmonic))
        z_effect = row.get("z_effect")

        def detected(q_value: float) -> bool:
            return bool(
                q_value <= 0.05
                and float(row["observed_minus_null"]) > 0.0
                and z_effect is not None
                and float(z_effect) >= 1.5
            )

        row["fdr_bh_q_value"] = bh_q_value
        row["detected_fdr_bh"] = detected(bh_q_value)
        row["fdr_by_q_value"] = by_q_value
        row["detected_fdr_by"] = detected(by_q_value)
        selected_q = bh_q_value if selected_method == "bh" else by_q_value
        row["fdr_q_value"] = selected_q
        row["detected_fdr"] = detected(selected_q)


def _independent_time_cluster_count(
    starts: Sequence[str],
    *,
    separation_seconds: float,
) -> int:
    """Count timestamp clusters separated by at least the declared interval."""

    timestamps = sorted(datetime.fromisoformat(str(start)).timestamp() for start in starts)
    clusters: list[list[float]] = []
    for timestamp in timestamps:
        if (
            not clusters
            or timestamp - clusters[-1][-1] >= separation_seconds
        ):
            clusters.append([timestamp])
        else:
            clusters[-1].append(timestamp)
    return len(clusters)


def _identity_persistence_summary(
    pair_occurrences: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    profile_key: str,
    layer_name: str,
    scope: str,
    cutoff_timestamp: float | None,
    null_repeats: int,
    seed: int,
    replication_separation_seconds: float = 0.0,
    minimum_independent_time_clusters: int = 1,
) -> dict[str, Any]:
    """Test whether pair identities retain stable local rank across contexts."""

    contexts: dict[
        tuple[str, int, int, str],
        list[tuple[tuple[str, str], float, str]],
    ] = defaultdict(list)
    for pair, occurrences in pair_occurrences.items():
        for occurrence in occurrences:
            timestamp = datetime.fromisoformat(str(occurrence["start"])).timestamp()
            if scope == "early" and cutoff_timestamp is not None and timestamp > cutoff_timestamp:
                continue
            if scope == "late" and cutoff_timestamp is not None and timestamp <= cutoff_timestamp:
                continue
            context = (
                str(occurrence["segment"]),
                int(occurrence["group_index"]),
                int(occurrence["window_index"]),
                str(occurrence["corpus_stratum"]),
            )
            contexts[context].append(
                (pair, float(occurrence[profile_key]), str(occurrence["start"]))
            )
    contexts = {
        context: values
        for context, values in contexts.items()
        if len(values) >= 2
    }
    pair_counts = Counter(
        pair
        for values in contexts.values()
        for pair, _, _ in values
    )
    pair_time_clusters = {
        pair: _independent_time_cluster_count(
            [
                start
                for values in contexts.values()
                for candidate_pair, _, start in values
                if candidate_pair == pair
            ],
            separation_seconds=replication_separation_seconds,
        )
        for pair in pair_counts
    }
    eligible_pairs = sorted(
        pair
        for pair, count in pair_counts.items()
        if count >= 2
        and pair_time_clusters[pair] >= minimum_independent_time_clusters
    )
    pair_index = {pair: index for index, pair in enumerate(eligible_pairs)}
    normalized_contexts = []
    source_contexts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for values in contexts.values():
        scores = np.asarray([score for _, score, _ in values], dtype=np.float64)
        ranks = rankdata(scores, method="average")
        scale = (len(values) - 1) / 2.0
        normalized = (ranks - (len(values) + 1) / 2.0) / scale
        identities = [pair_index.get(pair) for pair, _, _ in values]
        normalized_contexts.append((identities, normalized))
        for pair, _, start in values:
            if pair in pair_index:
                source_contexts[pair].append(start)

    counts = np.zeros(len(eligible_pairs), dtype=np.float64)
    observed_sums = np.zeros(len(eligible_pairs), dtype=np.float64)
    for identities, normalized in normalized_contexts:
        for identity, value in zip(identities, normalized):
            if identity is None:
                continue
            counts[identity] += 1.0
            observed_sums[identity] += float(value)

    def statistic(sums: np.ndarray) -> float:
        if not len(sums) or not np.any(counts):
            return 0.0
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return float(np.sum(counts * means**2) / np.sum(counts))

    observed = statistic(observed_sums)
    null_values = []
    if len(eligible_pairs) >= 2 and normalized_contexts:
        rng = _rng_for(
            seed,
            layer_name,
            profile_key,
            scope,
            minimum_independent_time_clusters,
            "pair_identity_persistence",
        )
        for _ in range(null_repeats):
            sums = np.zeros(len(eligible_pairs), dtype=np.float64)
            for identities, normalized in normalized_contexts:
                assigned = normalized[rng.permutation(len(normalized))]
                for identity, value in zip(identities, assigned):
                    if identity is not None:
                        sums[identity] += float(value)
            null_values.append(statistic(sums))
    summary = _null_summary(observed, null_values)
    summary.update(
        {
            "layer": layer_name,
            "profile": profile_key,
            "scope": scope,
            "cutoff_timestamp": cutoff_timestamp,
            "contexts": len(normalized_contexts),
            "eligible_pair_identities": len(eligible_pairs),
            "pair_occurrences": int(np.sum(counts)),
            "replication_separation_seconds": replication_separation_seconds,
            "minimum_independent_time_clusters": minimum_independent_time_clusters,
            "statistic": (
                "occurrence-weighted between-pair variance of mean within-context "
                "spectral-alignment rank"
            ),
            "null_generation": (
                "permutes complete local score-rank sets among pair identities "
                "within each exact segment/group/window/acquisition-stratum context"
            ),
            "top_pair_effects": sorted(
                [
                    {
                        "entities": list(pair),
                        "occurrences": int(counts[index]),
                        "independent_time_clusters": pair_time_clusters[pair],
                        "mean_local_rank": float(
                            observed_sums[index] / counts[index]
                        ),
                        "source_starts": sorted(set(source_contexts[pair])),
                    }
                    for pair, index in pair_index.items()
                ],
                key=lambda row: abs(float(row["mean_local_rank"])),
                reverse=True,
            )[:25],
        }
    )
    return summary


def _dyadic_residual_persistence_summary(
    pair_occurrences: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    profile_key: str,
    layer_name: str,
    scope: str,
    cutoff_timestamp: float | None,
    null_repeats: int,
    seed: int,
    entity_effect_scope: str = "global",
    replication_separation_seconds: float = 0.0,
    minimum_independent_time_clusters: int = 1,
) -> dict[str, Any]:
    """Test persistent dyadic effects beyond an additive entity-effect model."""

    if entity_effect_scope not in {"global", "segment", "exact_context"}:
        raise ValueError(f"unknown entity effect scope: {entity_effect_scope}")

    candidate_contexts: dict[
        tuple[str, int, int, str],
        list[tuple[tuple[str, str], float, str]],
    ] = defaultdict(list)
    for pair, occurrences in pair_occurrences.items():
        for occurrence in occurrences:
            timestamp = datetime.fromisoformat(str(occurrence["start"])).timestamp()
            if scope == "early" and cutoff_timestamp is not None and timestamp > cutoff_timestamp:
                continue
            if scope == "late" and cutoff_timestamp is not None and timestamp <= cutoff_timestamp:
                continue
            context = (
                str(occurrence["segment"]),
                int(occurrence["group_index"]),
                int(occurrence["window_index"]),
                str(occurrence["corpus_stratum"]),
            )
            candidate_contexts[context].append(
                (pair, float(occurrence[profile_key]), str(occurrence["start"]))
            )
    candidate_contexts = {
        context: values
        for context, values in candidate_contexts.items()
        if len(values) >= 2
        and float(np.std([score for _, score, _ in values])) > 1e-12
    }
    context_pair_count_distribution = Counter(
        len(values) for values in candidate_contexts.values()
    )
    locally_identifiable_contexts = {}
    for context, values in candidate_contexts.items():
        local_entities = sorted({entity for pair, _, _ in values for entity in pair})
        local_entity_index = {
            entity: index for index, entity in enumerate(local_entities)
        }
        local_design = np.ones(
            (len(values), len(local_entities) + 1),
            dtype=np.float64,
        )
        for row, (pair, _, _) in enumerate(values):
            local_design[row, 1:] = 0.0
            local_design[row, 1 + local_entity_index[pair[0]]] = 1.0
            local_design[row, 1 + local_entity_index[pair[1]]] = 1.0
        local_rank = int(np.linalg.matrix_rank(local_design))
        if len(values) > local_rank:
            locally_identifiable_contexts[context] = values
    contexts = (
        locally_identifiable_contexts
        if entity_effect_scope == "exact_context"
        else candidate_contexts
    )
    pair_counts = Counter(
        pair
        for values in contexts.values()
        for pair, _, _ in values
    )
    pair_time_clusters = {
        pair: _independent_time_cluster_count(
            [
                start
                for values in contexts.values()
                for candidate_pair, _, start in values
                if candidate_pair == pair
            ],
            separation_seconds=replication_separation_seconds,
        )
        for pair in pair_counts
    }
    eligible_pairs = sorted(
        pair
        for pair, count in pair_counts.items()
        if count >= 2
        and pair_time_clusters[pair] >= minimum_independent_time_clusters
    )
    pair_index = {pair: index for index, pair in enumerate(eligible_pairs)}

    observations = []
    context_observation_indices: dict[
        tuple[str, int, int, str], list[int]
    ] = defaultdict(list)
    for context, values in sorted(contexts.items()):
        scores = np.asarray([score for _, score, _ in values], dtype=np.float64)
        normalized = (scores - np.mean(scores)) / np.std(scores)
        for pair, value in zip((pair for pair, _, _ in values), normalized):
            observation_index = len(observations)
            observations.append((context, pair, float(value)))
            context_observation_indices[context].append(observation_index)

    context_index = {context: index for index, context in enumerate(sorted(contexts))}

    def entity_effect_key(
        context: tuple[str, int, int, str],
        entity: str,
    ) -> tuple[str, str]:
        if entity_effect_scope == "global":
            return ("global", entity)
        if entity_effect_scope == "segment":
            return (context[0], entity)
        return (repr(context), entity)

    entity_effect_keys = sorted(
        {
            entity_effect_key(context, entity)
            for context, pair, _ in observations
            for entity in pair
        }
    )
    entity_effect_index = {
        key: index for index, key in enumerate(entity_effect_keys)
    }
    n_observations = len(observations)
    n_columns = len(context_index) + len(entity_effect_index)
    design = np.zeros((n_observations, n_columns), dtype=np.float64)
    values = np.zeros(n_observations, dtype=np.float64)
    pair_assignment = np.full(n_observations, -1, dtype=np.int64)
    for index, (context, pair, value) in enumerate(observations):
        design[index, context_index[context]] = 1.0
        design[
            index,
            len(context_index) + entity_effect_index[entity_effect_key(context, pair[0])],
        ] = 1.0
        design[
            index,
            len(context_index) + entity_effect_index[entity_effect_key(context, pair[1])],
        ] = 1.0
        values[index] = value
        if pair in pair_index:
            pair_assignment[index] = pair_index[pair]

    pair_projection = np.zeros((len(eligible_pairs), n_observations), dtype=np.float64)
    counts = np.zeros(len(eligible_pairs), dtype=np.float64)
    for observation_index, identity in enumerate(pair_assignment):
        if identity < 0:
            continue
        pair_projection[identity, observation_index] = 1.0
        counts[identity] += 1.0

    def statistics(residuals: np.ndarray) -> np.ndarray:
        pair_sums = pair_projection @ residuals
        if residuals.ndim == 1:
            return np.asarray(
                [float(np.sum(pair_sums**2 / counts) / np.sum(counts))]
            )
        return np.sum(pair_sums**2 / counts[:, None], axis=0) / np.sum(counts)

    null_values: list[float] = []
    observed = 0.0
    reduced_residual: np.ndarray | None = None
    rank = int(np.linalg.matrix_rank(design)) if n_observations else 0
    if len(eligible_pairs) >= 2 and n_observations > rank:
        design_pinv = np.linalg.pinv(design)
        fitted = design @ (design_pinv @ values)
        reduced_residual = values - fitted
        observed = float(statistics(reduced_residual)[0])
        rng = _rng_for(
            seed,
            layer_name,
            profile_key,
            scope,
            entity_effect_scope,
            minimum_independent_time_clusters,
            "dyadic_residual_persistence",
        )
        batch_size = min(500, null_repeats)
        for batch_start in range(0, null_repeats, batch_size):
            repeats = min(batch_size, null_repeats - batch_start)
            permuted = np.empty((n_observations, repeats), dtype=np.float64)
            for indices in context_observation_indices.values():
                index_array = np.asarray(indices, dtype=np.int64)
                orders = np.argsort(
                    rng.random((len(index_array), repeats)),
                    axis=0,
                )
                permuted[index_array, :] = reduced_residual[index_array][orders]
            synthetic = fitted[:, None] + permuted
            synthetic_residual = synthetic - design @ (design_pinv @ synthetic)
            null_values.extend(
                float(value)
                for value in statistics(synthetic_residual)
            )
    summary = _null_summary(observed, null_values)
    pair_residual_effects = sorted(
        [
            {
                "entities": list(pair),
                "occurrences": int(counts[index]),
                "independent_time_clusters": pair_time_clusters[pair],
                "mean_residual": float(
                    (pair_projection @ reduced_residual)[index] / counts[index]
                )
                if reduced_residual is not None
                else None,
            }
            for pair, index in pair_index.items()
        ],
        key=lambda row: abs(float(row["mean_residual"] or 0.0)),
        reverse=True,
    )
    summary.update(
        {
            "layer": layer_name,
            "profile": profile_key,
            "scope": scope,
            "entity_effect_scope": entity_effect_scope,
            "cutoff_timestamp": cutoff_timestamp,
            "contexts": len(contexts),
            "candidate_contexts": len(candidate_contexts),
            "locally_identifiable_contexts": len(locally_identifiable_contexts),
            "context_pair_count_distribution": {
                str(size): int(count)
                for size, count in sorted(context_pair_count_distribution.items())
            },
            "distinct_entities": len({entity for pair in pair_counts for entity in pair}),
            "eligible_pair_identities": len(eligible_pairs),
            "replication_separation_seconds": replication_separation_seconds,
            "minimum_independent_time_clusters": minimum_independent_time_clusters,
            "observations": n_observations,
            "reduced_model_columns": n_columns,
            "reduced_model_rank": rank,
            "residual_degrees_of_freedom": n_observations - rank,
            "statistic": (
                "occurrence-weighted between-pair variance of mean residual "
                "alignment after the declared additive entity-effect model"
            ),
            "reduced_model": (
                "linearly standardizes complete pair scores within each exact "
                "context, fits exact-context intercepts, and fits additive "
                f"entity-incidence main effects at {entity_effect_scope} scope"
            ),
            "informative_context_gate": (
                "all nonconstant exact contexts are used for global and segment "
                "entity effects; exact-context entity effects require more pair "
                "observations than the local additive model rank"
            ),
            "null_generation": (
                "Freedman-Lane: permutes reduced-model residuals only within exact "
                "contexts, adds fitted context and entity effects, refits the "
                "declared reduced model, and recomputes dyadic persistence"
            ),
            "top_pair_residual_effects": pair_residual_effects[:25],
        }
    )
    return summary


def _mean_group_alignment(
    records: Sequence[Mapping[str, Any]],
    *,
    profile_key: str,
    within_stratum_only: bool = False,
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
            if (
                within_stratum_only
                and left.get("corpus_stratum") != right.get("corpus_stratum")
            ):
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
    entity_strata: Mapping[str, str] | None = None,
    within_stratum_pairs_only: bool = False,
    pair_domain_control: str = "global_pair_bootstrap",
    require_complete_matched_occurrences: bool = False,
    pair_fdr_method: str = "bh",
    compute_identity_persistence: bool = False,
    dyadic_persistence_hypotheses: Sequence[tuple[str, str]] = (),
    dyadic_entity_effect_scopes: Sequence[str] = ("global",),
    persistence_minimum_time_clusters: Sequence[int] = (1,),
    progress: bool = False,
) -> dict[str, Any]:
    """Test spectral hypotheses while preserving the obvious domain spectrum."""

    if pair_domain_control not in {
        "global_pair_bootstrap",
        "matched_occurrence_regroup",
    }:
        raise ValueError(f"unknown pair domain control: {pair_domain_control}")
    if require_complete_matched_occurrences and (
        pair_domain_control != "matched_occurrence_regroup"
    ):
        raise ValueError(
            "complete matched-occurrence coverage requires matched occurrence control"
        )

    def report(message: str) -> None:
        if progress:
            print(f"cdip_relationship corpus={message}", file=sys.stderr, flush=True)

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        for layer in window["layers"]:
            for profile in layer.get("spectral_profiles", []):
                entity = str(profile["entity"])
                by_layer[str(layer["layer"])].append(
                    {
                        "segment": str(window["segment"]),
                        "group_index": int(window["group_index"]),
                        "window_index": int(window["window_index"]),
                        "start": str(window["source"]["start"]),
                        "entity": entity,
                        "corpus_stratum": (
                            str(entity_strata.get(entity, "unassigned"))
                            if entity_strata is not None
                            else "all"
                        ),
                        "power": np.asarray(profile["power"], dtype=np.float64),
                    }
                )

    layer_summaries = []
    temporal_pairs = []
    pair_summaries = []
    identity_persistence = []
    dyadic_residual_persistence = []
    dyadic_hypotheses = {
        (str(layer), str(profile))
        for layer, profile in dyadic_persistence_hypotheses
    }
    dyadic_effect_scopes = tuple(str(scope) for scope in dyadic_entity_effect_scopes)
    persistence_time_clusters = tuple(
        int(count) for count in persistence_minimum_time_clusters
    )
    if not persistence_time_clusters or any(
        count < 1 for count in persistence_time_clusters
    ):
        raise ValueError("persistence minimum time clusters must be positive")
    invalid_effect_scopes = set(dyadic_effect_scopes) - {
        "global",
        "segment",
        "exact_context",
    }
    if invalid_effect_scopes:
        raise ValueError(
            f"unknown dyadic entity effect scopes: {sorted(invalid_effect_scopes)}"
        )
    for layer_name, records in sorted(by_layer.items()):
        if not records:
            continue
        report(f"layer={layer_name} phase=envelope_profiles records={len(records)}")
        _add_envelope_profiles(
            records,
            stratum_key="corpus_stratum" if entity_strata is not None else None,
        )
        domain_observed = {
            profile_key: _mean_group_alignment(
                records,
                profile_key=profile_key,
                within_stratum_only=within_stratum_pairs_only,
            )
            for profile_key in SPECTRAL_PROFILE_KEYS
        }
        strata: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            strata[
                (
                    row["segment"],
                    row["window_index"],
                    (
                        row["corpus_stratum"]
                        if within_stratum_pairs_only
                        else "all"
                    ),
                )
            ].append(row)
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
                        within_stratum_only=within_stratum_pairs_only,
                    )
                )
        report(f"layer={layer_name} phase=layer_domain_regroup complete")

        temporal_groups: dict[tuple[str, int], dict[str, dict[int, dict[str, Any]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        entity_stratum = {
            str(row["entity"]): str(row["corpus_stratum"])
            for row in records
        }

        def pair_allowed(left_entity: str, right_entity: str) -> bool:
            return bool(
                not within_stratum_pairs_only
                or entity_stratum[left_entity] == entity_stratum[right_entity]
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
                    if not pair_allowed(left_entity, right_entity):
                        continue
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
                if not pair_allowed(left_entity, right_entity):
                    continue
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
                    "corpus_stratum": entity_stratum[left_entity],
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
                if not pair_allowed(str(left["entity"]), str(right["entity"])):
                    continue
                occurrence = {
                    "segment": segment,
                    "group_index": group_index,
                    "window_index": window_index,
                    "start": left["start"],
                    "corpus_stratum": left["corpus_stratum"],
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
                (
                    pair,
                    str(occurrence["corpus_stratum"]),
                    float(occurrence[profile_key]),
                )
                for pair, occurrences in pair_occurrences.items()
                for occurrence in occurrences
            ]
            for profile_key in SPECTRAL_PROFILE_KEYS
        }
        matched_occurrence_scores: dict[
            str,
            dict[tuple[str, int, int, str], list[tuple[tuple[str, str], float]]],
        ] = {
            profile_key: defaultdict(list)
            for profile_key in SPECTRAL_PROFILE_KEYS
        }
        for candidate_pair, occurrences in pair_occurrences.items():
            for occurrence in occurrences:
                context = (
                    str(occurrence["segment"]),
                    int(occurrence["group_index"]),
                    int(occurrence["window_index"]),
                    str(occurrence["corpus_stratum"]),
                )
                for profile_key in SPECTRAL_PROFILE_KEYS:
                    matched_occurrence_scores[profile_key][context].append(
                        (candidate_pair, float(occurrence[profile_key]))
                    )
        if compute_identity_persistence or any(
            layer_name == candidate_layer
            for candidate_layer, _ in dyadic_hypotheses
        ):
            timestamps = [
                datetime.fromisoformat(str(occurrence["start"])).timestamp()
                for occurrences in pair_occurrences.values()
                for occurrence in occurrences
            ]
            cutoff_timestamp = float(np.median(timestamps)) if timestamps else None
            for scope in ("full", "early", "late"):
                for profile_key in SPECTRAL_PROFILE_KEYS:
                    for minimum_time_clusters in persistence_time_clusters:
                        if compute_identity_persistence:
                            identity_persistence.append(
                                _identity_persistence_summary(
                                    pair_occurrences,
                                    profile_key=profile_key,
                                    layer_name=layer_name,
                                    scope=scope,
                                    cutoff_timestamp=cutoff_timestamp,
                                    null_repeats=null_repeats,
                                    seed=seed,
                                    replication_separation_seconds=(
                                        replication_separation_seconds
                                    ),
                                    minimum_independent_time_clusters=(
                                        minimum_time_clusters
                                    ),
                                )
                            )
                        if (layer_name, profile_key) not in dyadic_hypotheses:
                            continue
                        for entity_effect_scope in dyadic_effect_scopes:
                            dyadic_residual_persistence.append(
                                _dyadic_residual_persistence_summary(
                                    pair_occurrences,
                                    profile_key=profile_key,
                                    layer_name=layer_name,
                                    scope=scope,
                                    cutoff_timestamp=cutoff_timestamp,
                                    null_repeats=null_repeats,
                                    seed=seed,
                                    entity_effect_scope=entity_effect_scope,
                                    replication_separation_seconds=(
                                        replication_separation_seconds
                                    ),
                                    minimum_independent_time_clusters=(
                                        minimum_time_clusters
                                    ),
                                )
                            )
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
                "corpus_stratum": str(occurrences[0]["corpus_stratum"]),
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
                all_occurrence_observed = float(
                    np.mean([row[profile_key] for row in occurrences])
                )
                domain_rng_parts = [
                    layer_name,
                    pair[0],
                    pair[1],
                    profile_key,
                    "pair_domain_regroup",
                ]
                if pair_domain_control != "global_pair_bootstrap":
                    domain_rng_parts.append(pair_domain_control)
                domain_rng = _rng_for(seed, *domain_rng_parts)
                domain_observed_value = all_occurrence_observed
                matched_count = len(occurrences)
                matched_coverage = 1.0
                complete_matched_coverage = True
                candidate_pool_sizes: list[int] = []
                domain_null_exact = False
                domain_permutation_space = 0
                if pair_domain_control == "global_pair_bootstrap":
                    other_scores = [
                        score
                        for candidate_pair, candidate_stratum, score in all_pair_scores[
                            profile_key
                        ]
                        if candidate_pair != pair
                        and (
                            not within_stratum_pairs_only
                            or candidate_stratum == pair_summary["corpus_stratum"]
                        )
                    ]
                    pair_domain_null = (
                        np.mean(
                            domain_rng.choice(
                                np.asarray(other_scores, dtype=np.float64),
                                size=(null_repeats, len(occurrences)),
                                replace=True,
                            ),
                            axis=1,
                        ).tolist()
                        if other_scores
                        else []
                    )
                    candidate_pool_sizes = [len(other_scores)] * len(occurrences)
                else:
                    alternative_scores = []
                    matched_observed = []
                    for occurrence in occurrences:
                        context = (
                            str(occurrence["segment"]),
                            int(occurrence["group_index"]),
                            int(occurrence["window_index"]),
                            str(occurrence["corpus_stratum"]),
                        )
                        candidates = [
                            score
                            for candidate_pair, score in matched_occurrence_scores[
                                profile_key
                            ][context]
                        ]
                        candidate_pool_sizes.append(len(candidates))
                        if len(candidates) >= 2:
                            alternative_scores.append(
                                np.asarray(candidates, dtype=np.float64)
                            )
                            matched_observed.append(float(occurrence[profile_key]))
                    matched_count = len(matched_observed)
                    matched_coverage = matched_count / len(occurrences)
                    complete_matched_coverage = matched_count == len(occurrences)
                    eligible = bool(
                        matched_count >= 2
                        and (
                            complete_matched_coverage
                            or not require_complete_matched_occurrences
                        )
                    )
                    if eligible:
                        domain_observed_value = float(np.mean(matched_observed))
                        domain_permutation_space = int(
                            np.prod(
                                [len(values) for values in alternative_scores],
                                dtype=object,
                            )
                        )
                        if domain_permutation_space <= null_repeats:
                            pair_domain_null = [
                                float(sum(values) / matched_count)
                                for values in product(*alternative_scores)
                            ]
                            domain_null_exact = True
                        else:
                            totals = np.zeros(null_repeats, dtype=np.float64)
                            for candidates in alternative_scores:
                                totals += candidates[
                                    domain_rng.integers(
                                        len(candidates),
                                        size=null_repeats,
                                    )
                                ]
                            pair_domain_null = (totals / matched_count).tolist()
                    else:
                        pair_domain_null = []
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
                domain_summary = _null_summary(
                    domain_observed_value,
                    pair_domain_null,
                    exact=domain_null_exact,
                )
                domain_summary.update(
                    {
                        "null_generation": (
                            pair_domain_control
                            if pair_domain_control == "global_pair_bootstrap"
                            else "matched_occurrence_exact_permutation_space"
                            if domain_null_exact
                            else "matched_occurrence_sampled_permutation_space"
                            if pair_domain_null
                            else "matched_occurrence_untestable"
                        ),
                        "all_occurrence_observed": all_occurrence_observed,
                        "matched_occurrences": matched_count,
                        "matched_occurrence_coverage": matched_coverage,
                        "complete_matched_occurrence_coverage": complete_matched_coverage,
                        "candidate_pool_sizes": candidate_pool_sizes,
                        "permutation_space_size": domain_permutation_space,
                        "require_complete_matched_occurrences": (
                            require_complete_matched_occurrences
                        ),
                    }
                )
                pair_summary[profile_key] = {
                    "domain_regroup": domain_summary,
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
                "corpus_strata": {
                    str(stratum): int(count)
                    for stratum, count in sorted(
                        Counter(
                            str(row["corpus_stratum"])
                            for row in records
                        ).items()
                    )
                },
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
    _apply_pair_fdr(pair_summaries, selected_method=pair_fdr_method)
    _apply_identity_persistence_fdr(
        identity_persistence,
        selected_method=pair_fdr_method,
    )
    _apply_identity_persistence_fdr(
        dyadic_residual_persistence,
        selected_method=pair_fdr_method,
    )
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
    identity_replication = []
    identity_by_hypothesis: dict[
        tuple[str, str, int], dict[str, dict[str, Any]]
    ] = (
        defaultdict(dict)
    )
    for row in identity_persistence:
        identity_by_hypothesis[
            (
                str(row["layer"]),
                str(row["profile"]),
                int(row["minimum_independent_time_clusters"]),
            )
        ][str(row["scope"])] = row
    for (
        layer_name,
        profile_key,
        minimum_time_clusters,
    ), scopes in sorted(identity_by_hypothesis.items()):
        full = scopes.get("full")
        early = scopes.get("early")
        late = scopes.get("late")
        identity_replication.append(
            {
                "layer": layer_name,
                "profile": profile_key,
                "minimum_independent_time_clusters": minimum_time_clusters,
                "full_detected_fdr": bool(full and full["detected_fdr"]),
                "early_detected_fdr": bool(early and early["detected_fdr"]),
                "late_detected_fdr": bool(late and late["detected_fdr"]),
                "replicated_across_chronological_halves": bool(
                    full
                    and early
                    and late
                    and full["detected_fdr"]
                    and early["detected_fdr"]
                    and late["detected_fdr"]
                ),
            }
        )
    dyadic_replication = []
    dyadic_by_hypothesis: dict[
        tuple[str, str, str, int], dict[str, dict[str, Any]]
    ] = (
        defaultdict(dict)
    )
    for row in dyadic_residual_persistence:
        dyadic_by_hypothesis[
            (
                str(row["layer"]),
                str(row["profile"]),
                str(row["entity_effect_scope"]),
                int(row["minimum_independent_time_clusters"]),
            )
        ][str(row["scope"])] = row
    for (
        layer_name,
        profile_key,
        entity_effect_scope,
        minimum_time_clusters,
    ), scopes in sorted(dyadic_by_hypothesis.items()):
        full = scopes.get("full")
        early = scopes.get("early")
        late = scopes.get("late")
        dyadic_replication.append(
            {
                "layer": layer_name,
                "profile": profile_key,
                "entity_effect_scope": entity_effect_scope,
                "minimum_independent_time_clusters": minimum_time_clusters,
                "full_detected_fdr": bool(full and full["detected_fdr"]),
                "early_detected_fdr": bool(early and early["detected_fdr"]),
                "late_detected_fdr": bool(late and late["detected_fdr"]),
                "replicated_across_chronological_halves": bool(
                    full
                    and early
                    and late
                    and full["detected_fdr"]
                    and early["detected_fdr"]
                    and late["detected_fdr"]
                ),
            }
        )
    return {
        "method": {
            "domain_regroup": (
                "preserves every spectral profile and view stratum while "
                "randomizing candidate group membership"
            ),
            "envelope_normalization_strata": (
                "known entity strata"
                if entity_strata is not None
                else "complete corpus"
            ),
            "within_stratum_pairs_only": within_stratum_pairs_only,
            "pair_domain_control": pair_domain_control,
            "require_complete_matched_occurrences": (
                require_complete_matched_occurrences
            ),
            "matched_occurrence_regroup": (
                "for each candidate pair occurrence, samples only alternative "
                "pairs from the same segment, group, window index, and known "
                "entity stratum; candidates without complete matched-occurrence "
                "coverage are untestable when the complete-coverage gate is enabled"
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
                "Benjamini-Hochberg and Benjamini-Yekutieli FDR corrections are "
                "computed jointly across every supported pair, waveform view, and "
                "tried attenuation profile, separately for the domain-regroup and "
                "temporal-regroup controls"
            ),
            "selected_pair_fdr_method": pair_fdr_method,
            "identity_persistence": (
                "tests whether pair identities retain consistently high or low "
                "within-context alignment rank; nulls permute the complete local "
                "rank set only within exact context and acquisition stratum"
            ),
            "identity_persistence_scopes": ["full", "early", "late"],
            "persistence_minimum_independent_time_clusters": list(
                persistence_time_clusters
            ),
            "persistence_replication_separation_seconds": (
                replication_separation_seconds
            ),
            "dyadic_residual_persistence": (
                "confirmatory nested control on explicitly selected hypotheses; "
                "removes independent additive entity effects within every exact "
                "context before testing persistent pair identity"
            ),
            "dyadic_persistence_hypotheses": [
                {"layer": layer, "profile": profile}
                for layer, profile in sorted(dyadic_hypotheses)
            ],
            "dyadic_entity_effect_scopes": list(dyadic_effect_scopes),
        },
        "layer_summary": layer_summaries,
        "pair_summary": pair_summaries,
        "top_temporal_pairs": temporal_pairs[:50],
        "identity_persistence": identity_persistence,
        "identity_persistence_replication": identity_replication,
        "dyadic_residual_persistence": dyadic_residual_persistence,
        "dyadic_residual_persistence_replication": dyadic_replication,
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
