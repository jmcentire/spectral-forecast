"""Post-hoc attribution audit for acquisition-stratified CDIP pair residuals.

The audit never participates in discovery. It asks whether confirmatory
independent-time pair residuals align with simple known station relationships:
geographic distance, depth difference, or mean depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata

from experiments.cdip_relationship_geography_audit import (
    _coordinate,
    _source_windows,
    _write_json,
    haversine_km,
)


DEFAULT_CONTROL_REPORT = Path(
    "experiments/results/"
    "2026-06-05-cdip-acquisition-stratified-residual-control-null4999.json"
)
CONFIRMATORY_PROFILES = (
    "envelope_attenuation_1.00",
    "signed_envelope_residual",
)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_rank = rankdata(np.asarray(left, dtype=np.float64), method="average")
    right_rank = rankdata(np.asarray(right, dtype=np.float64), method="average")
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator <= 1e-12:
        return None
    return float(np.dot(left_rank, right_rank) / denominator)


def _node_label_permutation_spearman(
    pairs: Sequence[tuple[str, str]],
    effects: Sequence[float],
    *,
    entity_values: Mapping[str, Any],
    relationship: Callable[[Any, Any], float],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Test pair-effect attribution while preserving the observed edge surface."""

    entities = sorted({entity for pair in pairs for entity in pair})
    if any(entity not in entity_values for entity in entities):
        return {
            "observed_spearman": None,
            "null_repeats": 0,
            "null_exceedances_two_sided": None,
            "empirical_p_two_sided": None,
        }
    observed_features = [
        relationship(entity_values[left], entity_values[right])
        for left, right in pairs
    ]
    observed = _spearman(effects, observed_features)
    if observed is None:
        return {
            "observed_spearman": None,
            "null_repeats": 0,
            "null_exceedances_two_sided": None,
            "empirical_p_two_sided": None,
        }
    rng = np.random.default_rng(seed)
    values = [entity_values[entity] for entity in entities]
    exceedances = 0
    unique_values = set()
    for _ in range(repeats):
        shuffled = rng.permutation(len(entities))
        assigned = {
            entity: values[int(shuffled[index])]
            for index, entity in enumerate(entities)
        }
        null_features = [
            relationship(assigned[left], assigned[right])
            for left, right in pairs
        ]
        null_value = _spearman(effects, null_features)
        if null_value is None:
            continue
        unique_values.add(round(null_value, 12))
        exceedances += int(abs(null_value) >= abs(observed) - 1e-12)
    return {
        "observed_spearman": observed,
        "null_repeats": repeats,
        "null_unique_values": len(unique_values),
        "null_exceedances_two_sided": exceedances,
        "empirical_p_two_sided": float((exceedances + 1) / (repeats + 1)),
    }


def _apply_by_fdr(rows: Sequence[dict[str, Any]]) -> None:
    eligible = sorted(
        (
            float(row["empirical_p_two_sided"]),
            index,
        )
        for index, row in enumerate(rows)
        if row.get("empirical_p_two_sided") is not None
    )
    total = len(eligible)
    harmonic = float(sum(1.0 / rank for rank in range(1, total + 1)))
    adjusted = {}
    running = 1.0
    for rank in range(total, 0, -1):
        p_value, index = eligible[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[index] = float(min(1.0, running * harmonic))
    for index, row in enumerate(rows):
        row["fdr_by_q_value"] = adjusted.get(index)
        row["detected_fdr_by"] = bool(
            index in adjusted and float(adjusted[index]) <= 0.05
        )


def run_attribution(
    report_path: Path,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paths = {}
    names = {}
    for window in _source_windows(report, report_path):
        for source in window["source"]["sources"]:
            entity = str(source["entity"])
            paths[entity] = Path(source["path"])
            names[entity] = str(source["platform_name"])
    coordinates = {entity: _coordinate(path) for entity, path in paths.items()}
    depths = {
        entity: float(metadata["water_depth"])
        for entity, metadata in report["station_metadata"].items()
        if np.isfinite(float(metadata["water_depth"]))
    }
    confirmatory = [
        row
        for row in report["spectral_specificity"]["dyadic_residual_persistence"]
        if row["layer"] == "raw"
        and row["profile"] in CONFIRMATORY_PROFILES
        and row["scope"] == "full"
        and row["entity_effect_scope"] == "segment"
        and row["minimum_independent_time_clusters"] == 2
    ]
    hypotheses = []
    audits = []
    for row_index, row in enumerate(confirmatory):
        effects = row.get(
            "pair_residual_effects",
            row.get("top_pair_residual_effects", []),
        )
        if len(effects) != int(row["eligible_pair_identities"]):
            raise ValueError(
                f"{row['profile']} report truncates confirmatory pair effects"
            )
        pairs = [tuple(str(entity) for entity in effect["entities"]) for effect in effects]
        signed_effects = [float(effect["mean_residual"]) for effect in effects]
        pair_rows = [
            {
                "entities": list(pair),
                "platform_names": [names.get(pair[0], pair[0]), names.get(pair[1], pair[1])],
                "mean_residual": signed_effects[index],
                "absolute_mean_residual": abs(signed_effects[index]),
                "distance_km": haversine_km(coordinates[pair[0]], coordinates[pair[1]]),
                "absolute_depth_difference": abs(depths[pair[0]] - depths[pair[1]]),
                "mean_depth": (depths[pair[0]] + depths[pair[1]]) / 2.0,
            }
            for index, pair in enumerate(pairs)
        ]
        feature_specs = (
            ("distance_km", coordinates, haversine_km),
            ("absolute_depth_difference", depths, lambda left, right: abs(left - right)),
            ("mean_depth", depths, lambda left, right: (left + right) / 2.0),
        )
        for effect_name, effect_values in (
            ("mean_residual", signed_effects),
            ("absolute_mean_residual", [abs(value) for value in signed_effects]),
        ):
            for feature_name, entity_values, relationship in feature_specs:
                hypothesis = {
                    "profile": str(row["profile"]),
                    "effect": effect_name,
                    "known_relationship": feature_name,
                    "pairs": len(pairs),
                    **_node_label_permutation_spearman(
                        pairs,
                        effect_values,
                        entity_values=entity_values,
                        relationship=relationship,
                        repeats=repeats,
                        seed=seed + row_index * 100 + len(hypotheses),
                    ),
                }
                hypotheses.append(hypothesis)
        audits.append(
            {
                "profile": str(row["profile"]),
                "entity_effect_scope": str(row["entity_effect_scope"]),
                "minimum_independent_time_clusters": int(
                    row["minimum_independent_time_clusters"]
                ),
                "pairs": pair_rows,
            }
        )
    _apply_by_fdr(hypotheses)
    return {
        "relationship_report": str(report_path),
        "method": {
            "label_use": "post-hoc attribution only",
            "selection_use": "none",
            "confirmatory_surface": (
                "full-corpus, segment-buoy-adjusted, at-least-two-independent-time-"
                "cluster residual pair effects in the two deepest raw profiles"
            ),
            "null": (
                "QAP-like node-label permutation keeps the discovered edge surface "
                "and pair residuals fixed while reassigning station geography/depth"
            ),
            "multiplicity": "BY FDR across all post-hoc attribution hypotheses",
        },
        "signature": {
            "permutation_repeats": repeats,
            "seed": seed,
        },
        "hypotheses": hypotheses,
        "audits": audits,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_CONTROL_REPORT)
    parser.add_argument("--permutation-repeats", type=int, default=19_999)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.permutation_repeats < 1:
        parser.error("--permutation-repeats must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_attribution(
        args.report,
        repeats=args.permutation_repeats,
        seed=args.seed,
    )
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
