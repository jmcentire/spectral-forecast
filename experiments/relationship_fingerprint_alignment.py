"""Test cross-run alignment of frozen relationship-change fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def edge_matrix(values: Sequence[float], entity_count: int) -> np.ndarray:
    expected = entity_count * (entity_count - 1) // 2
    if len(values) != expected:
        raise ValueError("edge vector length does not match entity count")
    matrix = np.zeros((entity_count, entity_count), dtype=np.float64)
    upper = np.triu_indices(entity_count, k=1)
    matrix[upper] = np.asarray(values, dtype=np.float64)
    return matrix + matrix.T


def cosine(left: np.ndarray, right: np.ndarray, *, absolute: bool) -> float:
    upper = np.triu_indices(len(left), k=1)
    a = left[upper]
    b = right[upper]
    if absolute:
        a = np.abs(a)
        b = np.abs(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def apply_by_fdr(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        (float(row["empirical_p_ge_observed"]), index)
        for index, row in enumerate(rows)
    )
    total = len(ordered)
    harmonic = sum(1.0 / rank for rank in range(1, total + 1))
    running = 1.0
    adjusted: dict[int, float] = {}
    for rank in range(total, 0, -1):
        p_value, index = ordered[rank - 1]
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running * harmonic)
    for index, row in enumerate(rows):
        row["fdr_by_q_value"] = adjusted[index]
        row["detected_fdr_by"] = bool(adjusted[index] <= 0.05)


def run(
    development_path: Path,
    validation_path: Path,
    *,
    null_repeats: int,
    seed: int,
) -> dict[str, Any]:
    development = json.loads(development_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    development_rows = development["attributions"]
    validation_rows = validation["attributions"]
    entity_orders = {
        tuple(row["entities"]) for row in development_rows + validation_rows
    }
    if len(entity_orders) != 1:
        raise ValueError("all fingerprints must use the same canonical entity order")
    entities = entity_orders.pop()
    entity_count = len(entities)
    development_graphs = [
        edge_matrix(row["description"]["edge_delta_upper"], entity_count)
        for row in development_rows
    ]
    validation_graphs = [
        edge_matrix(row["description"]["edge_delta_upper"], entity_count)
        for row in validation_rows
    ]

    def statistic(graphs: Sequence[np.ndarray], *, absolute: bool) -> float:
        similarities = [
            cosine(left, right, absolute=absolute)
            for left in development_graphs
            for right in graphs
        ]
        return float(np.median(similarities))

    observed = {
        "signed": statistic(validation_graphs, absolute=False),
        "absolute": statistic(validation_graphs, absolute=True),
    }
    rng = np.random.default_rng(seed)
    nulls = {
        "signed": np.empty(null_repeats, dtype=np.float64),
        "absolute": np.empty(null_repeats, dtype=np.float64),
    }
    seen = set()
    repeat = 0
    while repeat < null_repeats:
        permutations = tuple(
            tuple(int(value) for value in rng.permutation(entity_count))
            for _ in validation_graphs
        )
        if permutations in seen:
            continue
        seen.add(permutations)
        permuted = [
            graph[np.ix_(order, order)]
            for graph, order in zip(validation_graphs, permutations, strict=True)
        ]
        nulls["signed"][repeat] = statistic(permuted, absolute=False)
        nulls["absolute"][repeat] = statistic(permuted, absolute=True)
        repeat += 1
    rows = []
    for metric in ("signed", "absolute"):
        null = nulls[metric]
        exceedances = int(np.sum(null >= observed[metric]))
        rows.append(
            {
                "metric": metric,
                "observed_median_cross_run_cosine": observed[metric],
                "null_mean": float(np.mean(null)),
                "null_std": float(np.std(null, ddof=1)),
                "null_exceedances": exceedances,
                "null_repeats": null_repeats,
                "unique_null_configurations": len(seen),
                "empirical_p_ge_observed": (exceedances + 1)
                / (null_repeats + 1),
            }
        )
    apply_by_fdr(rows)
    return {
        "method": {
            "statistic": "median all-pairs cross-run edge-fingerprint cosine",
            "null": (
                "independent canonical node-label permutation for each held-out "
                "fingerprint, shared across all of its development comparisons"
            ),
            "multiplicity": "Benjamini-Yekutieli across signed and absolute tests",
        },
        "development": str(development_path),
        "validation": str(validation_path),
        "entities": list(entities),
        "development_fingerprints": len(development_graphs),
        "validation_fingerprints": len(validation_graphs),
        "seed": seed,
        "evidence": rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--null-repeats", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(
        args.development,
        args.validation,
        null_repeats=args.null_repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in report["evidence"]:
        print(
            f"{row['metric']}: observed={row['observed_median_cross_run_cosine']:.4f} "
            f"p={row['empirical_p_ge_observed']:.4f} "
            f"q={row['fdr_by_q_value']:.4f} detected={row['detected_fdr_by']}"
        )


if __name__ == "__main__":
    main()
