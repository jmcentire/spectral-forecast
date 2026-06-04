"""Assess likely structural views before running organizational-network analysis.

The goal is to decide which projections are worth testing: temporal order,
partial order, graph topology, relation churn, group/multilayer structure, and
artifact-control orderings. The assessor is label-free for scoring except that
labels can indicate that a group/multilayer view exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.org_network_autotune import (
    DATASETS,
    build_network_series,
    read_edges,
    read_labels,
    read_simplicial_edges,
    relation_only_network,
    resolve_dataset_files,
)
from spectral_forecast.structure import StructureReadiness, structure_readiness


@dataclass(frozen=True)
class EdgeProfile:
    """Canonical edge/simplicial projection profile."""

    edge_count: int
    node_count: int
    unique_pair_count: int
    timestamp_count: int
    span: float
    timestamp_resolution: float
    simultaneity_fraction: float
    max_same_timestamp: int
    repeated_edge_fraction: float
    returning_pair_fraction: float
    graph_density: float
    degree_gini: float
    top_degree_fraction: float
    label_coverage: float
    group_count: int
    cross_group_fraction: float | None
    data_format: str
    directed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_count": self.edge_count,
            "node_count": self.node_count,
            "unique_pair_count": self.unique_pair_count,
            "timestamp_count": self.timestamp_count,
            "span": self.span,
            "timestamp_resolution": self.timestamp_resolution,
            "simultaneity_fraction": self.simultaneity_fraction,
            "max_same_timestamp": self.max_same_timestamp,
            "repeated_edge_fraction": self.repeated_edge_fraction,
            "returning_pair_fraction": self.returning_pair_fraction,
            "graph_density": self.graph_density,
            "degree_gini": self.degree_gini,
            "top_degree_fraction": self.top_degree_fraction,
            "label_coverage": self.label_coverage,
            "group_count": self.group_count,
            "cross_group_fraction": self.cross_group_fraction,
            "data_format": self.data_format,
            "directed": self.directed,
        }


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _gini(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    ordered = np.sort(x)
    n = len(ordered)
    return float((2.0 * np.sum((np.arange(n) + 1) * ordered) / (n * total)) - ((n + 1) / n))


def _pair(source: str, target: str, *, directed: bool) -> tuple[str, str]:
    return (source, target) if directed else tuple(sorted((source, target)))


def profile_edges(
    edges: Sequence[Any],
    labels: Mapping[str, str],
    *,
    directed: bool,
    data_format: str,
) -> EdgeProfile:
    """Profile canonical projected edges without choosing an observer."""

    if not edges:
        raise ValueError("no edges to profile")
    nodes: set[str] = set()
    timestamps: list[float] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    degree_counts: Counter[str] = Counter()
    cross_group_edges = 0
    labeled_edge_endpoints = 0

    for edge in edges:
        nodes.update((edge.source, edge.target))
        timestamps.append(float(edge.timestamp))
        pair_counts[_pair(edge.source, edge.target, directed=directed)] += 1
        degree_counts[edge.source] += 1
        degree_counts[edge.target] += 1
        source_labeled = edge.source in labels
        target_labeled = edge.target in labels
        labeled_edge_endpoints += int(source_labeled) + int(target_labeled)
        if source_labeled and target_labeled and labels[edge.source] != labels[edge.target]:
            cross_group_edges += 1

    timestamp_counts = Counter(timestamps)
    shared_timestamp_edges = sum(count for count in timestamp_counts.values() if count > 1)
    repeated_edges = sum(count - 1 for count in pair_counts.values() if count > 1)
    returning_pairs = sum(1 for count in pair_counts.values() if count > 1)
    if directed:
        possible_pairs = max(len(nodes) * (len(nodes) - 1), 1)
    else:
        possible_pairs = max(len(nodes) * (len(nodes) - 1) / 2, 1)
    degree_total = sum(degree_counts.values())
    top_degree_total = sum(count for _, count in degree_counts.most_common(min(10, len(degree_counts))))
    labeled_nodes = sum(1 for node in nodes if node in labels)
    span = max(timestamps) - min(timestamps)

    return EdgeProfile(
        edge_count=len(edges),
        node_count=len(nodes),
        unique_pair_count=len(pair_counts),
        timestamp_count=len(timestamp_counts),
        span=float(span),
        timestamp_resolution=len(timestamp_counts) / max(len(edges), 1),
        simultaneity_fraction=shared_timestamp_edges / max(len(edges), 1),
        max_same_timestamp=max(timestamp_counts.values()) if timestamp_counts else 0,
        repeated_edge_fraction=repeated_edges / max(len(edges), 1),
        returning_pair_fraction=returning_pairs / max(len(pair_counts), 1),
        graph_density=len(pair_counts) / possible_pairs,
        degree_gini=_gini(list(degree_counts.values())),
        top_degree_fraction=top_degree_total / max(degree_total, 1),
        label_coverage=labeled_nodes / max(len(nodes), 1),
        group_count=len(set(labels.values())) if labels else 0,
        cross_group_fraction=(
            cross_group_edges / max(len(edges), 1)
            if labels and labeled_edge_endpoints > 0
            else None
        ),
        data_format=data_format,
        directed=directed,
    )


def _readiness_dict(readiness: StructureReadiness | None) -> dict[str, object] | None:
    return readiness.to_dict() if readiness is not None else None


def _safe_structure_readiness(
    series: Mapping[str, np.ndarray],
    *,
    null_repeats: int,
    seed: int,
) -> StructureReadiness | None:
    try:
        return structure_readiness(series, null_repeats=null_repeats, seed=seed)
    except Exception:
        return None


def _readiness_score(readiness: StructureReadiness | None) -> float:
    return float(readiness.structure_score) if readiness is not None and readiness.ready else 0.0


def _temporal_z_score(readiness: StructureReadiness | None) -> float:
    if readiness is None or readiness.temporal_z_effect is None:
        return 0.0
    return _clip01(float(readiness.temporal_z_effect) / 5.0)


def _covariance_z_score(readiness: StructureReadiness | None) -> float:
    if readiness is None or readiness.covariance_z_effect is None:
        return 0.0
    return _clip01(float(readiness.covariance_z_effect) / 5.0)


def _candidate(
    name: str,
    score: float,
    reason: str,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    score = _clip01(score)
    if score >= 0.70:
        grade = "strong"
    elif score >= 0.45:
        grade = "promising"
    elif score >= 0.20:
        grade = "weak"
    else:
        grade = "poor"
    return {
        "name": name,
        "score": score,
        "grade": grade,
        "reason": reason,
        "evidence": dict(evidence),
    }


def grade_view_candidates(
    profile: EdgeProfile,
    *,
    full_readiness: StructureReadiness | None,
    relation_readiness: StructureReadiness | None,
) -> list[dict[str, object]]:
    """Rank structural views worth testing before expensive analysis."""

    sequence_score = (
        0.30 * _clip01(profile.timestamp_count / 64.0)
        + 0.25 * profile.timestamp_resolution
        + 0.25 * _temporal_z_score(full_readiness)
        + 0.20 * (1.0 - profile.simultaneity_fraction)
    )
    graph_score = (
        0.30 * profile.returning_pair_fraction
        + 0.25 * _clip01(profile.degree_gini)
        + 0.25 * _covariance_z_score(relation_readiness or full_readiness)
        + 0.20 * _clip01(profile.top_degree_fraction * 4.0)
    )
    relation_score = (
        0.40 * _readiness_score(relation_readiness)
        + 0.25 * profile.returning_pair_fraction
        + 0.20 * profile.repeated_edge_fraction
        + 0.15 * _temporal_z_score(relation_readiness)
    )
    partial_order_score = (
        0.35 * profile.simultaneity_fraction
        + 0.25 * _clip01(profile.max_same_timestamp / 10.0)
        + 0.25 * (1.0 if profile.data_format == "simplices" else 0.0)
        + 0.15 * profile.returning_pair_fraction
    )
    group_score = (
        0.35 * profile.label_coverage
        + 0.25 * _clip01(profile.group_count / 6.0)
        + 0.20 * _clip01((profile.cross_group_fraction or 0.0) * 4.0)
        + 0.20 * _readiness_score(relation_readiness or full_readiness)
    )

    candidates = [
        _candidate(
            "temporal_total_order",
            sequence_score,
            "Use timestamp order as a primary sequence only if it beats order-control nulls.",
            {
                "timestamp_count": profile.timestamp_count,
                "timestamp_resolution": profile.timestamp_resolution,
                "simultaneity_fraction": profile.simultaneity_fraction,
                "full_temporal_z": None
                if full_readiness is None
                else full_readiness.temporal_z_effect,
            },
        ),
        _candidate(
            "relationship_graph",
            graph_score,
            "Use graph adjacency and node/pair topology without forcing a total order.",
            {
                "returning_pair_fraction": profile.returning_pair_fraction,
                "degree_gini": profile.degree_gini,
                "top_degree_fraction": profile.top_degree_fraction,
                "relation_covariance_z": None
                if relation_readiness is None
                else relation_readiness.covariance_z_effect,
            },
        ),
        _candidate(
            "relation_churn",
            relation_score,
            "Use novelty, returning ties, lost ties, and neighbor turnover as the main surface.",
            {
                "relation_ready": None if relation_readiness is None else relation_readiness.ready,
                "relation_score": None if relation_readiness is None else relation_readiness.structure_score,
                "repeated_edge_fraction": profile.repeated_edge_fraction,
                "returning_pair_fraction": profile.returning_pair_fraction,
            },
        ),
        _candidate(
            "partial_order_or_hyperedge",
            partial_order_score,
            "Preserve simultaneity or higher-order events instead of serializing them.",
            {
                "data_format": profile.data_format,
                "simultaneity_fraction": profile.simultaneity_fraction,
                "max_same_timestamp": profile.max_same_timestamp,
            },
        ),
        _candidate(
            "group_multilayer",
            group_score,
            "Use group/department layers when labels are canonical and available at analysis time.",
            {
                "label_coverage": profile.label_coverage,
                "group_count": profile.group_count,
                "cross_group_fraction": profile.cross_group_fraction,
            },
        ),
    ]
    return sorted(candidates, key=lambda row: float(row["score"]), reverse=True)


def order_candidates(profile: EdgeProfile, view_candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """List useful and adversarial orderings to test."""

    scores = {str(row["name"]): float(row["score"]) for row in view_candidates}
    orders: list[dict[str, object]] = [
        {
            "name": "timestamp",
            "role": "candidate" if scores.get("temporal_total_order", 0.0) >= 0.45 else "control",
            "reason": "Chronological order is plausible only if time carries structure.",
        },
        {
            "name": "first_seen_pair",
            "role": "candidate" if scores.get("relation_churn", 0.0) >= 0.45 else "control",
            "reason": "Orders ties by relationship emergence; useful for churn surfaces.",
        },
        {
            "name": "degree_descending",
            "role": "candidate" if scores.get("relationship_graph", 0.0) >= 0.45 else "control",
            "reason": "Topology-aligned order; tests whether hubs organize the signal.",
        },
        {
            "name": "group_then_time",
            "role": "candidate" if scores.get("group_multilayer", 0.0) >= 0.45 else "unavailable",
            "reason": "Only valid when canonical group labels exist and are allowed.",
        },
        {
            "name": "stable_id_or_alphabetic",
            "role": "artifact_control",
            "reason": "Arbitrary stable order; signal here is suspect unless explained as a proxy.",
        },
        {
            "name": "hash_order",
            "role": "artifact_control",
            "reason": "Stable but semantically arbitrary order-control.",
        },
        {
            "name": "random_order",
            "role": "null_control",
            "reason": "Destroys order while preserving the event set.",
        },
        {
            "name": "silly_proxy_order",
            "role": "artifact_probe",
            "reason": "Sort by a nonsense/proxy key such as string length or digit frequency to expose order artifacts.",
        },
    ]
    if profile.simultaneity_fraction > 0.10 or profile.data_format == "simplices":
        orders.append(
            {
                "name": "timestamp_bucket_partial_order",
                "role": "candidate",
                "reason": "Preserve within-bucket simultaneity instead of imposing arbitrary tie order.",
            }
        )
    return orders


def _load_edges_and_labels(args: argparse.Namespace) -> tuple[list[Any], dict[str, str], dict[str, Any], Path, Path | None]:
    edge_path, label_path, dataset_metadata = resolve_dataset_files(args)
    data_format = str(dataset_metadata.get("data_format", args.data_format))
    if data_format == "edges":
        edges = read_edges(
            edge_path,
            source_col=args.source_col,
            target_col=args.target_col,
            time_col=args.time_col,
            max_edges=args.max_edges,
        )
    elif data_format == "simplices":
        edges = read_simplicial_edges(
            edge_path,
            prefix=args.simplex_prefix,
            max_edges=args.max_edges,
        )
    else:
        raise ValueError(f"unknown data format: {data_format}")
    return edges, read_labels(label_path), {**dataset_metadata, "data_format": data_format}, edge_path, label_path


def assess_org_network(args: argparse.Namespace) -> dict[str, object]:
    edges, labels, dataset_metadata, edge_path, label_path = _load_edges_and_labels(args)
    profile = profile_edges(
        edges,
        labels,
        directed=args.directed,
        data_format=str(dataset_metadata.get("data_format", args.data_format)),
    )
    full_surface = build_network_series(
        edges,
        labels,
        bin_seconds=args.bin_seconds,
        top_nodes=args.top_nodes,
        top_groups=args.top_groups,
        include_global=True,
        include_nodes=not args.no_node_series,
        include_groups=not args.no_group_series,
        include_relation=True,
        directed=args.directed,
        transform=args.transform,
        max_bins=args.max_bins,
    )
    relation_surface = relation_only_network(full_surface)
    full_ready = _safe_structure_readiness(
        full_surface.series,
        null_repeats=args.structure_null_repeats,
        seed=args.seed,
    )
    relation_ready = _safe_structure_readiness(
        relation_surface.series,
        null_repeats=args.structure_null_repeats,
        seed=args.seed + 1,
    )
    views = grade_view_candidates(
        profile,
        full_readiness=full_ready,
        relation_readiness=relation_ready,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "dataset": {
            **dataset_metadata,
            "edge_path": str(edge_path),
            "label_path": str(label_path) if label_path else None,
        },
        "profile": profile.to_dict(),
        "surfaces": {
            "full": {
                "metadata": full_surface.metadata,
                "structure_readiness": _readiness_dict(full_ready),
            },
            "relation_only": {
                "metadata": relation_surface.metadata,
                "structure_readiness": _readiness_dict(relation_ready),
            },
        },
        "view_candidates": views,
        "order_candidates": order_candidates(profile, views),
        "recommended_views": [row for row in views if row["grade"] in {"strong", "promising"}],
        "guardrails": [
            "Treat view/order as a hypothesis, not as ground truth.",
            "Run artifact-control orders before trusting any ordered signal.",
            "Map all surfaced evidence back to canonical event/entity IDs.",
            "Do not use labels or future-derived information in the feature path.",
            "Prediction features must be available with the same distribution at serve time.",
        ],
    }


def print_report(report: Mapping[str, object]) -> None:
    profile = dict(report["profile"])  # type: ignore[index]
    print("Organizational structure assessment")
    print("  dataset=%s" % dict(report["dataset"]).get("dataset", "custom"))  # type: ignore[arg-type]
    print(
        "  edges=%d nodes=%d pairs=%d timestamps=%d simultaneity=%.3f repeated=%.3f returning_pairs=%.3f"
        % (
            profile["edge_count"],
            profile["node_count"],
            profile["unique_pair_count"],
            profile["timestamp_count"],
            profile["simultaneity_fraction"],
            profile["repeated_edge_fraction"],
            profile["returning_pair_fraction"],
        )
    )
    print("  views")
    for row in report["view_candidates"]:  # type: ignore[index]
        item = dict(row)
        print("    %-24s grade=%-10s score=%.3f" % (item["name"], item["grade"], item["score"]))
    print("  orders")
    for row in report["order_candidates"]:  # type: ignore[index]
        item = dict(row)
        print("    %-28s role=%s" % (item["name"], item["role"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/org_networks"))
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--data-format", choices=["edges", "simplices"], default="edges")
    parser.add_argument("--simplex-prefix", default="email-Enron")
    parser.add_argument("--source-col", type=int, default=0)
    parser.add_argument("--target-col", type=int, default=1)
    parser.add_argument("--time-col", type=int, default=2)
    parser.add_argument("--directed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--max-bins", type=int, default=0)
    parser.add_argument("--bin-seconds", type=float, default=86400.0)
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--no-node-series", action="store_true")
    parser.add_argument("--no-group-series", action="store_true")
    parser.add_argument("--transform", choices=["none", "log", "robust", "log-robust"], default="log-robust")
    parser.add_argument("--structure-null-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = assess_org_network(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
