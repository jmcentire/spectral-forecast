"""Probe candidate structural views for organizational network datasets.

This sits between assessment and expensive autotune runs. The assessor says
which structures look plausible; this script turns those structures into cheap
view surfaces and checks whether candidate surfaces separate from artifact and
random controls under the existing structure-readiness diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.org_network_assess import (
    _load_edges_and_labels,
    assess_org_network,
)
from experiments.org_network_autotune import (
    DATASETS,
    NetworkSeries,
    TemporalEdge,
    build_network_series,
    relation_only_network,
)
from spectral_forecast.structure import StructureReadiness, structure_readiness


@dataclass(frozen=True)
class ViewSpec:
    """One candidate/control view to materialize and score."""

    name: str
    view: str
    role: str
    order: str
    real_time: bool
    feature_family: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "view": self.view,
            "role": self.role,
            "order": self.order,
            "real_time": self.real_time,
            "feature_family": self.feature_family,
            "reason": self.reason,
        }


def _pair(edge: TemporalEdge, *, directed: bool) -> tuple[str, str]:
    return (edge.source, edge.target) if directed else tuple(sorted((edge.source, edge.target)))


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def remap_edges_by_order(
    edges: Sequence[TemporalEdge],
    *,
    order: str,
    labels: Mapping[str, str],
    directed: bool,
    seed: int,
) -> list[TemporalEdge]:
    """Replace timestamps with event ranks after sorting by an enforced order."""

    indexed = list(enumerate(edges))
    if order == "random_order":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(indexed))
        ordered = [indexed[int(index)] for index in permutation]
    else:
        degrees: dict[str, int] = {}
        first_seen: dict[tuple[str, str], float] = {}
        for edge in edges:
            degrees[edge.source] = degrees.get(edge.source, 0) + 1
            degrees[edge.target] = degrees.get(edge.target, 0) + 1
            key = _pair(edge, directed=directed)
            first_seen[key] = min(first_seen.get(key, float("inf")), float(edge.timestamp))

        def sort_key(item: tuple[int, TemporalEdge]) -> tuple[object, ...]:
            index, edge = item
            pair = _pair(edge, directed=directed)
            pair_text = "|".join(pair)
            source_group = labels.get(edge.source, "unknown")
            target_group = labels.get(edge.target, "unknown")
            if order in {"timestamp", "timestamp_bucket_partial_order"}:
                return (float(edge.timestamp), index)
            if order == "first_seen_pair":
                return (first_seen[pair], float(edge.timestamp), pair_text, index)
            if order == "degree_descending":
                degree = degrees.get(edge.source, 0) + degrees.get(edge.target, 0)
                return (-degree, float(edge.timestamp), pair_text, index)
            if order == "group_then_time":
                group_pair = tuple(sorted((source_group, target_group)))
                return (group_pair, float(edge.timestamp), pair_text, index)
            if order == "stable_id_or_alphabetic":
                return (pair_text, float(edge.timestamp), index)
            if order == "hash_order":
                return (_stable_hash(pair_text), float(edge.timestamp), index)
            if order == "silly_proxy_order":
                digit_sevens = pair_text.count("7")
                return (len(pair_text), -digit_sevens, pair_text[::-1], float(edge.timestamp), index)
            raise ValueError(f"unknown enforced order: {order}")

        ordered = sorted(indexed, key=sort_key)

    return [
        TemporalEdge(source=edge.source, target=edge.target, timestamp=float(rank))
        for rank, (_, edge) in enumerate(ordered)
    ]


def _surface_for_spec(
    edges: Sequence[TemporalEdge],
    labels: Mapping[str, str],
    spec: ViewSpec,
    *,
    args: argparse.Namespace,
    directed: bool,
    seed_offset: int,
) -> NetworkSeries:
    ordered_edges = list(edges) if spec.real_time else remap_edges_by_order(
        edges,
        order=spec.order,
        labels=labels,
        directed=directed,
        seed=args.seed + seed_offset,
    )
    bin_seconds = args.bin_seconds if spec.real_time else float(args.event_bin_size)

    include_global = True
    include_nodes = spec.feature_family in {"full", "graph", "relation"}
    include_groups = bool(labels) and spec.feature_family in {"full", "group"}
    include_relation = spec.feature_family in {"full", "relation", "group"}
    surface = build_network_series(
        ordered_edges,
        labels,
        bin_seconds=bin_seconds,
        top_nodes=args.top_nodes,
        top_groups=args.top_groups,
        include_global=include_global,
        include_nodes=include_nodes,
        include_groups=include_groups,
        include_relation=include_relation,
        directed=directed,
        transform=args.transform,
        max_bins=args.max_bins,
    )
    if spec.feature_family == "relation":
        return relation_only_network(surface)
    return surface


def _score_surface(
    surface: NetworkSeries,
    *,
    args: argparse.Namespace,
    seed_offset: int,
) -> StructureReadiness:
    return structure_readiness(
        surface.series,
        null_repeats=args.structure_null_repeats,
        seed=args.seed + 10_000 + seed_offset,
        active_z_threshold=args.structure_active_z,
        min_series=args.structure_min_series,
        min_score=args.structure_min_score,
    )


def _positive_log_z(value: object) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or number <= 0.0:
        return 0.0
    return float(np.log1p(number))


def probe_evidence_score(readiness: StructureReadiness | Mapping[str, Any] | None) -> float:
    """Unsaturated score for comparing candidate surfaces to controls."""

    if readiness is None:
        return 0.0
    payload = readiness.to_dict() if isinstance(readiness, StructureReadiness) else dict(readiness)
    active_fraction = float(payload.get("active_fraction", 0.0))
    if active_fraction >= 0.95:
        saturation = 0.0
    elif active_fraction <= 0.75:
        saturation = 1.0
    else:
        saturation = 1.0 - ((active_fraction - 0.75) / 0.20)
    return float(
        saturation
        * (
            0.45 * _positive_log_z(payload.get("covariance_z_effect"))
            + 0.20 * _positive_log_z(payload.get("pattern_z_effect"))
            + 0.35 * _positive_log_z(payload.get("temporal_z_effect"))
        )
    )


def _candidate_view_names(assessment: Mapping[str, Any], *, include_weak: bool) -> set[str]:
    accepted = {"strong", "promising"} if not include_weak else {"strong", "promising", "weak"}
    names: set[str] = set()
    for row in assessment.get("view_candidates", []):
        item = dict(row)
        if item.get("grade") in accepted:
            names.add(str(item["name"]))
    return names


def build_view_specs(assessment: Mapping[str, Any], *, include_weak: bool) -> list[ViewSpec]:
    """Create candidate and control surfaces from assessor output."""

    views = _candidate_view_names(assessment, include_weak=include_weak)
    specs: list[ViewSpec] = []

    if "temporal_total_order" in views:
        specs.extend(
            [
                ViewSpec(
                    "temporal_real_time",
                    "temporal_total_order",
                    "candidate",
                    "timestamp",
                    True,
                    "full",
                    "Canonical timestamp-binned full surface.",
                ),
                ViewSpec(
                    "temporal_event_timestamp",
                    "temporal_total_order",
                    "candidate",
                    "timestamp",
                    False,
                    "full",
                    "Equal-event bins with chronological event order.",
                ),
                ViewSpec(
                    "temporal_event_random_control",
                    "temporal_total_order",
                    "null_control",
                    "random_order",
                    False,
                    "full",
                    "Equal-event bins with random event order.",
                ),
                ViewSpec(
                    "temporal_silly_order_probe",
                    "temporal_total_order",
                    "artifact_probe",
                    "silly_proxy_order",
                    False,
                    "full",
                    "Equal-event bins sorted by a nonsense/proxy key.",
                ),
            ]
        )

    if "relation_churn" in views:
        specs.extend(
            [
                ViewSpec(
                    "relation_first_seen_pair",
                    "relation_churn",
                    "candidate",
                    "first_seen_pair",
                    False,
                    "relation",
                    "Relationship emergence order with relation-change features.",
                ),
                ViewSpec(
                    "relation_timestamp",
                    "relation_churn",
                    "candidate",
                    "timestamp",
                    False,
                    "relation",
                    "Chronological equal-event relation-change surface.",
                ),
                ViewSpec(
                    "relation_random_control",
                    "relation_churn",
                    "null_control",
                    "random_order",
                    False,
                    "relation",
                    "Random event order relation-change control.",
                ),
                ViewSpec(
                    "relation_silly_order_probe",
                    "relation_churn",
                    "artifact_probe",
                    "silly_proxy_order",
                    False,
                    "relation",
                    "Nonsense/proxy order relation-change artifact probe.",
                ),
            ]
        )

    if "relationship_graph" in views:
        specs.extend(
            [
                ViewSpec(
                    "graph_degree_descending",
                    "relationship_graph",
                    "candidate",
                    "degree_descending",
                    False,
                    "graph",
                    "Topology-aligned event bins sorted by endpoint degree.",
                ),
                ViewSpec(
                    "graph_stable_id_control",
                    "relationship_graph",
                    "artifact_control",
                    "stable_id_or_alphabetic",
                    False,
                    "graph",
                    "Arbitrary stable ID graph-order control.",
                ),
                ViewSpec(
                    "graph_hash_control",
                    "relationship_graph",
                    "artifact_control",
                    "hash_order",
                    False,
                    "graph",
                    "Stable hash graph-order control.",
                ),
                ViewSpec(
                    "graph_random_control",
                    "relationship_graph",
                    "null_control",
                    "random_order",
                    False,
                    "graph",
                    "Random graph-order control.",
                ),
            ]
        )

    if "partial_order_or_hyperedge" in views:
        specs.extend(
            [
                ViewSpec(
                    "partial_real_time_bucket",
                    "partial_order_or_hyperedge",
                    "candidate",
                    "timestamp_bucket_partial_order",
                    True,
                    "full",
                    "Real timestamp buckets preserve simultaneity instead of tie-breaking.",
                ),
                ViewSpec(
                    "partial_random_event_control",
                    "partial_order_or_hyperedge",
                    "null_control",
                    "random_order",
                    False,
                    "full",
                    "Random event order control for simultaneity-heavy data.",
                ),
            ]
        )

    if "group_multilayer" in views:
        specs.extend(
            [
                ViewSpec(
                    "group_real_time",
                    "group_multilayer",
                    "candidate",
                    "timestamp",
                    True,
                    "group",
                    "Canonical group-layer surface in real time.",
                ),
                ViewSpec(
                    "group_then_time",
                    "group_multilayer",
                    "candidate",
                    "group_then_time",
                    False,
                    "group",
                    "Equal-event bins grouped by canonical label before time.",
                ),
                ViewSpec(
                    "group_random_control",
                    "group_multilayer",
                    "null_control",
                    "random_order",
                    False,
                    "group",
                    "Random event order group-layer control.",
                ),
                ViewSpec(
                    "group_silly_order_probe",
                    "group_multilayer",
                    "artifact_probe",
                    "silly_proxy_order",
                    False,
                    "group",
                    "Nonsense/proxy group-layer artifact probe.",
                ),
            ]
        )

    seen: set[str] = set()
    deduped: list[ViewSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def _score_from_row(row: Mapping[str, Any]) -> float:
    if "probe_score" in row:
        return float(row.get("probe_score", 0.0))
    readiness = row.get("structure_readiness")
    if not isinstance(readiness, Mapping):
        return 0.0
    return probe_evidence_score(readiness)


def compare_surfaces(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_signal: float,
    min_gap: float,
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    views = sorted({str(row["view"]) for row in rows})
    for view in views:
        view_rows = [row for row in rows if row["view"] == view]
        candidates = [row for row in view_rows if row["role"] == "candidate"]
        controls = [row for row in view_rows if row["role"] != "candidate"]
        best_candidate = max(candidates, key=_score_from_row, default=None)
        best_control = max(controls, key=_score_from_row, default=None)
        candidate_score = _score_from_row(best_candidate) if best_candidate else 0.0
        control_score = _score_from_row(best_control) if best_control else 0.0
        gap = candidate_score - control_score
        if best_candidate is None:
            verdict = "no_candidate_surface"
        elif best_control is None:
            verdict = "candidate_only"
        elif candidate_score < min_signal and control_score < min_signal:
            verdict = "no_clear_signal"
        elif gap >= min_gap and candidate_score >= min_signal:
            verdict = "candidate_separates_from_controls"
        elif gap <= -min_gap and control_score >= min_signal:
            verdict = "control_dominates"
        else:
            verdict = "ambiguous"
        comparisons.append(
            {
                "view": view,
                "verdict": verdict,
                "candidate_minus_control": gap,
                "best_candidate": None
                if best_candidate is None
                else {
                    "name": best_candidate["name"],
                    "score": candidate_score,
                    "ready": dict(best_candidate.get("structure_readiness") or {}).get("ready"),
                },
                "best_control": None
                if best_control is None
                else {
                    "name": best_control["name"],
                    "role": best_control["role"],
                    "score": control_score,
                    "ready": dict(best_control.get("structure_readiness") or {}).get("ready"),
                },
            }
        )
    return comparisons


def run_view_probe(args: argparse.Namespace) -> dict[str, Any]:
    edges, labels, dataset_metadata, edge_path, label_path = _load_edges_and_labels(args)
    time_bin_adjustment = adjust_real_time_bin_seconds(args, edges)
    assessment = assess_org_network(args)
    directed = bool(args.directed)
    specs = build_view_specs(assessment, include_weak=args.include_weak_views)
    if args.max_surfaces > 0:
        specs = specs[: args.max_surfaces]

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        row: dict[str, Any] = spec.to_dict()
        try:
            surface = _surface_for_spec(
                edges,
                labels,
                spec,
                args=args,
                directed=directed,
                seed_offset=1_000 * index,
            )
            readiness = _score_surface(surface, args=args, seed_offset=2_000 * index)
            row.update(
                {
                    "series_metadata": surface.metadata,
                    "dropped_series": surface.dropped_series[:25],
                    "structure_readiness": readiness.to_dict(),
                    "probe_score": probe_evidence_score(readiness),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - failed candidate is part of the diagnostic.
            row.update(
                {
                    "series_metadata": None,
                    "dropped_series": [],
                    "structure_readiness": None,
                    "probe_score": 0.0,
                    "error": str(exc),
                }
            )
        rows.append(row)

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
        "time_bin_adjustment": time_bin_adjustment,
        "assessment": {
            "profile": assessment["profile"],
            "view_candidates": assessment["view_candidates"],
            "order_candidates": assessment["order_candidates"],
        },
        "surfaces": rows,
        "comparisons": compare_surfaces(
            rows,
            min_signal=args.min_signal,
            min_gap=args.min_gap,
        ),
        "interpretation_guardrails": [
            "A candidate view separating from controls earns a larger run; it is not a discovery by itself.",
            "A control-dominates result means the enforced structure is probably introducing an artifact.",
            "Ambiguous ties should be tested downstream in parallel or treated as unresolved.",
            "Map any surfaced anomalies back to canonical events/entities before naming them.",
        ],
    }


def _fmt_z(value: object) -> str:
    if value is None:
        return "None"
    try:
        return "%.2f" % float(value)
    except (TypeError, ValueError):
        return "None"


def adjust_real_time_bin_seconds(
    args: argparse.Namespace,
    edges: Sequence[TemporalEdge],
) -> dict[str, object] | None:
    """Bound real-time bin count before assessment/probing can allocate heavily."""

    if not args.auto_bin_real_time or args.max_real_time_bins <= 0 or args.max_bins > 0:
        return None
    if not edges:
        return None
    timestamps = [float(edge.timestamp) for edge in edges]
    span = max(timestamps) - min(timestamps)
    estimated_bins = int(span // float(args.bin_seconds)) + 1
    if estimated_bins <= args.max_real_time_bins:
        return None
    old = float(args.bin_seconds)
    adjusted = max(old, span / max(args.max_real_time_bins - 1, 1))
    args.bin_seconds = float(adjusted)
    return {
        "reason": "bounded_real_time_bins",
        "old_bin_seconds": old,
        "new_bin_seconds": float(args.bin_seconds),
        "estimated_bins_before": estimated_bins,
        "max_real_time_bins": int(args.max_real_time_bins),
    }


def print_report(report: Mapping[str, Any], *, top: int) -> None:
    dataset = dict(report["dataset"])
    print("Organizational network view probe")
    print("  dataset=%s" % dataset.get("dataset", dataset.get("edge_path", "custom")))
    print("  surfaces=%d" % len(report.get("surfaces", [])))
    adjustment = report.get("time_bin_adjustment")
    if adjustment:
        item = dict(adjustment)
        print(
            "  adjusted real-time bin_seconds %.3f -> %.3f to bound %s bins"
            % (
                float(item["old_bin_seconds"]),
                float(item["new_bin_seconds"]),
                item["max_real_time_bins"],
            )
        )
    print("  comparisons")
    for row in report.get("comparisons", []):
        item = dict(row)
        best_candidate = dict(item.get("best_candidate") or {})
        best_control = dict(item.get("best_control") or {})
        print(
            "    %-24s verdict=%-34s gap=% .3f candidate=%s:%.3f control=%s:%.3f"
            % (
                item["view"],
                item["verdict"],
                float(item["candidate_minus_control"]),
                best_candidate.get("name", "-"),
                float(best_candidate.get("score", 0.0)),
                best_control.get("name", "-"),
                float(best_control.get("score", 0.0)),
            )
        )
    print("  top surfaces")
    surfaces = sorted(report.get("surfaces", []), key=_score_from_row, reverse=True)[:top]
    for row in surfaces:
        item = dict(row)
        readiness = dict(item.get("structure_readiness") or {})
        print(
            "    %-34s view=%-24s role=%-16s probe=%.3f ready_score=%.3f ready=%s cov_z=%s pat_z=%s temp_z=%s error=%s"
            % (
                item["name"],
                item["view"],
                item["role"],
                float(item.get("probe_score", 0.0)),
                float(readiness.get("structure_score", 0.0)),
                readiness.get("ready"),
                _fmt_z(readiness.get("covariance_z_effect")),
                _fmt_z(readiness.get("pattern_z_effect")),
                _fmt_z(readiness.get("temporal_z_effect")),
                item.get("error"),
            )
        )


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
    parser.add_argument("--event-bin-size", type=int, default=1024)
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--no-node-series", action="store_true")
    parser.add_argument("--no-group-series", action="store_true")
    parser.add_argument("--transform", choices=["none", "log", "robust", "log-robust"], default="log-robust")
    parser.add_argument("--structure-null-repeats", type=int, default=50)
    parser.add_argument("--structure-active-z", type=float, default=1.5)
    parser.add_argument("--structure-min-series", type=int, default=3)
    parser.add_argument("--structure-min-score", type=float, default=0.35)
    parser.add_argument("--include-weak-views", action="store_true")
    parser.add_argument("--max-surfaces", type=int, default=0)
    parser.add_argument("--auto-bin-real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-real-time-bins", type=int, default=5000)
    parser.add_argument("--min-signal", type=float, default=1.0)
    parser.add_argument("--min-gap", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    if args.event_bin_size <= 0:
        raise ValueError("--event-bin-size must be positive")
    if args.min_gap < 0:
        raise ValueError("--min-gap must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_view_probe(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, top=args.top)


if __name__ == "__main__":
    main()
