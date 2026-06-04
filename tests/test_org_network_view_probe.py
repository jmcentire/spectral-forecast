"""Tests for organizational-network view probing."""

import argparse
from pathlib import Path

from experiments.org_network_view_probe import (
    build_view_specs,
    recommendation_state,
    remap_edges_by_order,
    run_view_probe,
)
from experiments.org_network_autotune import TemporalEdge


def test_remap_edges_by_order_replaces_time_with_event_rank() -> None:
    edges = [
        TemporalEdge("b", "c", 30.0),
        TemporalEdge("a", "b", 10.0),
        TemporalEdge("a", "b", 20.0),
    ]

    remapped = remap_edges_by_order(
        edges,
        order="first_seen_pair",
        labels={},
        directed=False,
        seed=1,
    )

    assert [edge.timestamp for edge in remapped] == [0.0, 1.0, 2.0]
    assert [(edge.source, edge.target) for edge in remapped][:2] == [("a", "b"), ("a", "b")]


def test_build_view_specs_adds_candidates_and_controls() -> None:
    assessment = {
        "view_candidates": [
            {"name": "relation_churn", "grade": "strong"},
            {"name": "relationship_graph", "grade": "promising"},
            {"name": "group_multilayer", "grade": "poor"},
        ]
    }

    specs = build_view_specs(assessment, include_weak=False)
    names = {spec.name for spec in specs}
    roles = {spec.role for spec in specs}

    assert "relation_first_seen_pair" in names
    assert "graph_degree_descending" in names
    assert "group_real_time" not in names
    assert "candidate" in roles
    assert "null_control" in roles
    assert "artifact_control" in roles


def test_recommendation_state_can_refuse_all_views() -> None:
    recommendation = recommendation_state(
        [
            {
                "view": "temporal_total_order",
                "verdict": "ambiguous",
                "best_control": {"score": 2.0},
            },
            {
                "view": "relationship_graph",
                "verdict": "control_dominates",
                "best_control": {"score": 3.0},
            },
        ],
        min_signal=1.0,
    )

    assert recommendation["state"] == "controls_too_strong"
    assert not recommendation["proceed"]
    assert recommendation["recommended_views"] == []
    assert recommendation["rejected_views"] == ["relationship_graph"]
    assert recommendation["unresolved_views"] == ["temporal_total_order"]


def test_recommendation_state_can_select_subset() -> None:
    recommendation = recommendation_state(
        [
            {
                "view": "group_multilayer",
                "verdict": "candidate_separates_from_controls",
                "best_control": {"score": 2.0},
            },
            {
                "view": "relation_churn",
                "verdict": "ambiguous",
                "best_control": {"score": 2.0},
            },
        ],
        min_signal=1.0,
    )

    assert recommendation["state"] == "selective_candidate_set"
    assert recommendation["proceed"]
    assert recommendation["recommended_views"] == ["group_multilayer"]


def test_run_view_probe_reports_candidate_control_comparison(tmp_path: Path) -> None:
    edge_path = tmp_path / "edges.txt"
    edge_path.write_text(
        "\n".join(
            [
                "a b 0",
                "a b 1",
                "a c 2",
                "a c 3",
                "b c 4",
                "b c 5",
                "d e 6",
                "d e 7",
                "d f 8",
                "d f 9",
                "e f 10",
                "e f 11",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    label_path = tmp_path / "labels.txt"
    label_path.write_text("a G1 b G1 c G1 d G2 e G2 f G2", encoding="utf-8")
    args = argparse.Namespace(
        dataset=None,
        download=False,
        force_download=False,
        data_dir=tmp_path,
        edges=edge_path,
        labels=label_path,
        data_format="edges",
        simplex_prefix="email-Enron",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=False,
        max_edges=0,
        max_bins=0,
        bin_seconds=2.0,
        event_bin_size=2,
        top_nodes=6,
        top_groups=2,
        no_node_series=False,
        no_group_series=False,
        transform="log-robust",
        structure_null_repeats=4,
        structure_active_z=1.5,
        structure_min_series=2,
        structure_min_score=0.1,
        include_weak_views=True,
        max_surfaces=6,
        auto_bin_real_time=True,
        max_real_time_bins=5000,
        min_signal=0.1,
        min_gap=0.05,
        seed=11,
        top=5,
        output=None,
        format="text",
    )

    report = run_view_probe(args)

    assert report["surfaces"]
    assert report["comparisons"]
    assert report["recommendation"]["state"]
    assert any(row["role"] == "candidate" for row in report["surfaces"])
    assert any(row["role"] != "candidate" for row in report["surfaces"])
