"""Tests for temporal organizational-network adapter."""

import argparse
from pathlib import Path

import numpy as np

import experiments.org_network_autotune as org_network_autotune
from experiments.org_network_autotune import (
    TemporalEdge,
    build_selected_network,
    build_network_series,
    canonical_context_for_interval,
    effective_null_block_size,
    read_edges,
    read_labels,
    read_simplicial_edges,
    relation_only_network,
    remap_edges_by_order_with_mapping,
    top_windows,
)
from spectral_forecast.autotune import AutoTuneConfig, AutoTuneObservation


def test_read_edges_supports_reordered_columns(tmp_path: Path) -> None:
    path = tmp_path / "edges.txt"
    path.write_text("10 a b\n20 b c\n", encoding="utf-8")

    edges = read_edges(path, source_col=1, target_col=2, time_col=0, max_edges=0)

    assert edges == [
        TemporalEdge(source="a", target="b", timestamp=10.0),
        TemporalEdge(source="b", target="c", timestamp=20.0),
    ]


def test_read_labels_accepts_single_line_pair_list(tmp_path: Path) -> None:
    path = tmp_path / "labels.txt"
    path.write_text("a G1 b G1 c G2", encoding="utf-8")

    assert read_labels(path) == {"a": "G1", "b": "G1", "c": "G2"}


def test_build_network_series_creates_node_group_and_global_features() -> None:
    edges = [
        TemporalEdge("a", "b", 0.0),
        TemporalEdge("b", "a", 10.0),
        TemporalEdge("a", "c", 70.0),
        TemporalEdge("c", "a", 130.0),
        TemporalEdge("a", "b", 190.0),
    ]
    labels = {"a": "G1", "b": "G1", "c": "G2"}

    network = build_network_series(
        edges,
        labels,
        bin_seconds=60.0,
        top_nodes=2,
        top_groups=2,
        include_global=True,
        include_nodes=True,
        include_groups=True,
        include_relation=True,
        directed=True,
        transform="none",
        max_bins=0,
    )

    assert network.metadata["bin_count"] == 4
    assert network.metadata["selected_groups"] == ["G1", "G2"]
    assert "global_total_edges" in network.series
    assert "global_new_pairs" in network.series
    assert "node_a_total" in network.series
    assert "node_a_new_neighbors" in network.series
    assert "group_G1_internal_edges" in network.series
    assert "group_G2_new_external_pairs" in network.series
    assert "group_G2_received" in network.series
    assert np.allclose(network.raw_series["global_total_edges"], [2.0, 1.0, 1.0, 1.0])
    assert np.allclose(network.raw_series["global_cross_group_edges"], [0.0, 1.0, 1.0, 0.0])
    assert np.allclose(network.raw_series["global_new_pairs"], [2.0, 1.0, 1.0, 0.0])
    assert np.allclose(network.raw_series["global_returning_pairs"], [0.0, 0.0, 0.0, 1.0])

    relation = relation_only_network(network)
    assert "global_total_edges" not in relation.series
    assert "global_new_pairs" in relation.series
    assert "node_a_new_neighbors" in relation.series


def test_read_simplicial_edges_projects_sets_to_pairs(tmp_path: Path) -> None:
    directory = tmp_path / "simplices"
    directory.mkdir()
    (directory / "toy-nverts.txt").write_text("3\n2\n", encoding="utf-8")
    (directory / "toy-simplices.txt").write_text("1\n2\n3\n2\n3\n", encoding="utf-8")
    (directory / "toy-times.txt").write_text("10\n20\n", encoding="utf-8")

    edges = read_simplicial_edges(directory, prefix="toy", max_edges=0)

    assert edges == [
        TemporalEdge(source="1", target="2", timestamp=10.0),
        TemporalEdge(source="1", target="3", timestamp=10.0),
        TemporalEdge(source="2", target="3", timestamp=10.0),
        TemporalEdge(source="2", target="3", timestamp=20.0),
    ]


def test_remap_edges_retains_mapping_to_canonical_order() -> None:
    edges = [
        TemporalEdge("b", "c", 30.0),
        TemporalEdge("a", "b", 10.0),
        TemporalEdge("a", "b", 20.0),
    ]

    remapped, mapping = remap_edges_by_order_with_mapping(
        edges,
        order="first_seen_pair",
        labels={},
        directed=False,
        seed=1,
    )

    assert mapping == [1, 2, 0]
    assert [edge.timestamp for edge in remapped] == [0.0, 1.0, 2.0]
    assert [edges[index].timestamp for index in mapping] == [10.0, 20.0, 30.0]


def test_effective_null_block_size_keeps_multiple_blocks_when_possible() -> None:
    assert effective_null_block_size(8, 4) == 2
    assert effective_null_block_size(8, 20) == 8
    assert effective_null_block_size(8, 1) == 1


def test_build_selected_network_limits_feature_family() -> None:
    edges = [
        TemporalEdge("a", "b", 0.0),
        TemporalEdge("a", "c", 1.0),
        TemporalEdge("b", "c", 2.0),
        TemporalEdge("a", "b", 3.0),
    ]
    labels = {"a": "G1", "b": "G1", "c": "G2"}
    common = {
        "relation_only": False,
        "no_global_series": False,
        "no_node_series": False,
        "no_group_series": False,
        "no_relation_series": False,
        "bin_seconds": 1.0,
        "top_nodes": 3,
        "top_groups": 2,
        "directed": False,
        "transform": "none",
        "max_bins": 0,
        "event_order": "real-time",
    }

    graph = build_selected_network(
        edges,
        labels,
        args=argparse.Namespace(**common, surface_mode="graph"),
    )
    group = build_selected_network(
        edges,
        labels,
        args=argparse.Namespace(**common, surface_mode="group"),
    )

    assert graph.metadata["surface_mode"] == "graph"
    assert any(name.startswith("node_") for name in graph.series)
    assert not any(name.startswith("group_") for name in graph.series)
    assert "global_new_pairs" not in graph.series
    assert group.metadata["surface_mode"] == "group"
    assert not any(name.startswith("node_") for name in group.series)
    assert any(name.startswith("group_") for name in group.series)
    assert "global_new_pairs" in group.series


def test_canonical_context_maps_interval_to_original_edges() -> None:
    edges = [
        TemporalEdge("a", "b", 0.0),
        TemporalEdge("a", "c", 60.0),
        TemporalEdge("c", "b", 120.0),
    ]
    labels = {"a": "G1", "b": "G1", "c": "G2"}

    context = canonical_context_for_interval(
        edges,
        labels,
        absolute_start_timestamp=0.0,
        bin_seconds=60.0,
        start_bin=0,
        end_bin=2,
        directed=False,
        max_events=5,
        max_entities=5,
        max_pairs=5,
    )

    assert context["event_count"] == 2
    assert context["unique_entity_count"] == 3
    assert context["cross_group_event_count"] == 1
    assert [event["canonical_edge_index"] for event in context["sampled_events"]] == [0, 1]
    assert context["top_entities"][0] == {
        "entity": "a",
        "event_endpoint_count": 2,
        "group": "G1",
    }


def test_top_windows_attaches_bounded_canonical_context(monkeypatch) -> None:
    observation = AutoTuneObservation(
        anchors=[12],
        matrix=np.asarray([[4.0, 3.0]], dtype=np.float64),
        readiness_score=1.0,
        series_count=2,
    )
    monkeypatch.setattr(
        org_network_autotune,
        "build_autotune_observation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("observation should be reused")),
    )
    config = AutoTuneConfig(
        baseline_size=10,
        adaptive_window=9,
        stride=1,
        emission_threshold=1.0,
        decay=0.9,
        min_active_series=1,
    )
    args = argparse.Namespace(
        sample_rate=1.0,
        bin_seconds=1.0,
        top_windows=1,
        canonical_max_events=2,
        canonical_max_entities=2,
        canonical_max_pairs=2,
    )
    edges = [
        TemporalEdge("a", "b", float(index))
        for index in range(20)
    ]

    rows = top_windows(
        {"left": np.arange(20, dtype=np.float64), "right": np.arange(20, dtype=np.float64)},
        {"left": np.arange(20, dtype=np.float64), "right": np.arange(20, dtype=np.float64)},
        config,
        args=args,
        segment_name="calibration",
        segment_bin_offset=0,
        absolute_start_timestamp=0.0,
        canonical_edges=edges,
        labels={"a": "G1", "b": "G2"},
        directed=False,
        observation=observation,
    )

    assert rows[0]["analysis_bins"]["adaptive_start"] == 3
    assert rows[0]["canonical_context"]["anchor_bin"]["event_count"] == 1
    assert rows[0]["canonical_context"]["adaptive_history"]["event_count"] == 9
    assert len(rows[0]["canonical_context"]["adaptive_history"]["sampled_events"]) == 2
    assert rows[0]["canonical_context"]["adaptive_history"]["sampled_events_truncated"]


def test_canonical_context_maps_ordered_analysis_back_to_original_edges() -> None:
    canonical = [
        TemporalEdge("a", "b", 100.0),
        TemporalEdge("c", "d", 200.0),
        TemporalEdge("e", "f", 300.0),
    ]
    analysis, mapping = remap_edges_by_order_with_mapping(
        canonical,
        order="stable_id_or_alphabetic",
        labels={},
        directed=False,
        seed=1,
    )

    context = canonical_context_for_interval(
        analysis,
        {},
        absolute_start_timestamp=0.0,
        bin_seconds=1.0,
        start_bin=0,
        end_bin=1,
        directed=False,
        max_events=2,
        max_entities=2,
        max_pairs=2,
        canonical_edges=canonical,
        canonical_edge_indices=mapping,
        coordinate_kind="event_rank",
    )

    event = context["sampled_events"][0]
    assert context["coordinate_kind"] == "event_rank"
    assert context["timestamp_start"] is None
    assert context["analysis_coordinate_start"] == 0.0
    assert event["analysis_edge_index"] == 0
    assert event["canonical_edge_index"] == mapping[0]
    assert event["timestamp"] == canonical[mapping[0]].timestamp
