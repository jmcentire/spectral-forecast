"""Tests for temporal organizational-network adapter."""

from pathlib import Path

import numpy as np

from experiments.org_network_autotune import (
    TemporalEdge,
    build_network_series,
    read_edges,
    read_labels,
    read_simplicial_edges,
    relation_only_network,
)


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
