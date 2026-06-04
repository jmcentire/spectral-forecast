"""Tests for organizational-network structure assessment."""

from experiments.org_network_assess import (
    grade_view_candidates,
    order_candidates,
    profile_edges,
)
from experiments.org_network_autotune import TemporalEdge, build_network_series, relation_only_network
from spectral_forecast.structure import structure_readiness


def test_profile_edges_detects_recurrence_labels_and_simultaneity() -> None:
    edges = [
        TemporalEdge("a", "b", 0.0),
        TemporalEdge("a", "b", 0.0),
        TemporalEdge("a", "c", 1.0),
        TemporalEdge("c", "a", 2.0),
    ]
    labels = {"a": "G1", "b": "G1", "c": "G2"}

    profile = profile_edges(edges, labels, directed=False, data_format="edges")

    assert profile.edge_count == 4
    assert profile.node_count == 3
    assert profile.unique_pair_count == 2
    assert profile.simultaneity_fraction == 0.5
    assert profile.repeated_edge_fraction == 0.5
    assert profile.returning_pair_fraction == 1.0
    assert profile.label_coverage == 1.0
    assert profile.group_count == 2
    assert profile.cross_group_fraction == 0.5


def test_grade_view_candidates_prefers_graph_and_relation_for_repeated_ties() -> None:
    edges = [
        TemporalEdge("a", "b", float(index))
        for index in range(40)
    ] + [
        TemporalEdge("a", "c", float(index))
        for index in range(40, 80)
    ] + [
        TemporalEdge("d", "e", float(index))
        for index in range(80, 120)
    ]
    labels: dict[str, str] = {}
    profile = profile_edges(edges, labels, directed=False, data_format="edges")
    surface = build_network_series(
        edges,
        labels,
        bin_seconds=4.0,
        top_nodes=5,
        top_groups=0,
        include_global=True,
        include_nodes=True,
        include_groups=False,
        include_relation=True,
        directed=False,
        transform="log-robust",
        max_bins=0,
    )
    relation = relation_only_network(surface)
    full_ready = structure_readiness(surface.series, null_repeats=8, seed=1)
    relation_ready = structure_readiness(relation.series, null_repeats=8, seed=2)

    candidates = grade_view_candidates(
        profile,
        full_readiness=full_ready,
        relation_readiness=relation_ready,
    )
    by_name = {candidate["name"]: candidate for candidate in candidates}

    assert by_name["relationship_graph"]["score"] > 0.45
    assert by_name["relation_churn"]["score"] > 0.45
    orders = order_candidates(profile, candidates)
    order_roles = {order["name"]: order["role"] for order in orders}
    assert order_roles["degree_descending"] == "candidate"
    assert order_roles["random_order"] == "null_control"
    assert order_roles["silly_proxy_order"] == "artifact_probe"
