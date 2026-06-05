"""Known-answer tests for identity, role, motif, step, and drift structure."""

import numpy as np
import pytest

from spectral_forecast.relationship_dynamics import (
    RelationshipGraphSequence,
    analyze_relationship_dynamics,
    describe_step_change,
    estimate_relationship_block_size,
    exact_identity_distance,
    global_motif_distance,
    structural_role_distance,
    window_relationship_graphs,
)


def _matching_graph() -> np.ndarray:
    graph = np.zeros((4, 4), dtype=np.float64)
    graph[0, 1] = graph[1, 0] = 1.0
    graph[2, 3] = graph[3, 2] = 1.0
    return graph


def _star_graph() -> np.ndarray:
    graph = np.zeros((4, 4), dtype=np.float64)
    for node in (1, 2, 3):
        graph[0, node] = graph[node, 0] = 1.0
    return graph


def _sequence(graphs: list[np.ndarray]) -> RelationshipGraphSequence:
    return RelationshipGraphSequence(
        entities=("a", "b", "c", "d"),
        anchors=tuple(range(1, len(graphs) + 1)),
        family="correlation",
        window_size=8,
        stride=1,
        graphs=np.asarray(graphs),
    )


def _evidence(result, mode: str, metric: str):
    return next(
        row for row in result.evidence if row.mode == mode and row.metric == metric
    )


def test_node_substitution_changes_identity_but_preserves_roles_and_motifs() -> None:
    base = _star_graph()
    permutation = np.asarray([1, 0, 2, 3])
    substituted = base[np.ix_(permutation, permutation)]

    assert exact_identity_distance(base, substituted) > 0.0
    assert structural_role_distance(base, substituted)[0] == 0.0
    assert global_motif_distance(base, substituted) < 1e-12


def test_step_audit_separates_actor_substitution_from_topology_change() -> None:
    base = _star_graph()
    permutation = np.asarray([1, 0, 2, 3])
    substituted = base[np.ix_(permutation, permutation)]
    substitution = analyze_relationship_dynamics(
        _sequence([base] * 20 + [substituted] * 20),
        min_segment_windows=6,
        null_repeats=999,
        null_block_size=2,
        seed=12,
    )
    topology = analyze_relationship_dynamics(
        _sequence([_matching_graph()] * 20 + [_star_graph()] * 20),
        min_segment_windows=6,
        null_repeats=999,
        null_block_size=2,
        seed=13,
    )

    assert _evidence(substitution, "step", "exact_identity").detected_fdr_by
    assert not _evidence(substitution, "step", "structural_role").detected_fdr_by
    assert not _evidence(substitution, "step", "global_motif").detected_fdr_by
    assert _evidence(topology, "step", "structural_role").detected_fdr_by
    assert _evidence(topology, "step", "global_motif").detected_fdr_by


def test_drift_audit_detects_gradual_role_and_motif_change() -> None:
    base = _matching_graph()
    target = _star_graph()
    graphs = [
        (1.0 - fraction) * base + fraction * target
        for fraction in np.linspace(0.0, 1.0, 60)
    ]

    result = analyze_relationship_dynamics(
        _sequence(graphs),
        min_segment_windows=5,
        null_repeats=999,
        null_block_size=3,
        seed=14,
    )

    assert _evidence(result, "drift", "structural_role").detected_fdr_by
    assert _evidence(result, "drift", "global_motif").detected_fdr_by


def test_raw_noisy_series_recovers_injected_topology_change() -> None:
    rng = np.random.default_rng(42)
    n = 4096
    midpoint = n // 2
    matrix = np.zeros((n, 4), dtype=np.float64)
    left_a = rng.normal(size=midpoint)
    left_b = rng.normal(size=midpoint)
    matrix[:midpoint, 0] = left_a + rng.normal(0.0, 0.2, midpoint)
    matrix[:midpoint, 1] = left_a + rng.normal(0.0, 0.2, midpoint)
    matrix[:midpoint, 2] = left_b + rng.normal(0.0, 0.2, midpoint)
    matrix[:midpoint, 3] = left_b + rng.normal(0.0, 0.2, midpoint)
    leaves = rng.normal(size=(midpoint, 3))
    matrix[midpoint:, 1:] = leaves
    matrix[midpoint:, 0] = (
        np.sum(leaves, axis=1) / np.sqrt(3.0)
        + rng.normal(0.0, 0.2, midpoint)
    )
    sequence = window_relationship_graphs(
        matrix,
        ("a", "b", "c", "d"),
        window_size=256,
        stride=128,
    )

    result = analyze_relationship_dynamics(
        sequence,
        min_segment_windows=6,
        null_repeats=999,
        null_block_size=2,
        seed=43,
    )

    role = _evidence(result, "step", "structural_role")
    motif = _evidence(result, "step", "global_motif")
    assert role.detected_fdr_by
    assert motif.detected_fdr_by
    assert abs(role.attributes["best_split_anchor"] - midpoint) <= 256


def test_stationary_graph_sequence_does_not_report_change() -> None:
    rng = np.random.default_rng(99)
    base = _matching_graph() * 0.8
    graphs = []
    for _ in range(40):
        noise = rng.normal(0.0, 0.05, (4, 4))
        noise = (noise + noise.T) / 2.0
        np.fill_diagonal(noise, 0.0)
        graphs.append(base + noise)

    result = analyze_relationship_dynamics(
        _sequence(graphs),
        min_segment_windows=6,
        null_repeats=399,
        null_block_size=2,
        seed=100,
    )

    assert not any(row.detected_fdr_by for row in result.evidence)


def test_window_relationship_graphs_requires_robust_window() -> None:
    matrix = np.ones((64, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 32 rows"):
        window_relationship_graphs(matrix, ("a", "b", "c", "d"), window_size=12)


def test_null_repeats_are_unique_and_capped_by_permutation_space() -> None:
    result = analyze_relationship_dynamics(
        _sequence([_matching_graph()] * 8),
        min_segment_windows=2,
        null_repeats=999,
        null_block_size=2,
        seed=101,
    )

    assert result.requested_null_repeats == 999
    assert result.null_permutation_space == 23
    assert result.null_repeats == 23
    assert result.null_exhaustive
    assert {row.null_repeats for row in result.evidence} == {23}


def test_automatic_block_size_is_recorded() -> None:
    base = _matching_graph()
    graphs = [base * (1.0 + 0.02 * index) for index in range(24)]
    sequence = _sequence(graphs)

    estimated = estimate_relationship_block_size(sequence)
    result = analyze_relationship_dynamics(
        sequence,
        min_segment_windows=4,
        null_repeats=19,
        null_block_size=None,
        seed=102,
    )

    assert 2 <= estimated <= 6
    assert result.null_block_estimated
    assert result.null_block_size == estimated


def test_step_description_identifies_changed_entities() -> None:
    sequence = _sequence([_matching_graph()] * 4 + [_star_graph()] * 4)

    description = describe_step_change(sequence, 4, top_k=3)

    assert description["split_anchor"] == 5
    assert "a" in {
        row["entity"] for row in description["top_changed_entities"][:3]
    }
    assert len(description["edge_delta_upper"]) == 6
    assert len(description["motif_signature_delta"]) == 13
    assert description["global_motif_distance"] > 0.0
