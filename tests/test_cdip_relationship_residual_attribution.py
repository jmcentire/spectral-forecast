"""Tests for post-hoc CDIP residual attribution."""

from itertools import combinations

import pytest

from experiments.cdip_relationship_residual_attribution import (
    _node_label_permutation_spearman,
    _spearman,
)


def test_spearman_detects_monotonic_order() -> None:
    assert _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_node_label_permutation_detects_pair_distance_attribution() -> None:
    entity_values = {str(index): float(index) for index in range(7)}
    pairs = list(combinations(sorted(entity_values), 2))
    effects = [
        -abs(entity_values[left] - entity_values[right])
        for left, right in pairs
    ]

    result = _node_label_permutation_spearman(
        pairs,
        effects,
        entity_values=entity_values,
        relationship=lambda left, right: abs(left - right),
        repeats=999,
        seed=41,
    )

    assert result["observed_spearman"] == pytest.approx(-1.0)
    assert result["empirical_p_two_sided"] <= 0.01
