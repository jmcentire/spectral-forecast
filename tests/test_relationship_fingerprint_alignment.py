import numpy as np
import pytest

from experiments.relationship_fingerprint_alignment import cosine, edge_matrix


def test_edge_matrix_and_cosine_preserve_canonical_edges() -> None:
    graph = edge_matrix([1.0, 0.0, -1.0], 3)

    assert graph[0, 1] == 1.0
    assert graph[1, 2] == -1.0
    assert cosine(graph, graph, absolute=False) == pytest.approx(1.0)
    assert cosine(graph, -graph, absolute=True) == pytest.approx(1.0)


def test_edge_matrix_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="does not match"):
        edge_matrix(np.ones(4), 3)
