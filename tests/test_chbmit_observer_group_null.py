"""Tests for CHB-MIT observer group-null helpers."""

import numpy as np

from experiments.chbmit_observer_group_null import _dominant_shape_matrices


def test_dominant_shape_matrices_excludes_short_files() -> None:
    rows = [
        ("a.edf", np.zeros((221, 6))),
        ("b.edf", np.zeros((221, 6))),
        ("short.edf", np.zeros((56, 6))),
    ]

    matrices, summary = _dominant_shape_matrices(rows)

    assert len(matrices) == 2
    assert summary["dominant_shape"] == [221, 6]
    assert summary["excluded_non_dominant_shape"] == [
        {"file": "short.edf", "shape": [56, 6]},
    ]
