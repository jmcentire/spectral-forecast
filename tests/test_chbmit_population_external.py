"""Tests for frozen external CHB-MIT population evaluation helpers."""

import numpy as np

from experiments.chbmit_population_external import (
    _axis_comparison,
    _candidate_from_source,
    _ordered_population_profile,
    _pairwise_fingerprint_relations,
    _prefix_split,
)
from experiments.chbmit_population_nominal import FeatureFile


def test_candidate_from_source_uses_only_frozen_shared_candidate() -> None:
    source = {
        "shared_population_instrument": {
            "selected_candidate": {
                "window_samples": 256,
                "stride": 128,
                "spectral_bins": 8,
                "feature_quantile": 0.9,
                "emission_threshold": 2.5,
                "min_active_channels": 3,
            }
        }
    }

    candidate = _candidate_from_source(source)

    assert candidate.geometry.window_samples == 256
    assert candidate.geometry.stride == 128
    assert candidate.emission_threshold == 2.5
    assert candidate.min_active_channels == 3


def test_axis_comparison_distinguishes_population_and_self_structure() -> None:
    source = {
        "shared_population_instrument": {
            "selected_candidate": {
                "window_samples": 256,
                "stride": 256,
                "spectral_bins": 8,
                "feature_quantile": 0.9,
                "emission_threshold": 2.0,
                "min_active_channels": 2,
            }
        }
    }
    candidate = _candidate_from_source(source)
    file = FeatureFile(
        subject="heldout",
        path="heldout_01.edf",
        anchors=np.arange(4, dtype=np.int64),
        tensor=np.zeros((4, 3, 2), dtype=np.float64),
        entity_names=("a", "b", "c"),
        feature_names=("x", "y"),
    )
    population = np.asarray(
        [
            [3.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    self_matrix = np.asarray(
        [
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    comparison = _axis_comparison(
        [file],
        [population],
        [self_matrix],
        candidate=candidate,
    )

    assert comparison["both_positive_fraction"] == 0.25
    assert comparison["population_only_fraction"] == 0.25
    assert comparison["self_only_fraction"] == 0.25
    assert comparison["neither_fraction"] == 0.25
    assert comparison["positive_jaccard"] == 1 / 3


def test_prefix_split_preserves_order() -> None:
    files = [
        FeatureFile(
            subject="heldout",
            path=f"heldout_{index:02d}.edf",
            anchors=np.arange(4, dtype=np.int64),
            tensor=np.zeros((4, 3, 2), dtype=np.float64),
            entity_names=("a", "b", "c"),
            feature_names=("x", "y"),
        )
        for index in range(1, 7)
    ]

    calibration, validation = _prefix_split(list(reversed(files)))

    assert [item.path for item in calibration] == [
        "heldout_01.edf",
        "heldout_02.edf",
        "heldout_03.edf",
    ]
    assert [item.path for item in validation] == [
        "heldout_04.edf",
        "heldout_05.edf",
        "heldout_06.edf",
    ]


def test_ordered_profile_reports_step_and_drift() -> None:
    source = {
        "shared_population_instrument": {
            "selected_candidate": {
                "window_samples": 256,
                "stride": 256,
                "spectral_bins": 8,
                "feature_quantile": 0.9,
                "emission_threshold": 2.0,
                "min_active_channels": 2,
            }
        }
    }
    candidate = _candidate_from_source(source)
    files = [
        FeatureFile(
            subject="heldout",
            path=f"heldout_{index:02d}.edf",
            anchors=np.arange(4, dtype=np.int64),
            tensor=np.zeros((4, 3, 2), dtype=np.float64),
            entity_names=("a", "b", "c"),
            feature_names=("x", "y"),
        )
        for index in range(1, 5)
    ]
    matrices = [
        np.zeros((4, 3)),
        np.zeros((4, 3)),
        np.full((4, 3), 3.0),
        np.full((4, 3), 3.0),
    ]

    profile = _ordered_population_profile(files, matrices, candidate=candidate)

    assert profile["second_minus_first_half"] == 1.0
    assert profile["largest_adjacent_step"]["from_file"] == "heldout_02.edf"
    assert profile["largest_adjacent_step"]["to_file"] == "heldout_03.edf"
    assert profile["largest_adjacent_step"]["delta"] == 1.0


def test_pairwise_fingerprints_rank_nearest_subjects() -> None:
    rows = _pairwise_fingerprint_relations(
        {
            "a": np.asarray([0.0, 0.0, 0.0]),
            "b": np.asarray([0.1, 0.0, 0.0]),
            "c": np.asarray([4.0, 4.0, 4.0]),
        }
    )

    assert (rows[0]["subject_a"], rows[0]["subject_b"]) == ("a", "b")
    assert rows[0]["rms_difference"] < rows[-1]["rms_difference"]


def test_pairwise_fingerprints_preserve_opposite_directions() -> None:
    rows = _pairwise_fingerprint_relations(
        {
            "positive": np.asarray([1.0, 1.0, 1.0]),
            "negative": np.asarray([-1.0, -1.0, -1.0]),
        }
    )

    assert rows[0]["cosine_similarity"] == -1.0
