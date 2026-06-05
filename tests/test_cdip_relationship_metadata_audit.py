"""Tests for the post-hoc CDIP relationship metadata audit."""

from experiments.cdip_relationship_metadata_audit import (
    _processing_family,
    audit_metadata,
)


def _pair(left, right, *, detected, occurrences=2, clusters=1):
    return {
        "entities": [left, right],
        "layer": "raw",
        "occurrences": occurrences,
        "independent_time_clusters": clusters,
        "replicated_across_distinct_times": clusters > 1,
        "raw_profile": {"domain_regroup": {"detected_fdr": detected}},
    }


def test_processing_family_reads_source_argument_prefix() -> None:
    history = "dataset created; program, arguments: process v3.2, iv74269_20260602"
    assert _processing_family(history) == "iv"


def test_metadata_audit_detects_matched_acquisition_family() -> None:
    rows = [
        _pair("a", "b", detected=True),
        _pair("a", "c", detected=True),
        _pair("a", "d", detected=False),
        _pair("b", "c", detected=False),
        _pair("b", "d", detected=False),
        _pair("c", "d", detected=False),
    ]
    metadata = {
        "a": {"sample_rate": 1.28, "processing_family": "lx", "water_depth": 10.0},
        "b": {"sample_rate": 1.28, "processing_family": "lx", "water_depth": 12.0},
        "c": {"sample_rate": 1.28, "processing_family": "lx", "water_depth": 14.0},
        "d": {"sample_rate": 2.56, "processing_family": "iv", "water_depth": 100.0},
    }

    result = audit_metadata(
        rows,
        metadata=metadata,
        profile_key="raw_profile",
        matched_permutation_repeats=99,
        seed=3,
    )

    metrics = {row["metric"]: row for row in result["metrics"]}
    assert metrics["sample_rate_match"]["detected_matches"] == 2
    assert metrics["processing_family_match"]["detected_matches"] == 2
    assert result["detected_pairs"][0]["sample_rate_and_processing_family_match"]
