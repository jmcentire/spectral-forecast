"""Tests for the post-hoc CDIP relationship geography audit."""

from experiments.cdip_relationship_geography_audit import audit_geography, haversine_km


def _pair(left, right, *, detected, layer="raw", replicated=False):
    return {
        "entities": [left, right],
        "layer": layer,
        "replicated_across_segments": replicated,
        "raw_profile": {
            "domain_regroup": {
                "detected_fdr": detected,
            }
        },
    }


def test_haversine_km_is_zero_for_identical_coordinates() -> None:
    assert haversine_km((32.0, -117.0), (32.0, -117.0)) == 0.0


def test_geography_audit_uses_only_fdr_detected_pair_identities() -> None:
    rows = [
        _pair("a", "b", detected=True, replicated=True),
        _pair("a", "b", detected=True, layer="dominant:1.0", replicated=True),
        _pair("a", "c", detected=False),
        _pair("b", "c", detected=False),
    ]
    result = audit_geography(
        rows,
        coordinates={
            "a": (0.0, 0.0),
            "b": (0.0, 0.5),
            "c": (40.0, 40.0),
        },
        names={"a": "A", "b": "B", "c": "C"},
        thresholds_km=(100.0,),
        matched_permutation_repeats=99,
        seed=1,
    )

    assert result["summary"]["population_pair_identities"] == 3
    assert result["summary"]["detected_pair_identities"] == 1
    assert result["thresholds"][0]["detected_close"] == 1
    assert result["thresholds"][0]["matched_permutation_repeats"] == 99
    assert result["detected_pairs"][0]["layers"] == ["dominant:1.0", "raw"]
