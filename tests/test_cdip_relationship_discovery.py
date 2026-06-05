"""Tests for the bounded CDIP relationship-discovery runner."""

import numpy as np

from experiments.cdip_relationship_discovery import (
    SIGNED_ENVELOPE_PROFILE,
    SPECTRAL_PROFILE_KEYS,
    _add_envelope_profiles,
    _aggregate,
    _apply_pair_fdr,
    _layer_view,
    _parse_layers,
    _rng_for,
    _spectral_specificity,
)


def test_parse_layers_requires_raw_and_accepts_residual_branches() -> None:
    assert _parse_layers("common:0.5,dominant:1.0") == [
        "raw",
        "common:0.5",
        "dominant:1.0",
    ]


def test_layer_view_keeps_raw_and_removes_common_mode() -> None:
    rng = np.random.default_rng(10)
    latent = rng.normal(0.0, 1.0, 512)
    matrix = np.column_stack([latent + rng.normal(0.0, 0.1, 512) for _ in range(3)])

    raw, raw_info = _layer_view(matrix, "raw")
    residual, residual_info = _layer_view(matrix, "common:1.0")

    assert np.allclose(raw, matrix)
    assert raw_info["parent"] is None
    assert residual_info["parent"] == "raw"
    raw_singular = np.linalg.svd(raw - np.mean(raw, axis=0), compute_uv=False)
    residual_singular = np.linalg.svd(
        residual - np.mean(residual, axis=0),
        compute_uv=False,
    )
    raw_fraction = raw_singular[0] ** 2 / np.sum(raw_singular**2)
    residual_fraction = residual_singular[0] ** 2 / np.sum(residual_singular**2)
    assert residual_fraction < raw_fraction


def test_envelope_attenuation_ladder_preserves_raw_endpoint() -> None:
    records = [
        {"power": np.asarray([0.7, 0.2, 0.1])},
        {"power": np.asarray([0.1, 0.2, 0.7])},
    ]

    _add_envelope_profiles(records)

    assert np.allclose(records[0]["raw_profile"], records[0]["power"])
    assert np.isclose(np.sum(records[0]["envelope_attenuation_1.00"]), 1.0)
    assert not np.allclose(
        records[0]["envelope_attenuation_1.00"],
        records[0]["raw_profile"],
    )
    assert len(records[0][SIGNED_ENVELOPE_PROFILE]) == 3


def test_pair_fdr_corrects_jointly_across_every_tried_profile() -> None:
    row = {"spectral_alignment_calibrated_supported": True}
    for profile_key in SPECTRAL_PROFILE_KEYS:
        row[profile_key] = {}
        for control in ("domain_regroup", "temporal_regroup"):
            row[profile_key][control] = {
                "empirical_p_ge_observed": (
                    0.01
                    if profile_key == "raw_profile" and control == "domain_regroup"
                    else 1.0
                ),
                "observed_minus_null": 1.0,
                "z_effect": 3.0,
            }

    _apply_pair_fdr([row])

    raw = row["raw_profile"]["domain_regroup"]
    assert np.isclose(raw["fdr_q_value"], 0.06)
    assert not raw["detected_fdr"]


def test_null_stream_is_independent_of_other_hypothesis_consumption() -> None:
    expected = _rng_for(12, "raw", "a", "b", "raw_profile").random(10)
    unrelated = _rng_for(12, "raw", "a", "b", "new_profile")
    unrelated.random(10_000)
    observed = _rng_for(12, "raw", "a", "b", "raw_profile").random(10)

    assert np.allclose(observed, expected)


def test_aggregate_retains_hypothesis_identity_and_source_mapping() -> None:
    evidence = {
        "family": "spectral_alignment",
        "entities": ["a", "b"],
        "observed": 0.9,
        "null_mean": 0.2,
        "null_std": 0.1,
        "observed_minus_null": 0.7,
        "z_effect": 7.0,
        "null_exceedances": 0,
        "empirical_p_ge_observed": 0.05,
        "empirical_p_floor": 0.05,
        "detected": True,
        "attributes": {"shared_peak_frequency": 0.1},
    }
    windows = [
        {
            "window_key": f"{segment}:1:0:t",
            "segment": segment,
            "group_index": 1,
            "window_index": 0,
            "source": {"start": f"{segment}-start", "sources": [{"path": "source.nc"}]},
            "layers": [
                {
                    "layer": "raw",
                    "discovery": {"evidence": [evidence]},
                }
            ],
        }
        for segment in ("calibration", "validation")
    ]

    aggregate = _aggregate(
        windows,
        supported_by_layer={"raw": ["spectral_alignment"]},
        hypothesis_decay=0.8,
        corpus_null_repeats=9,
        seed=11,
        replication_separation_seconds=24.0 * 3600.0,
    )

    hypothesis = aggregate["hypothesis_field"][0]
    assert hypothesis["replicated_across_segments"]
    assert hypothesis["entities"] == ["a", "b"]
    assert aggregate["top_source_mappings"][0]["source"]["sources"][0]["path"] == "source.nc"


def test_spectral_specificity_separates_generic_envelope_from_concurrence() -> None:
    rng = np.random.default_rng(30)
    windows = []
    bins = 32
    baseline = np.exp(-np.linspace(-2.0, 2.0, bins) ** 2)
    for group_index in range(4):
        for window_index in range(4):
            shared = np.zeros(bins)
            shared[(3 * window_index + group_index) % bins] = 0.8
            profiles = []
            for entity in ("a", "b", "c"):
                structure = shared
                if entity == "c":
                    structure = np.zeros(bins)
                    structure[(3 * window_index + group_index + 11) % bins] = 0.8
                power = baseline + structure + rng.uniform(0.0, 0.01, bins)
                power = power / np.sum(power)
                profiles.append({"entity": entity, "power": power.tolist()})
            windows.append(
                {
                    "segment": "validation",
                    "group_index": group_index,
                    "window_index": window_index,
                    "source": {
                        "start": (
                            f"2026-01-{group_index + 1:02d}"
                            f"T{window_index:02d}:00:00+00:00"
                        )
                    },
                    "layers": [{"layer": "raw", "spectral_profiles": profiles}],
                }
            )

    result = _spectral_specificity(
        windows,
        supported_by_layer={"raw": ["spectral_alignment"]},
        null_repeats=99,
        seed=31,
        replication_separation_seconds=24.0 * 3600.0,
    )

    summary = result["layer_summary"][0]
    assert summary["temporal_regroup"][SIGNED_ENVELOPE_PROFILE]["detected"]
    assert (
        summary["temporal_regroup"][SIGNED_ENVELOPE_PROFILE]["observed_minus_null"]
        > 0
    )
    pair = next(row for row in result["pair_summary"] if row["entities"] == ["a", "b"])
    assert pair["classification"].endswith("_concurrent_pair")
    assert pair["attenuation_survival"]["concurrent_fractions"]


def test_pair_temporal_null_enumerates_small_permutation_space_once() -> None:
    windows = []
    for window_index in range(2):
        left = np.asarray([0.8, 0.1, 0.1]) if window_index == 0 else np.asarray([0.1, 0.8, 0.1])
        windows.append(
            {
                "segment": "validation",
                "group_index": 0,
                "window_index": window_index,
                "source": {"start": f"2026-01-01T0{window_index}:00:00+00:00"},
                "layers": [
                    {
                        "layer": "raw",
                            "spectral_profiles": [
                                {"entity": "a", "power": left.tolist()},
                                {"entity": "b", "power": left.tolist()},
                                {"entity": "c", "power": [0.1, 0.1, 0.8]},
                            ],
                    }
                ],
            }
        )

    result = _spectral_specificity(
        windows,
        supported_by_layer={"raw": ["spectral_alignment"]},
        null_repeats=99,
        seed=32,
        replication_separation_seconds=24.0 * 3600.0,
    )

    pair = next(row for row in result["pair_summary"] if row["entities"] == ["a", "b"])
    temporal = pair["raw_profile"]["temporal_regroup"]
    assert temporal["null_generation"] == "exact_permutation_space"
    assert temporal["permutation_space_size"] == 2
    assert temporal["null_repeats"] == 2
