"""Tests for the bounded CDIP relationship-discovery runner."""

import numpy as np

from experiments.cdip_relationship_discovery import (
    _aggregate,
    _layer_view,
    _parse_layers,
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
    assert summary["temporal_regroup"]["envelope_residual"]["detected"]
    assert summary["temporal_regroup"]["envelope_residual"]["observed_minus_null"] > 0
    pair = next(row for row in result["pair_summary"] if row["entities"] == ["a", "b"])
    assert pair["classification"] == "beyond_envelope_concurrent_pair"
