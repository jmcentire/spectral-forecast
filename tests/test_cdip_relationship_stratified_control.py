"""Tests for the acquisition-stratified CDIP relationship control."""

from experiments.cdip_relationship_stratified_control import (
    acquisition_stratum,
    collapse_exact_start_contexts,
)


def test_acquisition_stratum_combines_sample_rate_and_processing_family() -> None:
    assert acquisition_stratum(
        {"sample_rate": 1.279999971, "processing_family": "lx"}
    ) == "1.28:lx"


def _window(
    group_index: int,
    start: str,
    entities: list[str],
    *,
    target_sample_rate: float = 1.28,
) -> dict[str, object]:
    return {
        "segment": "validation",
        "group_index": group_index,
        "window_index": 0,
        "source": {
            "start": start,
            "end": "2026-01-01T01:00:00+00:00",
            "samples": 4096,
            "target_sample_rate": target_sample_rate,
            "sources": [
                {"entity": entity, "path": f"{entity}.nc"}
                for entity in entities
            ],
        },
        "layers": [
            {
                "layer": "raw",
                "spectral_profiles": [
                    {"entity": entity, "power": [float(ord(entity)), 1.0]}
                    for entity in entities
                ],
            },
            {
                "layer": "dominant:1.0",
                "spectral_profiles": [],
            },
        ],
    }


def test_collapse_exact_start_contexts_merges_and_deduplicates_raw_profiles() -> None:
    windows = [
        _window(0, "2026-01-01T00:00:00+00:00", ["a", "b", "c"]),
        _window(1, "2026-01-01T00:00:00+00:00", ["b", "c", "d"]),
        _window(2, "2026-01-02T00:00:00+00:00", ["a", "c", "d"]),
    ]

    collapsed, summary = collapse_exact_start_contexts(windows)

    assert len(collapsed) == 2
    assert collapsed[0]["entities"] == ["a", "b", "c", "d"]
    assert [layer["layer"] for layer in collapsed[0]["layers"]] == ["raw"]
    assert summary["overlapping_contexts"] == 1
    assert summary["duplicate_profiles_verified_equal"] == 2
    assert summary["entity_count_distribution"] == {"3": 1, "4": 1}


def test_collapse_exact_start_contexts_keeps_sampling_geometries_separate() -> None:
    windows = [
        _window(0, "2026-01-01T00:00:00+00:00", ["a", "b", "c"]),
        _window(
            1,
            "2026-01-01T00:00:00+00:00",
            ["a", "b", "d"],
            target_sample_rate=2.56,
        ),
    ]

    collapsed, summary = collapse_exact_start_contexts(windows)

    assert len(collapsed) == 2
    assert summary["overlapping_contexts"] == 0
    assert summary["starts_with_multiple_sampling_geometries"] == 1
