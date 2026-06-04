"""Tests for the organizational-network directional-quality adapter."""

import argparse
from pathlib import Path

from experiments.org_network_directional_quality import (
    compare_candidate_to_controls,
    run_org_network_directional_quality,
    run_synthetic_validation,
)
from experiments.org_network_directional_order_null import summarize_order_null


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=None,
        download=False,
        force_download=False,
        data_dir=tmp_path,
        edges=tmp_path / "edges.txt",
        labels=tmp_path / "labels.txt",
        data_format="edges",
        simplex_prefix="email-Enron",
        source_col=0,
        target_col=1,
        time_col=2,
        directed=False,
        max_edges=0,
        max_bins=0,
        bin_seconds=1.0,
        event_bin_size=4,
        top_nodes=6,
        top_groups=2,
        no_global_series=False,
        no_node_series=False,
        no_group_series=False,
        no_relation_series=False,
        relation_only=False,
        surface_mode="group",
        event_order="real-time",
        transform="log-robust",
        validation_start_fraction=0.5,
        null_repeats=8,
        synthetic_null_repeats=16,
        null_block_size=4,
        active_z_threshold=1.5,
        max_lag=6,
        aggregate_quantile=0.9,
        max_series=32,
        significance_level=0.1,
        min_z_effect=1.0,
        skip_synthetic_validation=True,
        seed=17,
        output=None,
        format="json",
    )


def test_synthetic_validation_recovers_known_mechanisms(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.synthetic_null_repeats = 32
    args.null_block_size = 8
    args.max_lag = 16
    args.significance_level = 0.05
    args.min_z_effect = 2.0

    report = run_synthetic_validation(args)

    assert report["passed"]


def test_org_directional_quality_reports_calibration_validation_and_replication(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    rows = []
    for index in range(96):
        left = "a" if index % 4 < 2 else "d"
        right = "b" if left == "a" else "e"
        rows.append(f"{left} {right} {index}")
    args.edges.write_text("\n".join(rows) + "\n", encoding="utf-8")
    args.labels.write_text("a G1 b G1 d G2 e G2", encoding="utf-8")

    report = run_org_network_directional_quality(args)

    assert report["calibration_bins"] == 48
    assert report["validation_bins"] == 48
    assert set(report["calibration"]["metrics"]) == {
        "coactivation",
        "exclusion",
        "lagged_succession",
        "phase_offset",
    }
    assert report["replication"]["grade"]


def test_candidate_control_comparison_distinguishes_selective_mechanism() -> None:
    def report(stable: list[str], coactivation_delta: float, phase_delta: float):
        metrics = {
            "coactivation": {"observed_minus_null": coactivation_delta, "z_effect": 4.0},
            "phase_offset": {"observed_minus_null": phase_delta, "z_effect": 3.0},
        }
        return {
            "replication": {"stable_detected_mechanisms": stable},
            "calibration": {"metrics": metrics},
            "validation": {"metrics": metrics},
        }

    comparison = compare_candidate_to_controls(
        report(["coactivation", "phase_offset"], 0.4, 0.3),
        [report(["coactivation"], 0.2, -0.1)],
    )

    assert comparison["grade"] == "selective_replicated_structure"
    assert comparison["selective_candidate_mechanisms"] == ["phase_offset"]
    assert comparison["shared_but_descriptively_stronger_mechanisms"] == ["coactivation"]


def test_order_null_summary_confirms_replicated_candidate_above_controls() -> None:
    metrics = {
        "coactivation": {"observed_minus_null": 0.4, "z_effect": 4.0},
        "phase_offset": {"observed_minus_null": 0.3, "z_effect": 3.0},
    }
    candidate = {
        "replication": {"stable_detected_mechanisms": ["coactivation", "phase_offset"]},
        "calibration": {"metrics": metrics},
        "validation": {"metrics": metrics},
    }

    summary = summarize_order_null(
        candidate,
        {
            "coactivation": [0.1, 0.12, 0.15, 0.2],
            "phase_offset": [0.1, 0.35, 0.4, 0.45],
        },
        significance_level=0.25,
    )

    assert summary["grade"] == "confirmed_order_sensitive_structure"
    assert summary["confirmed_order_sensitive_mechanisms"] == ["coactivation"]
