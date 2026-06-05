from experiments.csv_subset_audit import apply_by_fdr, replicated_subset_names


def test_apply_by_fdr_requires_positive_effect() -> None:
    rows = [
        {"empirical_p_ge_observed": 0.001, "observed_minus_null": 1.0},
        {"empirical_p_ge_observed": 0.01, "observed_minus_null": -1.0},
        {"empirical_p_ge_observed": 0.5, "observed_minus_null": 1.0},
    ]

    apply_by_fdr(rows)

    assert rows[0]["detected_fdr_by"]
    assert not rows[1]["detected_fdr_by"]
    assert not rows[2]["detected_fdr_by"]


def test_replicated_subset_names_requires_exact_corrected_replication() -> None:
    segments = {
        "a": [
            {"subset": ["x", "y", "z"], "detected_fdr_by": True},
            {"subset": ["a", "b", "c"], "detected_fdr_by": True},
        ],
        "b": [
            {"subset": ["x", "y", "z"], "detected_fdr_by": True},
            {"subset": ["a", "b", "c"], "detected_fdr_by": False},
        ],
    }

    assert replicated_subset_names(segments) == [["x", "y", "z"]]
