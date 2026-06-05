from experiments.merge_relationship_families import apply_by_fdr


def test_cross_family_by_correction_covers_all_rows() -> None:
    rows = [
        {
            "empirical_p_ge_observed": 0.001 if index == 0 else 0.5,
            "observed_minus_null": 1.0,
        }
        for index in range(18)
    ]

    apply_by_fdr(rows)

    assert not rows[0]["detected_cross_family_fdr_by"]
    assert rows[0]["cross_family_fdr_by_q_value"] > 0.05
