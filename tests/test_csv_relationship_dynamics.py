from experiments.csv_relationship_dynamics import detected_keys, parse_segment_spec


def test_parse_segment_spec() -> None:
    spec = parse_segment_spec("heldout:data/ETTh1.csv:8192:8192")

    assert spec.name == "heldout"
    assert str(spec.path) == "data/ETTh1.csv"
    assert spec.offset == 8192
    assert spec.limit == 8192


def test_detected_keys_keeps_only_corrected_results() -> None:
    result = {
        "evidence": [
            {"mode": "step", "metric": "exact_identity", "detected_fdr_by": True},
            {"mode": "step", "metric": "structural_role", "detected_fdr_by": False},
        ]
    }

    assert detected_keys(result) == {("step", "exact_identity")}
