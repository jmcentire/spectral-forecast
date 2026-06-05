from experiments.pmubage_relationship_dynamics import detected_keys, recurring_keys


def _result(*detected: tuple[str, str]) -> dict:
    rows = []
    for mode in ("step", "drift"):
        for metric in ("exact_identity", "structural_role", "global_motif"):
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "detected_fdr_by": (mode, metric) in detected,
                }
            )
    return {"evidence": rows}


def test_recurring_keys_requires_event_support() -> None:
    exact = ("step", "exact_identity")
    role = ("step", "structural_role")
    results = [_result(exact, role), _result(exact), _result(exact), _result()]

    assert detected_keys(results[0]) == {exact, role}
    assert recurring_keys(results, 0.5) == [exact]
