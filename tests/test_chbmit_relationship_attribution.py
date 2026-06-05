from experiments.chbmit_relationship_attribution import seizure_distance


def test_seizure_distance_handles_inside_and_outside_intervals() -> None:
    intervals = ((10.0, 20.0), (50.0, 60.0))

    assert seizure_distance(15.0, intervals) == 0.0
    assert seizure_distance(35.0, intervals) == 15.0
    assert seizure_distance(1.0, ()) is None
