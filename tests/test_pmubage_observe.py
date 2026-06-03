import numpy as np
import pytest

from experiments.pmubage_observe import DATATYPES, robust_standardize


def test_datatype_axis_matches_pmubage_readme_order() -> None:
    assert DATATYPES == ("real_power", "reactive_power", "voltage_magnitude", "frequency")


def test_robust_standardize_centers_and_scales_signal() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=np.float64)

    scaled = robust_standardize(values)

    assert float(np.median(scaled)) == pytest.approx(0.0)
    assert scaled[-1] > 10.0
