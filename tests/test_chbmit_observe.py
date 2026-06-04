from pathlib import Path

import numpy as np
import pytest

from experiments.chbmit_observe import (
    _multi_channel_emissions,
    parse_chbmit_summary,
    permutation_null_summary,
)
from spectral_forecast.observation import observe_series


def test_parse_chbmit_summary_extracts_seizure_intervals(tmp_path: Path) -> None:
    summary = tmp_path / "chbxx-summary.txt"
    summary.write_text(
        """Data Sampling Rate: 256 Hz

File Name: chbxx_01.edf
Number of Seizures in File: 0

File Name: chbxx_02.edf
Number of Seizures in File: 1
Seizure Start Time: 12 seconds
Seizure End Time: 34 seconds

File Name: chbxx_03.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 56 seconds
Seizure 1 End Time: 78 seconds
Seizure 2 Start Time: 90 seconds
Seizure 2 End Time: 123 seconds
""",
        encoding="utf-8",
    )

    labels = parse_chbmit_summary(summary)

    assert labels["chbxx_01.edf"].seizures == ()
    assert labels["chbxx_02.edf"].seizures == ((12.0, 34.0),)
    assert labels["chbxx_03.edf"].seizures == ((56.0, 78.0), (90.0, 123.0))


def test_multi_channel_emissions_require_minimum_active_channels() -> None:
    matrix = np.array(
        [
            [4.0, 2.0, 1.0],
            [4.0, 5.0, 2.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )

    emissions = _multi_channel_emissions(matrix, threshold=3.0, min_active_channels=2)

    assert emissions.tolist() == pytest.approx([0.0, 3.0, 6.0])


def test_permutation_null_reports_exceedance_count() -> None:
    base = np.sin(np.linspace(0, 8 * np.pi, 160))
    results = [
        observe_series(base + offset, series=f"s{offset}", baseline_size=48, adaptive_window=24, stride=8)
        for offset in (0.0, 0.1, 0.2)
    ]

    summary = permutation_null_summary(
        results,
        score="max",
        threshold=1.0,
        min_active_channels=2,
        repeats=5,
        seed=1,
    )

    assert summary["null_repeats"] == 5
    assert "null_exceedances" in summary
    assert 0 <= summary["null_exceedances"] <= 5
