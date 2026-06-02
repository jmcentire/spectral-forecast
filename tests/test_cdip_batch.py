"""Tests for CDIP batch experiment helpers."""

from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest

from experiments.cdip_batch import (
    _checkpoint_signature,
    _discover_windows,
    _load_checkpoint,
    _multi_platform_emission_sum,
    _normal_survival_from_z,
    _shift_observation_result,
    _write_checkpoint,
)
from experiments.cdip_observe import CdipRawRecord
from spectral_forecast.observation import ObservationPoint, ObservationResult


def _point(series: str, index: int) -> ObservationPoint:
    return ObservationPoint(
        series=series,
        index=index,
        actual=0.0,
        frozen_prediction=0.0,
        sliding_prediction=0.0,
        frozen_residual=0.0,
        sliding_residual=0.0,
        frozen_score=1.0,
        sliding_score=0.0,
        conditional_drift_score=0.0,
        state_drift_score=0.0,
        state=None,  # type: ignore[arg-type]
    )


def _record(platform: str, start: float, n: int, rate: float = 1.0) -> CdipRawRecord:
    return CdipRawRecord(
        path=Path(f"{platform}_xy.nc"),
        arrays={"z": np.zeros(n, dtype=np.float64)},
        valid=np.ones(n, dtype=bool),
        sample_rate=rate,
        first_sample_time=start,
        station_id=platform[:3],
        platform_id=platform,
        platform_name=platform,
    )


def test_shift_observation_result_wraps_anchor_indexes():
    result = ObservationResult(
        series="a",
        points=[_point("a", 100), _point("a", 200), _point("a", 300)],
        baseline_size=100,
        adaptive_window=50,
        stride=100,
        residual_center=0.0,
        residual_scale=1.0,
    )

    shifted = _shift_observation_result(result, stride=100, shift_steps=1)

    assert [point.index for point in shifted.points] == [200, 300, 100]


def test_multi_platform_emission_sum_ignores_single_platform_rows():
    rows = [
        {"active_platforms": "a", "emission_sum": 10.0},
        {"active_platforms": "a,b", "emission_sum": 2.5},
        {"active_platforms": "b,c", "emission_sum": 3.5},
    ]

    assert _multi_platform_emission_sum(rows) == 6.0


def test_normal_survival_from_z_is_one_sided_tail():
    assert _normal_survival_from_z(0.0) == 0.5
    assert 0.0004 < _normal_survival_from_z(3.28) < 0.0006
    assert _normal_survival_from_z(float("nan")) == 1.0


def test_checkpoint_round_trip_validates_signature(tmp_path):
    args = Namespace(
        files=[Path("a_xy.nc")],
        keep_flags=[2],
        group_size=3,
        group_strategy="balanced",
        max_groups=32,
        max_windows_per_group=8,
        max_total_windows=0,
        baseline=1024,
        adaptive_window=512,
        stride=512,
        window_samples=4096,
        window_step=2048,
        min_clean_samples=None,
        score="max",
        emission_threshold=3.0,
        decay=0.9,
        null_repeats=50,
        posthoc_window=None,
    )
    signature = _checkpoint_signature(args, ["z"])
    path = tmp_path / "checkpoint.json"

    _write_checkpoint(path, {"signature": signature, "next_window_index": 7})

    assert _load_checkpoint(path, signature)["next_window_index"] == 7
    mismatched = dict(signature)
    mismatched["max_groups"] = 64
    with pytest.raises(ValueError, match="signature"):
        _load_checkpoint(path, mismatched)


def test_discover_windows_finds_overlapping_record_groups():
    records = [
        _record("a", 0.0, 1000),
        _record("b", 100.0, 1000),
        _record("c", 200.0, 1000),
    ]

    windows = _discover_windows(
        records,
        group_size=3,
        min_clean_samples=128,
        window_samples=256,
        window_step=128,
        max_groups=2,
        max_windows_per_group=2,
    )

    assert len(windows) == 2
    assert windows[0].platforms == ("a", "b", "c")
    assert windows[0].start_time == 200.0


def test_discover_windows_balanced_spreads_platform_usage():
    records = [_record(platform, 0.0, 1000) for platform in ["a", "b", "c", "d"]]

    windows = _discover_windows(
        records,
        group_size=2,
        min_clean_samples=128,
        window_samples=256,
        window_step=128,
        max_groups=2,
        max_windows_per_group=1,
        group_strategy="balanced",
    )

    assert len(windows) == 2
    assert set(windows[0].platforms).isdisjoint(windows[1].platforms)
