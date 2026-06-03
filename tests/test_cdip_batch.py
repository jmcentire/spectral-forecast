"""Tests for CDIP batch experiment helpers."""

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cdip_batch import (
    _checkpoint_signature,
    _discover_windows,
    _load_window_manifest,
    _load_checkpoint,
    _multi_platform_emission_sum,
    _normal_survival_from_z,
    _permute_observation_result,
    _select_shard_items,
    _shift_observation_result,
    _window_manifest_signature,
    _write_window_manifest,
    _write_checkpoint,
    merge_reports,
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


def test_permute_observation_result_preserves_scores_and_changes_indexes():
    result = ObservationResult(
        series="a",
        points=[_point("a", 100), _point("a", 200), _point("a", 300), _point("a", 400)],
        baseline_size=100,
        adaptive_window=50,
        stride=100,
        residual_center=0.0,
        residual_scale=1.0,
    )

    permuted = _permute_observation_result(result, repeat=7)

    assert sorted(point.index for point in permuted.points) == [100, 200, 300, 400]
    assert [point.index for point in permuted.points] != [100, 200, 300, 400]
    assert [point.frozen_score for point in permuted.points] == [1.0, 1.0, 1.0, 1.0]


def test_multi_platform_emission_sum_ignores_single_platform_rows():
    rows = [
        {"active_platforms": "a", "emission_sum": 10.0},
        {"active_platforms": "a,b", "emission_sum": 2.5},
        {"active_platforms": "b,c", "emission_sum": 3.5},
    ]

    assert _multi_platform_emission_sum(rows) == 6.0


def test_select_shard_items_partitions_by_stable_index():
    items = list(range(10))
    shards = [_select_shard_items(items, 3, index) for index in range(3)]

    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(item for shard in shards for item in shard) == items
    with pytest.raises(ValueError, match="shard_count"):
        _select_shard_items(items, 0, 0)
    with pytest.raises(ValueError, match="shard_index"):
        _select_shard_items(items, 3, 3)


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
        null_mode="shift",
        preprocess="none",
        highpass_period_minutes=30.0,
        mask_dominant_bins=3,
        mask_bin_radius=1,
        phase_surrogate_seed=20260602,
        posthoc_window=None,
        shard_count=1,
        shard_index=0,
    )
    signature = _checkpoint_signature(args, ["z"])
    assert signature["mask_dominant_bins"] == 3
    assert signature["mask_bin_radius"] == 1
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


def test_window_manifest_round_trip_validates_signature(tmp_path):
    records = [_record(platform, 0.0, 1000) for platform in ["a", "b", "c"]]
    args = Namespace(
        files=[record.path for record in records],
        keep_flags=[2],
        group_size=3,
        group_strategy="balanced",
        max_groups=1,
        max_windows_per_group=2,
        max_total_windows=0,
        window_samples=256,
        window_step=128,
    )
    signature = _window_manifest_signature(args, ["z"], 128)
    windows = _discover_windows(
        records,
        group_size=3,
        min_clean_samples=128,
        window_samples=256,
        window_step=128,
        max_groups=1,
        max_windows_per_group=2,
        group_strategy="balanced",
    )
    path = tmp_path / "windows.json"

    _write_window_manifest(path, signature=signature, windows=windows)
    loaded = _load_window_manifest(
        path,
        signature=signature,
        records_by_path={str(record.path): record for record in records},
    )

    assert [window.platforms for window in loaded] == [window.platforms for window in windows]
    assert [window.start_time for window in loaded] == [window.start_time for window in windows]
    mismatched = dict(signature)
    mismatched["max_groups"] = 2
    with pytest.raises(ValueError, match="signature"):
        _load_window_manifest(
            path,
            signature=mismatched,
            records_by_path={str(record.path): record for record in records},
        )


def test_merge_reports_reconstructs_totals_and_combinations(tmp_path):
    parameters = {
        "preset": "custom",
        "files": ["a_xy.nc", "b_xy.nc"],
        "channels": ["z"],
        "keep_flags": [2],
        "group_size": 2,
        "group_strategy": "balanced",
        "max_groups": 2,
        "max_windows_per_group": 1,
        "max_total_windows": 0,
        "baseline": 8,
        "adaptive_window": 4,
        "stride": 4,
        "window_samples": 16,
        "window_step": 8,
        "min_clean_samples": 16,
        "score": "max",
        "emission_threshold": 3.0,
        "decay": 0.9,
        "null_repeats": 2,
        "null_mode": "shift",
        "preprocess": "none",
        "highpass_period_minutes": 30.0,
        "mask_dominant_bins": 3,
        "mask_bin_radius": 1,
        "phase_surrogate_seed": 20260602,
        "posthoc_window": 4,
        "shard_count": 2,
    }

    def report(
        shard_index: int,
        observed: float,
        null_totals: list[float],
        combo_count: int,
        combo_emission: float,
    ) -> dict[str, object]:
        window_row = {
            "group": ["a", "b"],
            "paths": ["a_xy.nc", "b_xy.nc"],
            "start": f"1970-01-01T00:00:0{shard_index}+00:00",
            "end": f"1970-01-01T00:00:1{shard_index}+00:00",
            "sample_rate": 1.0,
            "samples": 16,
            "observed_multi_emission": observed,
            "null_multi_emission_mean": float(np.mean(null_totals)),
            "null_multi_emission_std": float(np.std(null_totals)),
            "observed_minus_null": observed - float(np.mean(null_totals)),
            "top_combinations": [],
        }
        params = dict(parameters)
        params["shard_index"] = shard_index
        merge_state = {
            "aggregate_combinations": [
                {
                    "active_platforms": "a,b",
                    "count": combo_count,
                    "emission_sum": combo_emission,
                    "max_pheromone": combo_emission,
                    "max_score": observed,
                    "examples": [{"group": "a,b", "window_start": window_row["start"]}],
                }
            ],
            "metric_rows": [],
            "window_rows": [window_row],
            "observed_multi": [observed],
            "null_totals": null_totals,
            "skipped": [],
        }
        if shard_index == 0:
            merge_state["null_multi"] = null_totals
        else:
            merge_state["null_window_stats"] = {
                "count": float(len(null_totals)),
                "sum": float(np.sum(null_totals)),
                "sumsq": float(np.sum(np.asarray(null_totals) ** 2)),
            }
        return {
            "parameters": params,
            "summary": {
                "records_loaded": 2,
                "all_windows_discovered": 2,
                "windows_discovered": 1,
            },
            "merge_state": merge_state,
        }

    paths = []
    for index, payload in enumerate(
        [
            report(0, 10.0, [8.0, 9.0], 1, 5.0),
            report(1, 20.0, [12.0, 14.0], 2, 7.0),
        ]
    ):
        path = tmp_path / f"shard{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)

    merged = merge_reports(paths, top=10, top_windows=10)

    summary = merged["summary"]
    assert summary["all_windows_discovered"] == 2
    assert summary["windows_discovered"] == 2
    assert summary["windows_run"] == 2
    assert summary["observed_multi_emission_total"] == 30.0
    assert summary["null_multi_emission_total_estimate"] == 21.5
    assert summary["null_multi_emission_total_std"] == 1.5
    assert summary["null_window_emission_std"] == pytest.approx(np.std([8.0, 9.0, 12.0, 14.0]))
    assert summary["observed_total_z"] == pytest.approx((30.0 - 21.5) / 1.5)
    assert summary["null_total_empirical_p_floor"] == pytest.approx(1 / 3)
    assert summary["null_total_empirical_p_ge_observed"] == pytest.approx(1 / 3)
    assert summary["null_total_unique_repeats"] == 2
    assert summary["null_total_unique_empirical_p_floor"] == pytest.approx(1 / 3)
    assert merged["parameters"]["shard_index"] == "merged"
    assert merged["top_combinations"][0]["active_platforms"] == "a,b"
    assert merged["top_combinations"][0]["count"] == 3
    assert merged["top_combinations"][0]["emission_sum"] == 12.0
