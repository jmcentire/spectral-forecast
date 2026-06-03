"""Tests for CDIP-specific autotune aggregation."""

import numpy as np

from experiments.cdip_batch import BatchWindow
from experiments.cdip_autotune import (
    CdipAutoTuneCandidate,
    MatrixCache,
    _aggregate_scores,
    _combine_null_mode_reports,
    _phase_surrogate_preprocess,
    _take_group_windows,
)
from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneNullSummary,
    AutoTuneObservation,
    AutoTuneScore,
)


def _score(
    *,
    accepted_quality: bool = True,
    observed: float = 20.0,
    null: float = 10.0,
    z: float = 2.0,
) -> AutoTuneScore:
    config = AutoTuneConfig(
        baseline_size=64,
        adaptive_window=32,
        stride=8,
        emission_threshold=3.0,
        decay=0.9,
        min_active_series=2,
    )
    return AutoTuneScore(
        config=config,
        quality=0.2 if accepted_quality else -0.1,
        readiness_score=1.0,
        null_lift_score=0.5 if observed > null else 0.0,
        stability_score=0.5,
        compression_score=0.5,
        residual_activity_score=0.5,
        saturation_penalty=0.2,
        fragility_penalty=0.0,
        null_summary=AutoTuneNullSummary(
            anchors=10,
            observed_total=observed,
            observed_active_windows=3,
            null_repeats=5,
            null_mean=null,
            null_std=(observed - null) / z if z else 1.0,
            observed_minus_null=observed - null,
            z_effect=z,
            null_exceedances=0,
            empirical_p_ge_observed=1 / 6,
            empirical_p_floor=1 / 6,
            unique_null_totals=5,
        ),
    )


def test_aggregate_rejects_positive_delta_with_too_few_accepted_windows() -> None:
    candidate = CdipAutoTuneCandidate(
        preprocess="highpass",
        config=AutoTuneConfig(baseline_size=64, adaptive_window=32, stride=8),
    )

    report = _aggregate_scores(
        candidate,
        [_score(accepted_quality=True), _score(accepted_quality=False)],
        skipped=[],
        min_accepted_fraction=0.75,
        min_z_effect=1.5,
    )

    assert report["observed_minus_null_total"] > 0.0
    assert report["accepted"] is False


def test_aggregate_accepts_when_lift_fraction_and_z_clear_gate() -> None:
    candidate = CdipAutoTuneCandidate(
        preprocess="highpass",
        config=AutoTuneConfig(baseline_size=64, adaptive_window=32, stride=8),
    )

    report = _aggregate_scores(
        candidate,
        [_score(), _score()],
        skipped=[],
        min_accepted_fraction=0.5,
        min_z_effect=1.5,
    )

    assert report["accepted"] is True


def test_aggregate_uses_repeat_aligned_batch_null_totals() -> None:
    candidate = CdipAutoTuneCandidate(
        preprocess="highpass",
        config=AutoTuneConfig(baseline_size=64, adaptive_window=32, stride=8),
    )

    report = _aggregate_scores(
        candidate,
        [
            _score(observed=20.0, null=10.0, z=2.0),
            _score(observed=30.0, null=10.0, z=2.0),
        ],
        skipped=[],
        min_accepted_fraction=0.0,
        min_positive_window_fraction=0.5,
        min_z_effect=1.5,
        null_totals_by_window=[
            [9.0, 10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
    )

    assert report["observed_total"] == 50.0
    assert report["null_mean_total"] == 22.0
    assert report["null_total_repeats"] == 4
    assert report["null_total_exceedances"] == 0
    assert report["null_total_empirical_p_floor"] == 0.2
    assert report["accepted"] is True


def test_combined_null_report_requires_every_null_family_to_pass() -> None:
    candidate = CdipAutoTuneCandidate(
        preprocess="highpass",
        config=AutoTuneConfig(baseline_size=64, adaptive_window=32, stride=8),
    )
    passing = _aggregate_scores(
        candidate,
        [_score(), _score()],
        skipped=[],
        min_accepted_fraction=0.5,
        min_z_effect=1.5,
    )
    passing["null_mode"] = "permute"
    failing = _aggregate_scores(
        candidate,
        [_score(observed=5.0, null=10.0, z=-1.0)],
        skipped=[],
        min_accepted_fraction=0.5,
        min_z_effect=1.5,
    )
    failing["null_mode"] = "shift"

    combined = _combine_null_mode_reports(candidate, [passing, failing])

    assert combined["accepted"] is False
    assert combined["worst_null_mode"] == "shift"
    assert combined["observed_minus_null_total"] < 0.0


def test_take_group_windows_selects_disjoint_group_ranges() -> None:
    windows = [
        BatchWindow(
            records=(),
            start_time=float(group * 10 + window_index),
            n_samples=16,
            target_rate=1.0,
            group_index=group,
            window_index=window_index,
        )
        for group in range(4)
        for window_index in range(3)
    ]

    selected = _take_group_windows(
        windows,
        group_start=2,
        group_count=2,
        window_offset=1,
        windows_per_group=1,
    )

    assert [(row.group_index, row.window_index) for row in selected] == [(2, 1), (3, 1)]


def test_matrix_cache_round_trips_observation_to_disk(tmp_path) -> None:
    observation = AutoTuneObservation(
        anchors=[4, 8],
        matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        readiness_score=0.75,
        series_count=2,
    )
    key = {"window": "a", "config": {"baseline": 64}}

    writer = MatrixCache(directory=tmp_path)
    assert writer.get(key) is None
    writer.put(key, observation)
    reader = MatrixCache(directory=tmp_path)
    loaded = reader.get(key)

    assert loaded is not None
    assert loaded.anchors == observation.anchors
    assert np.allclose(loaded.matrix, observation.matrix)
    assert loaded.readiness_score == observation.readiness_score
    assert reader.disk_hits == 1


def test_phase_surrogate_preprocess_preserves_matching_controls() -> None:
    assert _phase_surrogate_preprocess("none") == "phase-randomize"
    assert _phase_surrogate_preprocess("highpass") == "highpass-phase-randomize"
    assert (
        _phase_surrogate_preprocess("highpass-dominant-mask")
        == "highpass-dominant-mask-phase-randomize"
    )
