"""Tests for CDIP-specific autotune aggregation."""

from spectral_forecast.autotune import (
    AutoTuneConfig,
    AutoTuneNullSummary,
    AutoTuneScore,
)
from experiments.cdip_autotune import CdipAutoTuneCandidate, _aggregate_scores


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
