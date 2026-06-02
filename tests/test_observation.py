"""Tests for observational anomaly/drift tooling."""

import numpy as np

from spectral_forecast.observation import (
    ObservationPoint,
    ObservationResult,
    build_stigmergy,
    decomposition_state,
    observe_series,
)
from spectral_forecast.forecast import SpectralForecaster


def _stable_then_shifted(n=420):
    t = np.arange(n, dtype=np.float64)
    signal = 2.0 * np.sin(2 * np.pi * 0.05 * t)
    signal[260:] += np.linspace(0.0, 4.0, n - 260)
    return signal


def test_decomposition_state_from_fitted_model():
    t = np.arange(240, dtype=np.float64)
    signal = 3.0 * np.sin(2 * np.pi * 0.04 * t)
    model = SpectralForecaster()
    model.fit(signal)

    state = decomposition_state(model)

    assert state.component_count >= 1
    assert state.component_amplitude_sum > 0
    assert state.residual_std >= 0


def test_observe_series_surfaces_synthetic_shift():
    signal = _stable_then_shifted()

    result = observe_series(
        signal,
        series="synthetic",
        baseline_size=160,
        adaptive_window=96,
        stride=8,
    )

    early = [p.frozen_score for p in result.points if p.index < 240]
    late = [p.frozen_score for p in result.points if p.index >= 300]

    assert result.points
    assert max(late) > max(early) * 2.0
    assert any(p.index >= 260 for p in result.top("drift", 5))


def test_stigmergy_accumulates_cross_series_emissions():
    base = _stable_then_shifted()
    shifted_later = np.roll(base, 16)

    r1 = observe_series(base, "a", baseline_size=160, adaptive_window=96, stride=8)
    r2 = observe_series(shifted_later, "b", baseline_size=160, adaptive_window=96, stride=8)
    stig = build_stigmergy([r1, r2], score="max", emission_threshold=3.0, decay=0.8)

    top = stig.top(5)

    assert top
    assert top[0].pheromone > 0
    assert any(point.active_series_count > 0 for point in stig.points)


def test_stigmergy_ignores_nonfinite_scores():
    point = ObservationPoint(
        series="bad",
        index=1,
        actual=0.0,
        frozen_prediction=0.0,
        sliding_prediction=0.0,
        frozen_residual=0.0,
        sliding_residual=0.0,
        frozen_score=float("nan"),
        sliding_score=0.0,
        conditional_drift_score=0.0,
        state_drift_score=0.0,
        state=None,  # type: ignore[arg-type]
    )
    result = ObservationResult(
        series="bad",
        points=[point],
        baseline_size=10,
        adaptive_window=5,
        stride=1,
        residual_center=0.0,
        residual_scale=1.0,
    )

    stig = build_stigmergy([result], score="max", emission_threshold=3.0)

    assert stig.points[0].emission == 0.0
    assert stig.points[0].active_series_count == 0


def test_observation_scores_are_past_only_shapes():
    signal = _stable_then_shifted()
    result = observe_series(
        signal,
        series="shape",
        baseline_size=150,
        adaptive_window=80,
        stride=10,
    )

    point = result.points[0]
    assert point.index == 150
    assert np.isfinite(point.frozen_score)
    assert np.isfinite(point.sliding_score)
    assert np.isfinite(point.conditional_drift_score)
    assert np.isfinite(point.state_drift_score)
