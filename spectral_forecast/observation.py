"""Observational anomaly and drift tooling.

This module does not decide what an anomaly means. It produces comparable
signals from past-only baselines:

- frozen residual score: deviation from an initial spectral nominal
- sliding residual score: deviation from a local adaptive spectral nominal
- conditional drift: frozen score minus sliding score
- state drift: robust distance from baseline decomposition states
- stigmergy: decayed accumulation of anomaly emissions across series
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.forecast import SpectralForecaster


ScoreName = Literal["frozen", "sliding", "drift", "state", "max"]


@dataclass
class DecompositionState:
    """Internal state produced by the spectral decomposition."""

    component_count: float
    component_amplitude_sum: float
    component_power_sum: float
    component_snr_mean: float
    component_snr_max: float
    residual_std: float
    residual_ac1: float
    trend_slope: float
    shock_count: float
    local_order: float
    wavelet_used: float


@dataclass
class ObservationPoint:
    """One anchor-time observation for one series."""

    series: str
    index: int
    actual: float
    frozen_prediction: float
    sliding_prediction: float
    frozen_residual: float
    sliding_residual: float
    frozen_score: float
    sliding_score: float
    conditional_drift_score: float
    state_drift_score: float
    state: DecompositionState

    def score(self, name: ScoreName) -> float:
        """Return the requested score for ranking or emission."""
        if name == "frozen":
            return self.frozen_score
        if name == "sliding":
            return self.sliding_score
        if name == "drift":
            return self.conditional_drift_score
        if name == "state":
            return self.state_drift_score
        if name == "max":
            return max(
                self.frozen_score,
                self.sliding_score,
                self.conditional_drift_score,
                self.state_drift_score,
            )
        raise ValueError(f"Unknown score: {name}")

    def to_dict(self) -> dict[str, object]:
        """Flatten for CSV/JSON output."""
        row = asdict(self)
        state = row.pop("state")
        for key, value in state.items():
            row[f"state_{key}"] = value
        return row


@dataclass
class ObservationResult:
    """Observation results for one series."""

    series: str
    points: list[ObservationPoint]
    baseline_size: int
    adaptive_window: int
    stride: int
    residual_center: float
    residual_scale: float

    def top(self, score: ScoreName = "max", n: int = 10) -> list[ObservationPoint]:
        """Return the highest-scoring points."""
        return sorted(self.points, key=lambda p: p.score(score), reverse=True)[:n]


@dataclass
class StigmergyPoint:
    """Cross-series decayed anomaly evidence at one index."""

    index: int
    pheromone: float
    emission: float
    active_series_count: int
    active_series: tuple[str, ...]
    max_score: float
    mean_score: float

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["active_series"] = ",".join(self.active_series)
        return row


@dataclass
class StigmergyResult:
    """Cross-series stigmergic accumulation."""

    points: list[StigmergyPoint]
    score: ScoreName
    emission_threshold: float
    decay: float

    def top(self, n: int = 10) -> list[StigmergyPoint]:
        """Return the strongest pheromone points."""
        return sorted(self.points, key=lambda p: p.pheromone, reverse=True)[:n]


def _robust_center_scale(values: NDArray[np.floating]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if scale < 1e-10:
        scale = float(np.std(finite))
    if scale < 1e-10:
        scale = 1.0
    return center, scale


def _abs_robust_z(value: float, center: float, scale: float) -> float:
    return abs((float(value) - center) / max(scale, 1e-10))


def decomposition_state(model: SpectralForecaster) -> DecompositionState:
    """Extract agnostic internal state from a fitted forecaster."""
    if model._extraction is None or model._trend is None or model._local is None:
        raise RuntimeError("Forecaster must be fit before extracting state")

    components = model._extraction.components
    amplitudes = np.array([c.amplitude for c in components], dtype=np.float64)
    snrs = np.array([c.snr for c in components], dtype=np.float64)
    if len(amplitudes) == 0:
        amplitudes = np.array([], dtype=np.float64)
    if len(snrs) == 0:
        snrs = np.array([], dtype=np.float64)

    trend_slope = 0.0
    params = model._trend.model.params
    if "a" in params:
        trend_slope = float(params["a"])
    elif "b" in params:
        trend_slope = float(params["b"])

    return DecompositionState(
        component_count=float(len(components)),
        component_amplitude_sum=float(np.sum(np.abs(amplitudes))) if len(amplitudes) else 0.0,
        component_power_sum=float(np.sum(amplitudes**2)) if len(amplitudes) else 0.0,
        component_snr_mean=float(np.mean(snrs)) if len(snrs) else 0.0,
        component_snr_max=float(np.max(snrs)) if len(snrs) else 0.0,
        residual_std=float(np.std(model._residual)) if model._residual is not None else 0.0,
        residual_ac1=float(model._residual_ac1),
        trend_slope=trend_slope,
        shock_count=float(len(model._shocks.shocks)) if model._shocks is not None else 0.0,
        local_order=float(model._local.model.order),
        wavelet_used=1.0 if model._wavelet is not None else 0.0,
    )


def _state_matrix(states: Iterable[DecompositionState]) -> NDArray[np.floating]:
    rows = []
    for state in states:
        rows.append(
            [
                state.component_count,
                state.component_amplitude_sum,
                state.component_power_sum,
                state.component_snr_mean,
                state.component_snr_max,
                state.residual_std,
                state.residual_ac1,
                state.trend_slope,
                state.shock_count,
                state.local_order,
                state.wavelet_used,
            ]
        )
    return np.array(rows, dtype=np.float64)


def _baseline_state_reference(
    data: NDArray[np.floating],
    baseline_size: int,
    adaptive_window: int,
    sample_rate: float,
    forecaster_kwargs: dict[str, object],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    starts = range(0, baseline_size - adaptive_window + 1, max(1, adaptive_window // 4))
    states: list[DecompositionState] = []
    for start in starts:
        model = SpectralForecaster(sample_rate=sample_rate, **forecaster_kwargs)
        model.fit(data[start : start + adaptive_window])
        states.append(decomposition_state(model))
    if not states:
        model = SpectralForecaster(sample_rate=sample_rate, **forecaster_kwargs)
        model.fit(data[:baseline_size])
        states.append(decomposition_state(model))
    matrix = _state_matrix(states)
    centers = np.median(matrix, axis=0)
    scales = np.array([_robust_center_scale(matrix[:, i])[1] for i in range(matrix.shape[1])])
    scales = np.maximum(scales, 1e-10)
    return centers, scales


def _state_drift_score(state: DecompositionState, centers: NDArray, scales: NDArray) -> float:
    row = _state_matrix([state])[0]
    z = np.abs((row - centers) / scales)
    # Median avoids one unstable internal field dominating the whole observation.
    return float(np.median(z))


def observe_series(
    values: NDArray[np.floating],
    series: str = "series",
    baseline_size: int = 512,
    adaptive_window: int = 256,
    stride: int = 16,
    sample_rate: float = 1.0,
    forecaster_kwargs: dict[str, object] | None = None,
) -> ObservationResult:
    """Observe anomaly and drift signals for a single time series.

    All scores are computed from past-only information. The frozen nominal is
    fit on ``values[:baseline_size]`` once. The sliding nominal is refit on
    ``values[index-adaptive_window:index]`` for each anchor.
    """
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("Series contains NaN or Inf")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if baseline_size <= adaptive_window:
        raise ValueError("baseline_size must be greater than adaptive_window")
    if adaptive_window <= 8:
        raise ValueError("adaptive_window must be greater than 8")
    if len(y) <= max(baseline_size, adaptive_window):
        raise ValueError("Series is too short for the requested windows")

    kwargs = forecaster_kwargs or {}
    start_index = max(baseline_size, adaptive_window)
    anchors = list(range(start_index, len(y), stride))
    if not anchors:
        raise ValueError("No observation anchors for the requested windows")

    frozen = SpectralForecaster(sample_rate=sample_rate, **kwargs)
    frozen.fit(y[:baseline_size])
    frozen_horizon = max(anchors) - baseline_size + 1
    frozen_forecast = frozen.forecast(frozen_horizon).point_forecast
    resid_center, resid_scale = _robust_center_scale(frozen._residual)
    state_centers, state_scales = _baseline_state_reference(
        y, baseline_size, adaptive_window, sample_rate, kwargs
    )

    points: list[ObservationPoint] = []
    for anchor in anchors:
        frozen_pred = float(frozen_forecast[anchor - baseline_size])
        actual = float(y[anchor])
        frozen_resid = actual - frozen_pred
        frozen_score = _abs_robust_z(frozen_resid, resid_center, resid_scale)

        adaptive_context = y[anchor - adaptive_window : anchor]
        sliding = SpectralForecaster(sample_rate=sample_rate, **kwargs)
        sliding.fit(adaptive_context)
        sliding_pred = float(sliding.forecast(1).point_forecast[0])
        sliding_resid = actual - sliding_pred
        sliding_center, sliding_scale = _robust_center_scale(sliding._residual)
        sliding_score = _abs_robust_z(sliding_resid, sliding_center, sliding_scale)

        state = decomposition_state(sliding)
        state_score = _state_drift_score(state, state_centers, state_scales)

        points.append(
            ObservationPoint(
                series=series,
                index=anchor,
                actual=actual,
                frozen_prediction=frozen_pred,
                sliding_prediction=sliding_pred,
                frozen_residual=frozen_resid,
                sliding_residual=sliding_resid,
                frozen_score=frozen_score,
                sliding_score=sliding_score,
                conditional_drift_score=frozen_score - sliding_score,
                state_drift_score=state_score,
                state=state,
            )
        )

    return ObservationResult(
        series=series,
        points=points,
        baseline_size=baseline_size,
        adaptive_window=adaptive_window,
        stride=stride,
        residual_center=resid_center,
        residual_scale=resid_scale,
    )


def build_stigmergy(
    results: list[ObservationResult],
    score: ScoreName = "max",
    emission_threshold: float = 3.0,
    decay: float = 0.9,
) -> StigmergyResult:
    """Accumulate cross-series anomaly evidence with exponential decay."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    by_index: dict[int, list[ObservationPoint]] = {}
    for result in results:
        for point in result.points:
            by_index.setdefault(point.index, []).append(point)

    pheromone = 0.0
    out: list[StigmergyPoint] = []
    for index in sorted(by_index):
        points = by_index[index]
        scores = np.array([p.score(score) for p in points], dtype=np.float64)
        active = [p.series for p, s in zip(points, scores) if s > emission_threshold]
        emissions = np.maximum(scores - emission_threshold, 0.0)
        emission = float(np.sum(emissions))
        pheromone = decay * pheromone + emission
        out.append(
            StigmergyPoint(
                index=index,
                pheromone=pheromone,
                emission=emission,
                active_series_count=len(active),
                active_series=tuple(active),
                max_score=float(np.max(scores)) if len(scores) else 0.0,
                mean_score=float(np.mean(scores)) if len(scores) else 0.0,
            )
        )

    return StigmergyResult(
        points=out,
        score=score,
        emission_threshold=emission_threshold,
        decay=decay,
    )


def load_csv_series(
    path: str | Path,
    columns: list[str] | None = None,
    all_numeric: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """Load one or more numeric columns from a CSV file."""
    path = Path(path)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")

        fieldnames = list(reader.fieldnames)
        wanted = columns or []
        if all_numeric:
            wanted = []
        elif not wanted:
            wanted = ["OT"] if "OT" in fieldnames else [fieldnames[-1]]

        values: dict[str, list[float]] = {name: [] for name in wanted}
        numeric_candidates: dict[str, list[float]] = {name: [] for name in fieldnames}
        numeric_valid: dict[str, bool] = {name: True for name in fieldnames}

        for row in reader:
            if all_numeric:
                for name in fieldnames:
                    try:
                        numeric_candidates[name].append(float(row[name]))
                    except (TypeError, ValueError):
                        numeric_valid[name] = False
                continue

            for name in wanted:
                if name not in row:
                    raise ValueError(f"Column not found: {name}")
                try:
                    values[name].append(float(row[name]))
                except (TypeError, ValueError):
                    pass

    if all_numeric:
        values = {
            name: vals
            for name, vals in numeric_candidates.items()
            if numeric_valid[name] and vals
        }

    out = {
        name: np.asarray(vals, dtype=np.float64)
        for name, vals in values.items()
        if vals
    }
    if not out:
        raise ValueError("No numeric series found")
    return out


def observation_rows_jsonl(results: list[ObservationResult]) -> str:
    """Serialize observation points as JSONL."""
    rows = []
    for result in results:
        rows.extend(json.dumps(point.to_dict(), sort_keys=True) for point in result.points)
    return "\n".join(rows)


def stigmergy_rows_jsonl(result: StigmergyResult) -> str:
    """Serialize stigmergy points as JSONL."""
    return "\n".join(json.dumps(point.to_dict(), sort_keys=True) for point in result.points)
