"""ForecastEngine: unified ensemble forecasting tool.

One tool. Multiple workers consuming the same observation feed. Each worker
tracks its own confidence via running accuracy. Predictions are confidence-
weighted blends with calibrated uncertainty from inter-worker disagreement.

Usage:
    engine = ForecastEngine()
    engine.observe_batch(historical_data)
    result = engine.predict(horizon=96)
    # result.values, result.lower, result.upper, result.confidence
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Prediction:
    """Forecast result from a single worker or the ensemble."""

    values: NDArray[np.floating]
    confidence: float  # 0-1
    lower: NDArray[np.floating] | None = None
    upper: NDArray[np.floating] | None = None


@dataclass
class EnsembleResult:
    """Full ensemble forecast with diagnostics."""

    values: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]
    confidence: float  # overall ensemble confidence
    agreement: float  # 0-1, how much workers agree
    worker_predictions: dict[str, Prediction] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            "Forecast: %d steps, confidence=%.3f, agreement=%.3f"
            % (len(self.values), self.confidence, self.agreement)
        ]
        for name, wp in self.worker_predictions.items():
            lines.append("  %s: conf=%.3f" % (name, wp.confidence))
        return "\n".join(lines)


class Worker:
    """Base class for ensemble workers.

    Each worker maintains a history buffer, produces forecasts, and tracks
    its own accuracy via a sliding window of (predicted, actual) pairs.
    """

    name: str = "base"
    min_observations: int = 48

    def __init__(self, confidence_window: int = 50):
        self._history: list[float] = []
        self._last_prediction: float | None = None  # 1-step-ahead for confidence
        self._errors: deque[float] = deque(maxlen=confidence_window)
        self._signal_var: float = 1.0
        self._ready: bool = False

    def observe(self, value: float) -> None:
        """Add one observation and update confidence tracking."""
        # Score the last 1-step prediction against this actual value
        if self._last_prediction is not None:
            error = (value - self._last_prediction) ** 2
            self._errors.append(error)

        self._history.append(value)

        if len(self._history) >= self.min_observations:
            if not self._ready:
                self._refit()
                self._ready = True
            # Make a 1-step-ahead prediction for next confidence update
            try:
                pred = self._predict_impl(1)
                self._last_prediction = float(pred[0])
            except Exception:
                self._last_prediction = None
        else:
            self._last_prediction = None

    def observe_batch(self, values: NDArray[np.floating]) -> None:
        """Add multiple observations. Fits once at the end."""
        for v in values:
            self._history.append(float(v))
        if len(self._history) >= self.min_observations:
            self._refit()
            self._ready = True
            # Compute confidence from recent in-sample accuracy
            self._warm_up_confidence()

    def predict(self, horizon: int) -> Prediction:
        """Generate forecast with confidence score."""
        if not self._ready:
            return Prediction(
                values=np.full(horizon, np.nan),
                confidence=0.0,
            )
        try:
            values = self._predict_impl(horizon)
            return Prediction(values=values, confidence=self.confidence)
        except Exception:
            return Prediction(
                values=np.full(horizon, np.nan),
                confidence=0.0,
            )

    @property
    def confidence(self) -> float:
        """Running confidence based on recent prediction accuracy.

        Uses exp(-mse/var) which spreads the range much more than
        1/(1+mse/var). A worker with MSE = signal_var gets confidence
        0.37 (not 0.5), and a worker with MSE = 2*var gets 0.14.
        This creates meaningful gaps between good and bad workers.
        """
        if len(self._errors) < 3:
            return 0.25  # prior: low confidence until proven
        recent_mse = float(np.mean(self._errors))
        sig_var = max(self._signal_var, 1e-10)
        # exp(-mse/var) then squared to force separation
        raw = float(np.exp(-recent_mse / sig_var))
        return raw * raw

    def _refit(self) -> None:
        """Refit the internal model on current history. Override in subclass."""
        arr = np.array(self._history, dtype=np.float64)
        self._signal_var = float(np.var(arr))

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        """Generate raw forecast. Override in subclass."""
        raise NotImplementedError

    def _warm_up_confidence(self, eval_horizon: int = 24) -> None:
        """Compute initial confidence from hold-out multi-step errors.

        Fits on first 80% of history, forecasts eval_horizon steps into
        the remaining 20%, computes per-step squared error. This produces
        meaningful discrimination between workers because multi-step
        errors diverge much more than 1-step errors.

        The confidence is then squared to force separation between
        workers that are close in absolute terms but differ in relative
        performance.
        """
        arr = np.array(self._history, dtype=np.float64)
        self._signal_var = float(np.var(arr))
        n = len(arr)

        # Hold-out validation: fit on prefix, evaluate on suffix
        split = max(self.min_observations, int(n * 0.75))
        holdout = arr[split:]
        if len(holdout) < eval_horizon:
            eval_horizon = max(len(holdout), 1)

        try:
            train_subset = arr[:split]
            pred = self._predict_from(train_subset, eval_horizon)
            actual = holdout[:eval_horizon]
            # Per-step errors
            for j in range(min(len(pred), len(actual))):
                error = (actual[j] - pred[j]) ** 2
                self._errors.append(error)
        except Exception:
            pass

        # Restore full fit
        self._refit()

    def _refit_on(self, data: NDArray) -> None:
        """Refit on a subset of data. Default: same as _refit."""
        pass

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        """Predict from a specific data window. Default: use _predict_impl."""
        return self._predict_impl(horizon)


class ARWorker(Worker):
    """Worker wrapping LinearARForecaster."""

    name = "AR"
    min_observations = 48

    def __init__(self, p: int = 24, **kwargs):
        super().__init__(**kwargs)
        self._p = p
        self._model = None

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.models.baseline import LinearARForecaster
        arr = np.array(self._history, dtype=np.float64)
        self._model = LinearARForecaster(p=self._p)
        self._model.fit(arr)

    def _refit_on(self, data: NDArray) -> None:
        from spectral_forecast.models.baseline import LinearARForecaster
        self._model = LinearARForecaster(p=self._p)
        self._model.fit(data)

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        arr = np.array(self._history, dtype=np.float64)
        return self._model.predict(arr, horizon)

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.models.baseline import LinearARForecaster
        m = LinearARForecaster(p=self._p)
        m.fit(data)
        return m.predict(data, horizon)


class FourierWorker(Worker):
    """Worker wrapping spectral-forecast's full decomposition pipeline."""

    name = "Fourier"
    min_observations = 64

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.forecast import SpectralForecaster
        arr = np.array(self._history, dtype=np.float64)
        self._model = SpectralForecaster()
        self._model.fit(arr)

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        result = self._model.forecast(horizon)
        return result.point_forecast

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.forecast import SpectralForecaster
        m = SpectralForecaster()
        m.fit(data)
        return m.forecast(horizon).point_forecast


class IterativeWorker(Worker):
    """Worker wrapping IterativeDecompositionForecaster."""

    name = "Iterative"
    min_observations = 100

    def __init__(self, period: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._period = period
        self._model = None

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.models.iterative import IterativeDecompositionForecaster
        arr = np.array(self._history, dtype=np.float64)
        self._model = IterativeDecompositionForecaster(
            period=self._period, ar_lags=12, max_iter=5
        )
        self._model.fit(arr)

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        arr = np.array(self._history, dtype=np.float64)
        return self._model.predict(arr, horizon)

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.models.iterative import IterativeDecompositionForecaster
        m = IterativeDecompositionForecaster(period=self._period, ar_lags=12, max_iter=5)
        m.fit(data)
        return m.predict(data, horizon)


class SRWorker(Worker):
    """Worker wrapping StochasticResonanceForecaster with online feedback.

    This worker uses the SR mechanism's self-regulating noise loop.
    Between observations, it updates its ewma_error from actual errors,
    which adjusts the noise injection level for the next prediction.
    """

    name = "SR"
    min_observations = 48

    def __init__(self, kappa: float = 0.5, noise_cap: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self._kappa = kappa
        self._noise_cap = noise_cap
        self._model = None
        self._ewma_error: float = 0.0
        self._sigma_max: float = 1.0

    def _refit(self) -> None:
        super()._refit()
        from spectral_forecast.models.stochastic_resonance import StochasticResonanceForecaster
        arr = np.array(self._history, dtype=np.float64)
        self._model = StochasticResonanceForecaster(
            ar_lags=24, kappa=self._kappa, noise_cap_fraction=self._noise_cap
        )
        self._model.fit(arr)
        self._sigma_max = self._noise_cap * float(np.std(arr))
        # Initialize ewma_error from in-sample residuals
        p = self._model.ar_lags
        errors = []
        for i in range(p, len(arr)):
            pred = self._model._ar.predict_one(arr[i - p : i])
            errors.append(abs(arr[i] - pred))
        self._ewma_error = float(np.mean(errors[-20:])) if errors else float(np.std(arr))

    def observe(self, value: float) -> None:
        """Override: update ewma_error with actual error (online SR feedback)."""
        if self._last_prediction is not None:
            actual_error = abs(value - self._last_prediction)
            self._ewma_error = 0.1 * actual_error + 0.9 * self._ewma_error
        super().observe(value)

    def _predict_impl(self, horizon: int) -> NDArray[np.floating]:
        arr = np.array(self._history, dtype=np.float64)
        # Use the current ewma_error to set noise level
        sigma = min(self._sigma_max, self._kappa * self._ewma_error)
        rng = np.random.default_rng(42)
        p = self._model.ar_lags
        window = list(arr[-p:])
        preds = []
        for _ in range(horizon):
            w = np.array(window[-p:])
            pred = self._model._predict_with_noise(w, sigma, rng)
            preds.append(pred)
            window.append(pred)
        return np.array(preds)

    def _predict_from(self, data: NDArray, horizon: int) -> NDArray:
        from spectral_forecast.models.stochastic_resonance import StochasticResonanceForecaster
        m = StochasticResonanceForecaster(ar_lags=24, kappa=self._kappa, noise_cap_fraction=self._noise_cap)
        m.fit(data)
        return m.predict(data, horizon)


class ForecastEngine:
    """Unified ensemble forecasting tool.

    One interface. Multiple internal workers. Confidence-weighted predictions.

    Usage:
        engine = ForecastEngine()
        engine.observe_batch(data)
        result = engine.predict(96)
        print(result.values, result.confidence)
    """

    def __init__(
        self,
        workers: list[Worker] | None = None,
        confidence_level: float = 0.8,
    ):
        if workers is None:
            workers = [
                ARWorker(p=24),
                FourierWorker(),
                IterativeWorker(),
                SRWorker(),
            ]
        self.workers = workers
        self.confidence_level = confidence_level
        self._n_observed: int = 0
        self._signal_std: float = 1.0

    def observe(self, value: float) -> None:
        """Feed one observation to all workers."""
        for worker in self.workers:
            worker.observe(value)
        self._n_observed += 1

    def observe_batch(self, values: NDArray[np.floating]) -> None:
        """Feed multiple observations to all workers."""
        values = np.asarray(values, dtype=np.float64)
        self._signal_std = max(float(np.std(values)), 1e-10)
        for worker in self.workers:
            worker.observe_batch(values)
        self._n_observed += len(values)

    def predict(self, horizon: int) -> EnsembleResult:
        """Generate confidence-weighted ensemble forecast.

        Each ready worker produces a forecast and confidence score.
        The ensemble blends by confidence weight, with uncertainty
        derived from both individual worker uncertainty and inter-worker
        disagreement.
        """
        worker_preds: dict[str, Prediction] = {}
        for worker in self.workers:
            pred = worker.predict(horizon)
            if pred.confidence > 0 and not np.any(np.isnan(pred.values)):
                worker_preds[worker.name] = pred

        if not worker_preds:
            # No workers ready — return NaN with zero confidence
            return EnsembleResult(
                values=np.full(horizon, np.nan),
                lower=np.full(horizon, np.nan),
                upper=np.full(horizon, np.nan),
                confidence=0.0,
                agreement=0.0,
            )

        # Confidence-weighted blend with collapse detection.
        # If the best worker's confidence dominates (>3x the second-best),
        # collapse the ensemble to that single worker. Blending with weak
        # workers just dilutes a good prediction.
        names = list(worker_preds.keys())
        forecasts = np.array([worker_preds[n].values for n in names])
        confidences = np.array([worker_preds[n].confidence for n in names])

        # Check for collapse: best worker dominates
        sorted_conf = np.sort(confidences)[::-1]
        if len(sorted_conf) >= 2 and sorted_conf[1] > 0:
            dominance_ratio = sorted_conf[0] / sorted_conf[1]
        else:
            dominance_ratio = float("inf")

        if dominance_ratio > 2.0:
            # Collapse to the best worker
            best_idx = int(np.argmax(confidences))
            weights = np.zeros(len(names))
            weights[best_idx] = 1.0
        else:
            # Normalize confidence weights
            weights = confidences / confidences.sum()

        # Weighted mean forecast
        ensemble_values = np.zeros(horizon)
        for i, name in enumerate(names):
            ensemble_values += weights[i] * forecasts[i]

        # Inter-worker disagreement: weighted std of predictions at each step
        if len(names) > 1:
            weighted_var = np.zeros(horizon)
            for i, name in enumerate(names):
                weighted_var += weights[i] * (forecasts[i] - ensemble_values) ** 2
            disagreement_std = np.sqrt(weighted_var)
        else:
            disagreement_std = np.zeros(horizon)

        # Agreement metric: 1 when all workers agree, 0 when disagreement ~ signal std
        mean_disagreement = float(np.mean(disagreement_std))
        agreement = 1.0 / (1.0 + mean_disagreement / self._signal_std)

        # Overall confidence: product of best worker confidence and agreement
        overall_confidence = float(np.max(confidences)) * agreement

        # Error bounds: combination of disagreement and horizon uncertainty
        alpha = (1 - self.confidence_level) / 2
        z = 1.96 if self.confidence_level >= 0.95 else 1.28  # approximate
        horizon_growth = np.sqrt(1 + np.arange(horizon, dtype=np.float64) / max(self._n_observed, 1))
        bound_width = (disagreement_std + mean_disagreement * 0.5) * horizon_growth * z
        # Floor: at least some uncertainty
        min_bound = self._signal_std * 0.05 * horizon_growth
        bound_width = np.maximum(bound_width, min_bound)

        lower = ensemble_values - bound_width
        upper = ensemble_values + bound_width

        return EnsembleResult(
            values=ensemble_values,
            lower=lower,
            upper=upper,
            confidence=overall_confidence,
            agreement=agreement,
            worker_predictions=worker_preds,
        )
