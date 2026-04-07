"""
Stochastic Resonance (SR) Forecaster.

Core principle (from Cage and Mirror, Theorem 1):
  Adding noise to a nonlinear detector near threshold can improve detection
  of sub-threshold signals — the VG-WM inverted-U relationship. The same
  mechanism applies to forecasting:

  - The "signal" is weak periodic or structural content in the time series.
  - The "detector" is an AR model's ability to extract that structure.
  - When model accuracy is low, the signal is below the model's extraction
    floor. Injecting noise perturbs the input window across this threshold,
    occasionally revealing the hidden structure.

Self-regulating loop:
  sigma_t = min(sigma_max, kappa * ewma_error_t)

  where ewma_error_t is the exponentially-weighted moving average of recent
  absolute prediction errors.

  - High accuracy (low error) → low sigma → little noise → stable regime
  - Low accuracy (high error) → high sigma → more noise → may cross threshold
    → if pattern detected, accuracy improves → sigma decreases (convergence)
    → if noise overwhelms signal, accuracy degrades further → sigma increases
       but is bounded by sigma_max (DAMPING — prevents divergence)

Damping mechanism (CRITICAL):
  Without sigma_max, the loop is a positive feedback:
    low accuracy → more noise → worse predictions → even lower accuracy → ...

  sigma_max is the hard ceiling. We set it relative to the signal's standard
  deviation measured on the training set: sigma_max = noise_cap_fraction * std(train).

  Typical safe values: noise_cap_fraction in [0.3, 0.8]. Above 1.0, the noise
  amplitude exceeds the signal amplitude and divergence is likely.

Noise mechanism:
  At each prediction step, we generate K noisy copies of the input window,
  predict with each, and take the mean. This is ensemble-with-noise, not
  pure output perturbation — the noise acts on the INPUT to the AR model,
  which provides the nonlinear detection benefit.

Convergence (equilibrium noise level):
  At equilibrium, sigma* = kappa * ewma_error* where ewma_error* is the
  irreducible forecast error. This sigma* is itself a diagnostic:
  - Large sigma* → unpredictable signal (high noise floor)
  - Small sigma* → predictable signal (model has extracted most structure

This maps exactly to the maximum entropy stopping criterion: the system has
reached maximum extraction when sigma stabilizes.

Comparison with Cage and Mirror Theorem 1 conditions:
  1. Nonlinearity: AR prediction is effectively linear, but the ENSEMBLE
     over noisy inputs introduces nonlinearity (Jensen's inequality).
  2. Near-threshold signal: satisfied for weak periodic components.
  3. Bounded noise: enforced by sigma_max.
  4. Interior maximum: demonstrated empirically in FINDINGS.md noise curves.
  5. Severity inversion risk: present when noise_cap_fraction > 0.5 on
     low-SNR signals — documented in FINDINGS.md.
"""
import numpy as np
from typing import Optional, List

from .baseline import LinearARForecaster


class StochasticResonanceForecaster:
    """SR forecaster with adaptive noise injection and damping.

    Parameters
    ----------
    ar_lags : int
        AR order for the base predictor.
    n_ensemble : int
        Number of noisy ensemble members per prediction step.
    kappa : float
        Noise scale coefficient: sigma_t = min(sigma_max, kappa * ewma_error_t).
        Higher kappa → more aggressive noise injection.
    noise_cap_fraction : float
        sigma_max = noise_cap_fraction * std(train). Hard ceiling on noise.
        CRITICAL: keep <= 0.5 for stability; > 1.0 risks divergence.
    ewma_alpha : float
        Exponential smoothing factor for running error estimate.
        Higher alpha → faster adaptation; lower alpha → more stable.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        ar_lags: int = 24,
        n_ensemble: int = 20,
        kappa: float = 0.5,
        noise_cap_fraction: float = 0.4,
        ewma_alpha: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.ar_lags = ar_lags
        self.n_ensemble = n_ensemble
        self.kappa = kappa
        self.noise_cap_fraction = noise_cap_fraction
        self.ewma_alpha = ewma_alpha
        self.seed = seed

        self._ar: Optional[LinearARForecaster] = None
        self._sigma_max: float = 1.0
        self._fitted: bool = False

        # Diagnostics populated during predict()
        self.sigma_history: List[float] = []
        self.error_history: List[float] = []
        self.ewma_error_history: List[float] = []

    def fit(self, y: np.ndarray) -> "StochasticResonanceForecaster":
        """Fit the base AR model and set sigma_max from training data std."""
        self._ar = LinearARForecaster(p=self.ar_lags)
        self._ar.fit(y)
        # sigma_max relative to signal scale
        self._sigma_max = self.noise_cap_fraction * float(np.std(y))
        self._fitted = True
        return self

    def _predict_with_noise(
        self,
        window: np.ndarray,
        sigma: float,
        rng: np.random.Generator,
    ) -> float:
        """Make one noisy ensemble prediction from the last ar_lags values.

        Generates n_ensemble noisy copies of the window, predicts with each,
        returns the mean. When sigma = 0, this reduces to a single AR prediction.
        """
        if sigma < 1e-10 or self.n_ensemble <= 1:
            return self._ar.predict_one(window)
        preds = []
        for _ in range(self.n_ensemble):
            noisy_window = window + rng.normal(0, sigma, size=window.shape)
            preds.append(self._ar.predict_one(noisy_window))
        return float(np.mean(preds))

    def predict(
        self,
        y: np.ndarray,
        horizon: int,
        warm_up_steps: int = 10,
    ) -> np.ndarray:
        """Forecast `horizon` steps with adaptive noise.

        Strategy: step through the last `warm_up_steps + horizon` positions of
        the training window to warm up the running error estimate, then emit
        the final `horizon` predictions.

        Parameters
        ----------
        y : array
            Full training series.
        horizon : int
            Steps to forecast ahead.
        warm_up_steps : int
            Steps of in-sample walk-forward to calibrate ewma_error before
            the actual forecast horizon.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        rng = np.random.default_rng(self.seed)
        p = self.ar_lags

        self.sigma_history = []
        self.error_history = []
        self.ewma_error_history = []

        # Initialize ewma_error from in-sample AR residuals
        initial_preds = self._ar.predict_one
        in_sample_errors = []
        for i in range(p, len(y)):
            window = y[i - p : i]
            pred = self._ar.predict_one(window)
            in_sample_errors.append(abs(float(y[i]) - pred))
        if in_sample_errors:
            ewma_error = float(np.mean(in_sample_errors[-20:]))
        else:
            ewma_error = float(np.std(y))

        # Walk-forward warm-up to calibrate ewma_error on recent history
        warm_up_start = max(p, len(y) - warm_up_steps - p)
        for i in range(warm_up_start, len(y)):
            window = y[max(0, i - p) : i]
            if len(window) < p:
                continue
            sigma = min(self._sigma_max, self.kappa * ewma_error)
            pred_val = self._predict_with_noise(window, sigma, rng)
            error = abs(float(y[i]) - pred_val)
            ewma_error = (
                self.ewma_alpha * error + (1 - self.ewma_alpha) * ewma_error
            )

        # Recursive forecast for `horizon` steps
        window = list(y[-p:])
        predictions = []

        for _ in range(horizon):
            sigma = min(self._sigma_max, self.kappa * ewma_error)
            self.sigma_history.append(sigma)
            self.ewma_error_history.append(ewma_error)

            w = np.array(window)
            pred_val = self._predict_with_noise(w, sigma, rng)
            predictions.append(pred_val)

            # Update window (recursive: predicted value becomes next lag)
            window.append(pred_val)
            window = window[-p:]

            # Update ewma_error using predicted error as proxy.
            # We don't know the true future values, so we use the variance
            # of the ensemble as a proxy for prediction uncertainty.
            if sigma > 1e-10:
                ensemble_preds = [
                    self._ar.predict_one(np.array(window[-p:]) + rng.normal(0, sigma, p))
                    for _ in range(max(3, self.n_ensemble // 4))
                ]
                pred_std = float(np.std(ensemble_preds))
            else:
                pred_std = 0.0
            # ewma_error estimate evolves using ensemble spread as proxy
            ewma_error = self.ewma_alpha * pred_std + (1 - self.ewma_alpha) * ewma_error

        return np.array(predictions)

    def predict_online(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> np.ndarray:
        """One-step-ahead online predictions using true observed values.

        At each test step, the true value is revealed and used to update the
        running error estimate before the next prediction. This is the cleanest
        demonstration of the self-regulating loop.

        Returns
        -------
        predictions : array of shape (len(y_test),)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        rng = np.random.default_rng(self.seed + 1)
        p = self.ar_lags

        # Warm up ewma_error
        in_sample_errors = []
        for i in range(p, len(y_train)):
            window = y_train[i - p : i]
            pred = self._ar.predict_one(window)
            in_sample_errors.append(abs(float(y_train[i]) - pred))
        ewma_error = float(np.mean(in_sample_errors[-20:])) if in_sample_errors else float(np.std(y_train))

        self.sigma_history = []
        self.error_history = []
        self.ewma_error_history = []

        history = list(y_train)
        predictions = []

        for actual in y_test:
            window = np.array(history[-p:])
            sigma = min(self._sigma_max, self.kappa * ewma_error)
            self.sigma_history.append(sigma)
            self.ewma_error_history.append(ewma_error)

            pred_val = self._predict_with_noise(window, sigma, rng)
            predictions.append(pred_val)

            # True value revealed — update ewma_error
            error = abs(float(actual) - pred_val)
            self.error_history.append(error)
            ewma_error = self.ewma_alpha * error + (1 - self.ewma_alpha) * ewma_error

            history.append(float(actual))

        return np.array(predictions)

    @property
    def equilibrium_noise(self) -> Optional[float]:
        """The noise level at convergence — diagnostic of signal predictability.

        A large equilibrium noise level indicates high irreducible uncertainty.
        A small value indicates the signal is highly predictable.
        Returns None if no prediction has been made.
        """
        if not self.sigma_history:
            return None
        # Take mean of last 20% of history as the converged estimate
        tail = max(1, len(self.sigma_history) // 5)
        return float(np.mean(self.sigma_history[-tail:]))

    @property
    def equilibrium_error(self) -> Optional[float]:
        """The EWMA error at convergence."""
        if not self.ewma_error_history:
            return None
        tail = max(1, len(self.ewma_error_history) // 5)
        return float(np.mean(self.ewma_error_history[-tail:]))
