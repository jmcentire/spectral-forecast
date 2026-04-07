"""
Linear AR (Autoregressive) Baseline Forecaster.

Pure numpy implementation of an AR(p) model fit by ordinary least squares.
This is the conventional baseline against which the novel approaches are compared.

Design:
  - Fit: construct Toeplitz feature matrix from lags 1..p, solve via lstsq
  - Predict: recursive multi-step (each predicted value feeds back as a lag)
  - No differencing or seasonality handling — kept intentionally simple

Why this baseline?
  - Transparent and inspectable
  - Competitive on stationary AR processes
  - Fast: O(n*p) fit, O(h*p) predict
  - Published ARIMA/linear results on M4 hourly: SMAPE ~14-16%
"""
import numpy as np
from typing import Optional


class LinearARForecaster:
    """AR(p) forecaster fit by ordinary least squares.

    Parameters
    ----------
    p : int
        Number of autoregressive lags. Default 24 works well for hourly data
        with daily periodicity.
    include_bias : bool
        Whether to include a constant term (intercept).
    """

    def __init__(self, p: int = 24, include_bias: bool = True) -> None:
        self.p = p
        self.include_bias = include_bias
        self._coeffs: Optional[np.ndarray] = None  # shape (p+1,) or (p,)
        self._fitted: bool = False

    def _build_feature_matrix(self, y: np.ndarray) -> np.ndarray:
        """Build (n-p, p) matrix where row i contains [y[i], y[i-1], ..., y[i-p+1]]."""
        n = len(y)
        rows = []
        for i in range(self.p, n):
            row = y[i - self.p : i][::-1]  # most recent lag first
            rows.append(row)
        X = np.array(rows)
        if self.include_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, y: np.ndarray) -> "LinearARForecaster":
        """Fit the AR(p) model to training data y."""
        if len(y) <= self.p:
            raise ValueError(f"Need at least p+1={self.p+1} observations, got {len(y)}")
        X = self._build_feature_matrix(y)
        targets = y[self.p:]
        # OLS: coeffs = (X'X)^{-1} X'y via lstsq for numerical stability
        result = np.linalg.lstsq(X, targets, rcond=None)
        self._coeffs = result[0]
        self._fitted = True
        return self

    def predict_one(self, window: np.ndarray) -> float:
        """Predict one step ahead from a window of the last p values.

        Parameters
        ----------
        window : array of shape (p,)
            The p most recent observations, oldest first.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if len(window) < self.p:
            raise ValueError(f"Window must have at least p={self.p} values.")
        lags = window[-self.p:][::-1]  # most recent first
        if self.include_bias:
            x = np.concatenate([[1.0], lags])
        else:
            x = lags
        return float(np.dot(self._coeffs, x))

    def predict(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast `horizon` steps ahead recursively from the end of y.

        Recursive: each predicted value is appended to the window and used
        as a lag for subsequent predictions.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        window = list(y[-self.p:])
        preds = []
        for _ in range(horizon):
            val = self.predict_one(np.array(window))
            preds.append(val)
            window.append(val)
            window = window[-self.p:]  # keep only the last p values
        return np.array(preds)

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        return self._coeffs
