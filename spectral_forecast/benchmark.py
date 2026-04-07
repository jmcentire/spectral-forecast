"""Benchmark harness for standard time series datasets.

Implements rolling-window evaluation matching the protocol used by
TimesFM, PatchTST, and other standard benchmarks on ETT and similar datasets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spectral_forecast.forecast import SpectralForecaster


@dataclass
class BenchmarkMetrics:
    """Metrics for a single prediction length."""

    mse: float
    mae: float
    mape: float  # can be inf if actual values contain zeros
    rmse: float
    n_windows: int
    elapsed_seconds: float


@dataclass
class BenchmarkResult:
    """Full benchmark result across prediction lengths."""

    dataset_name: str
    column: str
    results: dict[int, BenchmarkMetrics]  # prediction_length -> metrics

    def summary(self) -> str:
        """Human-readable benchmark summary."""
        lines = [f"Benchmark: {self.dataset_name} (column: {self.column})"]
        lines.append(f"{'Horizon':>8} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'Windows':>8} {'Time':>8}")
        lines.append("-" * 60)
        for h in sorted(self.results.keys()):
            m = self.results[h]
            lines.append(
                f"{h:>8} {m.mse:>10.4f} {m.mae:>10.4f} {m.rmse:>10.4f} "
                f"{m.n_windows:>8} {m.elapsed_seconds:>7.1f}s"
            )
        return "\n".join(lines)


def load_csv_dataset(
    path: str | Path,
    column: str = "OT",
    date_column: str = "date",
) -> NDArray[np.floating]:
    """Load a time series from CSV (e.g., ETT datasets).

    Expects a CSV with a date column and numeric columns.
    Returns the specified column as a 1D numpy array.
    """
    import csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Column '{column}' not found. Available: {available}"
            )
        values = []
        for row in reader:
            try:
                values.append(float(row[column]))
            except (ValueError, KeyError):
                continue

    if not values:
        raise ValueError(f"No valid numeric values found in column '{column}'")

    return np.array(values, dtype=np.float64)


def _compute_metrics(
    actual: NDArray[np.floating], predicted: NDArray[np.floating]
) -> tuple[float, float, float]:
    """Compute MSE, MAE, MAPE."""
    mse = float(np.mean((actual - predicted) ** 2))
    mae = float(np.mean(np.abs(actual - predicted)))
    # MAPE: handle zeros
    nonzero = np.abs(actual) > 1e-10
    if nonzero.any():
        mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])))
    else:
        mape = float("inf")
    return mse, mae, mape


def run_benchmark(
    data: NDArray[np.floating],
    prediction_lengths: list[int] | None = None,
    context_length: int = 512,
    stride: int | None = None,
    dataset_name: str = "unknown",
    column: str = "unknown",
    **forecaster_kwargs,
) -> BenchmarkResult:
    """Run rolling-window benchmark on a time series.

    Standard protocol: use a sliding window of `context_length` to predict
    the next `prediction_length` steps. Slide by `stride` (default=prediction_length).

    Args:
        data: Full time series.
        prediction_lengths: List of horizons to test. Default [96, 192, 336, 720].
        context_length: Input window size.
        stride: Step size between windows. None = prediction_length.
        dataset_name: Name for reporting.
        column: Column name for reporting.
        **forecaster_kwargs: Passed to SpectralForecaster.

    Returns:
        BenchmarkResult with metrics per prediction length.
    """
    if prediction_lengths is None:
        prediction_lengths = [96, 192, 336, 720]

    results: dict[int, BenchmarkMetrics] = {}

    for pred_len in prediction_lengths:
        s = stride or pred_len
        all_actual = []
        all_predicted = []
        t0 = time.time()

        # Rolling windows
        start = 0
        while start + context_length + pred_len <= len(data):
            context = data[start : start + context_length]
            actual_future = data[start + context_length : start + context_length + pred_len]

            forecaster = SpectralForecaster(**forecaster_kwargs)
            result = forecaster.fit_forecast(context, pred_len)

            all_actual.append(actual_future)
            all_predicted.append(result.point_forecast)

            start += s

        elapsed = time.time() - t0

        if not all_actual:
            results[pred_len] = BenchmarkMetrics(
                mse=float("nan"),
                mae=float("nan"),
                mape=float("nan"),
                rmse=float("nan"),
                n_windows=0,
                elapsed_seconds=elapsed,
            )
            continue

        actual_all = np.concatenate(all_actual)
        predicted_all = np.concatenate(all_predicted)
        mse, mae, mape = _compute_metrics(actual_all, predicted_all)

        results[pred_len] = BenchmarkMetrics(
            mse=mse,
            mae=mae,
            mape=mape,
            rmse=float(np.sqrt(mse)),
            n_windows=len(all_actual),
            elapsed_seconds=elapsed,
        )

    return BenchmarkResult(
        dataset_name=dataset_name,
        column=column,
        results=results,
    )
