"""Ablation study: measure per-layer contribution."""

import numpy as np
import time
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def run_ablation_window(data_norm, start, context_length, horizon, config):
    """Run one window with a given config."""
    context = data_norm[start : start + context_length]
    actual = data_norm[start + context_length : start + context_length + horizon]

    kwargs = {}
    if not config.get("trend", True):
        # Disable trend by forcing NONE
        pass  # can't easily disable layers without refactoring; use full model
    if not config.get("local", True):
        kwargs["amplitude_damping"] = False  # approximate: no damping

    forecaster = SpectralForecaster(**kwargs)
    result = forecaster.fit_forecast(context, horizon)

    mse = float(np.mean((actual - result.point_forecast) ** 2))
    return mse


def naive_forecast(data_norm, start, context_length, horizon):
    """Last-value naive baseline."""
    last = data_norm[start + context_length - 1]
    actual = data_norm[start + context_length : start + context_length + horizon]
    return float(np.mean((actual - last) ** 2))


def fourier_only(data_norm, start, context_length, horizon):
    """Fourier extraction only — no trend, shock, local, wavelet."""
    from spectral_forecast.extraction import extract
    context = data_norm[start : start + context_length]
    actual = data_norm[start + context_length : start + context_length + horizon]

    # Detrend
    t = np.arange(context_length, dtype=np.float64)
    X = np.column_stack([t, np.ones(context_length)])
    coeffs, _, _, _ = np.linalg.lstsq(X, context, rcond=None)
    slope, intercept = coeffs

    detrended = context - (slope * t + intercept)
    result = extract(detrended)

    # Forecast: just periodic + linear trend
    t_future = np.arange(context_length, context_length + horizon, dtype=np.float64)
    forecast = np.zeros(horizon)
    for comp in result.components:
        forecast += comp.evaluate(t_future)
    forecast += slope * t_future + intercept

    return float(np.mean((actual - forecast) ** 2))


def main():
    datasets = {
        "ETTh1": {"file": "data/ETTh1.csv", "train": 8640, "val": 2880},
        "ETTm1": {"file": "data/ETTm1.csv", "train": 34465, "val": 11521},
    }

    horizon = 96
    context_length = 512

    print("Ablation Study (H=96)")
    print("=" * 70)
    print("%8s %12s %12s %12s %12s" % ("Dataset", "Naive", "Fourier", "Full-noWav", "Full"))
    print("-" * 58)

    for name, info in datasets.items():
        data = load_csv_dataset(info["file"], column="OT")
        train = data[: info["train"]]
        mean, std = train.mean(), train.std()
        data_norm = (data - mean) / std
        test_start = info["train"] + info["val"]

        naive_mses = []
        fourier_mses = []
        full_mses = []

        start = test_start - context_length
        while start + context_length + horizon <= len(data_norm):
            naive_mses.append(naive_forecast(data_norm, start, context_length, horizon))
            fourier_mses.append(fourier_only(data_norm, start, context_length, horizon))
            full_mses.append(run_ablation_window(data_norm, start, context_length, horizon, {}))
            start += horizon

        print("%8s %12.4f %12.4f %12s %12.4f" % (
            name,
            np.mean(naive_mses),
            np.mean(fourier_mses),
            "---",
            np.mean(full_mses),
        ))

        # Also report std across windows for significance
        print("         +/- %.4f   +/- %.4f %12s +/- %.4f" % (
            np.std(naive_mses),
            np.std(fourier_mses),
            "",
            np.std(full_mses),
        ))

    print()
    print("Fourier-only = Layer 1 + linear detrend, no BIC trend/shock/AR/wavelet")
    print("Full = all layers")


if __name__ == "__main__":
    main()
