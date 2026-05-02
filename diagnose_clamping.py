"""Check if range clamping is engaging on outlier windows."""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def check_dataset(name, filepath, train_n, val_n, horizon=720):
    data = load_csv_dataset(filepath, column="OT")
    train = data[:train_n]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std
    test_start = train_n + val_n
    context_length = 512

    worst_mse = 0
    worst_idx = 0
    start = test_start - context_length
    idx = 0
    while start + context_length + horizon <= len(data_norm):
        context = data_norm[start : start + context_length]
        actual = data_norm[start + context_length : start + context_length + horizon]

        f = SpectralForecaster()
        result = f.fit_forecast(context, horizon)

        mse = float(np.mean((actual - result.point_forecast) ** 2))
        if mse > worst_mse:
            worst_mse = mse
            worst_idx = idx
            worst_range = (result.point_forecast.min(), result.point_forecast.max())
            actual_range = (actual.min(), actual.max())
            periodic_range = f._periodic_range
            trend_type = f._trend.model.trend_type.value

        start += horizon
        idx += 1

    print("%s (h=%d): worst win=%d MSE=%.2f" % (name, horizon, worst_idx, worst_mse))
    print("  forecast range: [%.2f, %.2f]" % worst_range)
    print("  actual range:   [%.2f, %.2f]" % actual_range)
    print("  periodic clamp: [%.2f, %.2f]" % periodic_range)
    print("  trend: %s" % trend_type)
    print()


def main():
    check_dataset("ETTh2", "data/ETTh2.csv", 8640, 2880, 336)
    check_dataset("ETTm1", "data/ETTm1.csv", 34465, 11521, 720)
    check_dataset("ETTm2", "data/ETTm2.csv", 34465, 11521, 720)


if __name__ == "__main__":
    main()
