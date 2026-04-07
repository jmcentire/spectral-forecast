"""Run standardized ETTh1 benchmark with proper normalization and test split."""

import numpy as np
import time
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def main():
    data = load_csv_dataset("data/ETTh1.csv", column="OT")

    # Standard ETTh1 split
    train = data[:8640]
    val = data[8640 : 8640 + 2880]
    test = data[8640 + 2880 :]

    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    context_length = 512
    horizons = [96, 192, 336, 720]

    print("ETTh1 (OT) - Normalized MSE/MAE - Standard Test Split")
    print("Train: %d, Val: %d, Test: %d" % (len(train), len(val), len(test)))
    print("Normalization: mean=%.2f, std=%.2f" % (mean, std))
    print()
    print(
        "%8s %10s %10s %8s %8s" % ("Horizon", "MSE", "MAE", "Windows", "Time")
    )
    print("-" * 50)

    test_start = 8640 + 2880

    for horizon in horizons:
        t0 = time.time()
        all_mse = []
        all_mae = []
        n_windows = 0

        start = test_start - context_length
        while start + context_length + horizon <= len(data_norm):
            context = data_norm[start : start + context_length]
            actual = data_norm[
                start + context_length : start + context_length + horizon
            ]

            forecaster = SpectralForecaster()
            result = forecaster.fit_forecast(context, horizon)

            mse = float(np.mean((actual - result.point_forecast) ** 2))
            mae = float(np.mean(np.abs(actual - result.point_forecast)))
            all_mse.append(mse)
            all_mae.append(mae)
            n_windows += 1
            start += horizon

        elapsed = time.time() - t0
        avg_mse = np.mean(all_mse)
        avg_mae = np.mean(all_mae)
        print(
            "%8d %10.4f %10.4f %8d %7.1fs"
            % (horizon, avg_mse, avg_mae, n_windows, elapsed)
        )

    print()
    print("Reference (horizon=96):")
    print("  TimesFM zero-shot:   MSE=0.375, MAE=0.401")
    print("  PatchTST supervised: MSE=0.370, MAE=0.400")
    print("  ARIMA:               MSE=~0.600")
    print("  Naive (repeat last): MSE=~0.900")


if __name__ == "__main__":
    main()
