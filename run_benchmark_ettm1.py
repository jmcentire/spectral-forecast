"""Run standardized ETTm1 benchmark."""

import numpy as np
import time
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def main():
    data = load_csv_dataset("data/ETTm1.csv", column="OT")

    # Standard ETTm1 split: 34465 / 11521 / 11521
    train = data[:34465]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    context_length = 512
    horizons = [96, 192, 336, 720]

    test_start = 34465 + 11521

    print("ETTm1 (OT) - Normalized MSE/MAE - Standard Test Split")
    print("Train: 34465, Test start: %d, Total: %d" % (test_start, len(data)))
    print("Normalization: mean=%.2f, std=%.2f" % (mean, std))
    print()
    print(
        "%8s %10s %10s %8s %8s" % ("Horizon", "MSE", "MAE", "Windows", "Time")
    )
    print("-" * 50)

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
    print("  TimesFM zero-shot:   MSE=0.320")
    print("  PatchTST supervised: MSE=0.293")


if __name__ == "__main__":
    main()
