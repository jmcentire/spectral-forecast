"""Diagnose which windows have worst MSE at long horizons."""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def main():
    data = load_csv_dataset("data/ETTh1.csv", column="OT")
    train = data[:8640]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    test_start = 8640 + 2880
    context_length = 512

    for horizon in [336, 720]:
        print("=== Horizon %d ===" % horizon)
        window_mses = []
        start = test_start - context_length
        idx = 0
        while start + context_length + horizon <= len(data_norm):
            context = data_norm[start : start + context_length]
            actual = data_norm[start + context_length : start + context_length + horizon]

            forecaster = SpectralForecaster()
            result = forecaster.fit_forecast(context, horizon)

            mse = float(np.mean((actual - result.point_forecast) ** 2))
            trend_type = forecaster._trend.model.trend_type.value
            trend_end = forecaster._trend.model.predict(
                np.array([512.0 + horizon])
            )[0]
            actual_end = actual[-1]
            window_mses.append((idx, mse, trend_type, trend_end, actual_end))
            start += horizon
            idx += 1

        window_mses.sort(key=lambda x: x[1], reverse=True)
        print("Worst 5 windows:")
        for i, mse, tt, te, ae in window_mses[:5]:
            print("  win=%d MSE=%.3f trend=%s trend_end=%.2f actual_end=%.2f" % (
                i, mse, tt, te, ae
            ))
        print("Best 5 windows:")
        for i, mse, tt, te, ae in window_mses[-5:]:
            print("  win=%d MSE=%.3f trend=%s trend_end=%.2f actual_end=%.2f" % (
                i, mse, tt, te, ae
            ))
        avg = np.mean([x[1] for x in window_mses])
        median = np.median([x[1] for x in window_mses])
        print("Mean MSE: %.4f, Median MSE: %.4f" % (avg, median))
        print()


if __name__ == "__main__":
    main()
