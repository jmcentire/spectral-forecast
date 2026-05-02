"""Diagnose ETTm1 720 worst windows."""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def main():
    data = load_csv_dataset("data/ETTm1.csv", column="OT")
    train = data[:34465]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    test_start = 34465 + 11521
    context_length = 512
    horizon = 720

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
        window_mses.append((idx, mse, trend_type))
        start += horizon
        idx += 1

    window_mses.sort(key=lambda x: x[1], reverse=True)
    print("Worst 5 windows (horizon=%d):" % horizon)
    for i, mse, tt in window_mses[:5]:
        print("  win=%d MSE=%.3f trend=%s" % (i, mse, tt))

    avg = np.mean([x[1] for x in window_mses])
    median = np.median([x[1] for x in window_mses])
    print("Mean MSE: %.4f, Median MSE: %.4f, N=%d" % (avg, median, len(window_mses)))

    # What if we clip outlier windows?
    mses = sorted([x[1] for x in window_mses])
    p95 = np.percentile(mses, 95)
    clipped = [min(m, p95) for m in mses]
    print("P95-clipped mean: %.4f" % np.mean(clipped))


if __name__ == "__main__":
    main()
