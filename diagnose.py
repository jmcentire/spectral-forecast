"""Diagnose long-horizon trend divergence."""

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

    start = test_start - context_length
    context = data_norm[start : start + context_length]

    f = SpectralForecaster()
    f.fit(context)

    print("Trend type:", f._trend.model.trend_type)
    print("Trend params:", f._trend.model.params)

    t336 = np.arange(512, 512 + 336, dtype=np.float64)
    t720 = np.arange(512, 512 + 720, dtype=np.float64)
    trend336 = f._trend.model.predict(t336)
    trend720 = f._trend.model.predict(t720)
    print("Trend at t+1:", f._trend.model.predict(np.array([512.0]))[0])
    print("Trend at t+336:", trend336[-1])
    print("Trend at t+720:", trend720[-1])
    print("Trend range at 336:", trend336.min(), "to", trend336.max())
    print("Trend range at 720:", trend720.min(), "to", trend720.max())
    print()
    print(
        "Actual data range:",
        data_norm[test_start : test_start + 720].min(),
        "to",
        data_norm[test_start : test_start + 720].max(),
    )

    # Check component count and SNR distribution
    print()
    print("Components:", len(f._extraction.components))
    snrs = [c.snr for c in f._extraction.components]
    print("SNR range: %.1f to %.1f" % (min(snrs), max(snrs)))
    print("Components with SNR > 100:", sum(1 for s in snrs if s > 100))
    print("Components with SNR < 20:", sum(1 for s in snrs if s < 20))

    # Forecast and check error at each horizon
    for h in [96, 192, 336, 720]:
        result = f.forecast(h)
        actual = data_norm[test_start : test_start + h]
        n = min(len(actual), len(result.point_forecast))
        mse = float(np.mean((actual[:n] - result.point_forecast[:n]) ** 2))
        print("Horizon %d: MSE=%.4f, forecast range=[%.2f, %.2f]" % (
            h, mse, result.point_forecast.min(), result.point_forecast.max()
        ))


if __name__ == "__main__":
    main()
