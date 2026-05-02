"""Diagnose ETTm1 window 17 at horizon 720."""

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

    start = test_start - context_length + 17 * horizon
    context = data_norm[start : start + context_length]
    actual = data_norm[start + context_length : start + context_length + horizon]

    forecaster = SpectralForecaster()
    result = forecaster.fit_forecast(context, horizon)

    print("Trend:", forecaster._trend.model.trend_type.value)
    print("Params:", forecaster._trend.model.params)

    t_end = np.array([512.0 + 720])
    full = forecaster._trend.model.predict(t_end)[0]
    damped = forecaster._trend.model.predict_damped(t_end, 511.0, 128.0)[0]
    print("Full trend at t+720: %.2f" % full)
    print("Damped trend at t+720: %.2f" % damped)
    print("Forecast range: [%.2f, %.2f]" % (result.point_forecast.min(), result.point_forecast.max()))
    print("Actual range: [%.2f, %.2f]" % (actual.min(), actual.max()))
    print("MSE: %.4f" % float(np.mean((actual - result.point_forecast) ** 2)))

    # Check: what tangent line value at t+720?
    slope_at_boundary = (
        forecaster._trend.model.predict(np.array([512.0]))
        - forecaster._trend.model.predict(np.array([510.0]))
    ) / 2.0
    boundary_val = forecaster._trend.model.predict(np.array([511.0]))[0]
    tangent_720 = boundary_val + slope_at_boundary[0] * 720
    print("Tangent at t+720: %.2f" % tangent_720)


if __name__ == "__main__":
    main()
