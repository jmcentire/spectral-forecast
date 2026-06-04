"""Quick single-window diagnostic for a few forecaster settings."""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def _score(label: str, forecaster: SpectralForecaster, context: np.ndarray, actual: np.ndarray) -> None:
    result = forecaster.fit_forecast(context, 96)
    mse = float(np.mean((actual - result.point_forecast) ** 2))
    print(f"{label:<24} MSE={mse:.4f}")


def main() -> None:
    data = load_csv_dataset("data/ETTh1.csv", column="OT")
    train = data[:8640]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    context = data_norm[8640 + 2880 - 512 : 8640 + 2880]
    actual = data_norm[8640 + 2880 : 8640 + 2880 + 96]

    _score("no amplitude damping", SpectralForecaster(amplitude_damping=False), context, actual)
    _score("default", SpectralForecaster(), context, actual)
    _score(
        "recency/shock bounded",
        SpectralForecaster(recency_halflife=256, shock_lookback=128),
        context,
        actual,
    )


if __name__ == "__main__":
    main()
