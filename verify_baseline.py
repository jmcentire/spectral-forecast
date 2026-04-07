"""Verify v0.4 baseline numbers before applying fixes."""

import numpy as np
import time
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster

DATASETS = {
    "ETTh1": {"file": "data/ETTh1.csv", "train": 8640, "val": 2880},
    "ETTh2": {"file": "data/ETTh2.csv", "train": 8640, "val": 2880},
    "ETTm1": {"file": "data/ETTm1.csv", "train": 34465, "val": 11521},
    "ETTm2": {"file": "data/ETTm2.csv", "train": 34465, "val": 11521},
}


def run_h96(name, info):
    data = load_csv_dataset(info["file"], column="OT")
    train = data[: info["train"]]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std
    test_start = info["train"] + info["val"]

    all_mse = []
    start = test_start - 512
    while start + 512 + 96 <= len(data_norm):
        context = data_norm[start : start + 512]
        actual = data_norm[start + 512 : start + 512 + 96]
        f = SpectralForecaster()
        result = f.fit_forecast(context, 96)
        mse = float(np.mean((actual - result.point_forecast) ** 2))
        all_mse.append(mse)
        start += 96

    return np.mean(all_mse)


def main():
    print("v0.4 Baseline Verification (H=96)")
    for name, info in DATASETS.items():
        mse = run_h96(name, info)
        print("  %s: %.4f" % (name, mse))


if __name__ == "__main__":
    main()
