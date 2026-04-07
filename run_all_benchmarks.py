"""Run all four ETT benchmarks to validate against overfitting."""

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

HORIZONS = [96, 192, 336, 720]

# Published reference numbers (approximate normalized MSE, horizon=96)
REFERENCES = {
    "ETTh1": {"TimesFM": 0.375, "PatchTST": 0.370},
    "ETTh2": {"TimesFM": 0.289, "PatchTST": 0.274},
    "ETTm1": {"TimesFM": 0.320, "PatchTST": 0.293},
    "ETTm2": {"TimesFM": 0.175, "PatchTST": 0.166},
}


def run_one(name, info):
    data = load_csv_dataset(info["file"], column="OT")
    train = data[: info["train"]]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    test_start = info["train"] + info["val"]
    context_length = 512

    results = {}
    for horizon in HORIZONS:
        t0 = time.time()
        all_mse = []
        all_mae = []

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
            start += horizon

        elapsed = time.time() - t0
        results[horizon] = {
            "mse": np.mean(all_mse),
            "mae": np.mean(all_mae),
            "n": len(all_mse),
            "time": elapsed,
        }

    return results


def main():
    all_results = {}
    total_t0 = time.time()

    for name, info in DATASETS.items():
        print("Running %s..." % name)
        all_results[name] = run_one(name, info)

    total_elapsed = time.time() - total_t0

    # Print comparison table
    print()
    print("=" * 85)
    print("BENCHMARK RESULTS — Normalized MSE — Standard Train/Val/Test Splits")
    print("=" * 85)
    print()

    # Header
    print("%8s %8s | %10s %10s | %10s %10s" % (
        "Dataset", "Horizon", "Ours MSE", "Ours MAE", "TimesFM", "PatchTST"
    ))
    print("-" * 85)

    for name in DATASETS:
        refs = REFERENCES.get(name, {})
        for h in HORIZONS:
            r = all_results[name][h]
            tfm = refs.get("TimesFM", "")
            pts = refs.get("PatchTST", "")
            if h == 96:
                tfm_str = "%.3f" % tfm if tfm else "—"
                pts_str = "%.3f" % pts if pts else "—"
            else:
                tfm_str = "—"
                pts_str = "—"
            print(
                "%8s %8d | %10.4f %10.4f | %10s %10s"
                % (name, h, r["mse"], r["mae"], tfm_str, pts_str)
            )
        print("-" * 85)

    print()
    print("Total benchmark time: %.1fs" % total_elapsed)
    print()

    # Summary: how many horizons do we beat TimesFM at h=96?
    print("Summary vs TimesFM (horizon=96 MSE):")
    for name in DATASETS:
        ours = all_results[name][96]["mse"]
        tfm = REFERENCES.get(name, {}).get("TimesFM")
        if tfm:
            delta = (1 - ours / tfm) * 100
            print(
                "  %s: %.4f vs %.3f (%+.0f%%)"
                % (name, ours, tfm, delta)
            )


if __name__ == "__main__":
    main()
