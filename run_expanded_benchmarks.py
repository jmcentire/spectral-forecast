"""Run expanded benchmarks: ETT + Weather + Electricity."""

import numpy as np
import csv
import time
from spectral_forecast.forecast import SpectralForecaster


def load_nixtla_csv(path):
    """Load a single-series CSV with columns: unique_id, ds, y."""
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row["y"]))
            except (ValueError, KeyError):
                continue
    return np.array(values, dtype=np.float64)


def load_ett_csv(path, column="OT"):
    """Load ETT-format CSV."""
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row[column]))
            except (ValueError, KeyError):
                continue
    return np.array(values, dtype=np.float64)


DATASETS = {
    "ETTh1": {"loader": lambda: load_ett_csv("data/ETTh1.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "ETTh2": {"loader": lambda: load_ett_csv("data/ETTh2.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "ETTm1": {"loader": lambda: load_ett_csv("data/ETTm1.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "ETTm2": {"loader": lambda: load_ett_csv("data/ETTm2.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "Weather": {"loader": lambda: load_nixtla_csv("data/weather_series0.csv"), "train_frac": 0.7, "val_frac": 0.1},
    "ECL": {"loader": lambda: load_nixtla_csv("data/ecl_series0.csv"), "train_frac": 0.7, "val_frac": 0.1},
}

# Published reference MSE at H=96 (approximate, from various papers)
REFERENCES = {
    "ETTh1": {"TimesFM": 0.375, "PatchTST": 0.370},
    "ETTh2": {"TimesFM": 0.289, "PatchTST": 0.274},
    "ETTm1": {"TimesFM": 0.320, "PatchTST": 0.293},
    "ETTm2": {"TimesFM": 0.175, "PatchTST": 0.166},
    "Weather": {"PatchTST": 0.149},
    "ECL": {"PatchTST": 0.129},
}

HORIZONS = [96, 192, 336, 720]


def run_one(name, info):
    data = info["loader"]()
    n = len(data)
    train_end = int(n * info["train_frac"])
    val_end = int(n * (info["train_frac"] + info["val_frac"]))

    train = data[:train_end]
    mean, std = train.mean(), train.std()
    if std < 1e-10:
        std = 1.0
    data_norm = (data - mean) / std

    test_start = val_end
    context_length = 512

    results = {}
    for horizon in HORIZONS:
        t0 = time.time()
        all_mse = []

        start = test_start - context_length
        while start + context_length + horizon <= len(data_norm):
            context = data_norm[start : start + context_length]
            actual = data_norm[start + context_length : start + context_length + horizon]

            forecaster = SpectralForecaster()
            result = forecaster.fit_forecast(context, horizon)

            mse = float(np.mean((actual - result.point_forecast) ** 2))
            all_mse.append(mse)
            start += horizon

        elapsed = time.time() - t0
        results[horizon] = {
            "mse": np.mean(all_mse) if all_mse else float("nan"),
            "std": np.std(all_mse) if all_mse else float("nan"),
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

    print()
    print("=" * 90)
    print("EXPANDED BENCHMARK RESULTS -- Normalized MSE -- 6 Datasets")
    print("=" * 90)
    print()
    print("%10s %8s | %10s %10s | %10s %10s" % (
        "Dataset", "Horizon", "Ours MSE", "Ours Std", "PatchTST", "TimesFM"
    ))
    print("-" * 90)

    for name in DATASETS:
        refs = REFERENCES.get(name, {})
        for h in HORIZONS:
            r = all_results[name][h]
            pts = "%.3f" % refs["PatchTST"] if h == 96 and "PatchTST" in refs else "---"
            tfm = "%.3f" % refs["TimesFM"] if h == 96 and "TimesFM" in refs else "---"
            print(
                "%10s %8d | %10.4f %10.4f | %10s %10s  (n=%d, %.1fs)"
                % (name, h, r["mse"], r["std"], pts, tfm, r["n"], r["time"])
            )
        print("-" * 90)

    print()
    print("Total time: %.1fs" % total_elapsed)

    # Summary at H=96
    print()
    print("Summary vs baselines (H=96 MSE):")
    for name in DATASETS:
        ours = all_results[name][96]["mse"]
        refs = REFERENCES.get(name, {})
        parts = ["%s: %.4f" % (name, ours)]
        for ref_name, ref_val in refs.items():
            delta = (1 - ours / ref_val) * 100
            parts.append("vs %s %.3f (%+.0f%%)" % (ref_name, ref_val, delta))
        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
