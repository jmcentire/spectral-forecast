"""Benchmark: 6-worker ensemble (v2.0) vs 3-worker (v1.0)."""

import csv
import time
import numpy as np
from spectral_forecast.engine import ForecastEngine, ARWorker, FourierWorker, SRWorker
from spectral_forecast.workers_v2 import (
    ShockAwareFourierWorker,
    NormalityAwareFourierWorker,
    AggressiveDampingFourierWorker,
)


def load_ett(path):
    values = []
    with open(path) as f:
        for row in csv.DictReader(f):
            values.append(float(row["OT"]))
    return np.array(values, dtype=np.float64)


def load_nixtla(path):
    values = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                values.append(float(row["y"]))
            except (ValueError, KeyError):
                continue
    return np.array(values, dtype=np.float64)


DATASETS = {
    "ETTh1": {"loader": lambda: load_ett("data/ETTh1.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "ETTh2": {"loader": lambda: load_ett("data/ETTh2.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "ETTm1": {"loader": lambda: load_ett("data/ETTm1.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "ETTm2": {"loader": lambda: load_ett("data/ETTm2.csv"), "train_frac": 0.6, "val_frac": 0.2},
    "Weather": {"loader": lambda: load_nixtla("data/weather_series0.csv"), "train_frac": 0.7, "val_frac": 0.1},
    "ECL": {"loader": lambda: load_nixtla("data/ecl_series0.csv"), "train_frac": 0.7, "val_frac": 0.1},
}

HORIZON = 96
CONTEXT = 512


def run_ensemble(name, info, workers_factory):
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

    mses = []
    start = test_start - CONTEXT
    while start + CONTEXT + HORIZON <= len(data_norm):
        context = data_norm[start : start + CONTEXT]
        actual = data_norm[start + CONTEXT : start + CONTEXT + HORIZON]
        engine = ForecastEngine(workers=workers_factory())
        engine.observe_batch(context)
        result = engine.predict(HORIZON)
        mses.append(float(np.mean((actual - result.values) ** 2)))
        start += HORIZON

    return np.mean(mses), len(mses)


def v1_workers():
    return [ARWorker(p=24), FourierWorker(), SRWorker()]


def v2_workers():
    return [
        ARWorker(p=24),
        FourierWorker(),
        SRWorker(),
        ShockAwareFourierWorker(),
        NormalityAwareFourierWorker(),
        AggressiveDampingFourierWorker(),
    ]


def main():
    print("v1.0 (3 workers) vs v2.0 (6 workers) — H=%d" % HORIZON)
    print("=" * 70)
    print("%10s | %10s %10s | %10s %10s" % (
        "Dataset", "v1.0", "v2.0", "Delta", "Verdict"
    ))
    print("-" * 70)

    for name, info in DATASETS.items():
        t0 = time.time()
        mse_v1, n = run_ensemble(name, info, v1_workers)
        mse_v2, _ = run_ensemble(name, info, v2_workers)
        elapsed = time.time() - t0

        delta = (mse_v1 - mse_v2) / mse_v1 * 100
        verdict = "v2 wins" if mse_v2 < mse_v1 else "v1 wins" if mse_v1 < mse_v2 else "tie"
        print("%10s | %10.4f %10.4f | %+9.1f%% %10s  (%.0fs, n=%d)" % (
            name, mse_v1, mse_v2, delta, verdict, elapsed, n
        ))

    print()
    print("Reference — TimesFM (200M params):")
    print("  ETTh1: 0.375  ETTh2: 0.289  ETTm1: 0.320  ETTm2: 0.175")


if __name__ == "__main__":
    main()
