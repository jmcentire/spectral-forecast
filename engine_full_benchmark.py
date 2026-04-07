"""Full engine benchmark: all 6 datasets."""

import csv
import time
import numpy as np
from spectral_forecast.engine import ForecastEngine, ARWorker, FourierWorker, SRWorker


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

REFERENCES = {
    "ETTh1": {"TimesFM": 0.375, "PatchTST": 0.370, "SF_v04": 0.131},
    "ETTh2": {"TimesFM": 0.289, "PatchTST": 0.274, "SF_v04": 0.400},
    "ETTm1": {"TimesFM": 0.320, "PatchTST": 0.293, "SF_v04": 0.088},
    "ETTm2": {"TimesFM": 0.175, "PatchTST": 0.166, "SF_v04": 0.251},
    "Weather": {"PatchTST": 0.149, "SF_v04": 0.156},
    "ECL": {"PatchTST": 0.129, "SF_v04": 0.609},
}

HORIZON = 96
CONTEXT = 512


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

    t0 = time.time()
    ensemble_mses = []
    worker_mses = {}
    collapses = 0

    start = test_start - CONTEXT
    while start + CONTEXT + HORIZON <= len(data_norm):
        context = data_norm[start : start + CONTEXT]
        actual = data_norm[start + CONTEXT : start + CONTEXT + HORIZON]

        engine = ForecastEngine(workers=[ARWorker(p=24), FourierWorker(), SRWorker()])
        engine.observe_batch(context)
        result = engine.predict(HORIZON)

        ensemble_mses.append(float(np.mean((actual - result.values) ** 2)))

        for wname, wp in result.worker_predictions.items():
            mse = float(np.mean((actual - wp.values) ** 2))
            worker_mses.setdefault(wname, []).append(mse)

        confs = [wp.confidence for wp in result.worker_predictions.values()]
        if len(confs) >= 2:
            s = sorted(confs, reverse=True)
            if s[1] > 0 and s[0] / s[1] > 2.0:
                collapses += 1

        start += HORIZON

    elapsed = time.time() - t0
    return {
        "ensemble": np.mean(ensemble_mses),
        "workers": {k: np.mean(v) for k, v in worker_mses.items()},
        "n": len(ensemble_mses),
        "collapses": collapses,
        "time": elapsed,
    }


def main():
    print("ForecastEngine Full Benchmark (H=%d, C=%d)" % (HORIZON, CONTEXT))
    print("=" * 95)
    print("%10s | %10s %10s %10s | %10s %10s | %6s %5s" % (
        "Dataset", "Ensemble", "AR", "Fourier", "TimesFM", "PatchTST", "Collps", "Time"
    ))
    print("-" * 95)

    all_results = {}
    for name, info in DATASETS.items():
        r = run_one(name, info)
        all_results[name] = r
        refs = REFERENCES.get(name, {})
        tfm = "%.3f" % refs["TimesFM"] if "TimesFM" in refs else "---"
        pts = "%.3f" % refs["PatchTST"] if "PatchTST" in refs else "---"
        ar_mse = r["workers"].get("AR", float("nan"))
        fourier_mse = r["workers"].get("Fourier", float("nan"))
        print("%10s | %10.4f %10.4f %10.4f | %10s %10s | %5d%% %4.0fs" % (
            name, r["ensemble"], ar_mse, fourier_mse,
            tfm, pts,
            100 * r["collapses"] // r["n"], r["time"],
        ))

    print("-" * 95)
    print()

    # Summary
    print("Summary vs references (H=96 MSE):")
    for name in DATASETS:
        r = all_results[name]
        refs = REFERENCES.get(name, {})
        parts = ["Ensemble=%.4f" % r["ensemble"]]
        for ref_name in ["SF_v04", "TimesFM", "PatchTST"]:
            if ref_name in refs:
                ref_val = refs[ref_name]
                delta = (1 - r["ensemble"] / ref_val) * 100
                parts.append("vs %s %.3f (%+.0f%%)" % (ref_name, ref_val, delta))
        print("  %s: %s" % (name, " | ".join(parts)))


if __name__ == "__main__":
    main()
