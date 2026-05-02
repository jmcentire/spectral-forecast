"""Benchmark ForecastEngine ensemble against individual models on ETT."""

import csv
import time
import numpy as np

from spectral_forecast.engine import (
    ForecastEngine,
    ARWorker,
    FourierWorker,
    IterativeWorker,
    SRWorker,
)


def load_ett_csv(path, column="OT"):
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
    "ETTh1": {"file": "data/ETTh1.csv", "train": 8640, "val": 2880, "period": 24},
    "ETTh2": {"file": "data/ETTh2.csv", "train": 8640, "val": 2880, "period": 24},
    "ETTm1": {"file": "data/ETTm1.csv", "train": 34465, "val": 11521, "period": 96},
    "ETTm2": {"file": "data/ETTm2.csv", "train": 34465, "val": 11521, "period": 96},
}

HORIZON = 96
CONTEXT = 512


def main():
    print("ForecastEngine Ensemble Benchmark (H=%d)" % HORIZON)
    print("=" * 70)
    print("%8s | %10s %10s %10s | %10s" % (
        "Dataset", "Ensemble", "AR-only", "Fourier", "Confidence"
    ))
    print("-" * 70)

    for name, info in DATASETS.items():
        data = load_ett_csv(info["file"])
        train = data[: info["train"]]
        mean, std = train.mean(), train.std()
        data_norm = (data - mean) / std
        test_start = info["train"] + info["val"]

        # Run ensemble
        t0 = time.time()
        ensemble_mses = []
        ar_mses = []
        fourier_mses = []
        confidences = []
        agreements = []

        start = test_start - CONTEXT
        while start + CONTEXT + HORIZON <= len(data_norm):
            context = data_norm[start : start + CONTEXT]
            actual = data_norm[start + CONTEXT : start + CONTEXT + HORIZON]

            engine = ForecastEngine(workers=[
                ARWorker(p=24),
                FourierWorker(),
                IterativeWorker(period=info["period"]),
                SRWorker(),
            ])
            engine.observe_batch(context)
            result = engine.predict(HORIZON)

            ensemble_mse = float(np.mean((actual - result.values) ** 2))
            ensemble_mses.append(ensemble_mse)
            confidences.append(result.confidence)
            agreements.append(result.agreement)

            # Individual worker MSEs for comparison
            for wname, wpred in result.worker_predictions.items():
                mse = float(np.mean((actual - wpred.values) ** 2))
                if wname == "AR":
                    ar_mses.append(mse)
                elif wname == "Fourier":
                    fourier_mses.append(mse)

            start += HORIZON

        elapsed = time.time() - t0
        print(
            "%8s | %10.4f %10.4f %10.4f | %10.3f  (%.0fs, n=%d)"
            % (
                name,
                np.mean(ensemble_mses),
                np.mean(ar_mses) if ar_mses else float("nan"),
                np.mean(fourier_mses) if fourier_mses else float("nan"),
                np.mean(confidences),
                elapsed,
                len(ensemble_mses),
            )
        )

    print()
    print("Reference — previous best (spectral-forecast v0.4, standard splits):")
    print("  ETTh1: 0.131  ETTh2: 0.400  ETTm1: 0.088  ETTm2: 0.251")
    print("Reference — TimesFM (200M params):")
    print("  ETTh1: 0.375  ETTh2: 0.289  ETTm1: 0.320  ETTm2: 0.175")


if __name__ == "__main__":
    main()
