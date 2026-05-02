"""Profile worker confidence across datasets to find collapse threshold."""

import csv
import numpy as np
from spectral_forecast.engine import ForecastEngine, ARWorker, FourierWorker, SRWorker

DATASETS = {
    "ETTh1": {"file": "data/ETTh1.csv", "train": 8640, "val": 2880},
    "ETTh2": {"file": "data/ETTh2.csv", "train": 8640, "val": 2880},
    "ETTm1": {"file": "data/ETTm1.csv", "train": 34465, "val": 11521},
    "ETTm2": {"file": "data/ETTm2.csv", "train": 34465, "val": 11521},
}


def load(path):
    values = []
    with open(path) as f:
        for row in csv.DictReader(f):
            values.append(float(row["OT"]))
    return np.array(values, dtype=np.float64)


def main():
    print("Worker Confidence + MSE Profile (3 windows per dataset)")
    print("=" * 80)

    for name, info in DATASETS.items():
        data = load(info["file"])
        train = data[: info["train"]]
        mean, std = train.mean(), train.std()
        data_norm = (data - mean) / std
        test_start = info["train"] + info["val"]

        # Sample 3 windows spread across the test set
        n_test = len(data_norm) - test_start
        offsets = [0, n_test // 3, 2 * n_test // 3]

        print("\n%s:" % name)
        for offset in offsets:
            start = test_start - 512 + offset
            if start + 512 + 96 > len(data_norm):
                continue
            context = data_norm[start : start + 512]
            actual = data_norm[start + 512 : start + 512 + 96]

            engine = ForecastEngine(workers=[
                ARWorker(p=24),
                FourierWorker(),
                SRWorker(),
            ])
            engine.observe_batch(context)
            result = engine.predict(96)

            ensemble_mse = float(np.mean((actual - result.values) ** 2))
            print("  Window @%d: ensemble MSE=%.4f conf=%.3f agree=%.3f" % (
                offset, ensemble_mse, result.confidence, result.agreement
            ))

            best_name = ""
            best_mse = float("inf")
            for wname, wp in sorted(result.worker_predictions.items()):
                mse = float(np.mean((actual - wp.values) ** 2))
                gap = wp.confidence - min(
                    p.confidence for p in result.worker_predictions.values()
                )
                marker = ""
                if mse < best_mse:
                    best_mse = mse
                    best_name = wname
                print("    %-8s MSE=%.4f conf=%.3f gap=%.3f" % (
                    wname, mse, wp.confidence, gap
                ))

            # Would collapsing to best worker help?
            if best_mse < ensemble_mse:
                delta = (ensemble_mse - best_mse) / ensemble_mse * 100
                print("    -> %s wins by %.1f%% over ensemble" % (best_name, delta))
            else:
                print("    -> ensemble wins")


if __name__ == "__main__":
    main()
