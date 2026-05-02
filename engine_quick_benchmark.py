"""Quick engine benchmark: ETTh1 + ETTh2 only, with confidence breakdown."""

import csv
import time
import numpy as np
from spectral_forecast.engine import ForecastEngine, ARWorker, FourierWorker, SRWorker


def load(path):
    values = []
    with open(path) as f:
        for row in csv.DictReader(f):
            values.append(float(row["OT"]))
    return np.array(values, dtype=np.float64)


DATASETS = {
    "ETTh1": {"file": "data/ETTh1.csv", "train": 8640, "val": 2880},
    "ETTh2": {"file": "data/ETTh2.csv", "train": 8640, "val": 2880},
}

HORIZON = 96
CONTEXT = 512


def main():
    for name, info in DATASETS.items():
        data = load(info["file"])
        train = data[: info["train"]]
        mean, std = train.mean(), train.std()
        data_norm = (data - mean) / std
        test_start = info["train"] + info["val"]

        ensemble_mses = []
        ar_mses = []
        fourier_mses = []
        collapses = 0

        t0 = time.time()
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
                if wname == "AR":
                    ar_mses.append(mse)
                elif wname == "Fourier":
                    fourier_mses.append(mse)

            # Detect collapse
            confs = [wp.confidence for wp in result.worker_predictions.values()]
            if len(confs) >= 2:
                s = sorted(confs, reverse=True)
                if s[1] > 0 and s[0] / s[1] > 2.0:
                    collapses += 1

            start += HORIZON

        elapsed = time.time() - t0
        n = len(ensemble_mses)
        print("%s (n=%d, %.0fs):" % (name, n, elapsed))
        print("  Ensemble: %.4f" % np.mean(ensemble_mses))
        print("  AR alone: %.4f" % np.mean(ar_mses))
        print("  Fourier:  %.4f" % np.mean(fourier_mses))
        print("  Collapses: %d/%d (%.0f%%)" % (collapses, n, 100 * collapses / n))
        print("  Best individual: %.4f" % min(np.mean(ar_mses), np.mean(fourier_mses)))
        print()

    print("Reference:")
    print("  spectral-forecast v0.4: ETTh1=0.131, ETTh2=0.400")
    print("  TimesFM:                ETTh1=0.375, ETTh2=0.289")


if __name__ == "__main__":
    main()
