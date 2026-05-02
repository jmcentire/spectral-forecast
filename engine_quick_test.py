"""Quick 2-window engine test on ETTh1."""

import csv
import numpy as np
from spectral_forecast.engine import ForecastEngine, ARWorker, FourierWorker, SRWorker

values = []
with open("data/ETTh1.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        values.append(float(row["OT"]))
data = np.array(values, dtype=np.float64)
train = data[:8640]
mean, std = train.mean(), train.std()
data_norm = (data - mean) / std

test_start = 8640 + 2880
context = data_norm[test_start - 512 : test_start]
actual = data_norm[test_start : test_start + 96]

engine = ForecastEngine(workers=[ARWorker(p=24), FourierWorker(), SRWorker()])
engine.observe_batch(context)
result = engine.predict(96)

ensemble_mse = float(np.mean((actual - result.values) ** 2))
print("Ensemble MSE: %.4f" % ensemble_mse)
print("Confidence: %.3f, Agreement: %.3f" % (result.confidence, result.agreement))
for name, wp in result.worker_predictions.items():
    mse = float(np.mean((actual - wp.values) ** 2))
    print("  %s: MSE=%.4f, conf=%.3f" % (name, mse, wp.confidence))
