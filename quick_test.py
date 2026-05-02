"""Quick single-window test of each mode."""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster

data = load_csv_dataset("data/ETTh1.csv", column="OT")
train = data[:8640]
mean, std = train.mean(), train.std()
data_norm = (data - mean) / std

context = data_norm[8640 + 2880 - 512 : 8640 + 2880]
actual = data_norm[8640 + 2880 : 8640 + 2880 + 96]

# Baseline: no new features
f = SpectralForecaster(iterative_refinement=False, sr_noise=False)
result = f.fit_forecast(context, 96)
mse = float(np.mean((actual - result.point_forecast) ** 2))
print("No refinement, no SR: MSE=%.4f" % mse)

# Iterative only
f2 = SpectralForecaster(iterative_refinement=True, sr_noise=False, max_refinement_iter=2)
result2 = f2.fit_forecast(context, 96)
mse2 = float(np.mean((actual - result2.point_forecast) ** 2))
print("Iterative only:      MSE=%.4f" % mse2)

# SR only
f3 = SpectralForecaster(iterative_refinement=False, sr_noise=True, sr_n_ensemble=5)
result3 = f3.fit_forecast(context, 96)
mse3 = float(np.mean((actual - result3.point_forecast) ** 2))
print("SR only:             MSE=%.4f" % mse3)
