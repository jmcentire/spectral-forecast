"""Residual diagnostic: what fraction of signal does our model capture?

The key question isn't whether the raw signal is periodic — it's whether
our decomposition captures enough of the variance. The residual tells us
what we're missing.
"""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


DATASETS = {
    "ETTh1": {"file": "data/ETTh1.csv", "train": 8640, "val": 2880},
    "ETTh2": {"file": "data/ETTh2.csv", "train": 8640, "val": 2880},
    "ETTm1": {"file": "data/ETTm1.csv", "train": 34465, "val": 11521},
    "ETTm2": {"file": "data/ETTm2.csv", "train": 34465, "val": 11521},
}


def analyze_residuals(name, info):
    data = load_csv_dataset(info["file"], column="OT")
    train = data[: info["train"]]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std

    test_start = info["train"] + info["val"]
    context_length = 512

    # Analyze several windows
    variance_explained = []
    periodic_var_frac = []
    trend_var_frac = []
    local_var_frac = []
    residual_autocorr = []
    n_components = []

    start = test_start - context_length
    count = 0
    while start + context_length + 96 <= len(data_norm) and count < 30:
        context = data_norm[start : start + context_length]

        f = SpectralForecaster()
        f.fit(context)

        t = np.arange(512, dtype=np.float64)

        # Reconstruct each layer's contribution
        periodic = np.zeros(512)
        for comp in f._extraction.components:
            periodic += comp.evaluate(t)
        trend = f._trend.model.predict(t)
        shock = np.zeros(512)
        for s in f._shocks.shocks:
            shock += s.evaluate(t)
        local_vals = f._local.fitted_values

        # Variance of each component
        total_var = float(np.var(context))
        if total_var > 0:
            periodic_var_frac.append(float(np.var(periodic)) / total_var)
            trend_var_frac.append(float(np.var(trend)) / total_var)
            local_var_frac.append(float(np.var(local_vals)) / total_var)

            residual_var = float(np.var(f._residual))
            variance_explained.append(1 - residual_var / total_var)

        n_components.append(len(f._extraction.components))

        # Residual autocorrelation at lag 1 (measure of remaining structure)
        r = f._residual
        if len(r) > 1 and np.std(r) > 0:
            ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            residual_autocorr.append(ac1)

        start += 96
        count += 1

    print("%s:" % name)
    print("  Variance explained by model: %.1f%%" % (np.mean(variance_explained) * 100))
    print("    Periodic (Layer 1):  %.1f%% of total variance" % (np.mean(periodic_var_frac) * 100))
    print("    Trend (Layer 2):     %.1f%% of total variance" % (np.mean(trend_var_frac) * 100))
    print("    Local (Layer 3b):    %.1f%% of total variance" % (np.mean(local_var_frac) * 100))
    print("  Mean components extracted: %.1f" % np.mean(n_components))
    print("  Residual lag-1 autocorrelation: %.3f" % np.mean(residual_autocorr))
    print("    (high = remaining structure we're missing)")
    print()

    return {
        "var_explained": np.mean(variance_explained),
        "periodic_frac": np.mean(periodic_var_frac),
        "residual_ac1": np.mean(residual_autocorr),
    }


def main():
    results = {}
    for name, info in DATASETS.items():
        results[name] = analyze_residuals(name, info)

    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print()
    print("%8s %12s %12s %12s %10s" % (
        "Dataset", "Var Expl.", "Periodic %", "Resid AC(1)", "Ours h=96"
    ))
    print("-" * 58)

    our_mse = {"ETTh1": 0.132, "ETTh2": 0.406, "ETTm1": 0.089, "ETTm2": 0.253}
    for name in DATASETS:
        r = results[name]
        print("%8s %11.1f%% %11.1f%% %12.3f %10.3f" % (
            name, r["var_explained"] * 100, r["periodic_frac"] * 100,
            r["residual_ac1"], our_mse[name]
        ))

    print()
    print("The residual autocorrelation is the smoking gun.")
    print("High AC(1) = our residuals still have structure = we're leaving")
    print("performance on the table. That structure needs a different basis.")


if __name__ == "__main__":
    main()
