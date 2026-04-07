"""Spectral density diagnostic: why does Fourier dominate on some datasets?

Computes and compares the power spectral density of all four ETT datasets.
Clean peaks = periodic structure = Fourier wins.
Flat/broad spectrum = non-periodic = need a different basis.
"""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset


DATASETS = {
    "ETTh1": {"file": "data/ETTh1.csv", "train": 8640},
    "ETTh2": {"file": "data/ETTh2.csv", "train": 8640},
    "ETTm1": {"file": "data/ETTm1.csv", "train": 34465},
    "ETTm2": {"file": "data/ETTm2.csv", "train": 34465},
}


def spectral_profile(data, name):
    """Compute spectral density metrics for a dataset."""
    # Normalize
    mean, std = data.mean(), data.std()
    normed = (data - mean) / std

    # Power spectral density via FFT
    n = len(normed)
    fft_vals = np.fft.rfft(normed)
    power = np.abs(fft_vals) ** 2 / n
    freqs = np.fft.rfftfreq(n)

    # Skip DC
    power = power[1:]
    freqs = freqs[1:]

    # Metrics
    total_power = power.sum()

    # Spectral concentration: what fraction of power is in the top K peaks?
    sorted_power = np.sort(power)[::-1]
    top5_frac = sorted_power[:5].sum() / total_power
    top10_frac = sorted_power[:10].sum() / total_power
    top20_frac = sorted_power[:20].sum() / total_power

    # Peak-to-median ratio: how much do peaks stand out?
    median_power = np.median(power)
    peak_power = sorted_power[0]
    peak_ratio = peak_power / median_power if median_power > 0 else float("inf")

    # Spectral entropy: 0 = all power in one bin (pure sine), 1 = uniform (white noise)
    p_norm = power / total_power
    p_norm = p_norm[p_norm > 0]
    max_entropy = np.log(len(power))
    entropy = -np.sum(p_norm * np.log(p_norm)) / max_entropy

    # Top 5 peak frequencies and their periods
    top5_idx = np.argsort(power)[-5:][::-1]
    peaks = [(freqs[i], 1.0 / freqs[i] if freqs[i] > 0 else float("inf"),
              power[i] / total_power * 100) for i in top5_idx]

    print("%s (%d points):" % (name, n))
    print("  Spectral concentration:")
    print("    Top 5 peaks:  %.1f%% of total power" % (top5_frac * 100))
    print("    Top 10 peaks: %.1f%% of total power" % (top10_frac * 100))
    print("    Top 20 peaks: %.1f%% of total power" % (top20_frac * 100))
    print("  Peak-to-median ratio: %.0f" % peak_ratio)
    print("  Spectral entropy: %.3f (0=periodic, 1=white noise)" % entropy)
    print("  Top 5 frequencies:")
    for freq, period, pct in peaks:
        print("    freq=%.6f period=%.1f samples (%.1f%% power)" % (freq, period, pct))
    print()

    return {
        "top5_frac": top5_frac,
        "top10_frac": top10_frac,
        "entropy": entropy,
        "peak_ratio": peak_ratio,
    }


def main():
    results = {}
    for name, info in DATASETS.items():
        data = load_csv_dataset(info["file"], column="OT")
        train = data[: info["train"]]
        results[name] = spectral_profile(train, name)

    # Summary comparison
    print("=" * 70)
    print("SUMMARY: Spectral character vs forecast performance")
    print("=" * 70)
    print()
    print("%8s %12s %12s %10s %10s" % (
        "Dataset", "Top5 Power", "Entropy", "Peak Ratio", "Ours h=96"
    ))
    print("-" * 55)

    our_mse = {"ETTh1": 0.132, "ETTh2": 0.406, "ETTm1": 0.089, "ETTm2": 0.253}
    for name in DATASETS:
        r = results[name]
        print("%8s %11.1f%% %12.3f %10.0f %10.3f" % (
            name, r["top5_frac"] * 100, r["entropy"], r["peak_ratio"], our_mse[name]
        ))

    print()
    print("Interpretation:")
    print("  High spectral concentration + low entropy = periodic signal")
    print("  = Fourier basis is the right decomposition = we dominate")
    print()
    print("  Low concentration + high entropy = broadband/non-periodic")
    print("  = need wavelets, EMD, or learned basis = transformer wins")


if __name__ == "__main__":
    main()
