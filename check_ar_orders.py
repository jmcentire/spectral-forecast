"""Check what AR orders Layer 3b selects across datasets."""

import numpy as np
from spectral_forecast.benchmark import load_csv_dataset
from spectral_forecast.forecast import SpectralForecaster


def check_dataset(name, filepath, train_n, val_n):
    data = load_csv_dataset(filepath, column="OT")
    train = data[:train_n]
    mean, std = train.mean(), train.std()
    data_norm = (data - mean) / std
    test_start = train_n + val_n
    context_length = 512

    orders = []
    start = test_start - context_length
    while start + context_length + 96 <= len(data_norm):
        context = data_norm[start : start + context_length]
        f = SpectralForecaster()
        f.fit(context)
        orders.append(f._local.model.order)
        start += 96

    print("%s: AR orders across %d windows:" % (name, len(orders)))
    from collections import Counter
    counts = Counter(orders)
    for order in sorted(counts.keys()):
        print("  AR(%d): %d windows (%.0f%%)" % (order, counts[order], 100 * counts[order] / len(orders)))
    print("  Mean order: %.1f" % np.mean(orders))
    print()


def main():
    check_dataset("ETTh1", "data/ETTh1.csv", 8640, 2880)
    check_dataset("ETTh2", "data/ETTh2.csv", 8640, 2880)
    check_dataset("ETTm1", "data/ETTm1.csv", 34465, 11521)
    check_dataset("ETTm2", "data/ETTm2.csv", 34465, 11521)


if __name__ == "__main__":
    main()
