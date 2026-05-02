"""Debug: what does the pre-pass detect on the failing test signal?"""

import numpy as np
from spectral_forecast.prepass import detect_level_shifts

rng = np.random.default_rng(42)
n = 1000
t = np.arange(n, dtype=np.float64)
signal = 5.0 * np.cos(2 * np.pi * 0.03 * t) + 0.05 * t + rng.normal(0, 1.0, n)

result = detect_level_shifts(signal)
print("Level shifts detected:", len(result.level_shifts))
for ls in result.level_shifts:
    print("  idx=%d mag=%.3f" % (ls.index, ls.magnitude))
print("Shift signal at end:", result.shift_signal[-1])
print("Signal range: [%.2f, %.2f]" % (signal.min(), signal.max()))
print("Cleaned range: [%.2f, %.2f]" % (result.cleaned_signal.min(), result.cleaned_signal.max()))

# What's the adaptive threshold?
diff = np.diff(signal)
mad = float(np.median(np.abs(diff - np.median(diff))))
excess_k = float(np.mean((diff - np.mean(diff))**4) / np.mean((diff - np.mean(diff))**2)**2 - 3)
adaptive_sigma = 3.0 + np.sqrt(max(excess_k, 0))
print("MAD: %.4f, excess kurtosis: %.4f, adaptive sigma: %.2f" % (mad, excess_k, adaptive_sigma))
print("Threshold: %.4f" % (adaptive_sigma * mad / 0.6745))
