"""Pre-pass signal conditioning: detect and remove level shifts before FFT.

A step function (sudden permanent level change) has a broadband 1/f spectral
footprint. If it reaches the Fourier extractor, it gets smeared into dozens
of spurious sinusoidal components. This module detects and removes level
shifts BEFORE spectral analysis, so the FFT only sees continuous, periodic
structure.

Uses median filtering to identify sudden jumps, then fits step functions
at the detected changepoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter


@dataclass
class LevelShift:
    """A detected level shift (step function)."""

    index: int  # changepoint location
    magnitude: float  # size of the jump (signed)


@dataclass
class PrepassResult:
    """Result of pre-pass conditioning."""

    cleaned_signal: NDArray[np.floating]
    level_shifts: list[LevelShift]
    shift_signal: NDArray[np.floating]  # reconstructed level shift component


def detect_level_shifts(
    signal: NDArray[np.floating],
    median_window: int = 21,
    min_sigma: float | None = None,
) -> PrepassResult:
    """Detect and remove level shifts from the signal.

    Uses median filtering to create a smooth baseline, then identifies
    points where the difference between the signal and the baseline
    exceeds a threshold. The threshold adapts to the signal's local
    entropy via kurtosis of the first differences.

    Args:
        signal: 1D time series.
        median_window: Window size for median filter (must be odd).
            Larger = ignores wider transients. Default 21.
        min_sigma: Minimum deviation (in MAD units) for a changepoint.
            If None (default), derived adaptively from the excess kurtosis
            of the first differences. Heavy-tailed signals (high kurtosis)
            get a higher threshold because large single-step jumps are
            normal noise. Light-tailed signals get a lower threshold
            because sudden jumps are genuinely anomalous.

    Returns:
        PrepassResult with cleaned signal, detected shifts, and
        the reconstructed shift component.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if n < median_window * 2:
        return PrepassResult(
            cleaned_signal=signal.copy(),
            level_shifts=[],
            shift_signal=np.zeros(n),
        )

    # Median filter produces a smooth baseline that's robust to outliers
    # and level shifts (the median ignores them if they're narrower than
    # half the window)
    if median_window % 2 == 0:
        median_window += 1
    baseline = median_filter(signal, size=median_window)

    # Compute the difference signal — level shifts show as step functions here
    diff = np.diff(signal)
    # MAD of the difference (robust noise estimate)
    mad = float(np.median(np.abs(diff - np.median(diff))))
    if mad < 1e-10:
        mad = float(np.std(diff)) * 0.6745  # fallback to std-based estimate
    if mad < 1e-10:
        return PrepassResult(
            cleaned_signal=signal.copy(),
            level_shifts=[],
            shift_signal=np.zeros(n),
        )

    # Adaptive threshold from kurtosis of first differences.
    # Gaussian has excess kurtosis = 0. Heavy tails > 0.
    # Base sigma = 3.0 for Gaussian noise. Add kurtosis to raise it
    # for heavy-tailed signals where large jumps are normal.
    if min_sigma is None:
        excess_kurtosis = float(
            np.mean((diff - np.mean(diff)) ** 4) / np.mean((diff - np.mean(diff)) ** 2) ** 2 - 3
        )
        excess_kurtosis = max(excess_kurtosis, 0.0)
        min_sigma = 3.0 + np.sqrt(excess_kurtosis)
        # Floor at 3.0 (Gaussian), cap at 10.0 (extremely heavy tails)
        min_sigma = min(min_sigma, 10.0)

    threshold = min_sigma * mad / 0.6745  # convert MAD to std-equivalent

    # Find changepoints: points where the step size exceeds threshold
    abs_diff = np.abs(diff)
    exceedances = np.where(abs_diff > threshold)[0]

    if len(exceedances) == 0:
        return PrepassResult(
            cleaned_signal=signal.copy(),
            level_shifts=[],
            shift_signal=np.zeros(n),
        )

    # Group adjacent exceedances into single changepoints
    # (a level shift may span 2-3 samples due to smoothing)
    changepoints: list[int] = []
    min_gap = max(median_window // 2, 3)
    last_cp = -min_gap - 1
    for idx in exceedances:
        if idx - last_cp >= min_gap:
            changepoints.append(int(idx))
            last_cp = idx

    # Fit step functions at each changepoint.
    # A genuine level shift must exceed BOTH the noise threshold AND
    # the signal's typical oscillation range. This prevents periodic
    # components from being misidentified as level shifts — if the signal
    # oscillates ±5, a "jump" of 6 is within normal oscillation, not a
    # structural break.
    signal_iqr = float(np.percentile(signal, 75) - np.percentile(signal, 25))
    magnitude_floor = max(threshold, signal_iqr)

    level_shifts: list[LevelShift] = []
    shift_signal = np.zeros(n)

    for cp in changepoints:
        # Estimate magnitude: median after - median before (local windows)
        win = min(median_window, cp, n - cp - 1)
        if win < 3:
            continue
        before = float(np.median(signal[max(0, cp - win) : cp]))
        after = float(np.median(signal[cp + 1 : min(n, cp + 1 + win)]))
        magnitude = after - before

        if abs(magnitude) < magnitude_floor:
            continue

        level_shifts.append(LevelShift(index=cp + 1, magnitude=magnitude))
        shift_signal[cp + 1 :] += magnitude

    cleaned = signal - shift_signal

    return PrepassResult(
        cleaned_signal=cleaned,
        level_shifts=level_shifts,
        shift_signal=shift_signal,
    )
