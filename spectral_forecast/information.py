"""Finite-window information readiness diagnostics.

These functions estimate when a time-series prefix has enough structure to
justify spectral extraction. They are intentionally domain-agnostic: the only
question is whether the window contains concentrated spectral structure above
the finite-sample noise extremes expected from a random stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma


@dataclass(frozen=True)
class InformationReadiness:
    """Readiness metrics for one time-series window."""

    n: int
    usable_bins: int
    min_usable_bins: int
    sample_rate: float
    min_cycles: float
    spectral_entropy: float
    noise_entropy: float
    entropy_deficit: float
    peak_frequency: float
    peak_period_samples: float
    peak_cycles: float
    noise_floor: float
    peak_power: float
    peak_snr: float
    peak_surprise: float
    peak_p_value: float
    extraction_margin: float
    readiness_score: float
    ready: bool
    reason: str

    @property
    def seconds(self) -> float:
        """Window duration in sample-rate units."""
        return self.n / self.sample_rate if self.sample_rate > 0 else float("inf")

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Serialize readiness metrics for reports."""
        return {
            "n": self.n,
            "seconds": self.seconds,
            "usable_bins": self.usable_bins,
            "min_usable_bins": self.min_usable_bins,
            "sample_rate": self.sample_rate,
            "min_cycles": self.min_cycles,
            "spectral_entropy": self.spectral_entropy,
            "noise_entropy": self.noise_entropy,
            "entropy_deficit": self.entropy_deficit,
            "peak_frequency": self.peak_frequency,
            "peak_period_samples": self.peak_period_samples,
            "peak_cycles": self.peak_cycles,
            "noise_floor": self.noise_floor,
            "peak_power": self.peak_power,
            "peak_snr": self.peak_snr,
            "peak_surprise": self.peak_surprise,
            "peak_p_value": self.peak_p_value,
            "extraction_margin": self.extraction_margin,
            "readiness_score": self.readiness_score,
            "ready": self.ready,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReadinessScan:
    """Readiness curve across increasing prefix lengths."""

    points: list[InformationReadiness]
    first_ready_n: int | None
    stable_ready_n: int | None
    stable_windows: int
    min_snr: float
    min_entropy_deficit: float
    min_usable_bins: int

    @property
    def first_ready(self) -> InformationReadiness | None:
        if self.first_ready_n is None:
            return None
        for point in self.points:
            if point.n == self.first_ready_n:
                return point
        return None

    @property
    def stable_ready(self) -> InformationReadiness | None:
        if self.stable_ready_n is None:
            return None
        for point in self.points:
            if point.n == self.stable_ready_n:
                return point
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize scan metrics for reports."""
        return {
            "first_ready_n": self.first_ready_n,
            "stable_ready_n": self.stable_ready_n,
            "stable_windows": self.stable_windows,
            "min_snr": self.min_snr,
            "min_entropy_deficit": self.min_entropy_deficit,
            "min_usable_bins": self.min_usable_bins,
            "points": [point.to_dict() for point in self.points],
        }


def _detrend(values: NDArray[np.floating]) -> NDArray[np.float64]:
    n = len(values)
    if n < 2:
        return np.asarray(values, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    design = np.column_stack([t, np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return np.asarray(values - design @ coeffs, dtype=np.float64)


def _normalized_entropy(power: NDArray[np.floating]) -> float:
    total = float(np.sum(power))
    if total <= 0 or len(power) <= 1:
        return 1.0
    probs = np.asarray(power, dtype=np.float64) / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / np.log(len(power))


def _expected_white_noise_entropy(n_bins: int) -> float:
    """Expected normalized periodogram entropy for finite white-noise samples."""
    if n_bins <= 1:
        return 1.0
    # Normalized periodogram proportions for white noise follow Dirichlet(1).
    # E[H] in nats is psi(K + 1) - psi(2), then normalize by log(K).
    return float((digamma(n_bins + 1) - digamma(2)) / np.log(n_bins))


def information_readiness(
    values: NDArray[np.floating],
    *,
    sample_rate: float = 1.0,
    min_snr: float = 2.0,
    min_cycles: float = 2.0,
    min_entropy_deficit: float = 0.02,
    min_usable_bins: int = 16,
    detrend: bool = True,
) -> InformationReadiness:
    """Measure whether a finite window is ready for spectral extraction.

    Readiness is conservative: both the entropy gap and the strongest peak's
    finite-sample surprise must clear their thresholds. Entropy gap is measured
    against the expected periodogram entropy of finite white noise, not a
    perfect uniform spectrum.
    """

    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("Series contains NaN or Inf")
    if len(y) < 8:
        raise ValueError(f"Series too short for readiness (n={len(y)}, need >= 8)")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if min_usable_bins < 1:
        raise ValueError("min_usable_bins must be >= 1")

    n = len(y)
    work = _detrend(y) if detrend else y.copy()
    power = np.abs(np.fft.rfft(work)) ** 2
    freqs = np.fft.rfftfreq(n)

    min_freq = min_cycles / n
    min_bin = max(1, int(np.ceil(min_freq * n)))
    usable_power = power[min_bin:]
    usable_freqs = freqs[min_bin:]
    usable_bins = len(usable_power)

    if usable_bins == 0:
        return InformationReadiness(
            n=n,
            usable_bins=0,
            min_usable_bins=min_usable_bins,
            sample_rate=sample_rate,
            min_cycles=min_cycles,
            spectral_entropy=1.0,
            noise_entropy=1.0,
            entropy_deficit=0.0,
            peak_frequency=0.0,
            peak_period_samples=float("inf"),
            peak_cycles=0.0,
            noise_floor=0.0,
            peak_power=0.0,
            peak_snr=0.0,
            peak_surprise=0.0,
            peak_p_value=1.0,
            extraction_margin=0.0,
            readiness_score=0.0,
            ready=False,
            reason="no_usable_bins",
        )

    spectral_entropy = _normalized_entropy(usable_power)
    noise_entropy = _expected_white_noise_entropy(usable_bins)
    entropy_deficit = max(noise_entropy - spectral_entropy, 0.0)

    peak_local = int(np.argmax(usable_power))
    peak_power = float(usable_power[peak_local])
    peak_frequency = float(usable_freqs[peak_local])
    peak_period_samples = 1.0 / peak_frequency if peak_frequency > 0 else float("inf")
    peak_cycles = peak_frequency * n
    noise_floor = float(np.median(usable_power))
    extreme_value_threshold = noise_floor * (1.0 + np.log(max(usable_bins, 2)))
    peak_snr = peak_power / max(noise_floor, 1e-30)
    peak_surprise = peak_power / max(extreme_value_threshold, 1e-30)
    # Under a white-noise periodogram, each bin is approximately exponential.
    # The median estimates mean * ln(2), so convert before computing the
    # probability that any of K bins would exceed the observed peak.
    expected_noise_mean = noise_floor / np.log(2.0) if noise_floor > 0 else 1e-30
    single_bin_tail = float(np.exp(-peak_power / max(expected_noise_mean, 1e-30)))
    no_exceed = float(np.exp(usable_bins * np.log1p(-min(single_bin_tail, 1.0 - 1e-15))))
    peak_p_value = max(0.0, min(1.0, 1.0 - no_exceed))
    extraction_margin = peak_surprise / max(min_snr, 1e-30)

    entropy_score = (
        1.0
        if min_entropy_deficit <= 0
        else min(1.0, entropy_deficit / max(min_entropy_deficit, 1e-30))
    )
    peak_score = min(1.0, extraction_margin)
    bin_score = min(1.0, usable_bins / max(min_usable_bins, 1))
    readiness_score = min(entropy_score, peak_score, bin_score)

    ready = bool(
        usable_bins >= min_usable_bins
        and entropy_deficit >= min_entropy_deficit
        and peak_surprise >= min_snr
        and peak_cycles >= min_cycles
    )
    if ready:
        reason = "ready"
    elif usable_bins < min_usable_bins:
        reason = "too_few_bins"
    elif entropy_deficit < min_entropy_deficit:
        reason = "entropy_diffuse"
    elif peak_surprise < min_snr:
        reason = "peak_below_noise_extreme"
    else:
        reason = "insufficient_cycles"

    return InformationReadiness(
        n=n,
        usable_bins=usable_bins,
        min_usable_bins=min_usable_bins,
        sample_rate=sample_rate,
        min_cycles=min_cycles,
        spectral_entropy=spectral_entropy,
        noise_entropy=noise_entropy,
        entropy_deficit=entropy_deficit,
        peak_frequency=peak_frequency,
        peak_period_samples=peak_period_samples,
        peak_cycles=peak_cycles,
        noise_floor=noise_floor,
        peak_power=peak_power,
        peak_snr=peak_snr,
        peak_surprise=peak_surprise,
        peak_p_value=peak_p_value,
        extraction_margin=extraction_margin,
        readiness_score=readiness_score,
        ready=ready,
        reason=reason,
    )


def readiness_window_sizes(
    n: int,
    *,
    min_size: int = 64,
    max_size: int | None = None,
    step: int | None = None,
    growth: float = 1.5,
) -> list[int]:
    """Build increasing prefix sizes for a readiness scan."""

    if n < 8:
        raise ValueError(f"Series too short for readiness scan (n={n}, need >= 8)")
    if min_size < 8:
        min_size = 8
    max_size = n if max_size is None else min(max_size, n)
    if max_size < 8:
        raise ValueError("max_size must be >= 8")
    if max_size < min_size:
        return [max_size]

    sizes: list[int] = []
    if step is not None:
        if step < 1:
            raise ValueError("step must be positive")
        sizes = list(range(min_size, max_size + 1, step))
    else:
        if growth <= 1.0:
            raise ValueError("growth must be > 1")
        current = min_size
        while current < max_size:
            sizes.append(int(current))
            next_size = int(np.ceil(current * growth))
            current = max(next_size, current + 1)
    if not sizes or sizes[-1] != max_size:
        sizes.append(max_size)
    return sorted(set(int(size) for size in sizes if size >= 8))


def scan_information_readiness(
    values: NDArray[np.floating],
    *,
    sample_rate: float = 1.0,
    min_snr: float = 2.0,
    min_cycles: float = 2.0,
    min_entropy_deficit: float = 0.02,
    min_usable_bins: int = 16,
    min_size: int = 64,
    max_size: int | None = None,
    step: int | None = None,
    growth: float = 1.5,
    stable_windows: int = 2,
    window_sizes: Sequence[int] | None = None,
    detrend: bool = True,
) -> ReadinessScan:
    """Scan increasing prefixes and estimate the first informative window."""

    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {y.shape}")
    if stable_windows < 1:
        raise ValueError("stable_windows must be >= 1")

    sizes = (
        sorted(set(int(size) for size in window_sizes if int(size) >= 8))
        if window_sizes is not None
        else readiness_window_sizes(
            len(y),
            min_size=min_size,
            max_size=max_size,
            step=step,
            growth=growth,
        )
    )
    if not sizes:
        raise ValueError("No valid readiness window sizes")

    points = [
        information_readiness(
            y[:size],
            sample_rate=sample_rate,
            min_snr=min_snr,
            min_cycles=min_cycles,
            min_entropy_deficit=min_entropy_deficit,
            min_usable_bins=min_usable_bins,
            detrend=detrend,
        )
        for size in sizes
    ]

    first_ready_n = next((point.n for point in points if point.ready), None)
    stable_ready_n: int | None = None
    consecutive = 0
    run_start: int | None = None
    for point in points:
        if point.ready:
            if consecutive == 0:
                run_start = point.n
            consecutive += 1
            if consecutive >= stable_windows:
                stable_ready_n = run_start
                break
        else:
            consecutive = 0
            run_start = None

    return ReadinessScan(
        points=points,
        first_ready_n=first_ready_n,
        stable_ready_n=stable_ready_n,
        stable_windows=stable_windows,
        min_snr=min_snr,
        min_entropy_deficit=min_entropy_deficit,
        min_usable_bins=min_usable_bins,
    )
