"""Known-answer power calibration for directional-quality matrix shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spectral_forecast.directional import directional_quality


@dataclass(frozen=True)
class DirectionalResolutionCalibration:
    """Known-answer recovery rates for one matrix geometry."""

    n: int
    series_count: int
    trials: int
    null_repeats: int
    activation_mode: str
    noise_false_positive_rate: float
    mechanism_detection_rates: dict[str, float]
    mechanism_strongest_correct_rates: dict[str, float]
    supported_mechanisms: list[str]
    min_detection_rate: float
    max_false_positive_rate: float
    false_positive_controlled: bool
    usable: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _positive_noise(rng: np.random.Generator, n: int, count: int) -> dict[str, np.ndarray]:
    return {
        f"s{index}": rng.exponential(0.35, n)
        for index in range(count)
    }


def _event_starts(rng: np.random.Generator, n: int, *, count: int) -> np.ndarray:
    usable = np.arange(2, max(3, n - 2), dtype=np.int64)
    if len(usable) == 0:
        return np.asarray([], dtype=np.int64)
    return np.sort(rng.choice(usable, size=min(count, len(usable)), replace=False))


def _known_answer_series(
    mechanism: str,
    *,
    n: int,
    series_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    series = _positive_noise(rng, n, series_count)
    values = list(series.values())
    event_count = max(4, n // 6)
    starts = _event_starts(rng, n, count=event_count)

    if mechanism == "independent_noise":
        return series
    if mechanism == "coactivation":
        for start in starts:
            for y in values:
                y[int(start)] += 8.0
        return series
    if mechanism == "exclusion":
        for event_index, start in enumerate(starts):
            values[event_index % series_count][int(start)] += 8.0
        return series
    if mechanism == "lagged_succession":
        pair_count = max(1, min(series_count // 2, 8))
        for event_index, start in enumerate(starts):
            pair = event_index % pair_count
            left = 2 * pair
            right = left + 1
            lag = 1 + pair % max(1, min(3, n // 8))
            if int(start) + lag < n:
                values[left][int(start)] += 8.0
                values[right][int(start) + lag] += 8.0
        return series
    if mechanism == "phase_offset":
        t = np.arange(n, dtype=np.float64)
        period = max(8.0, min(24.0, n / 2.0))
        for index, y in enumerate(values):
            phase = 2.0 * np.pi * index / series_count
            y += 2.0 + 2.0 * np.sin(2.0 * np.pi * t / period + phase)
        return series
    raise ValueError(f"unknown known-answer mechanism: {mechanism}")


def directional_resolution_calibration(
    *,
    n: int,
    series_count: int,
    trials: int = 20,
    null_repeats: int = 24,
    seed: int = 20260604,
    active_z_threshold: float = 1.5,
    max_lag: int = 6,
    aggregate_quantile: float = 0.9,
    null_block_size: int = 4,
    significance_level: float = 0.05,
    min_z_effect: float = 2.0,
    min_detection_rate: float = 0.8,
    max_false_positive_rate: float = 0.1,
) -> DirectionalResolutionCalibration:
    """Calibrate which mechanisms a matrix geometry can recover independently."""

    if n < 16:
        return DirectionalResolutionCalibration(
            n=n,
            series_count=series_count,
            trials=trials,
            null_repeats=null_repeats,
            activation_mode="positive",
            noise_false_positive_rate=1.0,
            mechanism_detection_rates={},
            mechanism_strongest_correct_rates={},
            supported_mechanisms=[],
            min_detection_rate=min_detection_rate,
            max_false_positive_rate=max_false_positive_rate,
            false_positive_controlled=False,
            usable=False,
            reason="too_few_observations",
        )
    if series_count < 2:
        raise ValueError("series_count must be >= 2")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    mechanisms = ("coactivation", "exclusion", "lagged_succession", "phase_offset")
    detected = {name: 0 for name in mechanisms}
    strongest = {name: 0 for name in mechanisms}
    false_positives = 0
    for trial in range(trials):
        noise = directional_quality(
            _known_answer_series(
                "independent_noise",
                n=n,
                series_count=series_count,
                seed=seed + 10_000 * trial,
            ),
            null_repeats=null_repeats,
            seed=seed + 10_000 * trial + 1,
            active_z_threshold=active_z_threshold,
            activation_mode="positive",
            max_lag=max_lag,
            aggregate_quantile=aggregate_quantile,
            null_block_size=null_block_size,
            max_series=series_count,
            significance_level=significance_level,
            min_z_effect=min_z_effect,
        )
        false_positives += int(noise.verdict == "structured")
        for mechanism_index, mechanism in enumerate(mechanisms):
            result = directional_quality(
                _known_answer_series(
                    mechanism,
                    n=n,
                    series_count=series_count,
                    seed=seed + 10_000 * trial + 100 * (mechanism_index + 1),
                ),
                null_repeats=null_repeats,
                seed=seed + 10_000 * trial + 100 * (mechanism_index + 1) + 1,
                active_z_threshold=active_z_threshold,
                activation_mode="positive",
                max_lag=max_lag,
                aggregate_quantile=aggregate_quantile,
                null_block_size=null_block_size,
                max_series=series_count,
                significance_level=significance_level,
                min_z_effect=min_z_effect,
            )
            detected[mechanism] += int(result.metrics[mechanism].detected)
            strongest[mechanism] += int(result.strongest_mechanism == mechanism)

    detection_rates = {
        name: count / trials
        for name, count in detected.items()
    }
    strongest_rates = {
        name: count / trials
        for name, count in strongest.items()
    }
    false_positive_rate = false_positives / trials
    false_positive_controlled = false_positive_rate <= max_false_positive_rate
    supported = sorted(
        name
        for name, rate in detection_rates.items()
        if false_positive_controlled and rate >= min_detection_rate
    )
    return DirectionalResolutionCalibration(
        n=n,
        series_count=series_count,
        trials=trials,
        null_repeats=null_repeats,
        activation_mode="positive",
        noise_false_positive_rate=false_positive_rate,
        mechanism_detection_rates=detection_rates,
        mechanism_strongest_correct_rates=strongest_rates,
        supported_mechanisms=supported,
        min_detection_rate=min_detection_rate,
        max_false_positive_rate=max_false_positive_rate,
        false_positive_controlled=false_positive_controlled,
        usable=bool(supported),
        reason="known_answer_calibrated" if supported else "insufficient_known_answer_power",
    )
