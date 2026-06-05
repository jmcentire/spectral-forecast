"""Label-free discovery of explicit multichannel relationship hypotheses.

The module is agnostic to domain labels, but not to mathematical relationship
families. It preserves relationship identity and evaluates each family against
an appropriate null instead of collapsing every stream into one anomaly score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch


@dataclass(frozen=True)
class RelationshipEvidence:
    """Observed-versus-null evidence for one explicit relationship hypothesis."""

    family: str
    entities: tuple[str, ...]
    observed: float
    null_mean: float
    null_std: float
    observed_minus_null: float
    z_effect: float | None
    null_exceedances: int
    empirical_p_ge_observed: float
    empirical_p_floor: float
    detected: bool
    attributes: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationshipDiscovery:
    """Relationship hypotheses discovered on one canonical multichannel view."""

    n: int
    series_count: int
    sample_rate: float
    layer: str
    null_repeats: int
    nperseg: int
    max_lag: int
    evidence: tuple[RelationshipEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "series_count": self.series_count,
            "sample_rate": self.sample_rate,
            "layer": self.layer,
            "null_repeats": self.null_repeats,
            "nperseg": self.nperseg,
            "max_lag": self.max_lag,
            "evidence": [row.to_dict() for row in self.evidence],
        }


@dataclass(frozen=True)
class RelationshipCalibration:
    """Known-answer recovery and false-positive rates for one matrix geometry."""

    n: int
    series_count: int
    trials: int
    null_repeats: int
    family_detection_rates: dict[str, float]
    family_false_positive_rates: dict[str, float]
    supported_families: tuple[str, ...]
    min_detection_rate: float
    max_false_positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _robust_standardize_matrix(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2D")
    if x.shape[0] < 32:
        raise ValueError("matrix needs at least 32 rows")
    if x.shape[1] < 2:
        raise ValueError("matrix needs at least two series")
    if not np.all(np.isfinite(x)):
        raise ValueError("matrix must contain only finite values")
    centers = np.median(x, axis=0)
    mad = np.median(np.abs(x - centers[None, :]), axis=0)
    scales = 1.4826 * mad
    std = np.std(x, axis=0)
    scales = np.where(scales > 1e-10, scales, std)
    scales = np.where(scales > 1e-10, scales, 1.0)
    return ((x - centers[None, :]) / scales[None, :]).astype(np.float64)


def residualize_relationship_view(
    matrix: NDArray[np.floating],
    *,
    common_mode_fraction: float = 0.0,
    dominant_spectral_fraction: float = 0.0,
    dominant_bins: int = 3,
    bin_radius: int = 1,
) -> NDArray[np.float64]:
    """Return a non-destructive residual view for a relationship-discovery layer."""

    if not 0.0 <= common_mode_fraction <= 1.0:
        raise ValueError("common_mode_fraction must be in [0, 1]")
    if not 0.0 <= dominant_spectral_fraction <= 1.0:
        raise ValueError("dominant_spectral_fraction must be in [0, 1]")
    if dominant_bins < 0 or bin_radius < 0:
        raise ValueError("dominant_bins and bin_radius must be non-negative")

    x = np.asarray(matrix, dtype=np.float64).copy()
    if x.ndim != 2:
        raise ValueError("matrix must be 2D")
    means = np.mean(x, axis=0, keepdims=True)
    centered = x - means

    if common_mode_fraction > 0.0:
        u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        common = singular[0] * np.outer(u[:, 0], vt[0])
        centered = centered - common_mode_fraction * common

    if dominant_spectral_fraction > 0.0 and dominant_bins > 0:
        for column in range(centered.shape[1]):
            spectrum = np.fft.rfft(centered[:, column])
            magnitude = np.abs(spectrum)
            if len(magnitude) <= 1:
                continue
            magnitude[0] = 0.0
            selected = np.argsort(magnitude)[::-1][:dominant_bins]
            attenuation = np.ones(len(spectrum), dtype=np.float64)
            for index in selected:
                start = max(1, int(index) - bin_radius)
                stop = min(len(spectrum), int(index) + bin_radius + 1)
                attenuation[start:stop] *= 1.0 - dominant_spectral_fraction
            centered[:, column] = np.fft.irfft(
                spectrum * attenuation,
                n=len(centered),
            )

    return (centered + means).astype(np.float64)


def _effective_nperseg(n: int, requested: int) -> int:
    return max(32, min(requested, n // 2))


def _normalized_psd(
    values: NDArray[np.float64],
    *,
    sample_rate: float,
    nperseg: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    frequencies, power = welch(
        values,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )
    frequencies = frequencies[1:]
    power = np.maximum(power[1:], 0.0)
    total = float(np.sum(power))
    if total <= 1e-18:
        return frequencies, np.zeros_like(power)
    return frequencies, (power / total).astype(np.float64)


def normalized_spectral_profile(
    values: NDArray[np.floating],
    *,
    sample_rate: float,
    nperseg: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a normalized Welch spectrum for corpus-level relationship controls."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if len(x) < 32:
        raise ValueError("values needs at least 32 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must contain only finite values")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return _normalized_psd(
        x,
        sample_rate=sample_rate,
        nperseg=_effective_nperseg(len(x), nperseg),
    )


def _cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-18:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def spectral_profile_alignment(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> float:
    """Return cosine alignment between two equal-length spectral profiles."""

    left_profile = np.asarray(left, dtype=np.float64)
    right_profile = np.asarray(right, dtype=np.float64)
    if left_profile.ndim != 1 or right_profile.ndim != 1:
        raise ValueError("spectral profiles must be one-dimensional")
    if left_profile.shape != right_profile.shape:
        raise ValueError("spectral profiles must have equal shape")
    if not np.all(np.isfinite(left_profile)) or not np.all(np.isfinite(right_profile)):
        raise ValueError("spectral profiles must contain only finite values")
    return _cosine(left_profile, right_profile)


def _spectral_alignment_metric(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    sample_rate: float,
    nperseg: int,
) -> tuple[float, dict[str, Any], NDArray[np.float64], NDArray[np.float64]]:
    frequencies, left_power = _normalized_psd(
        left,
        sample_rate=sample_rate,
        nperseg=nperseg,
    )
    _, right_power = _normalized_psd(
        right,
        sample_rate=sample_rate,
        nperseg=nperseg,
    )
    shared = np.sqrt(left_power * right_power)
    peak_index = int(np.argmax(shared)) if len(shared) else 0
    return (
        _cosine(left_power, right_power),
        {
            "shared_peak_frequency": (
                float(frequencies[peak_index]) if len(frequencies) else None
            ),
            "shared_peak_period_seconds": (
                float(1.0 / frequencies[peak_index])
                if len(frequencies) and frequencies[peak_index] > 0
                else None
            ),
        },
        left_power,
        right_power,
    )


def _phase_coherence_metric(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    sample_rate: float,
    nperseg: int,
) -> tuple[float, dict[str, Any]]:
    segment_count = len(left) // nperseg
    if segment_count < 2:
        return 0.0, {
            "peak_frequency": None,
            "peak_period_seconds": None,
            "peak_phase_locking": None,
            "phase_radians": None,
            "phase_delay_seconds": None,
        }
    usable = segment_count * nperseg
    left_segments = left[:usable].reshape(segment_count, nperseg)
    right_segments = right[:usable].reshape(segment_count, nperseg)
    left_segments = left_segments - np.mean(left_segments, axis=1, keepdims=True)
    right_segments = right_segments - np.mean(right_segments, axis=1, keepdims=True)
    left_spectrum = np.fft.rfft(left_segments, axis=1)[:, 1:]
    right_spectrum = np.fft.rfft(right_segments, axis=1)[:, 1:]
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)[1:]
    cross = np.conj(left_spectrum) * right_spectrum
    shared_by_segment = np.sqrt(
        np.abs(left_spectrum) ** 2 * np.abs(right_spectrum) ** 2
    )
    unit_phase = cross / np.maximum(np.abs(cross), 1e-18)
    phase_vector = np.sum(shared_by_segment * unit_phase, axis=0)
    shared_power = np.sum(shared_by_segment, axis=0)
    phase_locking = np.abs(phase_vector) / np.maximum(shared_power, 1e-18)
    if len(shared_power) == 0 or float(np.sum(shared_power)) <= 1e-18:
        return 0.0, {
            "peak_frequency": None,
            "peak_period_seconds": None,
            "peak_phase_locking": None,
            "phase_radians": None,
            "phase_delay_seconds": None,
        }
    cutoff = float(np.quantile(shared_power, 0.75))
    mask = shared_power >= cutoff
    weights = np.where(mask, shared_power, 0.0)
    metric = float(np.sum(weights * phase_locking) / max(np.sum(weights), 1e-18))
    peak_index = int(np.argmax(weights * phase_locking))
    frequency = float(frequencies[peak_index])
    phase = float(np.angle(phase_vector[peak_index]))
    return metric, {
        "peak_frequency": frequency,
        "peak_period_seconds": float(1.0 / frequency) if frequency > 0 else None,
        "peak_phase_locking": float(phase_locking[peak_index]),
        "phase_radians": phase,
        "phase_delay_seconds": (
            float(phase / (2.0 * np.pi * frequency)) if frequency > 0 else None
        ),
    }


def _signed_correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _lagged_dependence_metric(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    max_lag: int,
    sample_rate: float,
) -> tuple[float, dict[str, Any]]:
    zero = _signed_correlation(left, right)
    best_abs = 0.0
    best_signed = 0.0
    best_lag = 0
    effective = min(max_lag, max(1, len(left) // 4))
    for lag in range(1, effective + 1):
        positive = _signed_correlation(left[:-lag], right[lag:])
        negative = _signed_correlation(left[lag:], right[:-lag])
        if abs(positive) > best_abs:
            best_abs = abs(positive)
            best_signed = positive
            best_lag = lag
        if abs(negative) > best_abs:
            best_abs = abs(negative)
            best_signed = negative
            best_lag = -lag
    gain = max(0.0, best_abs - abs(zero))
    return gain, {
        "zero_lag_correlation": zero,
        "best_lag_correlation": best_signed,
        "best_lag_samples": best_lag,
        "best_lag_seconds": float(best_lag / sample_rate),
    }


def _relation_change_metric(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    sample_rate: float,
    nperseg: int,
    max_lag: int,
) -> tuple[float, dict[str, Any]]:
    midpoint = len(left) // 2

    def vector(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        spectral, _, _, _ = _spectral_alignment_metric(
            a,
            b,
            sample_rate=sample_rate,
            nperseg=_effective_nperseg(len(a), nperseg),
        )
        phase, _ = _phase_coherence_metric(
            a,
            b,
            sample_rate=sample_rate,
            nperseg=_effective_nperseg(len(a), nperseg),
        )
        lagged, _ = _lagged_dependence_metric(
            a,
            b,
            max_lag=min(max_lag, max(1, len(a) // 4)),
            sample_rate=sample_rate,
        )
        return np.asarray(
            [_signed_correlation(a, b), spectral, phase, lagged],
            dtype=np.float64,
        )

    first = vector(left[:midpoint], right[:midpoint])
    second = vector(left[midpoint:], right[midpoint:])
    delta = second - first
    strongest_index = int(np.argmax(np.abs(delta)))
    components = [
        "signed_zero_lag_correlation",
        "spectral_alignment",
        "phase_coherence",
        "lagged_dependence_gain",
    ]
    return float(np.max(np.abs(delta))), {
        "components": components,
        "strongest_changed_component": components[strongest_index],
        "first_half": [float(value) for value in first],
        "second_half": [float(value) for value in second],
        "second_minus_first": [float(value) for value in delta],
    }


def _common_mode_metric(matrix: NDArray[np.float64]) -> tuple[float, dict[str, Any]]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    power = singular**2
    fraction = float(power[0] / max(np.sum(power), 1e-18))
    return fraction, {
        "first_component_variance_fraction": fraction,
        "singular_values": [float(value) for value in singular],
    }


def _circular_shift(
    values: NDArray[np.float64],
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if len(values) < 2:
        return values.copy()
    return np.roll(values, int(rng.integers(1, len(values))))


def _segment_circular_shift(
    values: NDArray[np.float64],
    *,
    segment_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    shifted = values.copy()
    for start in range(0, len(values), segment_size):
        stop = min(start + segment_size, len(values))
        segment = values[start:stop]
        if len(segment) > 1:
            shifted[start:stop] = np.roll(
                segment,
                int(rng.integers(1, len(segment))),
            )
    return shifted


def _evidence(
    *,
    family: str,
    entities: tuple[str, ...],
    observed: float,
    null_values: Sequence[float],
    attributes: dict[str, Any],
    significance_level: float,
    min_z_effect: float,
) -> RelationshipEvidence:
    null = np.asarray(null_values, dtype=np.float64)
    mean = float(np.mean(null))
    std = float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    delta = float(observed - mean)
    z_effect = delta / std if std > 1e-12 else None
    exceedances = int(np.sum(null >= observed))
    p_ge = float((exceedances + 1) / (len(null) + 1))
    detected = bool(
        delta > 0.0
        and p_ge <= significance_level
        and z_effect is not None
        and z_effect >= min_z_effect
    )
    return RelationshipEvidence(
        family=family,
        entities=entities,
        observed=float(observed),
        null_mean=mean,
        null_std=std,
        observed_minus_null=delta,
        z_effect=z_effect,
        null_exceedances=exceedances,
        empirical_p_ge_observed=p_ge,
        empirical_p_floor=float(1.0 / (len(null) + 1)),
        detected=detected,
        attributes=attributes,
    )


def discover_relationships(
    matrix: NDArray[np.floating],
    entity_names: Sequence[str],
    *,
    sample_rate: float,
    layer: str = "raw",
    null_repeats: int = 39,
    seed: int = 20260604,
    nperseg: int = 256,
    max_lag: int = 64,
    significance_level: float = 0.05,
    min_z_effect: float = 1.5,
) -> RelationshipDiscovery:
    """Discover explicit relationship hypotheses in one canonical data view."""

    if null_repeats < 1:
        raise ValueError("null_repeats must be >= 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    x = _robust_standardize_matrix(matrix)
    if len(entity_names) != x.shape[1]:
        raise ValueError("entity_names must match matrix columns")
    effective_nperseg = _effective_nperseg(len(x), nperseg)
    rng = np.random.default_rng(seed)
    rows: list[RelationshipEvidence] = []

    common_observed, common_attributes = _common_mode_metric(x)
    common_null = []
    for _ in range(null_repeats):
        null = np.column_stack(
            [
                _circular_shift(x[:, column], rng=rng)
                for column in range(x.shape[1])
            ]
        )
        common_null.append(_common_mode_metric(null)[0])
    rows.append(
        _evidence(
            family="common_mode",
            entities=tuple(entity_names),
            observed=common_observed,
            null_values=common_null,
            attributes=common_attributes,
            significance_level=significance_level,
            min_z_effect=min_z_effect,
        )
    )

    for left_index, right_index in combinations(range(x.shape[1]), 2):
        left = x[:, left_index]
        right = x[:, right_index]
        entities = (str(entity_names[left_index]), str(entity_names[right_index]))

        spectral, spectral_attributes, left_power, right_power = _spectral_alignment_metric(
            left,
            right,
            sample_rate=sample_rate,
            nperseg=effective_nperseg,
        )
        spectral_null = [
            _cosine(left_power, rng.permutation(right_power))
            for _ in range(null_repeats)
        ]
        rows.append(
            _evidence(
                family="spectral_alignment",
                entities=entities,
                observed=spectral,
                null_values=spectral_null,
                attributes=spectral_attributes,
                significance_level=significance_level,
                min_z_effect=min_z_effect,
            )
        )

        phase, phase_attributes = _phase_coherence_metric(
            left,
            right,
            sample_rate=sample_rate,
            nperseg=effective_nperseg,
        )
        phase_null = []
        lagged_null = []
        for _ in range(null_repeats):
            permuted = _segment_circular_shift(
                right,
                segment_size=effective_nperseg,
                rng=rng,
            )
            phase_null.append(
                _phase_coherence_metric(
                    left,
                    permuted,
                    sample_rate=sample_rate,
                    nperseg=effective_nperseg,
                )[0]
            )
            shifted = _circular_shift(right, rng=rng)
            lagged_null.append(
                _lagged_dependence_metric(
                    left,
                    shifted,
                    max_lag=max_lag,
                    sample_rate=sample_rate,
                )[0]
            )
        rows.append(
            _evidence(
                family="phase_coherence",
                entities=entities,
                observed=phase,
                null_values=phase_null,
                attributes=phase_attributes,
                significance_level=significance_level,
                min_z_effect=min_z_effect,
            )
        )

        lagged, lagged_attributes = _lagged_dependence_metric(
            left,
            right,
            max_lag=max_lag,
            sample_rate=sample_rate,
        )
        rows.append(
            _evidence(
                family="lagged_dependence",
                entities=entities,
                observed=lagged,
                null_values=lagged_null,
                attributes=lagged_attributes,
                significance_level=significance_level,
                min_z_effect=min_z_effect,
            )
        )

        change, change_attributes = _relation_change_metric(
            left,
            right,
            sample_rate=sample_rate,
            nperseg=effective_nperseg,
            max_lag=max_lag,
        )
        change_null = []
        for _ in range(null_repeats):
            shift = int(rng.integers(1, len(left)))
            null_left = np.roll(left, shift)
            null_right = np.roll(right, shift)
            change_null.append(
                _relation_change_metric(
                    null_left,
                    null_right,
                    sample_rate=sample_rate,
                    nperseg=effective_nperseg,
                    max_lag=max_lag,
                )[0]
            )
        rows.append(
            _evidence(
                family="relation_change",
                entities=entities,
                observed=change,
                null_values=change_null,
                attributes=change_attributes,
                significance_level=significance_level,
                min_z_effect=min_z_effect,
            )
        )

    rows.sort(
        key=lambda row: (
            row.detected,
            row.z_effect if row.z_effect is not None else float("-inf"),
            row.observed_minus_null,
        ),
        reverse=True,
    )
    return RelationshipDiscovery(
        n=len(x),
        series_count=x.shape[1],
        sample_rate=float(sample_rate),
        layer=layer,
        null_repeats=null_repeats,
        nperseg=effective_nperseg,
        max_lag=max_lag,
        evidence=tuple(rows),
    )


def _ar_noise(
    rng: np.random.Generator,
    n: int,
    *,
    coefficient: float = 0.85,
) -> NDArray[np.float64]:
    noise = rng.normal(0.0, 1.0, n)
    values = np.zeros(n, dtype=np.float64)
    for index in range(1, n):
        values[index] = coefficient * values[index - 1] + noise[index]
    return values


def _known_answer_matrix(
    family: str,
    *,
    n: int,
    series_count: int,
    seed: int,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    if family == "independent":
        frequencies = np.linspace(0.013, 0.21, series_count)
        return np.column_stack(
            [
                np.sin(2.0 * np.pi * frequency * t + rng.uniform(0.0, 2.0 * np.pi))
                + rng.normal(0.0, 0.45, n)
                for frequency in frequencies
            ]
        )
    matrix = np.column_stack(
        [
            _ar_noise(rng, n, coefficient=float(rng.uniform(-0.5, 0.9)))
            for _ in range(series_count)
        ]
    )
    if family == "common_mode":
        latent = _ar_noise(rng, n, coefficient=0.95)
        return np.column_stack(
            [2.5 * latent + rng.normal(0.0, 0.3, n) for _ in range(series_count)]
        )
    if family == "spectral_alignment":
        frequencies = (0.017, 0.043, 0.081)
        for column in range(series_count):
            phase = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
            matrix[:, column] = sum(
                np.sin(2.0 * np.pi * frequency * t + phase[index])
                for index, frequency in enumerate(frequencies)
            ) + rng.normal(0.0, 0.25, n)
        return matrix
    if family == "phase_coherence":
        latent = (
            np.sin(2.0 * np.pi * 0.031 * t)
            + 0.7 * np.sin(2.0 * np.pi * 0.067 * t)
        )
        for column in range(series_count):
            matrix[:, column] = np.roll(latent, 3 * column) + rng.normal(0.0, 0.2, n)
        return matrix
    if family == "lagged_dependence":
        source = _ar_noise(rng, n, coefficient=0.65)
        matrix[:, 0] = source
        for column in range(1, series_count):
            lag = 8 * column
            matrix[:, column] = np.roll(source, lag) + rng.normal(0.0, 0.2, n)
        return matrix
    if family == "relation_change":
        midpoint = n // 2
        source = _ar_noise(rng, n)
        matrix[:, 0] = source
        for column in range(1, series_count):
            matrix[:midpoint, column] = _ar_noise(rng, midpoint)
            matrix[midpoint:, column] = (
                source[midpoint:] + rng.normal(0.0, 0.2, n - midpoint)
            )
        return matrix
    raise ValueError(f"unknown known-answer family: {family}")


def calibrate_relationship_geometry(
    *,
    n: int,
    series_count: int,
    sample_rate: float = 1.0,
    trials: int = 6,
    null_repeats: int = 19,
    seed: int = 20260604,
    nperseg: int = 256,
    max_lag: int = 64,
    min_detection_rate: float = 0.75,
    max_false_positive_rate: float = 0.10,
    layer: str = "raw",
    matrix_transform: Callable[
        [NDArray[np.float64]], NDArray[np.float64]
    ] | None = None,
) -> RelationshipCalibration:
    """Calibrate exact-geometry recovery for each implemented relationship family."""

    families = (
        "common_mode",
        "spectral_alignment",
        "phase_coherence",
        "lagged_dependence",
        "relation_change",
    )
    detections = {family: 0 for family in families}
    false_positives = {family: 0 for family in families}
    names = tuple(f"s{index}" for index in range(series_count))

    def transform(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        if matrix_transform is None:
            return matrix
        transformed = np.asarray(matrix_transform(matrix), dtype=np.float64)
        if transformed.shape != matrix.shape:
            raise ValueError("matrix_transform must preserve matrix geometry")
        return transformed

    for trial in range(trials):
        noise = discover_relationships(
            transform(
                _known_answer_matrix(
                    "independent",
                    n=n,
                    series_count=series_count,
                    seed=seed + 100_000 * trial,
                )
            ),
            names,
            sample_rate=sample_rate,
            layer=layer,
            null_repeats=null_repeats,
            seed=seed + 100_000 * trial + 1,
            nperseg=nperseg,
            max_lag=max_lag,
        )
        for family in families:
            false_positives[family] += int(
                any(row.detected and row.family == family for row in noise.evidence)
            )
        for family_index, family in enumerate(families):
            result = discover_relationships(
                transform(
                    _known_answer_matrix(
                        family,
                        n=n,
                        series_count=series_count,
                        seed=seed + 100_000 * trial + 1000 * (family_index + 1),
                    )
                ),
                names,
                sample_rate=sample_rate,
                layer=layer,
                null_repeats=null_repeats,
                seed=seed + 100_000 * trial + 1000 * (family_index + 1) + 1,
                nperseg=nperseg,
                max_lag=max_lag,
            )
            detections[family] += int(
                any(row.detected and row.family == family for row in result.evidence)
            )
    detection_rates = {
        family: detections[family] / trials
        for family in families
    }
    false_positive_rates = {
        family: false_positives[family] / trials
        for family in families
    }
    supported = tuple(
        family
        for family in families
        if detection_rates[family] >= min_detection_rate
        and false_positive_rates[family] <= max_false_positive_rate
    )
    return RelationshipCalibration(
        n=n,
        series_count=series_count,
        trials=trials,
        null_repeats=null_repeats,
        family_detection_rates=detection_rates,
        family_false_positive_rates=false_positive_rates,
        supported_families=supported,
        min_detection_rate=min_detection_rate,
        max_false_positive_rate=max_false_positive_rate,
    )
