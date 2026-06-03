"""Run a bounded spectral+stigmergy observation on raw CDIP displacement files.

This is experiment tooling. It avoids wave-domain labels and published
rogue-wave predictor features; the optional mesh receives only terms derived
from the spectral observer's own scores and decomposition state.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numpy.typing import NDArray
from scipy.io import netcdf_file

from spectral_forecast.information import scan_information_readiness
from spectral_forecast.observation import (
    ObservationPoint,
    ObservationResult,
    ScoreName,
    build_stigmergy,
    observe_series,
)


CHANNEL_VARIABLES = {
    "x": "xyzXDisplacement",
    "y": "xyzYDisplacement",
    "z": "xyzZDisplacement",
}

PREPROCESS_MODES = (
    "none",
    "highpass",
    "dominant-mask",
    "highpass-dominant-mask",
    "phase-randomize",
    "highpass-phase-randomize",
    "dominant-mask-phase-randomize",
    "highpass-dominant-mask-phase-randomize",
)


@dataclass(frozen=True)
class CdipSeries:
    """One clean, aligned CDIP displacement segment."""

    name: str
    channel: str
    values: NDArray[np.float64]
    sample_rate: float
    start_time: float
    filter_delay: float
    span_start: int
    span_end: int
    station_id: str
    platform_id: str
    platform_name: str
    source_path: str

    def timestamp_for_index(self, index: int) -> datetime:
        seconds = self.start_time + (self.span_start + index) / self.sample_rate - self.filter_delay
        return datetime.fromtimestamp(seconds, tz=timezone.utc)


@dataclass(frozen=True)
class CdipRawRecord:
    """Raw CDIP displacement arrays and timing metadata for one file."""

    path: Path
    arrays: dict[str, NDArray[np.float64]]
    valid: NDArray[np.bool_]
    sample_rate: float
    first_sample_time: float
    station_id: str
    platform_id: str
    platform_name: str

    def sample_time(self, index: int) -> float:
        return self.first_sample_time + index / self.sample_rate


@dataclass(frozen=True)
class PosthocMetrics:
    """Audit-only wave and spectrum metrics computed after detection."""

    elevation_kurtosis: float
    max_wave_height: float
    significant_wave_height: float
    max_to_significant_wave_height: float
    spectral_bandwidth: float
    spectral_mean_period: float
    crest_trough_correlation_proxy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "elevation_kurtosis": self.elevation_kurtosis,
            "max_wave_height": self.max_wave_height,
            "significant_wave_height": self.significant_wave_height,
            "max_to_significant_wave_height": self.max_to_significant_wave_height,
            "spectral_bandwidth": self.spectral_bandwidth,
            "spectral_mean_period": self.spectral_mean_period,
            "crest_trough_correlation_proxy": self.crest_trough_correlation_proxy,
        }


def _decode_attr(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar(variable: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(variable.data).reshape(()))
    except Exception:
        return default


def _contiguous_spans(mask: NDArray[np.bool_], min_length: int) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("Expected a 1D validity mask")
    if len(mask) == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    spans = list(zip(changes[0::2], changes[1::2]))
    return [(int(start), int(end)) for start, end in spans if end - start >= min_length]


def _select_span(
    spans: Sequence[tuple[int, int]],
    sample_limit: int | None,
    segment_offset: int,
) -> tuple[int, int]:
    if not spans:
        raise ValueError("No contiguous clean segment satisfies the requested minimum length")

    start, end = max(spans, key=lambda span: span[1] - span[0])
    if segment_offset:
        start += segment_offset
    if start >= end:
        raise ValueError("Segment offset moves past the selected clean segment")
    if sample_limit is not None:
        end = min(end, start + sample_limit)
    if start >= end:
        raise ValueError("Selected segment is empty")
    return int(start), int(end)


def highpass_fft(
    values: NDArray[np.float64],
    *,
    sample_rate: float,
    cutoff_period_seconds: float,
) -> NDArray[np.float64]:
    """Remove frequencies with periods longer than cutoff_period_seconds."""

    y = np.asarray(values, dtype=np.float64)
    if len(y) == 0:
        return y.copy()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive for high-pass preprocessing")
    if cutoff_period_seconds <= 0:
        raise ValueError("cutoff_period_seconds must be positive")

    centered = y - float(np.mean(y))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
    spectrum = np.fft.rfft(centered)
    spectrum[freqs < 1.0 / cutoff_period_seconds] = 0.0
    return np.fft.irfft(spectrum, n=len(centered)).astype(np.float64)


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def phase_randomize_fft(
    values: NDArray[np.float64],
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Preserve the Fourier magnitudes while randomizing non-DC phases."""

    y = np.asarray(values, dtype=np.float64)
    if len(y) < 4:
        return y.copy()
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(y)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(spectrum))
    phases[0] = 0.0
    if len(y) % 2 == 0:
        phases[-1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    randomized[0] = spectrum[0]
    if len(y) % 2 == 0:
        randomized[-1] = spectrum[-1]
    return np.fft.irfft(randomized, n=len(y)).astype(np.float64)


def mask_dominant_fft(
    values: NDArray[np.float64],
    *,
    bins: int = 3,
    radius: int = 1,
) -> NDArray[np.float64]:
    """Remove the strongest non-DC Fourier bins from a series."""

    y = np.asarray(values, dtype=np.float64)
    if len(y) < 4 or bins <= 0:
        return y.copy()
    if radius < 0:
        raise ValueError("radius must be non-negative")

    mean = float(np.mean(y))
    centered = y - mean
    spectrum = np.fft.rfft(centered)
    if len(spectrum) <= 1:
        return y.copy()

    magnitudes = np.abs(spectrum)
    magnitudes[0] = 0.0
    blocked = np.zeros(len(spectrum), dtype=bool)
    blocked[0] = True
    masked = spectrum.copy()
    selected = 0
    for index in np.argsort(magnitudes)[::-1]:
        if selected >= bins:
            break
        if blocked[index] or magnitudes[index] <= 0.0:
            continue
        left = max(1, int(index) - radius)
        right = min(len(spectrum), int(index) + radius + 1)
        masked[left:right] = 0.0
        blocked[left:right] = True
        selected += 1

    return (np.fft.irfft(masked, n=len(centered)) + mean).astype(np.float64)


def preprocess_series_values(
    values: NDArray[np.float64],
    *,
    sample_rate: float,
    mode: str = "none",
    highpass_period_seconds: float = 30.0 * 60.0,
    mask_dominant_bins: int = 3,
    mask_bin_radius: int = 1,
    phase_seed: int = 0,
) -> NDArray[np.float64]:
    """Apply audit preprocessing before the agnostic observer sees a series."""

    if mode == "none":
        return np.asarray(values, dtype=np.float64)
    if mode == "highpass":
        return highpass_fft(
            values,
            sample_rate=sample_rate,
            cutoff_period_seconds=highpass_period_seconds,
        )
    if mode == "dominant-mask":
        return mask_dominant_fft(
            values,
            bins=mask_dominant_bins,
            radius=mask_bin_radius,
        )
    if mode == "highpass-dominant-mask":
        highpassed = highpass_fft(
            values,
            sample_rate=sample_rate,
            cutoff_period_seconds=highpass_period_seconds,
        )
        return mask_dominant_fft(
            highpassed,
            bins=mask_dominant_bins,
            radius=mask_bin_radius,
        )
    if mode == "phase-randomize":
        return phase_randomize_fft(values, seed=phase_seed)
    if mode == "highpass-phase-randomize":
        highpassed = highpass_fft(
            values,
            sample_rate=sample_rate,
            cutoff_period_seconds=highpass_period_seconds,
        )
        return phase_randomize_fft(highpassed, seed=phase_seed)
    if mode == "dominant-mask-phase-randomize":
        masked = mask_dominant_fft(
            values,
            bins=mask_dominant_bins,
            radius=mask_bin_radius,
        )
        return phase_randomize_fft(masked, seed=phase_seed)
    if mode == "highpass-dominant-mask-phase-randomize":
        highpassed = highpass_fft(
            values,
            sample_rate=sample_rate,
            cutoff_period_seconds=highpass_period_seconds,
        )
        masked = mask_dominant_fft(
            highpassed,
            bins=mask_dominant_bins,
            radius=mask_bin_radius,
        )
        return phase_randomize_fft(masked, seed=phase_seed)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def load_cdip_raw_record(
    path: str | Path,
    channels: Sequence[str],
    *,
    keep_flags: set[int],
) -> CdipRawRecord:
    """Load raw CDIP arrays and shared validity mask for selected channels."""

    path = Path(path)
    normalized_channels = [channel.lower() for channel in channels]
    unknown = [channel for channel in normalized_channels if channel not in CHANNEL_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown channels: {', '.join(unknown)}")

    with netcdf_file(path, mmap=False) as nc:
        sample_rate = _scalar(nc.variables["xyzSampleRate"])
        start_time = _scalar(nc.variables["xyzStartTime"])
        filter_delay = _scalar(nc.variables["xyzFilterDelay"])
        flags = np.asarray(nc.variables["xyzFlagPrimary"].data, dtype=np.int16)

        arrays: dict[str, NDArray[np.float64]] = {}
        valid = np.isin(flags, list(keep_flags))
        for channel in normalized_channels:
            variable = CHANNEL_VARIABLES[channel]
            values = np.asarray(nc.variables[variable].data, dtype=np.float64)
            arrays[channel] = values
            valid &= np.isfinite(values)
            valid &= values > -999.0

        station_id = _decode_attr(getattr(nc, "cdip_station_id", ""), "unknown")
        platform_id = _decode_attr(getattr(nc, "platform_id", station_id), station_id)
        platform_name = _decode_attr(getattr(nc, "platform_name", platform_id), platform_id)

    return CdipRawRecord(
        path=path,
        arrays=arrays,
        valid=valid,
        sample_rate=sample_rate,
        first_sample_time=start_time - filter_delay,
        station_id=station_id,
        platform_id=platform_id,
        platform_name=platform_name,
    )


def load_cdip_series(
    path: str | Path,
    channels: Sequence[str],
    *,
    keep_flags: set[int],
    min_clean_samples: int,
    sample_limit: int | None,
    segment_offset: int = 0,
    preprocess: str = "none",
    highpass_period_seconds: float = 30.0 * 60.0,
    mask_dominant_bins: int = 3,
    mask_bin_radius: int = 1,
    phase_surrogate_seed: int = 20260602,
) -> list[CdipSeries]:
    """Load aligned clean displacement segments from a CDIP NetCDF file."""

    raw = load_cdip_raw_record(path, channels, keep_flags=keep_flags)
    spans = _contiguous_spans(raw.valid, min_clean_samples)
    span_start, span_end = _select_span(spans, sample_limit, segment_offset)
    if span_end - span_start < min_clean_samples:
        raise ValueError(
            "Selected segment is shorter than the requested minimum after limiting/offset"
        )

    out: list[CdipSeries] = []
    for channel, values in raw.arrays.items():
        segment = np.asarray(values[span_start:span_end], dtype=np.float64)
        seed = _stable_seed(
            phase_surrogate_seed,
            f"{raw.platform_id}:{channel}:{span_start}:{span_end}",
        )
        segment = preprocess_series_values(
            segment,
            sample_rate=raw.sample_rate,
            mode=preprocess,
            highpass_period_seconds=highpass_period_seconds,
            mask_dominant_bins=mask_dominant_bins,
            mask_bin_radius=mask_bin_radius,
            phase_seed=seed,
        )
        out.append(
            CdipSeries(
                name=f"{raw.platform_id}:{channel}",
                channel=channel,
                values=segment,
                sample_rate=raw.sample_rate,
                start_time=raw.first_sample_time,
                filter_delay=0.0,
                span_start=span_start,
                span_end=span_end,
                station_id=raw.station_id,
                platform_id=raw.platform_id,
                platform_name=raw.platform_name,
                source_path=str(raw.path),
            )
        )
    return out


def _raw_valid_time_spans(
    record: CdipRawRecord,
    *,
    min_duration: float,
) -> list[tuple[float, float]]:
    spans = _contiguous_spans(record.valid, 1)
    out = []
    for start, end in spans:
        if end <= start:
            continue
        start_time = record.sample_time(start)
        end_time = record.sample_time(end - 1)
        if end_time >= start_time and (end_time - start_time) >= min_duration:
            out.append((start_time, end_time))
    return out


def _intersect_time_spans(
    span_lists: Sequence[Sequence[tuple[float, float]]],
    *,
    min_duration: float,
) -> list[tuple[float, float]]:
    if not span_lists:
        return []

    intervals = list(span_lists[0])
    for spans in span_lists[1:]:
        intersections: list[tuple[float, float]] = []
        for left_start, left_end in intervals:
            for right_start, right_end in spans:
                start = max(left_start, right_start)
                end = min(left_end, right_end)
                if end >= start and (end - start) >= min_duration:
                    intersections.append((start, end))
        intervals = intersections
        if not intervals:
            break
    return intervals


def _select_aligned_window(
    intervals: Sequence[tuple[float, float]],
    *,
    target_rate: float,
    min_clean_samples: int,
    sample_limit: int | None,
    segment_offset: int,
) -> tuple[float, int]:
    if not intervals:
        raise ValueError("No common clean UTC interval across CDIP files")
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")

    start_time, end_time = max(intervals, key=lambda item: item[1] - item[0])
    available = int(np.floor((end_time - start_time) * target_rate)) + 1
    if segment_offset:
        available -= segment_offset
        start_time += segment_offset / target_rate
    if available <= 0:
        raise ValueError("Segment offset moves past the selected common interval")

    n_samples = min(available, sample_limit) if sample_limit is not None else available
    if n_samples < min_clean_samples:
        raise ValueError(
            "Selected common interval is shorter than the requested minimum after limiting/offset"
        )
    return start_time, int(n_samples)


def load_aligned_cdip_series(
    paths: Sequence[str | Path],
    channels: Sequence[str],
    *,
    keep_flags: set[int],
    min_clean_samples: int,
    sample_limit: int | None,
    segment_offset: int = 0,
    preprocess: str = "none",
    highpass_period_seconds: float = 30.0 * 60.0,
    mask_dominant_bins: int = 3,
    mask_bin_radius: int = 1,
    phase_surrogate_seed: int = 20260602,
) -> list[CdipSeries]:
    """Load multiple CDIP files aligned on a common clean UTC grid."""

    if len(paths) == 1:
        return load_cdip_series(
            paths[0],
            channels,
            keep_flags=keep_flags,
            min_clean_samples=min_clean_samples,
            sample_limit=sample_limit,
            segment_offset=segment_offset,
            preprocess=preprocess,
            highpass_period_seconds=highpass_period_seconds,
            mask_dominant_bins=mask_dominant_bins,
            mask_bin_radius=mask_bin_radius,
            phase_surrogate_seed=phase_surrogate_seed,
        )

    records = [
        load_cdip_raw_record(path, channels, keep_flags=keep_flags)
        for path in paths
    ]
    target_rate = min(record.sample_rate for record in records)
    min_duration = (min_clean_samples - 1) / target_rate
    span_lists = [
        _raw_valid_time_spans(record, min_duration=min_duration)
        for record in records
    ]
    intervals = _intersect_time_spans(span_lists, min_duration=min_duration)
    aligned_start, n_samples = _select_aligned_window(
        intervals,
        target_rate=target_rate,
        min_clean_samples=min_clean_samples,
        sample_limit=sample_limit,
        segment_offset=segment_offset,
    )
    grid = aligned_start + np.arange(n_samples, dtype=np.float64) / target_rate

    out: list[CdipSeries] = []
    for record in records:
        source_index = (grid - record.first_sample_time) * record.sample_rate
        source_x = np.arange(len(record.valid), dtype=np.float64)
        for channel, values in record.arrays.items():
            segment = np.interp(source_index, source_x, values).astype(np.float64)
            seed = _stable_seed(
                phase_surrogate_seed,
                f"{record.platform_id}:{channel}:{aligned_start:.6f}:{n_samples}",
            )
            segment = preprocess_series_values(
                segment,
                sample_rate=target_rate,
                mode=preprocess,
                highpass_period_seconds=highpass_period_seconds,
                mask_dominant_bins=mask_dominant_bins,
                mask_bin_radius=mask_bin_radius,
                phase_seed=seed,
            )
            out.append(
                CdipSeries(
                    name=f"{record.platform_id}:{channel}",
                    channel=channel,
                    values=segment,
                    sample_rate=target_rate,
                    start_time=aligned_start,
                    filter_delay=0.0,
                    span_start=0,
                    span_end=n_samples,
                    station_id=record.station_id,
                    platform_id=record.platform_id,
                    platform_name=record.platform_name,
                    source_path=str(record.path),
                )
            )
    return out


def _level(value: float) -> str:
    value = abs(float(value))
    if value >= 10.0:
        return "extreme"
    if value >= 5.0:
        return "high"
    if value >= 3.0:
        return "elevated"
    if value >= 1.0:
        return "present"
    return "low"


def _direction(value: float, deadband: float = 1.0) -> str:
    if value >= deadband:
        return "positive"
    if value <= -deadband:
        return "negative"
    return "neutral"


def _safe_token(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def observation_signal_content(point: ObservationPoint, meta: CdipSeries) -> str:
    """Build mesh-facing content solely from detector output and source identity."""

    terms = [
        "spectral-observation",
        f"station-{_safe_token(meta.platform_id)}",
        f"channel-{meta.channel}disp",
        f"series-{_safe_token(point.series)}",
        f"frozen-{_level(point.frozen_score)}",
        f"sliding-{_level(point.sliding_score)}",
        f"drift-{_direction(point.conditional_drift_score)}",
        f"drift-{_level(point.conditional_drift_score)}",
        f"state-{_level(point.state_drift_score)}",
        f"residual-{_direction(point.frozen_residual, deadband=0.0)}",
    ]
    if point.score("max") >= 10.0:
        terms.append("score-extreme")
    elif point.score("max") >= 5.0:
        terms.append("score-high")
    elif point.score("max") >= 3.0:
        terms.append("score-elevated")
    else:
        terms.append("score-low")
    return " ".join(terms)


def observation_metadata(point: ObservationPoint, meta: CdipSeries, score: ScoreName) -> dict[str, Any]:
    timestamp = meta.timestamp_for_index(point.index)
    return {
        "series": point.series,
        "channel": meta.channel,
        "station_id": meta.station_id,
        "platform_id": meta.platform_id,
        "index": point.index,
        "timestamp": timestamp.isoformat(),
        "score_name": score,
        "score": point.score(score),
        "frozen_score": point.frozen_score,
        "sliding_score": point.sliding_score,
        "conditional_drift_score": point.conditional_drift_score,
        "state_drift_score": point.state_drift_score,
        "frozen_residual": point.frozen_residual,
        "sliding_residual": point.sliding_residual,
    }


def _zero_upcrossing_heights(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    crossings = np.flatnonzero((centered[:-1] <= 0.0) & (centered[1:] > 0.0)) + 1
    if len(crossings) < 2:
        return np.array([], dtype=np.float64)

    heights = []
    for start, end in zip(crossings[:-1], crossings[1:]):
        segment = centered[start:end]
        if len(segment) == 0:
            continue
        heights.append(float(np.max(segment) - np.min(segment)))
    return np.asarray(heights, dtype=np.float64)


def _autocorrelation_at(values: NDArray[np.float64], lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    left = centered[:-lag]
    right = centered[lag:]
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denom <= 1e-30:
        return 0.0
    return float(np.sum(left * right) / denom)


def posthoc_metrics(values: NDArray[np.float64], sample_rate: float) -> PosthocMetrics:
    """Compute research-style metrics after detection for audit only."""

    y = np.asarray(values, dtype=np.float64)
    if len(y) == 0:
        return PosthocMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    centered = y - float(np.mean(y))
    variance = float(np.mean(centered**2))
    elevation_kurtosis = (
        float(np.mean(centered**4) / max(variance**2, 1e-30))
        if variance > 0
        else 0.0
    )

    heights = _zero_upcrossing_heights(y)
    max_wave_height = float(np.max(heights)) if len(heights) else 0.0
    if len(heights):
        sorted_heights = np.sort(heights)[::-1]
        top_count = max(1, int(np.ceil(len(sorted_heights) / 3.0)))
        significant_wave_height = float(np.mean(sorted_heights[:top_count]))
    else:
        significant_wave_height = 0.0
    max_to_significant = (
        max_wave_height / significant_wave_height
        if significant_wave_height > 1e-30
        else 0.0
    )

    if len(centered) < 8 or sample_rate <= 0:
        spectral_bandwidth = 0.0
        spectral_mean_period = 0.0
        crest_trough = 0.0
    else:
        window = np.hanning(len(centered))
        spectrum = np.abs(np.fft.rfft(centered * window)) ** 2
        freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
        spectrum = spectrum[1:]
        freqs = freqs[1:]
        m0 = float(np.sum(spectrum))
        m1 = float(np.sum(freqs * spectrum))
        m2 = float(np.sum((freqs**2) * spectrum))
        spectral_mean_period = m0 / m1 if m1 > 1e-30 else 0.0
        if m1 > 1e-30:
            bandwidth_arg = max((m0 * m2) / (m1**2) - 1.0, 0.0)
            spectral_bandwidth = float(np.sqrt(bandwidth_arg))
        else:
            spectral_bandwidth = 0.0
        half_period_lag = int(round(0.5 * spectral_mean_period * sample_rate))
        crest_trough = -_autocorrelation_at(y, half_period_lag)

    return PosthocMetrics(
        elevation_kurtosis=elevation_kurtosis,
        max_wave_height=max_wave_height,
        significant_wave_height=significant_wave_height,
        max_to_significant_wave_height=max_to_significant,
        spectral_bandwidth=spectral_bandwidth,
        spectral_mean_period=spectral_mean_period,
        crest_trough_correlation_proxy=crest_trough,
    )


def _selected_points(
    results: Sequence[ObservationResult],
    *,
    score: ScoreName,
    threshold: float,
    max_signals: int,
) -> list[ObservationPoint]:
    points = [
        point
        for result in results
        for point in result.points
        if np.isfinite(point.score(score)) and point.score(score) > threshold
    ]
    if max_signals > 0 and len(points) > max_signals:
        points = sorted(points, key=lambda point: point.score(score), reverse=True)[:max_signals]
    return sorted(points, key=lambda point: (point.index, point.series))


async def run_stigmergy_mesh(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    threshold: float,
    max_signals: int,
    mesh_src: Path,
    max_workers: int,
    worker_capacity: int,
    base_threshold: float,
    gap_threshold: float,
    high_relevance_offset: float,
    prime: str,
) -> dict[str, Any]:
    """Route detector emissions through the sibling Stigmergy mesh."""

    sys.path.insert(0, str(mesh_src))
    from stigmergy.mesh.mesh import Mesh
    from stigmergy.pipeline.processor import AgentRegistry
    from stigmergy.primitives.signal import Signal

    selected = _selected_points(
        results,
        score=score,
        threshold=threshold,
        max_signals=max_signals,
    )

    mesh = Mesh(
        AgentRegistry(),
        dedup_enabled=False,
        max_workers=max_workers,
        worker_capacity=worker_capacity,
        base_threshold=base_threshold,
        gap_threshold=gap_threshold,
        high_relevance_offset=high_relevance_offset,
    )

    seed_count = max(2, len(series_by_name))
    previous = None
    seed_names = sorted(series_by_name) or ["spectral-observation"]
    while len(seed_names) < seed_count:
        seed_names.append(f"seed-{len(seed_names)}")
    for name in seed_names[:seed_count]:
        worker = mesh.spawn_worker(source_name=name, connect_to=previous)
        previous = worker.id
        if prime != "none" and name in series_by_name:
            meta = series_by_name[name]
            station_token = f"station-{_safe_token(meta.platform_id)}"
            series_token = f"series-{_safe_token(name)}"
            channel_token = f"channel-{meta.channel}disp"
            if prime == "station":
                content = f"spectral-observation {station_token} {channel_token} mesh-primer"
            else:
                content = f"spectral-observation {station_token} {channel_token} {series_token} mesh-primer"
            primer = Signal(
                content=content,
                source="spectral-observer-primer",
                channel=name,
                author="spectral-forecast",
                timestamp=meta.timestamp_for_index(0),
                metadata={"primer": True, "prime_mode": prime},
            )
            worker._context_summarize(primer, 0.1)
            worker.signals_accepted += 1
            worker.signals_received += 1
            worker._position_dirty = True

    traces = []
    for point in selected:
        meta = series_by_name[point.series]
        metadata = observation_metadata(point, meta, score)
        signal = Signal(
            content=observation_signal_content(point, meta),
            source="spectral-observer",
            channel=point.series,
            author="spectral-forecast",
            timestamp=meta.timestamp_for_index(point.index),
            metadata=metadata,
        )
        trace = await mesh.ingest(signal)
        familiarity = list(trace.familiarity_scores.values())
        traces.append(
            {
                "series": point.series,
                "index": point.index,
                "timestamp": metadata["timestamp"],
                "score": point.score(score),
                "accepted_workers": len(trace.accepted_workers),
                "hops": trace.total_hops,
                "max_familiarity": max(familiarity) if familiarity else 0.0,
                "content": signal.content,
            }
        )

    workers = []
    for worker in sorted(mesh.workers, key=lambda item: item.context.signal_count, reverse=True):
        workers.append(
            {
                "id": str(worker.id)[:8],
                "label": worker.label,
                "signals": worker.context.signal_count,
                "fullness": worker.fullness,
                "rolling_avg_familiarity": worker.rolling_avg_familiarity,
                "terms": sorted(worker.context.terms)[:20],
            }
        )

    low_familiarity = sorted(traces, key=lambda row: row["max_familiarity"])[:10]
    high_score = sorted(traces, key=lambda row: row["score"], reverse=True)[:10]
    return {
        "signals_routed": len(selected),
        "threshold": threshold,
        "score": score,
        "prime": prime,
        "base_threshold": base_threshold,
        "gap_threshold": gap_threshold,
        "worker_capacity": worker_capacity,
        "worker_count": mesh.worker_count,
        "workers": workers,
        "lowest_familiarity": low_familiarity,
        "highest_score": high_score,
    }


def _top_observation_rows(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    top: int,
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for point in result.top(score, top):
            meta = series_by_name[point.series]
            row = point.to_dict()
            row["timestamp"] = meta.timestamp_for_index(point.index).isoformat()
            row["score_name"] = score
            row["score"] = point.score(score)
            rows.append(row)
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _point_posthoc_metrics(
    point: ObservationPoint,
    meta: CdipSeries,
    *,
    window: int,
) -> PosthocMetrics:
    start = max(0, point.index - window + 1)
    end = min(len(meta.values), point.index + 1)
    return posthoc_metrics(meta.values[start:end], meta.sample_rate)


def _rankdata(values: Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if len(x_arr) < 3 or np.std(x_arr) <= 1e-30 or np.std(y_arr) <= 1e-30:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if len(x_arr) < 3:
        return 0.0
    return _pearson(_rankdata(x_arr), _rankdata(y_arr))


def _posthoc_correlation_rows(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    window: int,
) -> list[dict[str, Any]]:
    metric_values: dict[str, list[float]] = {}
    scores: list[float] = []

    for result in results:
        meta = series_by_name[result.series]
        for point in result.points:
            metrics = _point_posthoc_metrics(point, meta, window=window).to_dict()
            scores.append(point.score(score))
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(value)

    rows = []
    for name, values in metric_values.items():
        rows.append(
            {
                "metric": name,
                "n": len(values),
                "pearson": _pearson(scores, values),
                "spearman": _spearman(scores, values),
            }
        )
    return sorted(rows, key=lambda row: abs(row["spearman"]), reverse=True)


def _posthoc_top_rows(
    results: Sequence[ObservationResult],
    series_by_name: dict[str, CdipSeries],
    *,
    score: ScoreName,
    window: int,
    top: int,
) -> list[dict[str, Any]]:
    points = sorted(
        (point for result in results for point in result.points),
        key=lambda point: point.score(score),
        reverse=True,
    )[:top]
    rows = []
    for point in points:
        meta = series_by_name[point.series]
        row = {
            "series": point.series,
            "index": point.index,
            "timestamp": meta.timestamp_for_index(point.index).isoformat(),
            "score": point.score(score),
        }
        row.update(_point_posthoc_metrics(point, meta, window=window).to_dict())
        rows.append(row)
    return rows


def _combination_rows(stigmergy_points: Sequence[Any]) -> list[dict[str, Any]]:
    combos: dict[tuple[str, ...], dict[str, Any]] = {}
    for point in stigmergy_points:
        if not point.active_series:
            continue
        key = tuple(point.active_series)
        row = combos.setdefault(
            key,
            {
                "active_series": ",".join(key),
                "active_platforms": ",".join(sorted({item.split(":")[0] for item in key})),
                "count": 0,
                "emission_sum": 0.0,
                "max_pheromone": 0.0,
                "max_score": 0.0,
                "first_index": point.index,
                "last_index": point.index,
            },
        )
        row["count"] += 1
        row["emission_sum"] += point.emission
        row["max_pheromone"] = max(row["max_pheromone"], point.pheromone)
        row["max_score"] = max(row["max_score"], point.max_score)
        row["first_index"] = min(row["first_index"], point.index)
        row["last_index"] = max(row["last_index"], point.index)

    return sorted(
        combos.values(),
        key=lambda row: (row["emission_sum"], row["max_pheromone"]),
        reverse=True,
    )


def _readiness_rows(series: Sequence[CdipSeries], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for item in series:
        scan = scan_information_readiness(
            item.values,
            sample_rate=item.sample_rate,
            min_snr=args.readiness_min_snr,
            min_entropy_deficit=args.readiness_min_entropy_deficit,
            min_usable_bins=args.readiness_min_usable_bins,
            min_size=args.readiness_min_size,
            max_size=args.readiness_max_size,
            step=args.readiness_step,
            stable_windows=args.readiness_stable_windows,
        )
        final = scan.points[-1]
        first = scan.first_ready
        stable = scan.stable_ready
        rows.append(
            {
                "series": item.name,
                "first_ready_n": scan.first_ready_n,
                "first_ready_seconds": first.seconds if first is not None else None,
                "stable_ready_n": scan.stable_ready_n,
                "stable_ready_seconds": stable.seconds if stable is not None else None,
                "final_n": final.n,
                "final_ready": final.ready,
                "final_reason": final.reason,
                "final_readiness_score": final.readiness_score,
                "final_entropy_deficit": final.entropy_deficit,
                "final_peak_surprise": final.peak_surprise,
                "final_peak_p_value": final.peak_p_value,
                "final_peak_period_samples": final.peak_period_samples,
                "scan_points": [point.to_dict() for point in scan.points],
            }
        )
    return rows


def _print_text_report(report: dict[str, Any]) -> None:
    data = report["data"]
    print("CDIP observation")
    if len(data["paths"]) == 1:
        print(f"  path={data['paths'][0]}")
    else:
        print(f"  paths={len(data['paths'])} files")
        print("  platforms=%s" % ",".join(data["platform_ids"]))
    print(f"  platform={data['platform_id']} sample_rate={data['sample_rate']:.6g}Hz")
    print(
        "  segment=%d:%d samples=%d flags_kept=%s"
        % (
            data["span_start"],
            data["span_end"],
            data["samples"],
            ",".join(str(flag) for flag in data["flags_kept"]),
        )
    )
    if data["aligned"]:
        print(
            "  aligned_utc=%s to %s"
            % (
                data["aligned_start"].replace("+00:00", "Z"),
                data["aligned_end"].replace("+00:00", "Z"),
            )
        )
    print("  protocol=raw displacement, generic QC, past-only spectral observation")
    print("  mesh_terms=no labels and no published wave predictors")

    if report["readiness"]:
        print("\nInformation readiness")
        print(
            "%16s %8s %8s %8s %8s %10s %10s %10s %s"
            % ("series", "first", "stable", "score", "entropy", "peak", "p_peak", "period", "reason")
        )
        for row in report["readiness"]:
            first = "-" if row["first_ready_n"] is None else str(row["first_ready_n"])
            stable = "-" if row["stable_ready_n"] is None else str(row["stable_ready_n"])
            print(
                "%16s %8s %8s %8.3f %8.4f %10.3f %10.3g %10.1f %s"
                % (
                    row["series"],
                    first,
                    stable,
                    row["final_readiness_score"],
                    row["final_entropy_deficit"],
                    row["final_peak_surprise"],
                    row["final_peak_p_value"],
                    row["final_peak_period_samples"],
                    row["final_reason"],
                )
            )

    print("\nTop observations")
    print(
        "%16s %24s %8s %8s %8s %8s %8s %10s"
        % ("series", "timestamp", "index", "score", "frozen", "sliding", "state", "drift")
    )
    for row in report["top_observations"]:
        print(
            "%16s %24s %8d %8.3f %8.3f %8.3f %8.3f %10.3f"
            % (
                row["series"],
                row["timestamp"].replace("+00:00", "Z"),
                row["index"],
                row["score"],
                row["frozen_score"],
                row["sliding_score"],
                row["state_drift_score"],
                row["conditional_drift_score"],
            )
        )

    if report["stigmergy"]["points"]:
        print("\nStigmergy accumulator")
        print("%8s %10s %10s %8s %10s %s" % ("index", "pheromone", "emission", "series", "max", "active"))
        for row in report["stigmergy"]["top"]:
            print(
                "%8d %10.3f %10.3f %8d %10.3f %s"
                % (
                    row["index"],
                    row["pheromone"],
                    row["emission"],
                    row["active_series_count"],
                    row["max_score"],
                    row["active_series"],
                )
            )
        if report["stigmergy"]["combinations"]:
            print("\nActive combinations")
            print("%24s %5s %10s %10s %10s %s" % ("platforms", "count", "emission", "pheromone", "max_score", "series"))
            for row in report["stigmergy"]["combinations"][:8]:
                print(
                    "%24s %5d %10.3f %10.3f %10.3f %s"
                    % (
                        row["active_platforms"],
                        row["count"],
                        row["emission_sum"],
                        row["max_pheromone"],
                        row["max_score"],
                        row["active_series"],
                    )
                )

    posthoc = report.get("posthoc")
    if posthoc:
        print("\nPost-hoc research metrics")
        print("  audit_only=true window=%d" % posthoc["window"])
        print("%38s %5s %9s %9s" % ("metric", "n", "pearson", "spearman"))
        for row in posthoc["correlations"][:8]:
            print(
                "%38s %5d %9.3f %9.3f"
                % (row["metric"], row["n"], row["pearson"], row["spearman"])
            )

    mesh = report.get("mesh")
    if mesh:
        print("\nStigmergy mesh")
        print(
            "  routed=%d workers=%d score=%s threshold=%.3f prime=%s base=%.3f gap=%.3f"
            % (
                mesh["signals_routed"],
                mesh["worker_count"],
                mesh["score"],
                mesh["threshold"],
                mesh["prime"],
                mesh["base_threshold"],
                mesh["gap_threshold"],
            )
        )
        print("  workers")
        for worker in mesh["workers"]:
            terms = ",".join(worker["terms"][:8])
            print(
                "    %s signals=%d familiarity=%.3f label=%s terms=%s"
                % (
                    worker["id"],
                    worker["signals"],
                    worker["rolling_avg_familiarity"],
                    worker["label"],
                    terms,
                )
            )
        if mesh["lowest_familiarity"]:
            print("  lowest familiarity routed signals")
            for row in mesh["lowest_familiarity"][:5]:
                print(
                    "    %s index=%d score=%.3f familiarity=%.3f terms=%s"
                    % (
                        row["series"],
                        row["index"],
                        row["score"],
                        row["max_familiarity"],
                        row["content"],
                    )
                )


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    min_clean_samples = args.min_clean_samples or args.baseline + args.adaptive_window + args.stride
    series = load_aligned_cdip_series(
        args.files,
        args.channels,
        keep_flags=set(args.keep_flags),
        min_clean_samples=min_clean_samples,
        sample_limit=args.sample_limit,
        segment_offset=args.segment_offset,
        preprocess=args.preprocess,
        highpass_period_seconds=args.highpass_period_minutes * 60.0,
        mask_dominant_bins=args.mask_dominant_bins,
        mask_bin_radius=args.mask_bin_radius,
        phase_surrogate_seed=args.phase_surrogate_seed,
    )
    series_by_name = {item.name: item for item in series}

    results = [
        observe_series(
            item.values,
            series=item.name,
            baseline_size=args.baseline,
            adaptive_window=args.adaptive_window,
            stride=args.stride,
            sample_rate=item.sample_rate,
        )
        for item in series
    ]

    stig = build_stigmergy(
        results,
        score=args.score,
        emission_threshold=args.emission_threshold,
        decay=args.decay,
    )

    first = series[0]
    source_paths = sorted({item.source_path for item in series})
    platform_ids = sorted({item.platform_id for item in series})
    report: dict[str, Any] = {
        "data": {
            "path": source_paths[0] if len(source_paths) == 1 else source_paths,
            "paths": source_paths,
            "aligned": len(source_paths) > 1,
            "aligned_start": first.timestamp_for_index(0).isoformat(),
            "aligned_end": first.timestamp_for_index(len(first.values) - 1).isoformat(),
            "platform_id": first.platform_id if len(platform_ids) == 1 else "multi",
            "platform_ids": platform_ids,
            "platform_name": first.platform_name if len(platform_ids) == 1 else "multi-buoy aligned window",
            "station_id": first.station_id if len(platform_ids) == 1 else "multi",
            "sample_rate": first.sample_rate,
            "span_start": first.span_start,
            "span_end": first.span_end,
            "samples": first.span_end - first.span_start,
            "channels": [item.channel for item in series],
            "flags_kept": sorted(args.keep_flags),
        },
        "parameters": {
            "baseline": args.baseline,
            "adaptive_window": args.adaptive_window,
            "stride": args.stride,
            "score": args.score,
            "emission_threshold": args.emission_threshold,
            "decay": args.decay,
            "posthoc_window": args.posthoc_window or args.adaptive_window,
            "preprocess": args.preprocess,
            "highpass_period_minutes": args.highpass_period_minutes,
            "mask_dominant_bins": args.mask_dominant_bins,
            "mask_bin_radius": args.mask_bin_radius,
            "phase_surrogate_seed": args.phase_surrogate_seed,
        },
        "readiness": _readiness_rows(series, args),
        "top_observations": _top_observation_rows(
            results,
            series_by_name,
            score=args.score,
            top=args.top,
        ),
        "stigmergy": {
            "points": len(stig.points),
            "top": [point.to_dict() for point in stig.top(args.top)],
            "combinations": _combination_rows(stig.points),
        },
        "posthoc": {
            "note": "Audit-only metrics computed after detector emissions; not used by observer or mesh.",
            "window": args.posthoc_window or args.adaptive_window,
            "correlations": _posthoc_correlation_rows(
                results,
                series_by_name,
                score=args.score,
                window=args.posthoc_window or args.adaptive_window,
            ),
            "top": _posthoc_top_rows(
                results,
                series_by_name,
                score=args.score,
                window=args.posthoc_window or args.adaptive_window,
                top=args.top,
            ),
        },
    }

    if args.mesh:
        report["mesh"] = await run_stigmergy_mesh(
            results,
            series_by_name,
            score=args.score,
            threshold=args.emission_threshold,
            max_signals=args.mesh_max_signals,
            mesh_src=args.mesh_src,
            max_workers=args.mesh_max_workers,
            worker_capacity=args.mesh_worker_capacity,
            base_threshold=args.mesh_base_threshold,
            gap_threshold=args.mesh_gap_threshold,
            high_relevance_offset=args.mesh_high_relevance_offset,
            prime=args.mesh_prime,
        )

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="One or more paths to CDIP *_xy.nc displacement files",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["z"],
        choices=sorted(CHANNEL_VARIABLES),
        help="Displacement channels to observe",
    )
    parser.add_argument(
        "--keep-flags",
        type=int,
        nargs="+",
        default=[2],
        help="CDIP xyzFlagPrimary values to keep",
    )
    parser.add_argument("--baseline", type=int, default=4096, help="Frozen baseline samples")
    parser.add_argument("--adaptive-window", type=int, default=2048, help="Adaptive window samples")
    parser.add_argument("--stride", type=int, default=512, help="Observation stride")
    parser.add_argument("--sample-limit", type=int, default=32768, help="Limit selected clean samples")
    parser.add_argument("--segment-offset", type=int, default=0, help="Offset into selected clean segment")
    parser.add_argument("--min-clean-samples", type=int, default=None, help="Minimum clean segment length")
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default="none",
        help="Optional preprocessing applied before observation",
    )
    parser.add_argument(
        "--highpass-period-minutes",
        type=float,
        default=30.0,
        help="Remove periods longer than this when --preprocess=highpass",
    )
    parser.add_argument(
        "--mask-dominant-bins",
        type=int,
        default=3,
        help="Number of strongest non-DC Fourier bins to remove for dominant-mask modes",
    )
    parser.add_argument(
        "--mask-bin-radius",
        type=int,
        default=1,
        help="Neighbor radius around each selected dominant Fourier bin to remove",
    )
    parser.add_argument(
        "--phase-surrogate-seed",
        type=int,
        default=20260602,
        help="Base seed for deterministic phase-randomized preprocessing controls",
    )
    parser.add_argument(
        "--score",
        choices=["frozen", "sliding", "drift", "state", "max"],
        default="max",
        help="Score used for rankings and emissions",
    )
    parser.add_argument("--emission-threshold", type=float, default=3.0, help="Emission threshold")
    parser.add_argument("--decay", type=float, default=0.9, help="Stigmergy accumulator decay")
    parser.add_argument("--top", type=int, default=10, help="Rows to show")
    parser.add_argument("--posthoc-window", type=int, default=None, help="Past window for audit-only metrics")
    parser.add_argument("--readiness-min-size", type=int, default=512, help="Minimum prefix for readiness scan")
    parser.add_argument("--readiness-max-size", type=int, default=None, help="Maximum prefix for readiness scan")
    parser.add_argument("--readiness-step", type=int, default=512, help="Prefix step for readiness scan")
    parser.add_argument("--readiness-stable-windows", type=int, default=2, help="Consecutive ready prefixes required")
    parser.add_argument("--readiness-min-snr", type=float, default=2.0, help="Peak surprise threshold")
    parser.add_argument("--readiness-min-usable-bins", type=int, default=16, help="Minimum FFT bins for readiness")
    parser.add_argument(
        "--readiness-min-entropy-deficit",
        type=float,
        default=0.02,
        help="Minimum entropy gap below finite white-noise expectation",
    )
    parser.add_argument("--mesh", action="store_true", help="Route emissions through ../stigmergy mesh")
    parser.add_argument("--mesh-src", type=Path, default=Path("../stigmergy/src"), help="Stigmergy src path")
    parser.add_argument("--mesh-max-signals", type=int, default=160, help="Maximum emissions routed to mesh")
    parser.add_argument("--mesh-max-workers", type=int, default=12, help="Maximum mesh workers")
    parser.add_argument("--mesh-worker-capacity", type=int, default=80, help="Signals per mesh worker before full vigilance")
    parser.add_argument("--mesh-base-threshold", type=float, default=0.15, help="Mesh base ART vigilance threshold")
    parser.add_argument("--mesh-gap-threshold", type=float, default=0.08, help="Mesh gap-spawn threshold")
    parser.add_argument("--mesh-high-relevance-offset", type=float, default=0.2, help="Offset for full resonance")
    parser.add_argument(
        "--mesh-prime",
        choices=["none", "station", "series"],
        default="series",
        help="Neutral mesh priming mode to avoid empty-worker collapse",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Report format")
    args = parser.parse_args(argv)
    if args.mask_dominant_bins < 0:
        parser.error("--mask-dominant-bins must be >= 0")
    if args.mask_bin_radius < 0:
        parser.error("--mask-bin-radius must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(_main_async(args))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
