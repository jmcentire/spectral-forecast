"""Tests for CDIP multi-file alignment helpers."""

from experiments.cdip_observe import (
    _combination_rows,
    _intersect_time_spans,
    _select_aligned_window,
    highpass_fft,
    mask_dominant_fft,
    phase_randomize_fft,
    posthoc_metrics,
    preprocess_series_values,
)

import numpy as np


def test_intersect_time_spans_keeps_common_clean_interval():
    intervals = _intersect_time_spans(
        [
            [(0.0, 100.0), (200.0, 300.0)],
            [(50.0, 150.0), (220.0, 260.0)],
            [(40.0, 90.0), (230.0, 280.0)],
        ],
        min_duration=20.0,
    )

    assert intervals == [(50.0, 90.0), (230.0, 260.0)]


def test_select_aligned_window_applies_offset_and_sample_limit():
    start, n_samples = _select_aligned_window(
        [(10.0, 100.0)],
        target_rate=2.0,
        min_clean_samples=16,
        sample_limit=32,
        segment_offset=4,
    )

    assert start == 12.0
    assert n_samples == 32


def test_select_aligned_window_rejects_too_short_limit():
    try:
        _select_aligned_window(
            [(10.0, 20.0)],
            target_rate=1.0,
            min_clean_samples=16,
            sample_limit=8,
            segment_offset=0,
        )
    except ValueError as exc:
        assert "shorter" in str(exc)
    else:
        raise AssertionError("Expected ValueError for too-short selected window")


def test_posthoc_metrics_extracts_wave_and_spectrum_values():
    t = np.arange(512, dtype=np.float64) / 1.28
    values = 2.0 * np.sin(2 * np.pi * 0.08 * t)

    metrics = posthoc_metrics(values, sample_rate=1.28)

    assert metrics.elevation_kurtosis > 1.0
    assert metrics.max_wave_height > 3.0
    assert metrics.significant_wave_height > 3.0
    assert metrics.spectral_mean_period > 5.0
    assert metrics.crest_trough_correlation_proxy > 0.5


def test_highpass_fft_removes_sub_cutoff_component():
    sample_rate = 1.28
    t = np.arange(8192, dtype=np.float64) / sample_rate
    slow = 5.0 * np.sin(2 * np.pi * t / 3200.0)
    fast = np.sin(2 * np.pi * t / 10.0)

    filtered = highpass_fft(
        slow + fast,
        sample_rate=sample_rate,
        cutoff_period_seconds=30.0 * 60.0,
    )

    assert abs(np.corrcoef(filtered, slow)[0, 1]) < 0.05
    assert np.corrcoef(filtered, fast)[0, 1] > 0.95


def test_phase_randomize_fft_preserves_power_spectrum():
    sample_rate = 1.28
    t = np.arange(2048, dtype=np.float64) / sample_rate
    values = 2.0 * np.sin(2 * np.pi * 0.08 * t) + 0.5 * np.sin(2 * np.pi * 0.17 * t)

    randomized = phase_randomize_fft(values, seed=123)

    assert np.allclose(
        np.abs(np.fft.rfft(randomized)),
        np.abs(np.fft.rfft(values)),
        rtol=1e-10,
        atol=1e-10,
    )
    assert not np.allclose(randomized, values)
    assert np.allclose(randomized, phase_randomize_fft(values, seed=123))


def test_phase_surrogate_preprocess_modes_preserve_mode_spectrum():
    sample_rate = 1.28
    t = np.arange(2048, dtype=np.float64) / sample_rate
    values = 3.0 * np.sin(2 * np.pi * 0.08 * t) + 0.7 * np.sin(2 * np.pi * 0.17 * t)

    masked = preprocess_series_values(
        values,
        sample_rate=sample_rate,
        mode="dominant-mask",
        mask_dominant_bins=1,
        mask_bin_radius=0,
    )
    randomized_masked = preprocess_series_values(
        values,
        sample_rate=sample_rate,
        mode="dominant-mask-phase-randomize",
        mask_dominant_bins=1,
        mask_bin_radius=0,
        phase_seed=123,
    )

    assert np.allclose(
        np.abs(np.fft.rfft(randomized_masked)),
        np.abs(np.fft.rfft(masked)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_mask_dominant_fft_removes_strongest_non_dc_bin():
    n = 2048
    t = np.arange(n, dtype=np.float64)
    strong_bin = 64
    weak_bin = 211
    values = (
        2.5
        + 3.0 * np.sin(2 * np.pi * strong_bin * t / n)
        + 0.5 * np.sin(2 * np.pi * weak_bin * t / n)
    )

    masked = mask_dominant_fft(values, bins=1, radius=0)
    original_spectrum = np.abs(np.fft.rfft(values - np.mean(values)))
    masked_spectrum = np.abs(np.fft.rfft(masked - np.mean(masked)))

    assert np.isclose(np.mean(masked), np.mean(values))
    assert masked_spectrum[strong_bin] < original_spectrum[strong_bin] * 1e-10
    assert masked_spectrum[weak_bin] > original_spectrum[weak_bin] * 0.95


def test_combination_rows_summarizes_active_series_sets():
    class Point:
        def __init__(self, index, active_series, emission, pheromone, max_score):
            self.index = index
            self.active_series = active_series
            self.emission = emission
            self.pheromone = pheromone
            self.max_score = max_score

    rows = _combination_rows(
        [
            Point(10, ("045p1:z", "243p1:z"), 2.0, 4.0, 5.0),
            Point(20, ("045p1:z", "243p1:z"), 3.0, 6.0, 7.0),
            Point(30, ("430p1:z",), 1.0, 2.0, 3.0),
        ]
    )

    assert rows[0]["active_platforms"] == "045p1,243p1"
    assert rows[0]["count"] == 2
    assert rows[0]["emission_sum"] == 5.0
