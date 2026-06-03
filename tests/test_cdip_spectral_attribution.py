"""Tests for direct CDIP spectral attribution helpers."""

from datetime import UTC, datetime

import numpy as np

from experiments.cdip_spectral_attribution import (
    SpectrumRow,
    build_spectral_attribution,
    spectrum_metrics,
)


def _row(energy, *, station="036", time=None):
    return SpectrumRow(
        station_id=station,
        time=time or datetime(2026, 3, 1, tzinfo=UTC),
        frequency_hz=np.asarray([0.05, 0.07, 0.10, 0.15, 0.20], dtype=float),
        bandwidth_hz=np.asarray([0.02, 0.02, 0.03, 0.05, 0.05], dtype=float),
        energy_density=np.asarray(energy, dtype=float),
        directional_spread_deg=np.asarray([10, 12, 14, 16, 18], dtype=float),
    )


def _window(index, start, delta):
    return {
        "global_window_index": index,
        "start": start,
        "end": start.replace("00:00+00:00", "30:00+00:00"),
        "observed_minus_null": delta,
        "group": ["036p1", "153p1"],
    }


class FakeSpectraSource:
    def rows(self, station_id, start, end):
        if start.hour == 1:
            return [_row([0.1, 8.0, 0.2, 5.0, 0.1], station=station_id)]
        return [_row([0.1, 0.2, 8.0, 0.2, 0.1], station=station_id)]


def test_spectrum_metrics_detects_single_vs_multiple_peaks():
    single = spectrum_metrics(_row([0.1, 0.2, 8.0, 0.2, 0.1]))
    multi = spectrum_metrics(_row([0.1, 8.0, 0.2, 5.0, 0.1]))

    assert single["peak_count"] == 1
    assert single["multimodal_candidate"] == 0.0
    assert multi["peak_count"] == 2
    assert multi["second_peak_ratio"] > 0.5
    assert multi["multimodal_candidate"] == 1.0
    assert multi["bandwidth_hz"] > single["bandwidth_hz"]


def test_build_spectral_attribution_compares_true_spectral_shape():
    report = {
        "summary": {"windows_run": 4, "observed_minus_null_total": 15.0},
        "windows": [
            _window(1, "2026-03-01T00:00:00+00:00", 1.0),
            _window(2, "2026-03-01T01:00:00+00:00", 9.0),
            _window(3, "2026-03-01T02:00:00+00:00", 3.0),
            _window(4, "2026-03-02T00:00:00+00:00", 2.0),
        ],
    }

    attribution = build_spectral_attribution(
        report,
        source=FakeSpectraSource(),
        top_windows=1,
        baseline_windows=2,
        pad_minutes=0.0,
        peak_min_height_fraction=0.15,
        peak_min_prominence_fraction=0.10,
        multimodal_second_peak_ratio=0.35,
        progress_every=0,
    )

    comparison = attribution["comparisons"]["high_delta_vs_background_baseline"]
    assert comparison["peak_count_mean"]["high_delta"]["mean"] > comparison["peak_count_mean"][
        "baseline"
    ]["mean"]
    assert comparison["multimodal_candidate_mean"]["high_delta"]["mean"] == 1.0
    assert attribution["source"]["role"].startswith("post-hoc")
