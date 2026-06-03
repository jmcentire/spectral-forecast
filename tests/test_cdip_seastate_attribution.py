"""Tests for CDIP sea-state attribution helpers."""

from datetime import UTC, datetime

import pytest

from experiments.cdip_seastate_attribution import (
    WaveAggRow,
    build_attribution,
    circular_spread_degrees,
    select_window_sets,
)


def _window(index: int, start: str, delta: float, group: list[str]) -> dict[str, object]:
    return {
        "global_window_index": index,
        "start": start,
        "end": start.replace("00:00+00:00", "30:00+00:00"),
        "observed_minus_null": delta,
        "group": group,
    }


def test_circular_spread_handles_wraparound():
    assert circular_spread_degrees([350.0, 10.0]) < 20.0
    assert circular_spread_degrees([]) is None


def test_select_window_sets_uses_top_background_and_month_matched():
    report = {
        "windows": [
            _window(1, "2026-01-01T00:00:00+00:00", 1.0, ["036p1"]),
            _window(2, "2026-01-01T01:00:00+00:00", 9.0, ["036p1"]),
            _window(3, "2026-01-02T00:00:00+00:00", 3.0, ["036p1"]),
            _window(4, "2026-02-01T00:00:00+00:00", 2.0, ["036p1"]),
        ]
    }

    selected = select_window_sets(report, top_windows=1, baseline_windows=2)

    assert selected["high_delta"][0]["global_window_index"] == 2
    assert len(selected["background_baseline"]) == 2
    assert selected["month_matched_baseline"][0]["start"].startswith("2026-01")


def test_build_attribution_compares_published_parameter_proxies():
    report = {
        "summary": {"windows_run": 4, "observed_minus_null_total": 15.0},
        "windows": [
            _window(1, "2026-01-01T00:00:00+00:00", 1.0, ["036p1", "153p1"]),
            _window(2, "2026-01-01T01:00:00+00:00", 9.0, ["036p1", "153p1"]),
            _window(3, "2026-01-01T02:00:00+00:00", 3.0, ["036p1", "153p1"]),
            _window(4, "2026-02-01T00:00:00+00:00", 2.0, ["036p1", "153p1"]),
        ],
    }

    def fetcher(station_id, start, end):
        if start.hour == 0:
            hs, tp, ta, tz, psd = 1.0, 8.0, 7.5, 7.0, 8.0
        elif start.hour == 1:
            hs, tp, ta, tz, psd = 2.5, 14.0, 8.0, 7.0, 20.0
        else:
            hs, tp, ta, tz, psd = 1.2, 9.0, 7.5, 7.0, 9.0
        return [
            WaveAggRow(
                station_id=station_id,
                time=datetime(2026, 1, 1, max(start.hour, 0), tzinfo=UTC),
                waveHs=hs,
                waveTp=tp,
                waveTa=ta,
                waveTz=tz,
                waveDp=350.0 if station_id == "036" else 10.0,
                wavePeakPSD=psd,
                waveFlagPrimary=1,
            )
        ]

    attribution = build_attribution(
        report,
        top_windows=1,
        baseline_windows=2,
        pad_minutes=0.0,
        fetcher=fetcher,
        progress_every=0,
    )

    comparison = attribution["comparisons"]["high_delta_vs_background_baseline"]
    assert comparison["tp_ta_gap_mean"]["high_delta"]["mean"] > comparison["tp_ta_gap_mean"][
        "baseline"
    ]["mean"]
    assert attribution["source"]["role"].startswith("post-hoc")


def test_build_attribution_rejects_bad_platform_ids():
    report = {
        "windows": [_window(1, "2026-01-01T00:00:00+00:00", 1.0, ["bad-platform"])]
    }

    with pytest.raises(ValueError):
        build_attribution(
            report,
            top_windows=1,
            baseline_windows=0,
            pad_minutes=0.0,
            fetcher=lambda station, start, end: [],
            progress_every=0,
        )
