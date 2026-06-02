"""Tests for CDIP multi-file alignment helpers."""

from experiments.cdip_observe import (
    _intersect_time_spans,
    _select_aligned_window,
)


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
