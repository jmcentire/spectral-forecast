"""Tests for CDIP manifest subset helpers."""

import pytest

from experiments.cdip_manifest_split import (
    assert_disjoint,
    filter_manifest_windows,
    subset_manifest,
)


def _row(group: int, window: int) -> dict[str, object]:
    return {
        "source_paths": [f"{group}_a.nc", f"{group}_b.nc", f"{group}_c.nc"],
        "start_time": float(group * 100 + window),
        "n_samples": 4096,
        "target_rate": 1.28,
        "group_index": group,
        "window_index": window,
    }


def test_filter_manifest_windows_by_group_range_and_parity():
    rows = [_row(group, window) for group in range(6) for window in range(2)]

    selected = filter_manifest_windows(
        rows,
        group_start=2,
        group_end=6,
        parity="odd",
    )

    assert [row["group_index"] for row in selected] == [3, 3, 5, 5]


def test_subset_manifest_preserves_signature_and_records_split_metadata():
    manifest = {
        "version": 1,
        "created_at": "2026-06-02T00:00:00+00:00",
        "signature": {"max_groups": 6},
        "windows": [_row(group, 0) for group in range(6)],
    }

    subset = subset_manifest(
        manifest,
        group_start=3,
        group_end=6,
        label="last-half",
    )

    assert subset["signature"] == manifest["signature"]
    assert subset["split"]["label"] == "last-half"
    assert subset["split"]["groups"] == 3
    assert [row["group_index"] for row in subset["windows"]] == [3, 4, 5]


def test_assert_disjoint_rejects_overlapping_windows():
    left = [_row(1, 0), _row(2, 0)]
    right = [_row(2, 0), _row(3, 0)]

    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(left, right)

    assert_disjoint([_row(1, 0)], [_row(1, 1)])
