"""Tests for post-hoc CDIP anomaly catalog helpers."""

from experiments.cdip_anomaly_catalog import build_catalog


def _window(index: int, start: str, delta: float, group: list[str]) -> dict[str, object]:
    return {
        "global_window_index": index,
        "start": start,
        "end": start,
        "observed_minus_null": delta,
        "observed_multi_emission": delta + 10.0,
        "null_multi_emission_mean": 10.0,
        "group": group,
        "top_combinations": [
            {
                "active_platforms": ",".join(group[:2]),
                "emission_sum": delta,
                "max_score": delta / 2.0,
            }
        ],
    }


def test_build_catalog_summarizes_top_windows_without_detector_inputs(monkeypatch):
    def fake_locations(report):
        return {
            "a": {"region": "alpha"},
            "b": {"region": "alpha"},
            "c": {"region": "beta"},
        }

    monkeypatch.setattr("experiments.cdip_anomaly_catalog._platform_locations", fake_locations)
    report = {
        "parameters": {"preprocess": "highpass-dominant-mask"},
        "summary": {
            "windows_run": 3,
            "observed_minus_null_total": 9.0,
            "observed_total_z": 2.0,
            "null_total_unique_repeats": 10,
            "null_total_unique_exceedances": 0,
            "null_total_unique_p_ge_observed": 1 / 11,
        },
        "windows": [
            _window(1, "2026-01-01T00:00:00+00:00", 1.0, ["a", "b", "c"]),
            _window(2, "2026-01-01T01:00:00+00:00", 5.0, ["a", "b", "c"]),
            _window(3, "2026-01-02T00:00:00+00:00", 3.0, ["a", "b"]),
        ],
    }

    catalog = build_catalog(report, top_windows=2, cluster_gap_hours=6.0)

    assert [row["global_window_index"] for row in catalog["top_windows"]] == [2, 3]
    assert catalog["top_windows"][0]["region_label"] == "mixed:alpha+beta"
    assert catalog["recurring_combinations"][0]["active_platforms"] == "a,b"
    assert catalog["time_clusters"][0]["windows"] == 1
    assert catalog["interpretation_constraints"]["catalog_role"].startswith("post-hoc")
