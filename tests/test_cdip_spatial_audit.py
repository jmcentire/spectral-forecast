"""Tests for post-hoc CDIP spatial/tide audit helpers."""

from experiments.cdip_spatial_audit import (
    WindowPoint,
    bearing_bucket,
    classify_region,
    geographic_stratification,
    haversine_km,
    initial_bearing_degrees,
    spatial_surrogate_control,
)


def _point(index: int, delta: float, regions: tuple[str, ...]) -> WindowPoint:
    label = regions[0] if len(set(regions)) == 1 else "mixed:" + "+".join(sorted(set(regions)))
    return WindowPoint(
        index=index,
        start_epoch=float(index * 3600),
        delta=delta,
        z_delta=delta,
        latitude=0.0,
        longitude=0.0,
        platforms=tuple(f"p{idx}" for idx in range(len(regions))),
        platform_latitudes=tuple(float(idx) for idx in range(len(regions))),
        platform_longitudes=tuple(float(idx) for idx in range(len(regions))),
        regions=regions,
        region_label=label,
    )


def test_haversine_km_matches_one_degree_equator():
    assert 110.0 < haversine_km(0.0, 0.0, 0.0, 1.0) < 112.0


def test_initial_bearing_and_bucket_cardinals():
    assert 89.0 < initial_bearing_degrees(0.0, 0.0, 0.0, 1.0) < 91.0
    assert bearing_bucket(0.0) == "N"
    assert bearing_bucket(90.0) == "E"
    assert bearing_bucket(225.0) == "SW"


def test_classify_region_uses_coarse_cdip_basins():
    assert classify_region(33.8, -118.6) == "southern_california"
    assert classify_region(21.4, -157.7) == "hawaii"
    assert classify_region(44.0, -87.0) == "great_lakes"


def test_geographic_stratification_separates_same_and_mixed_regions():
    points = [
        _point(0, 3.0, ("southern_california", "southern_california")),
        _point(1, 1.0, ("southern_california", "southern_california")),
        _point(2, -2.0, ("hawaii", "southern_california")),
    ]

    result = geographic_stratification(points, high_delta_quantile=0.67)

    assert result["same_region"]["same_region"]["windows"] == 2
    assert result["same_region"]["same_region"]["delta_sum"] == 4.0
    assert result["same_region"]["mixed_region"]["windows"] == 1
    assert result["same_region"]["mixed_region"]["delta_sum"] == -2.0
    assert result["pair_exposure"]["same_region_pairs"] == 2
    assert result["pair_exposure"]["mixed_region_pairs"] == 1
    assert result["pair_exposure"]["same_region_delta_per_pair"] == 2.0
    assert result["pair_exposure"]["mixed_region_delta_per_pair"] == -2.0


def test_spatial_surrogate_control_can_be_disabled():
    result = spatial_surrogate_control(
        [_point(0, 1.0, ("hawaii", "southern_california"))],
        lag_edges_hours=[0.0, 1.0],
        distance_edges_km=[0.0, 100.0],
        high_delta_quantile=0.95,
        repeats=0,
        pair_count=10,
        seed=1,
        min_pairs=1,
    )

    assert result == {"repeats": 0, "metrics": {}}
