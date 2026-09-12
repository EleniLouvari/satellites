import geopandas as gpd
import numpy as np
from shapely.geometry import box

from satellites.data_preparation.features.temporal import reduce_annual_median_features


def test_reduce_annual_median_features_preserves_static_data_geometry_and_crs() -> None:
    periods = ("20231001", "20231101", "20231201")
    data = {
        "parcel_code": ["a", "b"],
        "label": [1, 2],
        "context": [10.0, 20.0],
        "B02_mean__20231001": [100.0, 200.0],
    }
    for period, first, second in zip(periods, [1.0, 2.0, 3.0], [4.0, 6.0, 8.0]):
        data[f"B02_median__{period}"] = [first, second]
    parcels = gpd.GeoDataFrame(data, geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:2100")

    reduced = reduce_annual_median_features(parcels, temporal_sources=["B02"], expected_periods=3)

    assert "B02_mean__20231001" not in reduced.columns
    assert all(f"B02_median__{period}" not in reduced.columns for period in periods)
    assert reduced["parcel_code"].tolist() == ["a", "b"]
    assert reduced["context"].tolist() == [10.0, 20.0]
    assert reduced.crs == parcels.crs
    assert reduced.geometry.equals(parcels.geometry)
    np.testing.assert_allclose(reduced["B02_median_annual_mean"], [2.0, 6.0])
    np.testing.assert_allclose(reduced["B02_median_annual_min"], [1.0, 4.0])
    np.testing.assert_allclose(reduced["B02_median_annual_max"], [3.0, 8.0])
    np.testing.assert_allclose(reduced["B02_median_annual_std"], [np.std([1, 2, 3]), np.std([4, 6, 8])])


def test_reduce_annual_median_features_rejects_missing_periods() -> None:
    parcels = gpd.GeoDataFrame(
        {"B02_median__20231001": [1.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:2100"
    )

    try:
        reduce_annual_median_features(parcels, temporal_sources=["B02"], expected_periods=2)
    except ValueError as exc:
        assert "Expected 2 monthly median columns" in str(exc)
    else:
        raise AssertionError("Missing monthly median columns should be rejected.")
