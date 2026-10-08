"""Tests for parcel-based CRS selection."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from data_preparation.parcel_stats import estimate_utm_epsg_from_parcels


@pytest.mark.parametrize(
    ("geometry", "expected_epsg"),
    [
        (box(22.8, 40.5, 23.3, 41.0), 32634),
        (box(22.8, -34.0, 23.3, -33.0), 32734),
    ],
)
def test_estimate_utm_epsg_uses_centroid_zone_and_hemisphere(geometry, expected_epsg) -> None:
    parcels = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")

    assert estimate_utm_epsg_from_parcels(parcels) == expected_epsg


def test_estimate_utm_epsg_accepts_projected_parcels() -> None:
    parcels = gpd.GeoDataFrame(geometry=[box(22.8, 40.5, 23.3, 41.0)], crs="EPSG:4326").to_crs(2100)

    assert estimate_utm_epsg_from_parcels(parcels) == 32634


def test_estimate_utm_epsg_requires_a_known_crs() -> None:
    parcels = gpd.GeoDataFrame(geometry=[box(22.8, 40.5, 23.3, 41.0)])

    with pytest.raises(ValueError, match="must have a CRS"):
        estimate_utm_epsg_from_parcels(parcels)
