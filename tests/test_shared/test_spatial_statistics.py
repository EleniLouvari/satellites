"""Regression checks for spatial-statistics imports and cross-module behavior."""

import importlib
import pickle

import geopandas as gpd
import pytest
from shapely.geometry import box

from shared import spatial_statistics
from shared.spatial_statistics import clustering
from tests.utils import expect_equal, expect_true


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("clustering", "spatial_clustering_using_buffer"),
    ],
)
def test_existing_imports_and_pickles_resolve_to_submodules(module, name):
    """Existing imports and function references serialized with the old path still resolve."""
    implementation = importlib.import_module(f"shared.spatial_statistics.{module}")
    exported = getattr(spatial_statistics, name)
    expect_true(exported is getattr(implementation, name))
    legacy_pickle = f"cshared.spatial_statistics\n{name}\n.".encode("ascii")
    expect_true(pickle.loads(legacy_pickle) is exported)
    expect_true(pickle.loads(pickle.dumps(exported)) is exported)


def test_buffer_clusters_follow_chains_and_preserve_parcels(monkeypatch):
    """Transitive buffer connectivity preserves source geometries and index-aligned labels."""
    monkeypatch.setattr(clustering, "display", lambda *_: None)
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b", "c", "d"]},
        geometry=[box(0, 0, 1, 1), box(2.5, 0, 3.5, 1), box(5, 0, 6, 1), box(30, 0, 31, 1)],
        crs="EPSG:3857",
        index=[10, 20, 30, 40],
    )
    original = parcels.copy()
    result = spatial_statistics.spatial_clustering_using_buffer(
        parcels, "parcel_id", min_cluster_distance_in_m=2, simplify_in_m=0, plot_results=False
    )
    expect_equal(result["cluster"].tolist(), ["Cluster Main", "Cluster Main", "Cluster Main", "Cluster 1"])
    expect_true(result.geometry.equals(original.geometry))
    expect_true(parcels.equals(original))
