"""Regression checks for spatial-statistics imports and cross-module behavior."""

import importlib
import pickle

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from shared import spatial_statistics
from shared.spatial_statistics import clustering
from tests.utils import expect_equal, expect_true


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("clustering", "spatial_clustering_using_buffer"),
        ("clustering", "spatial_clustering_using_grid"),
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


def test_grid_clusters_use_centroids_and_skip_empty_cells(monkeypatch):
    """Each parcel gets one consecutive row-major cell label; empty cells leave no gaps."""
    monkeypatch.setattr(clustering, "display", lambda *_: None)
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["sw", "se", "nw", "ne", "cross"], "cluster": ["old"] * 5},
        geometry=[
            box(0, 0, 1, 1),
            box(29, 0, 30, 1),
            box(0, 29, 1, 30),
            box(29, 29, 30, 30),
            # Crosses the west/middle column boundary; its centroid lies in the middle column.
            box(9, 0, 13, 1),
        ],
        crs="EPSG:3857",
        index=[5, 4, 3, 2, 1],
    )
    original = parcels.copy()
    result = spatial_statistics.spatial_clustering_using_grid(parcels, "parcel_id", rows=2, cols=3, plot_results=False)
    # Occupied cells in row-major order: (0,0)=sw, (0,1)=cross, (0,2)=se, (1,0)=nw, (1,2)=ne.
    expect_equal(result["cluster"].tolist(), [0, 2, 3, 4, 1])
    expect_true(result.index.equals(original.index))
    expect_true(result.geometry.equals(original.geometry))
    expect_true(parcels.equals(original))


@pytest.mark.parametrize(("rows", "cols"), [(0, 2), (2, 0), (True, 2), (1.5, 2)])
def test_grid_clusters_reject_invalid_shape(rows, cols):
    """Grid dimensions must be positive integers."""
    parcels = gpd.GeoDataFrame({"parcel_id": ["a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:3857")
    with pytest.raises(ValueError, match="positive integer"):
        spatial_statistics.spatial_clustering_using_grid(parcels, "parcel_id", rows=rows, cols=cols, plot_results=False)


def test_grid_clusters_handle_single_point_extent(monkeypatch):
    """A zero-width extent puts every parcel in the first cell instead of dividing by zero."""
    monkeypatch.setattr(clustering, "display", lambda *_: None)
    parcels = gpd.GeoDataFrame(
        {"parcel_id": ["a", "b"]}, geometry=[Point(5, 5), Point(5, 5)], crs="EPSG:3857"
    )
    result = spatial_statistics.spatial_clustering_using_grid(parcels, "parcel_id", rows=3, cols=3, plot_results=False)
    expect_equal(result["cluster"].tolist(), [0, 0])
