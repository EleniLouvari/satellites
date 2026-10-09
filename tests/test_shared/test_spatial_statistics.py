"""Regression checks for spatial-statistics imports and cross-module behavior."""

import importlib
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from shared import spatial_statistics
from shared.spatial_statistics import clustering, distributions, hotspots, scale_selection, time_series, weights
from tests.utils import expect_equal, expect_false, expect_true


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("clustering", "spatial_clustering_using_buffer"),
        ("weights", "calculate_spatial_weights_for_polygons"),
        ("hotspots", "spatial_outliers"),
        ("scale_selection", "process_dist"),
        ("distributions", "compare_distributions_ks"),
        ("trends", "check_spatial_trends"),
        ("similarity", "spatial_extent_similarity"),
        ("time_series", "time_sereis_cusum"),
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


def test_queen_weights_include_corner_neighbors_without_self_links():
    """Queen weights retain shared-corner adjacency and exclude disconnected polygons."""
    polygons = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(10, 10, 11, 11)],
        crs="EPSG:3857",
        index=[10, 20, 30],
    )
    matrix = weights.calculate_spatial_weights_for_polygons(polygons)
    expect_equal(set(matrix.neighbors[10]), {20})
    expect_equal(set(matrix.neighbors[20]), {10})
    expect_equal(matrix.neighbors[30], [])


def test_local_scores_use_other_observations_and_keep_input_unchanged():
    """The local outlier workflow still uses neighbor moments after the extraction."""
    data = gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 4.0, 5.0]}, geometry=[Point(x, 0) for x in range(4)], crs="EPSG:3857"
    )
    result, score_column, _ = hotspots.spatial_outliers(
        data, "value", "local", 30, 2.58, exclude_zeros=False, plot_results=False, verbose=False
    )
    expected = []
    for index in range(len(data)):
        neighbors = data["value"].drop(index)
        expected.append((data.loc[index, "value"] - neighbors.mean()) / neighbors.std())
    np.testing.assert_allclose(result[score_column], expected)
    expect_equal(result["neighbors_count"].tolist(), [3, 3, 3, 3])
    expect_equal(data.columns.tolist(), ["value", "geometry"])


def test_distance_worker_runs_in_spawned_process():
    """The relocated worker remains importable and serializable under Windows-style spawn."""
    data = gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 4.0, 5.0]}, geometry=[Point(x, 0) for x in range(4)], crs="EPSG:3857"
    )
    arguments = (30, data, "value", 0.05, 2.58)
    expected = scale_selection.process_dist(*arguments)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        actual = executor.submit(scale_selection.process_dist, *arguments).result(timeout=90)
    expect_equal(actual, expected)


def test_normality_helper_preserves_existing_decision(monkeypatch):
    """The structural change must not silently repair the documented legacy decision."""
    monkeypatch.setattr(distributions, "shapiro", lambda _: (0.99, 0.5))
    expect_false(distributions.qq_plot_with_distribution([1.0, 2.0, 3.0], "value", plot=False))


def test_distribution_dispatch_uses_its_module_helpers(monkeypatch):
    """The composite report finds every comparison helper in its new module."""
    called = []
    names = [
        "compare_distributions_ks", "compare_distributions_minkowski", "compare_distributions_ttest",
        "compare_distributions_expected", "compare_distributions_after_event",
    ]
    for name in names:
        monkeypatch.setattr(distributions, name, lambda *args, name=name, **kwargs: called.append(name))
    spatial_statistics.compare_two_distribution([1, 2, 3], [2, 3, 4], "value", plot=False)
    expect_equal(called, names)


def test_cusum_preserves_dead_band_and_continuing_alerts(capsys):
    """Small deviations decay toward zero; exceeding the threshold does not reset accumulation."""
    result = time_series.time_sereis_cusum([100, 12, 12, 8, 8], target=10, threshold=1.5, k=1)
    expect_equal(result, [0, 1, 2, 0, -1])
    expect_true("time step 2" in capsys.readouterr().out)


def test_zscore_helper_respects_supplied_neighborhood_moments():
    """Supplied local moments must not be replaced with whole-column summaries."""
    frame = pd.DataFrame({"value": [4.0, 10.0], "mean": [2.0, 4.0], "std": [1.0, 3.0]})
    result = hotspots.calc_zscore_and_pvalue(frame, "value", "mean", "std", "z", "p")
    np.testing.assert_allclose(result["z"], [2.0, 2.0])
    np.testing.assert_allclose(result["p"], [0.0455002639, 0.0455002639])
    expect_false("z" in frame)
