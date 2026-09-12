"""Regression tests for the shared cleaning package and legacy imports."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from satellites.shared.data_cleaning import OutlierAnalysis, fill_categorical_with_KNN, fill_null_values_using_interpolation


def test_nearest_interpolation_fills_a_missing_point() -> None:
    data = gpd.GeoDataFrame({"value": [10.0, np.nan, 30.0]}, geometry=[Point(0, 0), Point(1, 0), Point(10, 0)], crs="EPSG:3857")

    result = fill_null_values_using_interpolation(data, "nearest", "value")

    assert result.loc[1, "value"] == 10.0
    assert data.loc[1, "value"] is np.nan or np.isnan(data.loc[1, "value"])


def test_idw_respects_the_maximum_distance() -> None:
    data = gpd.GeoDataFrame({"value": [10.0, np.nan]}, geometry=[Point(0, 0), Point(100, 0)], crs="EPSG:3857")

    result = fill_null_values_using_interpolation(data, "idw", "value", max_distance_in_meters=50)

    assert np.isnan(result.loc[1, "value"])


def test_categorical_knn_uses_the_target_labels() -> None:
    data = pd.DataFrame({"class": ["A", "A", "B", "B", None], "x": [0.0, 0.2, 9.8, 10.0, 9.9]})

    result = fill_categorical_with_KNN(data, "class", ["x"], n_neighbors=1)

    assert result.loc[4, "class"] == "B"


def test_iqr_mask_is_index_aligned() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0]}, index=[10, 20, 30, 40, 50])

    mask = OutlierAnalysis(data).iqr_outlier_mask("value")

    assert mask.index.tolist() == data.index.tolist()
    assert mask.to_dict() == {10: False, 20: False, 30: False, 40: False, 50: True}


def test_outlier_masks_remain_row_aligned_with_duplicate_index_labels() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0]}, index=[10, 10, 20, 20, 20])

    analysis = OutlierAnalysis(data)
    iqr_mask = analysis.iqr_outlier_mask("value")
    summary = analysis.summarize_univariate(["value"])

    assert iqr_mask.index.tolist() == data.index.tolist()
    assert iqr_mask.tolist() == [False, False, False, False, True]
    assert summary.loc[0, "count"] == len(data)
    assert summary.loc[0, "iqr_outlier_count"] == 1
