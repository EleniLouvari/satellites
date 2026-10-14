"""Regression tests for the shared cleaning package and legacy imports."""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from shared.data_cleaning import OutlierAnalysis, fill_null_values_using_interpolation
from tests.utils import expect_equal, expect_true


def test_nearest_interpolation_fills_a_missing_point() -> None:
    data = gpd.GeoDataFrame({"value": [10.0, np.nan, 30.0]}, geometry=[Point(0, 0), Point(1, 0), Point(10, 0)], crs="EPSG:3857")

    result = fill_null_values_using_interpolation(data, "nearest", "value")

    expect_equal(result.loc[1, "value"], 10.0)
    expect_true(np.isnan(data.loc[1, "value"]))


def test_idw_respects_the_maximum_distance() -> None:
    data = gpd.GeoDataFrame({"value": [10.0, np.nan]}, geometry=[Point(0, 0), Point(100, 0)], crs="EPSG:3857")

    result = fill_null_values_using_interpolation(data, "idw", "value", max_distance_in_meters=50)

    expect_true(np.isnan(result.loc[1, "value"]))


def test_iqr_mask_is_index_aligned() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0]}, index=[10, 20, 30, 40, 50])

    mask = OutlierAnalysis(data).iqr_outlier_mask("value")

    expect_equal(mask.index.tolist(), data.index.tolist())
    expect_equal(mask.to_dict(), {10: False, 20: False, 30: False, 40: False, 50: True})


def test_outlier_masks_remain_row_aligned_with_duplicate_index_labels() -> None:
    data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0]}, index=[10, 10, 20, 20, 20])

    analysis = OutlierAnalysis(data)
    iqr_mask = analysis.iqr_outlier_mask("value")
    summary = analysis.summarize_univariate(["value"])

    expect_equal(iqr_mask.index.tolist(), data.index.tolist())
    expect_equal(iqr_mask.tolist(), [False, False, False, False, True])
    expect_equal(summary.loc[0, "count"], len(data))
    expect_equal(summary.loc[0, "iqr_outlier_count"], 1)


def test_combined_outliers_preserve_rows_and_leave_missing_rows_unscored() -> None:
    """Strong anomalies get a majority without treating absent data as normal."""
    random = np.random.default_rng(12)
    data = pd.DataFrame(random.normal(size=(100, 2)), columns=["x", "y"])
    data.iloc[-2] = [30, 30]
    data.iloc[-1] = [np.nan, np.inf]
    data.index = [1] * len(data)
    original = data.copy()
    result = OutlierAnalysis(data).detect_all_outliers()
    pd.testing.assert_frame_equal(data, original)
    expect_equal(result.index.tolist(), data.index.tolist())
    expect_true(result["is_outlier"].iloc[-2])
    expect_true(pd.isna(result["is_outlier"].iloc[-1]))
    expect_equal(result["eda_outlier_methods_available"].iloc[-1], 0)


def test_combined_outliers_require_strict_majority(monkeypatch) -> None:
    """Two of four votes is a tie; three of four is a majority."""
    from shared.data_cleaning import outliers

    data = pd.DataFrame({"x": [0., 1., 2., 3.], "y": [2., 0., 3., 1.]})
    monkeypatch.setattr(OutlierAnalysis, "iqr_outlier_mask", lambda self, column: pd.Series([True] * 4))
    monkeypatch.setattr(OutlierAnalysis, "zscore_outlier_mask", lambda self, column: pd.Series([False, True, False, True]))
    monkeypatch.setattr(outliers.IsolationForest, "fit_predict", lambda self, matrix: np.array([-1] * 4))
    result = OutlierAnalysis(data).detect_all_outliers()
    expect_equal(result["eda_outlier_methods_available"].tolist(), [4] * 4)
    expect_equal(result["is_outlier"].tolist(), [False, True, False, True])


def test_combined_outliers_without_usable_features_are_nullable() -> None:
    """Constant and absent features cannot supply detector votes."""
    result = OutlierAnalysis(pd.DataFrame({"x": [1.] * 6, "y": [np.nan] * 6})).detect_all_outliers()
    expect_true(result["is_outlier"].isna().all())
    expect_equal(result["eda_outlier_methods_available"].tolist(), [0] * 6)


def test_mahalanobis_uses_rank_for_collinear_features() -> None:
    """Duplicating a feature must not change its effective chi-square dimension."""
    from scipy import stats

    values = np.arange(20, dtype=float)
    result = OutlierAnalysis(pd.DataFrame({"x": values, "copy": 2 * values})).detect_multivariate_mahalanobis(limit=None)
    np.testing.assert_allclose(result["pvalue"], stats.chi2.sf(result["mahalanobis_distance_squared"], df=1))
