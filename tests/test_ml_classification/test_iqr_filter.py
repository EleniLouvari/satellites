"""Boolean flags are never part of outlier analysis, and is_outlier is never null."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml_classification.step_01_check.libraries.iqr import apply_iqr_filter
from shared.data_cleaning import OutlierAnalysis


def _flags(rows=100):
    return {
        "flag": np.arange(rows) % 7 == 0,
        "nullable_flag": pd.array([True, False, None, False] * (rows // 4), dtype="boolean"),
        "object_flag": pd.Series([True, None, False, True] * (rows // 4), dtype=object),
    }


def test_iqr_skips_boolean_features_with_custom_quantiles():
    data = pd.DataFrame({"ndvi": np.r_[np.linspace(0.2, 0.8, 99), 50.0], **_flags()})
    config = SimpleNamespace(apply_iqr=True, iqr_lower_quantile=0.11, iqr_upper_quantile=0.99, iqr_multiplier=1.5)
    features = ["ndvi", "flag", "nullable_flag", "object_flag"]
    result, summary = apply_iqr_filter(data, features, config)
    assert summary["skipped_non_numeric_features"] == ["flag", "nullable_flag", "object_flag"]
    assert set(summary["bounds"]) == {"ndvi"} and summary["outlier_counts"]["ndvi"] == 1
    for column in features[1:]:
        assert result[column].equals(data[column])


def test_outlier_analysis_excludes_boolean_columns_everywhere():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), **_flags()})
    analysis = OutlierAnalysis(data, iqr_lower_quantile=0.11, iqr_upper_quantile=0.99)
    for column in _flags():
        assert OutlierAnalysis.is_boolean(data[column])
        assert all(np.isnan(analysis.calculate_iqr_bounds(column)))
        assert not analysis.iqr_outlier_mask(column).any() and not analysis.zscore_outlier_mask(column).any()
    assert not OutlierAnalysis.is_boolean(data["a"])
    assert analysis.summarize_univariate(list(data))["column"].tolist() == ["a", "b"]
    assert analysis.prepared_numeric_matrix(list(data)).columns.tolist() == ["a", "b"]


def test_is_outlier_defaults_to_false_when_too_few_methods_vote():
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    data.loc[:4, ["a", "b"]] = np.nan  # Rows without any observed feature get no votes.
    result = OutlierAnalysis(data).detect_all_outliers()
    assert result["is_outlier"].dtype == bool and result["is_outlier"].notna().all()
    unscored = result["eda_outlier_methods_available"] < 2
    assert unscored.any() and not result.loc[unscored, "is_outlier"].any()
