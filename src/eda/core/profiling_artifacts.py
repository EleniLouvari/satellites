"""EDA artifact assembly entry points."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import EDAConfig
from .profiling_feature_selection import (
    _build_feature_selection_criteria,
    _build_feature_selection_message,
    _build_feature_selection_proposal,
    _build_feature_target_associations,
    _build_plot_feature_order,
)
from .profiling_helpers import (
    _categorical_columns,
    _datetime_columns,
    _multivariate_feature_columns,
    _numeric_columns,
    _numeric_feature_columns,
)
from .profiling_multivariate import _annotate_multivariate_outliers, _build_multivariate_outlier_scores, _build_pca_summary
from .profiling_quality import _build_data_quality_flags
from .profiling_statistics import (
    _build_categorical_associations,
    _build_categorical_numeric_target_tests,
    _build_categorical_target_tests,
    _build_correlation_matrices,
    _build_missingness_target_tests,
    _build_multicollinearity_summary,
    _build_normality_tests,
    _build_numeric_target_correlations,
    _build_numeric_target_tests,
    _build_outlier_summary,
    _build_strong_correlation_pairs,
)
from .profiling_summaries import (
    _build_categorical_summary,
    _build_column_profile,
    _build_dataset_summary,
    _build_datetime_summary,
    _build_geospatial_summary,
    _build_missingness_summary,
    _build_numeric_summary,
    _build_robust_numeric_summary,
    _build_target_summary,
)

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def build_eda_artifacts(df: pd.DataFrame, config: EDAConfig) -> dict[str, Any]:
    """Build all tabular and JSON-ready EDA artifacts for a dataframe."""
    # Keep annotation and diagnostic preparation isolated from the caller's input table.
    dataset = df.copy()
    if dataset.empty:
        raise ValueError("Error: EDA requires a dataframe with at least one row.")

    geospatial = _build_geospatial_summary(dataset, config)
    numeric_columns = _numeric_columns(dataset)
    # Feature-only diagnostics omit the target, identifier, and prior EDA diagnostic fields.
    numeric_feature_columns = _numeric_feature_columns(numeric_columns, config)
    categorical_columns = _categorical_columns(dataset)
    datetime_columns = _datetime_columns(dataset)

    summary = _build_dataset_summary(
        dataset,
        config,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        geospatial=geospatial,
    )
    column_profile = _build_column_profile(dataset)
    data_quality_flags = _build_data_quality_flags(dataset, column_profile, config)
    numeric_summary = _build_numeric_summary(dataset, numeric_columns, config)
    robust_numeric_summary = _build_robust_numeric_summary(dataset, numeric_columns)
    categorical_summary = _build_categorical_summary(dataset, categorical_columns, config)
    datetime_summary = _build_datetime_summary(dataset, datetime_columns)
    normality_tests = _build_normality_tests(dataset, numeric_feature_columns, config)
    outlier_summary = _build_outlier_summary(dataset, numeric_feature_columns, config)
    correlation_matrices = _build_correlation_matrices(dataset, numeric_columns)
    strong_pairs = _build_strong_correlation_pairs(correlation_matrices["pearson"], config)
    missingness_summary = _build_missingness_summary(dataset)
    categorical_associations = _build_categorical_associations(dataset, categorical_columns, config)
    categorical_target_tests = _build_categorical_target_tests(dataset, categorical_columns, config)
    numeric_target_tests = _build_numeric_target_tests(dataset, numeric_columns, config)
    numeric_target_correlations = _build_numeric_target_correlations(dataset, numeric_columns, config)
    categorical_numeric_target_tests = _build_categorical_numeric_target_tests(dataset, categorical_columns, config)
    feature_target_associations = _build_feature_target_associations(
        dataset,
        config,
        categorical_target_tests,
        numeric_target_tests,
        numeric_target_correlations,
        categorical_numeric_target_tests,
    )
    feature_selection_criteria = _build_feature_selection_criteria(config)
    # Combine quality gates and target evidence before applying redundancy and shortlist limits.
    feature_selection_proposal = _build_feature_selection_proposal(
        dataset,
        config,
        feature_target_associations,
        data_quality_flags,
        correlation_matrices["spearman"],
        categorical_associations,
    )
    feature_selection_message = _build_feature_selection_message(feature_selection_proposal, feature_selection_criteria)
    plot_feature_order = _build_plot_feature_order(
        dataset, config, feature_selection_proposal, column_profile, data_quality_flags
    )
    # Use the same feature priority for VIF analysis and the report plots.
    multicollinearity = _build_multicollinearity_summary(dataset, plot_feature_order["numeric"], config)
    summary["plot_feature_order"] = plot_feature_order
    missingness_target_tests = _build_missingness_target_tests(dataset, config)
    pca_summary = _build_pca_summary(dataset, numeric_columns, config)
    multivariate_scores = _build_multivariate_outlier_scores(dataset, numeric_columns, config)
    outlier_mask = multivariate_scores["is_outlier_97_5pct"].eq(True)
    # The outlier table contains flagged rows; annotated_data below retains the full input population.
    multivariate_outliers = multivariate_scores.loc[outlier_mask].reset_index(drop=True)
    annotated_data = _annotate_multivariate_outliers(dataset, multivariate_scores)
    summary["multivariate_rows_scored"] = len(multivariate_scores)
    summary["multivariate_outliers_found"] = len(multivariate_outliers)
    summary["multivariate_outlier_confidence"] = 0.975
    # Record the configured feature subset so consumers can interpret the multivariate diagnostic.
    summary["multivariate_features_used"] = _multivariate_feature_columns(numeric_columns, config)
    target_summary = _build_target_summary(dataset, config)

    return {
        "summary": summary,
        "column_profile": column_profile,
        "data_quality_flags": data_quality_flags,
        "numeric_summary": numeric_summary,
        "robust_numeric_summary": robust_numeric_summary,
        "categorical_summary": categorical_summary,
        "datetime_summary": datetime_summary,
        "normality_tests": normality_tests,
        "outlier_summary": outlier_summary,
        "correlation_matrices": correlation_matrices,
        "strong_correlation_pairs": strong_pairs,
        "multicollinearity": multicollinearity,
        "missingness_summary": missingness_summary,
        "categorical_associations": categorical_associations,
        "categorical_target_tests": categorical_target_tests,
        "numeric_target_tests": numeric_target_tests,
        "numeric_target_correlations": numeric_target_correlations,
        "categorical_numeric_target_tests": categorical_numeric_target_tests,
        "feature_target_associations": feature_target_associations,
        "feature_selection_criteria": feature_selection_criteria,
        "feature_selection_proposal": feature_selection_proposal,
        "feature_selection_message": feature_selection_message,
        "plot_feature_order": plot_feature_order,
        "missingness_target_tests": missingness_target_tests,
        "pca_summary": pca_summary,
        "multivariate_outliers": multivariate_outliers,
        "annotated_data": annotated_data,
        "target_summary": target_summary,
        "geospatial_summary": geospatial,
    }
