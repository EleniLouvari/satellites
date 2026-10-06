"""Configurable IQR filtering for modeling features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig
from satellites.shared.data_cleaning import OutlierAnalysis


def apply_iqr_filter(
    dataset: pd.DataFrame, feature_columns: list[str], config: ClassificationPipelineConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optionally replace numeric IQR outliers with null values."""
    # Work on a copy of the dataset to avoid mutating caller-owned frames.
    result = dataset.copy()
    # Restrict the IQR process to numeric features only.
    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(result[column])]
    summary: dict[str, Any] = {
        "applied": config.apply_iqr,
        "lower_quantile": config.iqr_lower_quantile,
        "upper_quantile": config.iqr_upper_quantile,
        "multiplier": config.iqr_multiplier,
        "outlier_counts": {},
        "total_outliers_replaced": 0,
        "bounds": {},
        "skipped_non_numeric_features": [column for column in feature_columns if column not in numeric_features],
    }
    # If IQR is disabled or there are no numeric features return early with an explanatory summary.
    if not config.apply_iqr or not numeric_features:
        return result, summary

    # Use the shared OutlierAnalysis helper to determine IQR bounds and masks.
    analysis = OutlierAnalysis(
        result[numeric_features],
        iqr_lower_quantile=config.iqr_lower_quantile,
        iqr_upper_quantile=config.iqr_upper_quantile,
        iqr_multiplier=config.iqr_multiplier,
    )
    for column in numeric_features:
        lower, upper = analysis.calculate_iqr_bounds(column)
        mask = analysis.iqr_outlier_mask(column)
        count = int(mask.sum())
        # Record bounds as None when undefined to keep JSON-friendly values.
        summary["bounds"][column] = {"lower": lower if pd.notna(lower) else None, "upper": upper if pd.notna(upper) else None}
        summary["outlier_counts"][column] = count
        if count:
            # Replace detected outliers with NaN so downstream processing treats
            # them consistently as missing values (e.g., for imputation).
            result.loc[mask, column] = float("nan")
            summary["total_outliers_replaced"] += count

    return result, summary
