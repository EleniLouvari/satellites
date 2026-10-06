"""Validate required input fields and normalize feature dtypes for the check step."""

from __future__ import annotations

import pandas as pd

from satellites.ml_classification.shared.config.config import ClassificationPipelineConfig


def validate_input(df: pd.DataFrame, config: ClassificationPipelineConfig) -> None:
    """Validate required columns, labels, and minimum class counts for training."""
    # Verify all configured feature columns are available in the input frame.
    missing_features = [column for column in config.feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' was not found.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    labeled = df[df[config.target_column].notna()]
    if labeled.empty:
        raise ValueError("No labeled rows found. The target column only contains missing values.")
    class_counts = labeled[config.target_column].astype(str).value_counts()
    if len(class_counts) < 2:
        raise ValueError("Classification requires at least two target classes.")
    if (class_counts < config.cv_folds).any():
        too_small = class_counts[class_counts < config.cv_folds].to_dict()
        raise ValueError(f"Each class needs at least {config.cv_folds} rows for stratified CV. Found: {too_small}")


def optimize_dataframe(
    df: pd.DataFrame, config: ClassificationPipelineConfig, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Cast feature columns to optimized numeric/category dtypes for efficiency."""
    # Work on a copy to keep caller-provided data unchanged.
    optimized = df.copy()
    for column in feature_columns or config.feature_columns:
        if pd.api.types.is_numeric_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype(config.float_dtype)
        else:
            optimized[column] = optimized[column].astype("category")
    return optimized
