"""Summarize feature types, missingness, and active-feature status."""

from __future__ import annotations

import pandas as pd

from ml_classification.shared.config.config import ClassificationPipelineConfig


def build_feature_profile(
    df: pd.DataFrame, active_features: list[str], config: ClassificationPipelineConfig, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Build a dataframe that summarizes feature dtypes, null rates, and activity."""
    # Collect one diagnostics row per requested feature column.
    rows = []
    for column in feature_columns or config.feature_columns:
        rows.append(
            {
                "feature": column,
                "dtype": str(df[column].dtype),
                "is_active": column in active_features,
                "null_pct": round(float(df[column].isna().mean() * 100), 4),
                "n_unique": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["is_active", "null_pct", "feature"], ascending=[False, False, True])
