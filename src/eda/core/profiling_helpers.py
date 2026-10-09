"""Shared helper functions for EDA profiling modules."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .config import EDAConfig
from .profiling_constants import _EDA_DIAGNOSTIC_COLUMNS

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric non-geometry columns."""
    return [column for column in df.columns if column != "geometry" and ptypes.is_numeric_dtype(df[column])]

def _categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return object-like categorical columns."""
    return [
        column
        for column in df.columns
        if column != "geometry" and not ptypes.is_numeric_dtype(df[column]) and not ptypes.is_datetime64_any_dtype(df[column])
    ]

def _datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return datetime columns."""
    return [column for column in df.columns if column != "geometry" and ptypes.is_datetime64_any_dtype(df[column])]

def _target_task(df: pd.DataFrame, config: EDAConfig) -> str | None:
    """Resolve whether target diagnostics should use classification or regression methods."""
    target = config.target_column
    if not target or target not in df.columns:
        return None
    # An explicit task handles numeric class codes that dtype inference would treat as regression.
    if config.target_task != "auto":
        return config.target_task
    return "regression" if ptypes.is_numeric_dtype(df[target]) else "classification"

def _logical_type(series: pd.Series, column: str) -> str:
    """Infer a friendly type label for report tables."""
    if column == "geometry":
        return "geometry"
    # Check booleans before numeric dtypes to preserve their distinct report label.
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"

def _safe_percent(numerator: float, denominator: float) -> float:
    """Calculate a percentage while avoiding division by zero."""
    return _round(100 * numerator / denominator) if denominator else 0.0

def _round(value: Any, digits: int = 4) -> Any:
    """Round finite numeric values and preserve missing values."""
    try:
        if pd.isna(value) or not np.isfinite(value):
            return np.nan
        return round(float(value), digits)
    except Exception:
        logging.getLogger(__name__).debug("Error: _round failed; using its fallback.", exc_info=True)
        return value

def _series_numeric_summary(series: pd.Series, config: EDAConfig) -> dict[str, Any]:
    """Return compact numeric statistics for a single series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "column": series.name,
        "count": int(clean.count()),
        "missing_count": int(series.isna().sum()),
        "mean": _round(clean.mean()),
        "std": _round(clean.std()),
        "min": _round(clean.min()),
        "median": _round(clean.median()),
        "max": _round(clean.max()),
    }

def _prepared_numeric_matrix(
    df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig, max_features: int | None = None
) -> pd.DataFrame:
    """Return a standardized numeric matrix for multivariate diagnostics."""
    feature_limit = config.max_multivariate_features if max_features is None else max_features
    columns = numeric_columns[:feature_limit]
    if not columns:
        return pd.DataFrame()
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    # An entirely missing column has no median for imputation and cannot enter the diagnostic matrix.
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        return pd.DataFrame()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    # Remove constant columns before division by their standard deviation.
    numeric_df = numeric_df.loc[:, numeric_df.std(ddof=0) > 0]
    if numeric_df.empty:
        return pd.DataFrame()
    # Standardization gives differently scaled features comparable weight in multivariate diagnostics.
    return (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)

def _numeric_feature_columns(numeric_columns: list[str], config: EDAConfig) -> list[str]:
    """Exclude target, identifier, and prior diagnostics from feature-only numeric analyses."""
    excluded = {config.target_column, config.id_column, *_EDA_DIAGNOSTIC_COLUMNS}
    return [column for column in numeric_columns if column not in excluded]

def _multivariate_feature_columns(numeric_columns: list[str], config: EDAConfig) -> list[str]:
    """Return the ordered numeric feature subset used for multivariate outlier scoring."""
    return _numeric_feature_columns(numeric_columns, config)[: config.max_multivariate_features]
