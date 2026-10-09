"""Summary table builders for EDA profiling."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import EDAConfig
from .profiling_helpers import (
    _logical_type,
    _numeric_columns,
    _round,
    _safe_percent,
    _series_numeric_summary,
    _target_task,
)

# Keep this helper focused on a single transformation so the reporting pipeline stays easy to follow.



def _build_dataset_summary(
    df: pd.DataFrame,
    config: EDAConfig,
    numeric_columns: list[str],
    categorical_columns: list[str],
    datetime_columns: list[str],
    geospatial: pd.DataFrame,
) -> dict[str, Any]:
    """Create a JSON-safe high-level dataset summary."""
    if "geometry" not in df.columns:
        duplicate_rows = int(df.duplicated().sum())
    else:
        # Compare parcel attributes without requiring geometry objects to participate in duplicate detection.
        duplicate_rows = int(df.drop(columns=["geometry"]).duplicated().sum())
    target_status = "not configured"
    if config.target_column:
        target_status = "present" if config.target_column in df.columns else "missing"
    return {
        "_schema": {"artifact": "eda_summary", "schema_version": config.schema_version},
        "config": asdict(config),
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "geometry_column_present": bool("geometry" in df.columns),
        "geospatial_summary_present": bool(not geospatial.empty),
        "target_column": config.target_column,
        "target_status": target_status,
        "target_task": _target_task(df, config),
        "id_column": config.id_column,
        "duplicate_rows_excluding_geometry": duplicate_rows,
        "total_missing_values": int(df.isna().sum().sum()),
        "total_missing_percent": _safe_percent(df.isna().sum().sum(), df.size),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / 1_000_000, 4),
    }

def _build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Profile every column in the input dataframe."""
    rows = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        # For geometry, this profile counts geometry types rather than distinct spatial shapes.
        unique = (
            int(series.nunique(dropna=True))
            if column != "geometry"
            else int(series.geom_type.nunique(dropna=True))
            if hasattr(series, "geom_type")
            else 0
        )
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "logical_type": _logical_type(series, column),
                "non_null_count": non_null,
                "missing_count": int(series.isna().sum()),
                "missing_percent": _safe_percent(series.isna().sum(), total_rows),
                "unique_count": unique,
                "unique_percent": _safe_percent(unique, max(non_null, 1)),
                "is_constant": bool(unique <= 1),
                "memory_usage_mb": round(float(series.memory_usage(deep=True)) / 1_000_000, 4),
            }
        )
    return pd.DataFrame(rows)

def _build_numeric_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Calculate descriptive statistics for numeric columns."""
    rows = []
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        clean = values.dropna()
        q1 = clean.quantile(0.25) if len(clean) else np.nan
        q3 = clean.quantile(0.75) if len(clean) else np.nan
        iqr = q3 - q1 if len(clean) else np.nan
        lower = q1 - config.iqr_multiplier * iqr if len(clean) else np.nan
        upper = q3 + config.iqr_multiplier * iqr if len(clean) else np.nan
        rows.append(
            {
                "column": column,
                "count": int(clean.count()),
                "missing_count": int(values.isna().sum()),
                "missing_percent": _safe_percent(values.isna().sum(), len(values)),
                "mean": _round(clean.mean()),
                "std": _round(clean.std()),
                "min": _round(clean.min()),
                "q1": _round(q1),
                "median": _round(clean.median()),
                "q3": _round(q3),
                "max": _round(clean.max()),
                "iqr": _round(iqr),
                "skew": _round(clean.skew()),
                "kurtosis": _round(clean.kurtosis()),
                "coefficient_of_variation": _round(clean.std() / clean.mean()) if len(clean) and clean.mean() else np.nan,
                "zeros_count": int((clean == 0).sum()),
                "negative_count": int((clean < 0).sum()),
                "iqr_lower_bound": _round(lower),
                "iqr_upper_bound": _round(upper),
                "iqr_outlier_count": int(((clean < lower) | (clean > upper)).sum()) if len(clean) else 0,
            }
        )
    return pd.DataFrame(rows)

def _build_robust_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Calculate robust univariate statistics for numeric columns."""
    rows = []
    for column in numeric_columns:
        clean = pd.to_numeric(df[column], errors="coerce").dropna()
        clean = clean[np.isfinite(clean)]
        if clean.empty:
            continue
        median = clean.median()
        # Normal scaling makes MAD comparable to standard deviation for normally distributed values.
        mad = stats.median_abs_deviation(clean, scale="normal", nan_policy="omit")
        trimmed = stats.trim_mean(clean, 0.1) if len(clean) >= 10 else clean.mean()
        sem = stats.sem(clean) if len(clean) > 1 else np.nan
        ci_low, ci_high = (np.nan, np.nan)
        if len(clean) > 1 and pd.notna(sem):
            ci_low, ci_high = stats.t.interval(0.95, len(clean) - 1, loc=clean.mean(), scale=sem)
        rows.append(
            {
                "column": column,
                "count": len(clean),
                "trimmed_mean_10pct": _round(trimmed),
                "median": _round(median),
                "mad_normalized": _round(mad),
                "range": _round(clean.max() - clean.min()),
                "percentile_1": _round(clean.quantile(0.01)),
                "percentile_5": _round(clean.quantile(0.05)),
                "percentile_95": _round(clean.quantile(0.95)),
                "percentile_99": _round(clean.quantile(0.99)),
                "mean_ci95_low": _round(ci_low),
                "mean_ci95_high": _round(ci_high),
            }
        )
    return pd.DataFrame(rows)

def _build_categorical_summary(df: pd.DataFrame, categorical_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Summarize categorical columns and their most frequent values."""
    rows = []
    for column in categorical_columns:
        series = df[column].dropna().astype(str)
        value_counts = series.value_counts()
        top_value = value_counts.index[0] if len(value_counts) else None
        top_count = int(value_counts.iloc[0]) if len(value_counts) else 0
        # The one-percent rarity cutoff uses observed values and never drops below a single occurrence.
        rare_threshold = max(1, int(0.01 * len(series)))
        probabilities = value_counts / value_counts.sum() if value_counts.sum() else pd.Series(dtype=float)
        entropy = float(stats.entropy(probabilities)) if len(probabilities) else np.nan
        rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "missing_count": int(df[column].isna().sum()),
                "missing_percent": _safe_percent(df[column].isna().sum(), len(df)),
                "unique_count": int(series.nunique()),
                "top_value": top_value,
                "top_count": top_count,
                "top_percent": _safe_percent(top_count, len(series)),
                "rare_value_count": int((value_counts <= rare_threshold).sum()),
                "entropy": _round(entropy),
                "top_values": dict(value_counts.head(config.max_categories)),
            }
        )
    return pd.DataFrame(rows)

def _build_datetime_summary(df: pd.DataFrame, datetime_columns: list[str]) -> pd.DataFrame:
    """Summarize datetime columns."""
    rows = []
    for column in datetime_columns:
        series = pd.to_datetime(df[column], errors="coerce")
        clean = series.dropna()
        rows.append(
            {
                "column": column,
                "count": int(clean.count()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": _safe_percent(series.isna().sum(), len(series)),
                "min": clean.min(),
                "max": clean.max(),
                "range_days": _round((clean.max() - clean.min()).days) if len(clean) else np.nan,
                "unique_count": int(clean.nunique()),
            }
        )
    return pd.DataFrame(rows)

def _build_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness and columns that tend to be missing together."""
    missing = df.isna()
    rows = []
    total_rows = len(df)
    for column in df.columns:
        missing_count = int(missing[column].sum())
        if missing_count == 0:
            continue
        other_missing = missing.drop(columns=[column], errors="ignore")
        paired = []
        if not other_missing.empty:
            # Measure co-missingness conditional on this column being missing, not across the whole dataset.
            rates = other_missing[missing[column]].mean().sort_values(ascending=False)
            paired = [f"{idx}: {_round(100 * value, 2)}%" for idx, value in rates.head(5).items() if value > 0]
        rows.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_percent": _safe_percent(missing_count, total_rows),
                "non_missing_count": int(total_rows - missing_count),
                "top_co_missing_columns": "; ".join(paired),
            }
        )
    return pd.DataFrame(rows)

def _build_target_summary(df: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Summarize target distribution and numeric features by target when configured."""
    if not config.target_column or config.target_column not in df.columns:
        return pd.DataFrame()
    target = df[config.target_column]
    numeric_columns = [column for column in _numeric_columns(df) if column != config.target_column]
    if _target_task(df, config) == "regression":
        rows = [{"section": "target_numeric", **_series_numeric_summary(target, config)}]
        return pd.DataFrame(rows)
    counts = target.dropna().astype(str).value_counts()
    rows = [
        {
            "section": "target_distribution",
            "target_value": value,
            "count": int(count),
            "percent": _safe_percent(count, target.notna().sum()),
        }
        for value, count in counts.items()
    ]
    for column in numeric_columns:
        grouped = (
            df.groupby(config.target_column, dropna=True)[column]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "section": "numeric_by_target",
                    "column": column,
                    "target_value": row[config.target_column],
                    "count": int(row["count"]),
                    "mean": _round(row["mean"]),
                    "median": _round(row["median"]),
                    "std": _round(row["std"]),
                    "min": _round(row["min"]),
                    "max": _round(row["max"]),
                }
            )
    return pd.DataFrame(rows)

def _build_geospatial_summary(df: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Create geospatial diagnostics when geometry is available."""
    if not config.include_geospatial or "geometry" not in df.columns:
        return pd.DataFrame()
    geometry = df["geometry"]
    if not hasattr(geometry, "geom_type"):
        return pd.DataFrame()
    rows = []
    crs = getattr(df, "crs", None)
    rows.append({"metric": "crs", "value": str(crs)})
    rows.append({"metric": "total_geometries", "value": len(geometry)})
    rows.append({"metric": "missing_geometries", "value": int(geometry.isna().sum())})
    rows.append({"metric": "empty_geometries", "value": int(geometry.is_empty.fillna(False).sum())})
    rows.append({"metric": "invalid_geometries", "value": int((~geometry.is_valid.fillna(False)).sum())})
    for geom_type, count in geometry.geom_type.value_counts(dropna=True).items():
        rows.append({"metric": f"geometry_type_{geom_type}", "value": int(count)})
    try:
        bounds = df.total_bounds
        rows.extend(
            [
                {"metric": "min_x", "value": _round(bounds[0])},
                {"metric": "min_y", "value": _round(bounds[1])},
                {"metric": "max_x", "value": _round(bounds[2])},
                {"metric": "max_y", "value": _round(bounds[3])},
            ]
        )
    except Exception:
        logging.getLogger(__name__).debug("Error: _build_geospatial_summary failed; using its fallback.", exc_info=True)
    try:
        is_projected = bool(getattr(crs, "is_projected", False))
        rows.append({"metric": "is_projected_crs", "value": is_projected})
        # Only projected coordinates support these planar area and length summaries.
        if is_projected:
            areas = geometry.area.replace([np.inf, -np.inf], np.nan).dropna()
            lengths = geometry.length.replace([np.inf, -np.inf], np.nan).dropna()
            rows.extend(
                [
                    {"metric": "area_mean", "value": _round(areas.mean())},
                    {"metric": "area_total", "value": _round(areas.sum())},
                    {"metric": "length_mean", "value": _round(lengths.mean())},
                    {"metric": "length_total", "value": _round(lengths.sum())},
                ]
            )
    except Exception:
        logging.getLogger(__name__).debug("Error: _build_geospatial_summary failed; using its fallback.", exc_info=True)
    return pd.DataFrame(rows)
