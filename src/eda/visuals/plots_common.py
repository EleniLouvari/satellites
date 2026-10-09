"""Shared helpers and constants for EDA plotting modules."""

from __future__ import annotations

import warnings

import matplotlib

# Use a noninteractive backend so report generation also works in unattended runs.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..core.config import EDAConfig

TARGET_PALETTE = ["#4C78A8", "#E3BA22", "#F58518", "#7A9A48", "#B279A2", "#72B7B2", "#9C755F", "#BAB0AC"]
PROFILE_LABEL_FONT_SIZE = 8

def _style_profile_labels(axis) -> None:
    """Use the median-profile label size across related report plots."""
    axis.tick_params(axis="both", labelsize=PROFILE_LABEL_FONT_SIZE)
    axis.xaxis.label.set_size(PROFILE_LABEL_FONT_SIZE)
    axis.yaxis.label.set_size(PROFILE_LABEL_FONT_SIZE)

def _boxplot(**kwargs) -> None:
    """Draw a seaborn box plot without its Matplotlib orientation transition warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="vert: bool will be deprecated.*", category=PendingDeprecationWarning)
        sns.boxplot(**kwargs)

def _ordered_available_features(ordered_features: list[str], table: pd.DataFrame, max_features: int) -> list[str]:
    """Keep the highest-priority features that are available in a plot-specific statistical table."""
    available = set(table["column"].tolist()) if "column" in table.columns else set()
    return [feature for feature in ordered_features if feature in available][:max_features]

def _artifact_has_rows(artifacts: dict, key: str) -> bool:
    """Return True when the named artifact table exists and has at least one row."""
    return not artifacts.get(key, pd.DataFrame()).empty

def _limited_categories(series: pd.Series, max_categories: int) -> pd.Series:
    """Keep the most frequent categories and group the remaining levels as Other."""
    values = series.astype("string")
    counts = values.value_counts()
    if len(counts) <= max_categories:
        return values
    # Reserve one displayed category for the tail while leaving missing values missing.
    top_categories = counts.head(max_categories - 1).index
    return values.where(values.isna() | values.isin(top_categories), "Other")

def _target_color_mapping(series: pd.Series, max_target_levels: int) -> dict[str, str]:
    """Assign stable report-wide colors to categorical target levels."""
    levels = series.astype("string").value_counts().index.astype(str).tolist()
    if len(levels) > max_target_levels:
        levels = [*levels[: max_target_levels - 1], "Other"]
    return {str(level): TARGET_PALETTE[index % len(TARGET_PALETTE)] for index, level in enumerate(levels)}

def _apply_target_color_mapping(series: pd.Series, target_colors: dict[str, str]) -> pd.Series:
    """Apply the report-wide target grouping without re-ranking levels in a subset."""
    values = series.astype("string")
    if "Other" not in target_colors:
        return values
    # Reuse the report-wide grouping so a subset cannot silently change class colors.
    retained_levels = [level for level in target_colors if level != "Other"]
    return values.where(values.isna() | values.isin(retained_levels), "Other")

def _format_vertical_category_labels(axis, max_characters: int = 15, font_size: int = 8) -> None:
    """Truncate crowded category labels and display them vertically."""
    labels = [tick_label.get_text()[:max_characters] for tick_label in axis.get_xticklabels()]
    axis.set_xticks(axis.get_xticks(), labels=labels)
    axis.tick_params(axis="x", labelrotation=90, labelsize=font_size)

def _target_median_percentile_rows(
    df: pd.DataFrame,
    columns: list[str],
    target_column: str,
    target_colors: dict[str, str],
    target_levels: list[str],
) -> dict[str, dict[str, float]]:
    """Build feature x target median-percentile lookup rows."""
    rows: dict[str, dict[str, float]] = {}
    for column in columns:
        plot_df = _numeric_target_plot_frame(df, target_column, column, target_colors)
        if plot_df.empty:
            continue
        feature_values = plot_df[column].to_numpy(dtype=float)
        rows[column] = {
            level: _median_percentile_for_level(plot_df, target_column, column, level, feature_values) for level in target_levels
        }
    return rows

def _numeric_target_plot_frame(
    df: pd.DataFrame, target_column: str, column: str, target_colors: dict[str, str]
) -> pd.DataFrame:
    """Return cleaned numeric-target frame for plotting."""
    plot_df = df[[target_column, column]].copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        return plot_df
    plot_df[target_column] = _apply_target_color_mapping(plot_df[target_column], target_colors)
    return plot_df

def _median_percentile_for_level(
    plot_df: pd.DataFrame, target_column: str, column: str, level: str, feature_values: np.ndarray
) -> float:
    """Return percentile location of the target-level median value within feature values."""
    if not (plot_df[target_column] == level).any():
        return np.nan
    median_value = plot_df.loc[plot_df[target_column] == level, column].median()
    # Express each class median on the pooled feature scale so different units can share a heatmap.
    return float(stats.percentileofscore(feature_values, median_value, kind="mean"))

def _prepared_numeric_matrix(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Return standardized numeric matrix for PCA plots."""
    columns = numeric_columns[: config.max_multivariate_features]
    if not columns:
        return pd.DataFrame()
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        return pd.DataFrame()
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    # Exclude zero-variance columns before standardizing the PCA input.
    numeric_df = numeric_df.loc[:, numeric_df.std(ddof=0) > 0]
    if numeric_df.empty:
        return pd.DataFrame()
    return (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
