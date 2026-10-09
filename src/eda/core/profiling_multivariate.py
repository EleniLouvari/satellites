"""Multivariate analyses for EDA profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_cleaning import OutlierAnalysis

from .config import EDAConfig
from .profiling_helpers import _multivariate_feature_columns, _prepared_numeric_matrix, _round


def _build_pca_summary(df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig) -> pd.DataFrame:
    """Build a PCA explained-variance summary using numeric columns."""
    numeric_df = _prepared_numeric_matrix(df, numeric_columns, config)
    if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 3:
        return pd.DataFrame()
    _, singular_values, _ = np.linalg.svd(numeric_df.to_numpy(), full_matrices=False)
    variances = (singular_values**2) / max(numeric_df.shape[0] - 1, 1)
    explained = variances / variances.sum() if variances.sum() else np.zeros_like(variances)
    cumulative = np.cumsum(explained)
    rows = []
    for idx, (ratio, cum_ratio) in enumerate(zip(explained, cumulative), start=1):
        rows.append(
            {
                "component": f"PC{idx}",
                "explained_variance_ratio": _round(ratio),
                "cumulative_explained_variance_ratio": _round(cum_ratio),
            }
        )
    return pd.DataFrame(rows)

def _build_multivariate_outlier_scores(
    df: pd.DataFrame, numeric_columns: list[str], config: EDAConfig
) -> pd.DataFrame:
    """Score every usable row and retain both positional and original-index identifiers."""
    output_columns = [
        "row_position",
        "source_index",
        "mahalanobis_distance",
        "mahalanobis_distance_squared",
        "pvalue",
        "is_outlier_97_5pct",
    ]
    feature_columns = _multivariate_feature_columns(numeric_columns, config)
    if len(feature_columns) < 2:
        return pd.DataFrame(columns=output_columns)

    # A positional RangeIndex makes annotation safe even when the source index contains duplicate labels.
    positional_df = df.reset_index(drop=True)
    scores = _outlier_analysis(positional_df, config).detect_multivariate_mahalanobis(feature_columns, limit=None)
    if scores.empty:
        return pd.DataFrame(columns=output_columns)
    scores = scores.rename(columns={"row_index": "row_position"})
    source_index = np.asarray(df.index, dtype=object)
    scores.insert(1, "source_index", source_index[scores["row_position"].astype(int).to_numpy()])
    return scores[output_columns]

def _annotate_multivariate_outliers(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Return the source DataFrame or GeoDataFrame with row-aligned multivariate diagnostic columns."""
    annotated = df.copy()
    flags = pd.array([pd.NA] * len(annotated), dtype="boolean")
    distances = np.full(len(annotated), np.nan, dtype=float)
    pvalues = np.full(len(annotated), np.nan, dtype=float)
    if not scores.empty:
        positions = scores["row_position"].astype(int).to_numpy()
        flags[positions] = scores["is_outlier_97_5pct"].astype(bool).to_numpy()
        distances[positions] = scores["mahalanobis_distance"].astype(float).to_numpy()
        pvalues[positions] = scores["pvalue"].astype(float).to_numpy()
    annotated["is_outlier_97_5pct"] = flags
    annotated["mahalanobis_distance"] = distances
    annotated["mahalanobis_pvalue"] = pvalues
    return annotated

def _outlier_analysis(df: pd.DataFrame, config: EDAConfig) -> OutlierAnalysis:
    """Create an outlier calculator configured identically to the EDA run."""
    # Pass every EDA-controlled outlier option through to prevent calculation
    # drift between this pipeline and direct OutlierAnalysis usage.
    return OutlierAnalysis(
        df,
        iqr_multiplier=config.iqr_multiplier,
        z_score_threshold=config.outlier_zscore_threshold,
        random_seed=config.random_state,
        max_multivariate_features=config.max_multivariate_features,
    )
