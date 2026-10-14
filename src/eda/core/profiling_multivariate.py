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
    feature_columns = _multivariate_feature_columns(numeric_columns, config)
    positional_df = df[feature_columns].reset_index(drop=True)
    combined = _outlier_analysis(positional_df, config).detect_all_outliers()
    diagnostics = [column for column in combined if column not in feature_columns]
    scores = combined[diagnostics].copy()
    scores.insert(0, "row_position", np.arange(len(df)))
    scores.insert(1, "source_index", pd.Series(list(df.index), dtype=object))
    return scores


def _annotate_multivariate_outliers(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Copy all combined diagnostics positionally, preserving geometry and duplicate indices."""
    annotated = df.copy()
    for column in scores.columns:
        if column not in {"row_position", "source_index"}:
            annotated[column] = scores[column].array
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
