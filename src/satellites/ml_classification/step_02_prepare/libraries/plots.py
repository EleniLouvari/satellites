"""Generate figures owned by the prepare step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from satellites.ml_classification.shared.reports.plotting import _save_figure


def save_split_distribution_plot(train_target: pd.Series, test_target: pd.Series, output_path: str | Path) -> None:
    """Save train-versus-test class-share comparison bars."""
    # Normalize counts per split so classes are comparable by share.
    plot_df = pd.concat(
        [
            train_target.astype(str).value_counts(normalize=True).rename("train"),
            test_target.astype(str).value_counts(normalize=True).rename("test"),
        ],
        axis=1,
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.5)))
    plot_df.sort_index().plot(kind="barh", ax=ax, color=["#1B998B", "#ED217C"])
    ax.set_title("Class Balance: Train vs Test")
    ax.set_xlabel("Share")
    ax.set_ylabel("")
    _save_figure(fig, output_path)


def save_train_test_feature_distributions(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str], output_path: str | Path, max_features: int = 9
) -> None:
    """Save feature distribution comparisons for train and test datasets."""
    # Cap plotted features to keep the grid compact and readable.
    plot_features = feature_columns[:max_features]
    if not plot_features:
        return
    n_cols = 3
    n_rows = (len(plot_features) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, feature in enumerate(plot_features):
        ax = axes[idx]
        train_series = train_df[feature]
        test_series = test_df[feature]
        if pd.api.types.is_numeric_dtype(train_series):
            sns.histplot(
                data=train_df, x=feature, bins=30, color="blue", label="Train Data", kde=True, ax=ax, alpha=0.35, stat="count"
            )
            sns.histplot(
                data=test_df, x=feature, bins=30, color="red", label="Test Data", kde=True, ax=ax, alpha=0.35, stat="count"
            )
        else:
            plot_df = (
                pd.concat(
                    [
                        train_series.astype(str).value_counts(normalize=True).rename("train"),
                        test_series.astype(str).value_counts(normalize=True).rename("test"),
                    ],
                    axis=1,
                )
                .fillna(0.0)
                .head(12)
            )
            plot_df.plot(kind="bar", ax=ax, color=["blue", "red"])
        ax.set_title(f"{feature}: Train vs Test")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency" if pd.api.types.is_numeric_dtype(train_series) else "Share")
        ax.legend()
    for idx in range(len(plot_features), len(axes)):
        axes[idx].set_visible(False)
    _save_figure(fig, output_path)


def save_correlation_heatmap(df: pd.DataFrame, output_path: str | Path, max_columns: int = 20) -> None:
    """Save a correlation heatmap for numeric features."""
    # Restrict very wide matrices to a bounded number of columns.
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return
    if numeric_df.shape[1] > max_columns:
        numeric_df = numeric_df.iloc[:, :max_columns]
    corr = numeric_df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fontsize = 6 if corr.shape[1] > 8 else 8
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.7},
        annot_kws={"fontsize": fontsize},
        ax=ax,
    )
    ax.set_title("Correlation Matrix with Heatmap")
    _save_figure(fig, output_path)
