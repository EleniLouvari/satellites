"""Generate figures owned by the train step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from satellites.ml_classification.shared.reports.plotting import _save_figure


def save_search_results_plot(results_df: pd.DataFrame, output_path: str | Path, metric_name: str) -> None:
    """Save a chart of top hyperparameter-search outcomes."""
    # Focus on the strongest candidates to keep the ranking chart concise.
    if results_df.empty:
        return
    plot_df = results_df.nlargest(15, "mean_test_score").copy()
    plot_df = plot_df.sort_values("mean_test_score", ascending=True)
    labels = [f"rank {rank}" for rank in plot_df["rank_test_score"]]
    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.45)))
    ax.barh(labels, plot_df["mean_test_score"], color="#2D3047")
    ax.set_title(f"Top Search Results by {metric_name}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("")
    _save_figure(fig, output_path)


def save_cv_fold_comparison_plot(fold_scores_df: pd.DataFrame, output_path: str | Path, metric_name: str) -> None:
    """Save per-fold line plots for best model CV scores."""
    # Compare fold stability across models on a shared axis.
    if fold_scores_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=fold_scores_df, x="fold", y="score", hue="model", style="model", markers=True, dashes=False, linewidth=2, ax=ax
    )
    ax.set_title(f"Best CV Score per Fold by Model ({metric_name})")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name)
    fold_values = sorted(pd.unique(fold_scores_df["fold"]))
    ax.set_xticks(fold_values)
    ax.set_xticklabels([str(int(value)) for value in fold_values])
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(fig, output_path)
