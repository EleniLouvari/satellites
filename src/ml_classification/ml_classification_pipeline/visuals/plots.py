"""Plotting helpers for class balance, metrics, confusion matrices, and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from ..core.persistence import ensure_dir


sns.set_theme(style="whitegrid")


def _confusion_matrix_tick_fontsize(labels: list[str]) -> float:
    """Return a readable tick-font size based on the number of labels."""
    # Shrink text as class count grows to avoid axis-label overlap.
    n_labels = len(labels)
    if n_labels >= 20:
        return 5
    if n_labels >= 15:
        return 6
    if n_labels >= 10:
        return 7
    if n_labels >= 7:
        return 8
    return 9


def _style_confusion_matrix_axes(ax, labels: list[str]) -> None:
    """Apply consistent tick and annotation styling to confusion-matrix axes."""
    # Reuse a dynamic font-size rule for both ticks and in-cell annotations.
    tick_fontsize = _confusion_matrix_tick_fontsize(labels)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    for text in ax.texts:
        text.set_fontsize(tick_fontsize)


def save_missing_values_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a horizontal bar chart of missing-value percentages by column."""
    # Plot only columns with nonzero missing percentage for clarity.
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    if missing_pct.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(4, len(missing_pct) * 0.35)))
    missing_pct.plot(kind="barh", ax=ax, color="#C66B3D")
    ax.set_title("Missing Values by Column (%)")
    ax.set_xlabel("Percent Missing")
    ax.set_ylabel("")
    _save_figure(fig, output_path)


def save_target_distribution_plot(series: pd.Series, output_path: str | Path, title: str) -> None:
    """Save a target-label distribution bar chart."""
    # Include missing values as an explicit class for transparent counts.
    plot_df = series.fillna("Missing").astype(str).value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.45)))
    plot_df.plot(kind="barh", ax=ax, color="#2E6F95")
    ax.set_title(title)
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    _save_figure(fig, output_path)


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
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    output_path: str | Path,
    max_features: int = 9,
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
                data=train_df,
                x=feature,
                bins=30,
                color="blue",
                label="Train Data",
                kde=True,
                ax=ax,
                alpha=0.35,
                stat="count",
            )
            sns.histplot(
                data=test_df,
                x=feature,
                bins=30,
                color="red",
                label="Test Data",
                kde=True,
                ax=ax,
                alpha=0.35,
                stat="count",
            )
        else:
            plot_df = pd.concat(
                [
                    train_series.astype(str).value_counts(normalize=True).rename("train"),
                    test_series.astype(str).value_counts(normalize=True).rename("test"),
                ],
                axis=1,
            ).fillna(0.0).head(12)
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


def save_cv_fold_comparison_plot(
    fold_scores_df: pd.DataFrame,
    output_path: str | Path,
    metric_name: str,
) -> None:
    """Save per-fold line plots for best model CV scores."""
    # Compare fold stability across models on a shared axis.
    if fold_scores_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=fold_scores_df,
        x="fold",
        y="score",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        linewidth=2,
        ax=ax,
    )
    ax.set_title(f"Best CV Score per Fold by Model ({metric_name})")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name)
    fold_values = sorted(pd.unique(fold_scores_df["fold"]))
    ax.set_xticks(fold_values)
    ax.set_xticklabels([str(int(value)) for value in fold_values])
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(fig, output_path)


def save_model_comparison_plot(metrics_df: pd.DataFrame, output_path: str | Path, metric_name: str) -> None:
    """Save a horizontal bar chart comparing model metric values."""
    # Sort ascending so best models appear at the bottom of the bar chart.
    if metrics_df.empty:
        return
    plot_df = metrics_df.sort_values(metric_name, ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.45)))
    ax.barh(plot_df["model"], plot_df[metric_name], color="#3B8EA5")
    ax.set_title(f"Model Comparison by {metric_name}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("")
    _save_figure(fig, output_path)


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Save a confusion matrix plot for predicted versus true labels."""
    # Use sklearn display helpers to keep label ordering consistent.
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    _style_confusion_matrix_axes(ax, labels)
    ax.set_title(title)
    _save_figure(fig, output_path)


def save_binary_evaluation_panel(
    y_true,
    y_pred,
    y_proba,
    labels: list[str],
    output_path: str | Path,
    model_name: str,
) -> None:
    """Save a two-panel binary evaluation figure with ROC and confusion matrix."""
    # Combine probability and classification diagnostics into one figure.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pos_label = labels[1] if len(labels) > 1 else None
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0], pos_label=pos_label)
    roc_legend = axes[0].get_legend()
    if roc_legend is not None:
        roc_legend.remove()
    axes[0].set_title(f"ROC Curve: {model_name}")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=axes[1],
        colorbar=False,
    )
    _style_confusion_matrix_axes(axes[1], labels)
    axes[1].set_title(f"Confusion Matrix: {model_name}")
    _save_figure(fig, output_path)


def save_multiclass_evaluation_panel(
    y_true,
    y_pred,
    y_proba,
    labels: list[str],
    output_path: str | Path,
    model_name: str,
) -> None:
    """Save a multiclass panel with one-vs-rest ROC and confusion matrix."""
    # Skip plotting when the problem is not truly multiclass.
    if len(labels) < 3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y_true_array = np.asarray(y_true).astype(str)
    for class_index, class_label in enumerate(labels):
        class_truth = (y_true_array == str(class_label)).astype(int)
        if class_truth.sum() == 0 or class_truth.sum() == len(class_truth):
            continue
        RocCurveDisplay.from_predictions(
            class_truth,
            y_proba[:, class_index],
            name=f"{class_label} vs rest",
            ax=axes[0],
        )
    roc_legend = axes[0].get_legend()
    if roc_legend is not None:
        roc_legend.remove()
    axes[0].set_title(f"Multiclass ROC: {model_name}")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=axes[1],
        colorbar=False,
    )
    _style_confusion_matrix_axes(axes[1], labels)
    axes[1].set_title(f"Confusion Matrix: {model_name}")
    _save_figure(fig, output_path)


def save_binary_curve_plots(
    y_true,
    y_proba,
    output_dir: str | Path,
    model_name: str,
    pos_label=None,
    include_roc: bool = True,
) -> None:
    """Save binary ROC and precision-recall curve plots."""
    # Ensure the output directory exists before writing curve images.
    output_dir = ensure_dir(output_dir)
    if include_roc:
        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, pos_label=pos_label)
        roc_legend = ax.get_legend()
        if roc_legend is not None:
            roc_legend.remove()
        ax.set_title(f"ROC Curve: {model_name}")
        _save_figure(fig, output_dir / f"{model_name}_roc_curve.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, pos_label=pos_label)
    ax.set_title(f"Precision-Recall Curve: {model_name}")
    _save_figure(fig, output_dir / f"{model_name}_pr_curve.png")


def save_multiclass_roc_plot(
    y_true,
    y_proba,
    labels: list[str],
    output_path: str | Path,
    model_name: str,
) -> None:
    """Save one-vs-rest ROC curves for multiclass predictions."""
    # Plot a curve per class only when positive and negative examples exist.
    if len(labels) < 3:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    y_true_array = np.asarray(y_true).astype(str)
    for class_index, class_label in enumerate(labels):
        class_truth = (y_true_array == str(class_label)).astype(int)
        if class_truth.sum() == 0 or class_truth.sum() == len(class_truth):
            continue
        RocCurveDisplay.from_predictions(
            class_truth,
            y_proba[:, class_index],
            name=f"{class_label} vs rest",
            ax=ax,
        )
    roc_legend = ax.get_legend()
    if roc_legend is not None:
        roc_legend.remove()
    ax.set_title(f"Multiclass ROC Curves: {model_name}")
    _save_figure(fig, output_path)


def save_prediction_fill_plot(original_target: pd.Series, filled_target: pd.Series, output_path: str | Path) -> None:
    """Save a bar chart summarizing known-versus-filled target rows."""
    # Aggregate counts by known/unknown origin and filled class label.
    plot_df = pd.DataFrame(
        {
            "original_known": original_target.notna().map({True: "Known", False: "Unknown"}),
            "filled_target": filled_target.astype(str),
        }
    )
    summary = plot_df.groupby(["original_known", "filled_target"]).size().reset_index(name="rows")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=summary, x="filled_target", y="rows", hue="original_known", ax=ax)
    ax.set_title("Final Target Distribution After Filling Unknown Labels")
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Save and close a matplotlib figure to the requested output path."""
    # Standardize layout and export settings across plotting helpers.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
