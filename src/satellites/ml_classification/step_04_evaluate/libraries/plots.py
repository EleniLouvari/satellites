"""Generate figures owned by the evaluate step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from satellites.ml_classification.shared.persistence import ensure_dir
from satellites.ml_classification.shared.reports.plotting import _save_figure


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
    # Shorten display text only; keep the complete class labels for evaluation.
    ax.set_xticks(ax.get_xticks(), labels=[label.get_text()[:20] for label in ax.get_xticklabels()])
    ax.set_yticks(ax.get_yticks(), labels=[label.get_text()[:20] for label in ax.get_yticklabels()])
    ax.tick_params(axis="x", labelsize=tick_fontsize, labelrotation=90)
    ax.tick_params(axis="y", labelsize=tick_fontsize, labelrotation=0)
    for text in ax.texts:
        text.set_fontsize(tick_fontsize)


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


def save_confusion_matrix_plot(y_true, y_pred, labels: list[str], output_path: str | Path, title: str) -> None:
    """Save a confusion matrix plot for predicted versus true labels."""
    # Use sklearn display helpers to keep label ordering consistent.
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, xticks_rotation=90, cmap="Blues", ax=ax, colorbar=False
    )
    _style_confusion_matrix_axes(ax, labels)
    ax.set_title(title)
    _save_figure(fig, output_path)


def save_binary_evaluation_panel(y_true, y_pred, y_proba, labels: list[str], output_path: str | Path, model_name: str) -> None:
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
        y_true, y_pred, display_labels=labels, xticks_rotation=90, cmap="Blues", ax=axes[1], colorbar=False
    )
    _style_confusion_matrix_axes(axes[1], labels)
    axes[1].set_title(f"Confusion Matrix: {model_name}")
    _save_figure(fig, output_path)


def save_multiclass_evaluation_panel(
    y_true, y_pred, y_proba, labels: list[str], output_path: str | Path, model_name: str
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
        RocCurveDisplay.from_predictions(class_truth, y_proba[:, class_index], name=f"{class_label} vs rest", ax=axes[0])
    roc_legend = axes[0].get_legend()
    if roc_legend is not None:
        roc_legend.remove()
    axes[0].set_title(f"Multiclass ROC: {model_name}")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, xticks_rotation=90, cmap="Blues", ax=axes[1], colorbar=False
    )
    _style_confusion_matrix_axes(axes[1], labels)
    axes[1].set_title(f"Confusion Matrix: {model_name}")
    _save_figure(fig, output_path)


def save_binary_curve_plots(
    y_true, y_proba, output_dir: str | Path, model_name: str, pos_label=None, include_roc: bool = True
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


def save_multiclass_roc_plot(y_true, y_proba, labels: list[str], output_path: str | Path, model_name: str) -> None:
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
        RocCurveDisplay.from_predictions(class_truth, y_proba[:, class_index], name=f"{class_label} vs rest", ax=ax)
    roc_legend = ax.get_legend()
    if roc_legend is not None:
        roc_legend.remove()
    ax.set_title(f"Multiclass ROC Curves: {model_name}")
    _save_figure(fig, output_path)
