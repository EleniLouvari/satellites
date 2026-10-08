"""Generate figures owned by the check step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ml_classification.shared.reports.plotting import _save_figure


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
