"""Generate figures owned by the predict step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml_classification.shared.reports.plotting import _save_figure


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
    ax.tick_params(axis="x", rotation=90)
    _save_figure(fig, output_path)
