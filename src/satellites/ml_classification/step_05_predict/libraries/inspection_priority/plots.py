"""Plot inspection priority against confidence and input-data reliability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from satellites.ml_classification.shared.reports.plotting import _save_figure


def save_inspection_relationship_plot(data: pd.DataFrame, output_path: str | Path, config) -> bool:
    """Plot confidence, data reliability, and their resulting inspection score."""
    # Skip the optional report figure when the inspection layer was unavailable.
    required = {
        config.prediction_confidence_level_column,
        config.inspection_data_reliability_column,
        "geometry_risk",
        "inspection_score",
    }
    if not required.issubset(data.columns):
        return False

    # Add declaration-aware fields when present while retaining legacy score-only support.
    plot_columns = list(required)
    if "label_prediction_status" in data:
        plot_columns.append("label_prediction_status")
    plot_data = data[plot_columns].dropna(subset=list(required)).copy()
    if plot_data.empty:
        return False
    # Bound rendering cost while keeping sampling deterministic across repeated runs.
    if len(plot_data) > 5_000:
        plot_data = plot_data.sample(5_000, random_state=config.random_state)

    confidence_column = config.prediction_confidence_level_column
    reliability_column = config.inspection_data_reliability_column
    confidence_order = ["HIGH", "MEDIUM", "LOW"]
    palette = {"HIGH": "#2A9D8F", "MEDIUM": "#E9C46A", "LOW": "#E76F51"}
    # Use a dedicated full-size canvas; related matrices are exported as separate figures.
    fig, ax = plt.subplots(figsize=(15, 9))

    # Marker shape exposes label agreement without changing confidence colors.
    scatter_options = {}
    if "label_prediction_status" in plot_data:
        scatter_options = {
            "style": "label_prediction_status",
            "style_order": ["SAME", "DIFFERENT", "NO_DECLARATION", "UNAVAILABLE"],
        }
    sns.scatterplot(
        data=plot_data,
        x=reliability_column,
        y="inspection_score",
        hue=confidence_column,
        hue_order=confidence_order,
        palette=palette,
        size="geometry_risk",
        sizes=(20, 180),
        alpha=0.55,
        linewidth=0.2,
        ax=ax,
        **scatter_options,
    )
    ax.axvline(config.inspection_low_data_reliability_threshold, color="#8D99AE", linestyle="--", linewidth=1)
    ax.axvline(config.inspection_medium_data_reliability_threshold, color="#8D99AE", linestyle="--", linewidth=1)
    ax.axhline(config.inspection_medium_score_threshold, color="#6C757D", linestyle=":", linewidth=1)
    ax.axhline(config.inspection_high_score_threshold, color="#6C757D", linestyle=":", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Parcel inspection score by confidence and data reliability", loc="left", fontweight="bold")
    ax.set_xlabel("Data reliability score (higher is better)")
    ax.set_ylabel("Inspection score (higher means greater inspection need)")
    ax.grid(color="#D9E2EC", linewidth=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True)
    _save_figure(fig, output_path)
    return True
