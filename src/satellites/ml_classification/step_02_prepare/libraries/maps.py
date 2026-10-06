"""Visualize the prepared spatial train/test split."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from satellites.ml_classification.shared.mapping import _marker_sizes_for_row_count, _sample_geometries_for_plot, _save_figure


def save_spatial_split_plot(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: str | Path, max_geometries: int = 1000, random_state: int = 42
) -> None:
    """Compare train/test locations using one representative point per geometry."""
    if "geometry" not in train_df.columns or "geometry" not in test_df.columns:
        return
    plot_frames = []
    for frame, seed in ((train_df, random_state), (test_df, random_state + 1)):
        geometry = frame["geometry"]
        points = frame.loc[geometry.notna() & ~geometry.is_empty, ["geometry"]].copy()
        # Convert before sampling so polygon area does not bias the displayed locations.
        # Work on copies: model inputs retain their original parcel geometries.
        points["geometry"] = points["geometry"].representative_point()
        plot_frames.append(_sample_geometries_for_plot(points, max_geometries=max_geometries, random_state=seed))
    train_plot_df, test_plot_df = plot_frames
    if train_plot_df.empty and test_plot_df.empty:
        return
    total_rows = len(train_plot_df) + len(test_plot_df)
    train_size, test_size = _marker_sizes_for_row_count(total_rows)
    fig, ax = plt.subplots(figsize=(10, 8))
    if not train_plot_df.empty:
        ax.scatter(
            train_plot_df.geometry.x,
            train_plot_df.geometry.y,
            s=train_size,
            c="#2E86AB",
            alpha=0.65,
            linewidths=0,
            marker="o",
            zorder=2,
            label=f"Train ({len(train_plot_df):,} shown)",
        )
    if not test_plot_df.empty:
        ax.scatter(
            test_plot_df.geometry.x,
            test_plot_df.geometry.y,
            s=1.8 * test_size,
            c="#D1495B",
            edgecolors="white",
            linewidths=0.4,
            marker="^",
            zorder=3,
            label=f"Test ({len(test_plot_df):,} shown)",
        )
    ax.set_title("Spatial Split: Train vs Test\nRepresentative points of parcel geometries")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_axis_off()
    _save_figure(fig, output_path)
