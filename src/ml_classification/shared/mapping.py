"""Label-map rendering and geometry sampling shared by spatial step plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from ml_classification.shared.reports.plotting import _save_figure as _save_figure  # noqa: PLC0414 - explicit public re-export


def save_predicted_labels_map(
    df: pd.DataFrame, label_column: str, output_path: str | Path, title: str = "Predicted Labels Map", max_geometries: int = 1000
) -> None:
    """Preserve the historical predicted-label map helper using the common renderer."""
    save_label_map(df, label_column, output_path, title=title, max_geometries=max_geometries)


def save_label_map(
    df: pd.DataFrame, label_column: str, output_path: str | Path, title: str = "Known Labels Map", max_geometries: int = 1000
) -> None:
    """Save a static label map for geometries in the dataframe."""
    # Exit early when geometry information is missing or unusable.
    if "geometry" not in df.columns or df.empty:
        return
    geometry = df["geometry"]
    if geometry.isna().all():
        return
    plot_df = df.loc[geometry.notna(), ["geometry", label_column]].copy()
    if plot_df.empty:
        return
    plot_df = _sample_geometries_for_plot(plot_df, max_geometries=max_geometries)
    labels = sorted(plot_df[label_column].astype(str).unique())
    palette = sns.color_palette("tab20", n_colors=max(3, len(labels)))
    color_map = {label: palette[idx] for idx, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(12, 10))
    for label in labels:
        class_df = plot_df.loc[plot_df[label_column].astype(str) == label]
        class_df.plot(ax=ax, color=color_map[label], markersize=14, linewidth=0.6, label=label)
    ax.set_title(title)
    ax.set_axis_off()
    legend_handles = [Patch(facecolor=color_map[label], edgecolor="none", label=label) for label in labels]
    ax.legend(handles=legend_handles, title=label_column, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    _save_figure(fig, output_path)


def _sample_geometries_for_plot(df: pd.DataFrame, max_geometries: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Sample geometries for plotting while preserving key spatial patterns."""
    # Prefer point sampling and polygon area-based selection depending on geometry type.
    if df.empty or len(df) <= max_geometries:
        return df
    geom_types = df["geometry"].geom_type.astype(str).str.lower()
    is_point_like = geom_types.isin(["point", "multipoint"]).all()
    is_polygon_like = geom_types.isin(["polygon", "multipolygon"]).all()

    if is_point_like:
        return df.sample(n=max_geometries, random_state=random_state).copy()

    if is_polygon_like:
        return _select_largest_polygons(df, max_geometries)

    polygon_mask = geom_types.isin(["polygon", "multipolygon"])
    if polygon_mask.any():
        polygon_df = _select_largest_polygons(df.loc[polygon_mask].copy(), min(max_geometries, int(max_geometries * 0.8)))
        remaining = max_geometries - len(polygon_df)
        other_df = df.loc[~polygon_mask].copy()
        if remaining > 0 and not other_df.empty:
            if len(other_df) > remaining:
                other_df = other_df.sample(n=remaining, random_state=random_state)
            return pd.concat([polygon_df, other_df], ignore_index=False)
        return polygon_df

    return df.sample(n=max_geometries, random_state=random_state).copy()


def _select_largest_polygons(df: pd.DataFrame, max_geometries: int) -> pd.DataFrame:
    """Select the largest polygons by area for map plotting."""
    # Measure areas in a projected CRS when available for better comparability.
    # Reproject to a metric CRS where possible to compute comparable polygon
    # areas (EPSG:3857 is a pragmatic choice for visualization sampling).
    area_df = df.copy()
    try:
        working = area_df
        if getattr(area_df, "crs", None) is not None and str(area_df.crs).lower() != "epsg:3857":
            working = area_df.to_crs(epsg=3857)
        areas = np.asarray(working.geometry.area)
    except Exception:
        logging.getLogger(__name__).debug("Error: _select_largest_polygons failed; using its fallback.", exc_info=True)
        areas = np.asarray(area_df.geometry.area)
    area_df["_plot_area"] = areas
    area_df = area_df.nlargest(max_geometries, "_plot_area").drop(columns=["_plot_area"])
    return area_df.copy()


def _geometry_is_points(df: pd.DataFrame) -> bool:
    """Return True when all geometries are point-like."""
    # Treat both point and multipoint geometries as point-like for plotting.
    if df.empty:
        return False
    return df["geometry"].geom_type.astype(str).str.lower().isin(["point", "multipoint"]).all()


def _marker_sizes_for_row_count(total_rows: int) -> tuple[float, float]:
    """Return train/test marker sizes tuned to total plotted row count."""
    # Reduce marker size for denser datasets to limit overlap.
    if total_rows >= 2000:
        return 8, 10
    if total_rows >= 1000:
        return 12, 15
    if total_rows >= 500:
        return 16, 20
    return 22, 28
