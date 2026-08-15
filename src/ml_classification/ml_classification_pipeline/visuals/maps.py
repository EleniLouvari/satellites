"""Map visualization helpers for static and interactive spatial outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.persistence import ensure_dir

# Map helpers keep spatial coordinates attached while converting model output into visuals.


def save_label_map(
    df: pd.DataFrame,
    label_column: str,
    output_path: str | Path,
    title: str = "Known Labels Map",
    max_geometries: int = 1000,
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
    ax.legend(
        handles=legend_handles,
        title=label_column,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )
    _save_figure(fig, output_path)


def save_label_map_with_osm_basemap(
    df: pd.DataFrame,
    label_column: str,
    output_path: str | Path,
    title: str = "Known Labels Map with OSM Basemap",
    max_geometries: int = 1000,
) -> None:
    """Save an interactive label map rendered with an OSM basemap."""
    # Delegate rendering to the shared interactive-map helper.
    if "geometry" not in df.columns or df.empty:
        return
    save_interactive_label_map(df, label_column, output_path, title=title, max_geometries=max_geometries)


def save_interactive_label_map(
    df: pd.DataFrame,
    label_column: str,
    output_path: str | Path,
    title: str = "Known Labels Map with OSM Basemap",
    max_geometries: int = 1000,
) -> None:
    """Create and save an interactive folium map of labeled geometries."""
    # Guard against missing geometry or CRS before folium rendering.
    if "geometry" not in df.columns or df.empty:
        return
    geometry = df["geometry"]
    if geometry.isna().all() or getattr(df, "crs", None) is None:
        return
    try:
        import folium
    except Exception:
        return
    plot_df = df.loc[geometry.notna(), ["geometry", label_column]].copy()
    if plot_df.empty:
        return
    plot_df = _sample_geometries_for_plot(plot_df, max_geometries=max_geometries)
    try:
        plot_df = plot_df.to_crs(epsg=4326)
    except Exception:
        return
    labels = sorted(plot_df[label_column].astype(str).unique())
    palette = sns.color_palette("tab20", n_colors=max(3, len(labels)))
    color_map = {
        label: "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        for label, color in zip(labels, palette)
    }
    centroid = plot_df.geometry.union_all().centroid
    fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    for label in labels:
        class_df = plot_df.loc[plot_df[label_column].astype(str) == label].copy()
        color = color_map[label]
        geojson = folium.GeoJson(
            data=class_df.__geo_interface__,
            name=str(label),
            style_function=lambda _feature, color=color: {
                "color": color,
                "weight": 2,
                "fillColor": color,
                "fillOpacity": 0.55,
            },
            marker=folium.CircleMarker(radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.85),
            tooltip=folium.GeoJsonTooltip(fields=[label_column], aliases=[f"{label_column}:"]),
        )
        geojson.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fmap.get_root().html.add_child(
        folium.Element(
            f"<h3 style='position: fixed; top: 8px; left: 56px; z-index: 9999; background: rgba(255,255,255,0.9); padding: 6px 10px; border-radius: 6px; font-family: sans-serif;'>{title}</h3>"
        )
    )
    fmap.save(str(output_path))


def save_spatial_split_plot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str | Path,
    max_geometries: int = 1000,
    random_state: int = 42,
) -> None:
    """Save a map comparing train and test geometry locations."""
    # Downsample geometries for plot readability and performance.
    if "geometry" not in train_df.columns or "geometry" not in test_df.columns:
        return
    train_plot_df = _sample_geometries_for_plot(
        train_df.loc[train_df["geometry"].notna(), ["geometry"]].copy(),
        max_geometries=max_geometries,
        random_state=random_state,
    )
    test_plot_df = _sample_geometries_for_plot(
        test_df.loc[test_df["geometry"].notna(), ["geometry"]].copy(),
        max_geometries=max_geometries,
        random_state=random_state + 1,
    )
    if train_plot_df.empty or test_plot_df.empty:
        return
    total_rows = len(train_plot_df) + len(test_plot_df)
    train_size, test_size = _marker_sizes_for_row_count(total_rows)
    fig, ax = plt.subplots(figsize=(10, 8))
    if _geometry_is_points(train_plot_df):
        train_plot_df.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#2E86AB",
            markersize=train_size,
            linewidth=0.9,
            marker="o",
            label="train",
        )
    else:
        train_plot_df.boundary.plot(ax=ax, color="#2E86AB", linewidth=0.7, label="train")

    if _geometry_is_points(test_plot_df):
        test_plot_df.plot(
            ax=ax,
            facecolor="none",
            edgecolor="#D1495B",
            markersize=test_size,
            linewidth=0.9,
            marker="o",
            label="test",
        )
    else:
        test_plot_df.plot(
            ax=ax,
            facecolor="#D1495B",
            edgecolor="#9B2226",
            alpha=0.35,
            linewidth=0.5,
            label="test",
        )
    ax.set_title("Spatial Split: Train vs Test")
    ax.legend()
    ax.set_axis_off()
    _save_figure(fig, output_path)


def save_predicted_labels_map(
    df: pd.DataFrame,
    label_column: str,
    output_path: str | Path,
    title: str = "Predicted Labels Map",
    max_geometries: int = 1000,
) -> None:
    """Save a static map for predicted labels."""
    # Reuse the generic static label-map helper for prediction outputs.
    save_label_map(df, label_column, output_path, title=title, max_geometries=max_geometries)


def _sample_geometries_for_plot(
    df: pd.DataFrame,
    max_geometries: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
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
    area_df = df.copy()
    try:
        working = area_df
        if getattr(area_df, "crs", None) is not None and str(area_df.crs).lower() != "epsg:3857":
            working = area_df.to_crs(epsg=3857)
        areas = np.asarray(working.geometry.area)
    except Exception:
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


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Save and close a matplotlib map figure."""
    # Apply consistent figure export options across map helpers.
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
