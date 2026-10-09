"""Interactive known-label maps owned by the check step."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import seaborn as sns

from ml_classification.shared.mapping import _sample_geometries_for_plot
from ml_classification.shared.persistence import ensure_dir


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
        logging.getLogger(__name__).debug("Error: save_interactive_label_map failed; using its fallback.", exc_info=True)
        return
    plot_df = df.loc[geometry.notna(), ["geometry", label_column]].copy()
    if plot_df.empty:
        return
    plot_df = _sample_geometries_for_plot(plot_df, max_geometries=max_geometries)
    try:
        plot_df = plot_df.to_crs(epsg=4326)
    except Exception:
        logging.getLogger(__name__).debug("Error: save_interactive_label_map failed; using its fallback.", exc_info=True)
        return
    labels = sorted(plot_df[label_column].astype(str).unique())
    palette = sns.color_palette("tab20", n_colors=max(3, len(labels)))
    color_map = {
        label: f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
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
            style_function=lambda _feature, color=color: {"color": color, "weight": 2, "fillColor": color, "fillOpacity": 0.55},
            marker=folium.CircleMarker(radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.85),
            tooltip=folium.GeoJsonTooltip(fields=[label_column], aliases=[f"{label_column}:"]),
        )
        geojson.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    # Keep the floating title readable without obscuring the map controls.
    title_html = (
        "<h3 style='position: fixed; top: 8px; left: 56px; z-index: 9999; "
        "background: rgba(255,255,255,0.9); padding: 6px 10px; border-radius: 6px; "
        f"font-family: sans-serif;'>{title}</h3>"
    )
    fmap.get_root().html.add_child(folium.Element(title_html))
    fmap.save(str(output_path))
