"""Buffer- and grid-based geometric clustering in the input coordinate system."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import shared.geometry as geom_l


def spatial_clustering_using_buffer(gdf, field_id, min_cluster_distance_in_m, simplify_in_m=24.384, field_cluster="cluster",
                                    plot_results=True, figsize=(15, 15)):
    """Perform clustering using buffer-based spatial analysis.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    field_id : str
        The column name with unique identifiers.
    min_cluster_distance_in_m : float
        The minimum distance to define clusters.
    simplify_in_m : float, optional
        The simplification distance in meters. Defaults to 24.384.
    field_cluster : str, optional
        The column name for cluster assignments. Defaults to 'cluster'.
    plot_results : bool, optional
        Whether to plot the clustering results. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 15).

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with cluster assignments.

    """
    if "geometry" not in gdf.columns:
        raise ValueError("Error: geometry column is not included in the dataframe")
    if field_id not in gdf.columns:
        raise ValueError(f"Error: {field_id} column is not included in the dataframe")

    gdf = gdf.copy()
    gdf[field_cluster] = None

    meter_to_df_units = geom_l.create_unit_conversion_factor(source_unit="m", target_unit=geom_l.get_unit_of_length(gdf))
    # Each geometry gets half the connection distance, so two expanding boundaries can bridge the full gap.
    buffer_distance = (min_cluster_distance_in_m / 2) * meter_to_df_units
    print("Buffer distance in data units: ", buffer_distance)
    buffered = gdf[["geometry"]].copy()
    buffered["geometry"] = buffered["geometry"].buffer(buffer_distance)
    buffered = geom_l.return_valid_geometries(buffered)

    if buffered.empty:
        return gdf

    # Union gives transitive connectivity: a chain can form one cluster much wider than the distance setting.
    merged = gpd.GeoDataFrame(geometry=[buffered["geometry"].union_all()], crs=gdf.crs)
    cluster = merged.explode(ignore_index=True)
    cluster = cluster[cluster.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    cluster = geom_l.return_valid_geometries(cluster)

    if cluster.empty:
        return gdf

    # Simplify only merged cluster boundaries; original parcel geometry stays intact for the returned frame.
    if simplify_in_m and simplify_in_m > 0:
        simplify_tolerance = simplify_in_m * meter_to_df_units
        cluster["geometry"] = cluster["geometry"].simplify(simplify_tolerance, preserve_topology=True)
        cluster = geom_l.return_valid_geometries(cluster)

    # Cluster Main is the largest merged area, which need not contain the most input features.
    cluster["area"] = cluster.area
    cluster = cluster.sort_values(by="area", ascending=False).reset_index(drop=True)
    cluster[field_cluster] = cluster.index.to_series().apply(lambda x: "Cluster Main" if x == 0 else f"Cluster {x}")

    # Preserve index labels for assignment after sjoin; callers should supply a unique, unnamed input index.
    gdf_to_join = gdf.reset_index().rename(columns={"index": "_orig_index"})
    # Avoid name collision in sjoin when <field_cluster> already exists on the left dataframe.
    if field_cluster in gdf_to_join.columns:
        gdf_to_join = gdf_to_join.drop(columns=[field_cluster])

    gdf_cluster = gpd.sjoin(gdf_to_join, cluster[[field_cluster, "area", "geometry"]], how="left", predicate="intersects")
    # Resolve multiple intersections in favor of the largest cluster, then restore one label per original index.
    gdf_cluster = gdf_cluster.sort_values(by=["_orig_index", "area"], ascending=[True, False])
    gdf_cluster = gdf_cluster.drop_duplicates(subset=["_orig_index"], keep="first")
    gdf_cluster = gdf_cluster.set_index("_orig_index")

    # Pandas aligns labels by index; features without a surviving spatial match retain a missing cluster.
    gdf[field_cluster] = gdf_cluster[field_cluster]

    gdf_cluster_group = gdf.groupby(field_cluster, as_index=False).agg({field_id: list})
    gdf_cluster_group["total"] = gdf_cluster_group[field_id].apply(lambda x: len(x))
    display(gdf_cluster_group)

    # Plot clustering results using centroids
    if plot_results:
        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        show_legend = len(gdf_cluster_group) <= 10
        gdf_centroids = gdf.copy()
        gdf_centroids["geometry"] = gdf_centroids["geometry"].centroid
        gdf_centroids.plot(column=field_cluster, cmap="tab20c", legend=show_legend, ax=ax, markersize=5)
        ax.set_title("Spatial Clustering Results (Centroids)")
        plt.show()

    return gdf


def _grid_axis_index(values, origin, cell_size, count):
    """Return clamped grid indices along one axis; a zero-width extent maps to index 0."""
    if count == 1 or cell_size <= 0:
        return np.zeros(len(values), dtype=int)
    return np.clip(np.floor((values - origin) / cell_size).astype(int), 0, count - 1)


def spatial_clustering_using_grid(gdf, field_id, rows, cols, field_cluster="cluster", plot_results=True, figsize=(15, 15)):
    """Assign each feature to a regular grid cell over the data extent.

    The grid spans ``gdf.total_bounds`` with ``rows`` x ``cols`` equal cells. Each
    feature belongs to exactly one cell, chosen by its centroid, so features that
    cross cell boundaries are not duplicated. This matches the assignment used by
    ``data_preparation.parcel_stats.multiuser.split_geodataframe_by_grid``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data. Use a projected CRS so
        centroids and cell sizes are meaningful.
    field_id : str
        The column name with unique identifiers.
    rows, cols : int
        The number of grid rows (south to north) and columns (west to east).
    field_cluster : str, optional
        The column name for cluster assignments. Defaults to 'cluster'. An
        existing column with this name is overwritten.
    plot_results : bool, optional
        Whether to plot the clustering results. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 15).

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of the input with integer cluster labels ``0..n-1``. Only non-empty
        cells receive a label, numbered row by row from the south-west cell.

    """
    if "geometry" not in gdf.columns:
        raise ValueError("Error: geometry column is not included in the dataframe")
    if field_id not in gdf.columns:
        raise ValueError(f"Error: {field_id} column is not included in the dataframe")
    if gdf.empty:
        raise ValueError("Error: gdf must contain at least one feature.")
    for name, value in (("rows", rows), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"Error: {name} must be a positive integer.")
    if (gdf.geometry.isna() | gdf.geometry.is_empty).any():
        raise ValueError("Error: All geometries must be non-empty before grid clustering.")

    gdf = gdf.copy()
    minx, miny, maxx, maxy = gdf.total_bounds
    cell_width = (maxx - minx) / cols
    cell_height = (maxy - miny) / rows
    print(f"Spatial grid: {rows}x{cols} cells of {cell_width:.0f} x {cell_height:.0f} data units")

    centroids = gdf.geometry.centroid
    col_idx = _grid_axis_index(centroids.x.to_numpy(), minx, cell_width, cols)
    row_idx = _grid_axis_index(centroids.y.to_numpy(), miny, cell_height, rows)
    # Renumber occupied cells consecutively in row-major order so empty cells leave no gaps.
    _cells, labels = np.unique(row_idx * cols + col_idx, return_inverse=True)
    gdf[field_cluster] = labels.astype(int)

    gdf_cluster_group = gdf.groupby(field_cluster, as_index=False).agg({field_id: list})
    gdf_cluster_group["total"] = gdf_cluster_group[field_id].apply(len)
    display(gdf_cluster_group)

    if plot_results:
        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        show_legend = len(gdf_cluster_group) <= 10
        gdf_centroids = gdf.copy()
        gdf_centroids["geometry"] = centroids
        gdf_centroids.plot(column=field_cluster, cmap="tab20c", categorical=True, legend=show_legend, ax=ax, markersize=5)
        ax.set_title("Grid Clustering Results (Centroids)")
        plt.show()

    return gdf
