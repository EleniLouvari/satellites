"""Buffer-based geometric clustering in the input coordinate system."""

import geopandas as gpd
import matplotlib.pyplot as plt
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
