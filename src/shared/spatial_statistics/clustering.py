"""Geometric clustering and Voronoi construction in the input coordinate system."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from shapely.geometry import Polygon
from shapely.ops import voronoi_diagram
from sklearn.cluster import DBSCAN, KMeans
from tqdm.notebook import tqdm

import shared.geometry as geom_l
import shared.tabular as cm_l
from shared.assertions import expect_true


def spatial_clustering_using_kmeans(gdf, num_clusters="auto", max_clusters=10, plot_results=True, figsize=(15, 15)):
    """Perform K-means clustering with automatic determination using the Elbow Method.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    num_clusters : int or str, optional
        The number of clusters. If 'auto', uses the Elbow Method. Defaults to 'auto'.
    max_clusters : int, optional
        The maximum number of clusters to consider. Defaults to 10.
    plot_results : bool, optional
        Whether to plot the clustering results. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 15).

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with an additional 'cluster_label' column indicating cluster assignments.

    """
    expect_true(max_clusters is not None, "Variable <max_clusters> is None; please provide a valid integer number!")
    expect_true(max_clusters > 0, "Variable <max_clusters> is zero; please provide a valid integer number!")
    expect_true(type(max_clusters) == int, "Variable <max_clusters> is not integer; please provide a valid integer number!")
    expect_true(
        max_clusters == int(max_clusters),
        "Variable <max_clusters> is not integer; please provide a valid integer number!",
    )

    # Extract spatial coordinates from the geodataframe
    gdf = gdf.copy()
    gdf = geom_l.convert_geometries_to_points(gdf)
    # Euclidean clustering uses representative points in native CRS units; no projection or coordinate scaling occurs.
    coordinates = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    if num_clusters == "auto":
        # Calculate inertia for different numbers of clusters
        inertias = []
        for candidate_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=candidate_clusters, random_state=42, n_init=20)
            kmeans.fit(coordinates)
            inertias.append(kmeans.inertia_)

        # Plot the Elbow curve
        if plot_results:
            plt.plot(range(1, max_clusters + 1), inertias, marker="o")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Inertia")
            plt.title("Elbow Method for Optimal Number of Clusters")
            plt.show()

        # Automatically determine the number of clusters using the Elbow Method
        # Inertia is the within-cluster sum of squares
        # We'll look for the "elbow" point where the inertia starts decreasing more slowly
        # This indicates the optimal number of clusters
        # Second differences select the largest discrete bend, a heuristic rather than a validated model-selection score.
        # This needs at least three candidate counts, and each candidate must fit within the number of observations.
        deltas = np.diff(inertias, 2)
        optimal_num_clusters = deltas.argmax() + 2  # Add 2 because of zero-based indexing

        print(f"Optimal number of clusters based on Elbow Method: {optimal_num_clusters}")
    else:
        optimal_num_clusters = num_clusters
        print(f"Number of user defined clusters: {optimal_num_clusters}")

    # Perform K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42, n_init=20)
    # The returned geometries remain the representative points created above, not the original polygon or line shapes.
    gdf['cluster_label'] = kmeans.fit_predict(coordinates)

    # Plot clustering results
    if plot_results:
        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        gdf.plot(column="cluster_label", cmap="viridis", legend=True, ax=ax)
        ax.set_title(f"Clustering Analysis with {optimal_num_clusters} clusters")
        plt.show()

    return gdf


def spatial_clustering_using_dbscan(gdf, epsilon=0.1, min_samples=5, field_cluster="cluster", plot_results=True, figsize=(15, 15)):
    """Perform DBSCAN clustering on a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    epsilon : float, optional
        The maximum distance between two samples. Defaults to 0.1.
    min_samples : int, optional
        The minimum number of samples in a neighborhood for a core point. Defaults to 5.
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
    # Unlike K-means above, this helper requires point geometries and uses epsilon directly in CRS units.
    coordinates = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
    # DBSCAN marks noise as -1; assignment mutates the supplied frame because this helper does not copy it.
    gdf[field_cluster] = clustering.fit_predict(coordinates)

    # Plot clustering results
    if plot_results:
        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        gdf.plot(column=field_cluster, cmap="viridis", legend=True, ax=ax)
        ax.set_title("DBSCAN Clustering Results")
        plt.show()

    return gdf


def create_voronoi_polygons(df_points, df_extend=None, col_id="pipe_id", max_distance_in_m=304.8, simplify_in_m=24.384,
                            plot_graphs=True):
    """Create Voronoi polygons from point geometries.

    Parameters
    ----------
    df_points : geopandas.GeoDataFrame
        The GeoDataFrame for which to create Voronoi polygons.
    df_extend : geopandas.GeoDataFrame, optional
        The GeoDataFrame used to clip Voronoi polygons. Defaults to None.
    col_id : str, optional
        The column name with the ID in df_points. Defaults to 'pipe_id'.
    max_distance_in_m : float, optional
        The maximum distance in meters for outer polygon creation. Defaults to 304.8.
    simplify_in_m : float, optional
        The simplification distance in meters. Defaults to 24.384.
    plot_graphs : bool, optional
        Whether to plot the Voronoi diagram. Defaults to True.

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame with clipped Voronoi polygons.

    """
    if df_points.empty:
        print("Error: Create voronoi on empty geodataframe")
        return None

    cm_l.check_needed_df_columns(df_points, [col_id, 'geometry'])
    cm_l.check_needed_df_columns(df_extend, ['geometry'])

    df_points = df_points.copy()
    # Although the signature allows None, the current implementation requires a geometry-bearing extent here.
    df_extend = df_extend[['geometry']].copy()

    df_points = geom_l.convert_geometries_to_points(df_points)
    points = df_points.unary_union
    if isinstance(df_extend, gpd.geodataframe.GeoDataFrame):
        clip_polygon = geom_l.create_outer_polygons(df_extend, max_distance_in_m, simplify_in_m)
        clip_polygon = clip_polygon.unary_union
        regions = voronoi_diagram(points, envelope=clip_polygon)
        voronoi_pols = gpd.GeoDataFrame(geometry=[Polygon(region) for region in regions.geoms], crs=df_points.crs)
        # The Voronoi envelope bounds construction; explicit clipping enforces the actual irregular study boundary.
        voronoi_pols = gpd.clip(voronoi_pols, clip_polygon)
    else:
        regions = voronoi_diagram(points)
        voronoi_pols = gpd.GeoDataFrame(geometry=[Polygon(region) for region in regions.geoms], crs=df_points.crs)

    voronoi_pols = geom_l.explode_multigeometries(voronoi_pols)
    voronoi_pols = geom_l.return_valid_geometries(voronoi_pols)
    # Recover source attributes by spatial membership, not by assuming Voronoi cell order matches point order.
    # Coincident points or boundary matches can produce multiple joined rows for a cell.
    voronoi_pols = gpd.sjoin(voronoi_pols, df_points)
    if plot_graphs:
        voronoi_pols.plot()
    return voronoi_pols


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


def spatial_clustering_using_network(df, field_id, min_cluster_distance_in_m=304.8, field_cluster="cluster",
                                     plot_results=True, figsize=(15,15)):
    """Identify spatially isolated clusters based on network connectivity.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    field_id : str
        The column name with unique identifiers.
    min_cluster_distance_in_m : float, optional
        The minimum distance in meters between clusters for isolation. Defaults to 304.8.
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
    if "geometry" not in df.columns:
        raise ValueError("Error: geometry column is not included in the <field_id> dataframe!")
    if field_id not in df.columns:
        raise ValueError(f"Error: {field_id} column is not included in the <field_id> dataframe!")

    df = df.copy()
    G = geom_l.create_nearby_network(df, field_id, min_cluster_distance_in_m)

    # Connected components allow indirect links; their ordering is by member count rather than polygon area.
    components = geom_l.order_components_bylen(G)
    len_components = len(components)
    len_largest_component = components[0][0]
    len_smallest_component = components[-1][0]

    df[field_cluster] = None
    # create labels
    for i in tqdm(range(len_components), desc="Assign Cluster labels"):
        cluster_ids = list(components[i][1])
        # Map graph membership through feature IDs rather than dataframe row positions.
        cond = (df[field_id].isin(cluster_ids))
        df.loc[cond, field_cluster] = f"Cluster {i}"

    df.loc[(df[field_cluster] == "Cluster 0"), field_cluster] = "Cluster Main"
    len_clusters = len(df[df[field_cluster] != "Cluster Main"].groupby(field_cluster))
    print(f"The total number of isolated pipe clusters are: {len_components}")
    print(f"The largest cluster has: {len_largest_component} geometries")
    print(f"The smallest cluster has: {len_smallest_component} geometries")
    print(f"The disconnected clusters based on minimum distance of {min_cluster_distance_in_m} are: {len_clusters}")
    components = None
    G = None

    gdf_cluster_group = df.groupby(field_cluster, as_index=False).agg({field_id: list})
    gdf_cluster_group['total'] = gdf_cluster_group[field_id].apply(lambda x: len(x))
    display(gdf_cluster_group)

    # Plot clustering results using centroids
    if plot_results:
        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        show_legend = len(gdf_cluster_group) <= 10
        df_centroids = df.copy()
        df_centroids["geometry"] = df_centroids["geometry"].centroid
        df_centroids.plot(column=field_cluster, cmap="tab20c", legend=show_legend, ax=ax, markersize=5)
        ax.set_title("Spatial Clustering Results (Centroids)")
        plt.show()
    return df
