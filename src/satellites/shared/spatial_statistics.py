
from IPython.display import display
from esda.moran import Moran
from esda.moran import Moran_Local
from libpysal.weights import DistanceBand
from libpysal.weights import Queen
from libpysal.weights.util import fill_diagonal
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.spatial.distance import minkowski
from scipy.stats import anderson
from scipy.stats import boxcox
from scipy.stats import chisquare
from scipy.stats import jarque_bera
from scipy.stats import ks_2samp
from scipy.stats import kurtosis
from scipy.stats import mannwhitneyu
from scipy.stats import norm
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import wilcoxon
from shapely.geometry import Polygon
from shapely.ops import voronoi_diagram
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from splot.esda import moran_scatterplot
from tqdm.notebook import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import warnings
import satellites.shared.constants as gb_l
import libraries.common_libraries.generic_library as cm_l
import libraries.common_libraries.geom_library as geom_l


def define_hotspot_columns(analysis):
    """Define column names for hotspot analysis results.

    Parameters
    ----------
    analysis : str
        The type of analysis (e.g., 'local', 'global', 'hotspot').

    Returns
    -------
    tuple
        A tuple containing column names for significant, p-value, z-score, hotspot class,
        high-low classification, mean, standard deviation, and outlier detection.

    """
    col_significant = f"{analysis}_significant"
    col_pvalue = f"{analysis}_pvalue"
    col_zscore = f"{analysis}_zscore"
    col_hotspot_class = f"{analysis}_class"
    col_highlow = f"{analysis}_high_low"
    col_mean = f"{analysis}_mean"
    col_stdev = f"{analysis}_stdev"
    col_outlier = f"{analysis}_outlier"
    return col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow, col_mean, col_stdev, col_outlier


def plot_outliers(gdf, col_outlier, attribute_column, col_hotspot_class, col_zscore, figsize=(10, 10), symbol_size=10):
    """Plot outliers and hotspot classifications for spatial data.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing the data to plot.
    col_outlier : str
        The column name indicating outliers.
    attribute_column : str
        The attribute column being analyzed.
    col_hotspot_class : str
        The column name containing hotspot classifications.
    col_zscore : str
        The column name containing z-score values.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (10, 10).
    symbol_size : int, optional
        The size of symbols in the plot. Defaults to 10.

    Returns
    -------
    None
        Displays the plot.

    """
    # Check geometry type of gdf
    gdf = gdf.copy()
    geometry_type = gdf.geometry.geom_type.iloc[0]

    if "point" in geometry_type.lower() or "line" in geometry_type.lower() or "polygon" in geometry_type.lower():
        color_mapping = {}
        gdf['class_color'] = "#9C9C9C"
        for z_class, (min_z, max_z, color) in gb_l.z_scores_dict.items():
            color_mapping[z_class] = color
            gdf.loc[gdf[col_hotspot_class] == z_class, 'class_color'] = color

        gdf['class_color'] = "#9C9C9C"
        for outlier, color in gb_l.outliers_dict.items():
            gdf.loc[gdf[col_outlier] == outlier, 'outlier_color'] = color

        figsize_w, figsize_h = figsize
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figsize_w, figsize_h - 4), sharex=False, sharey=True)

        if "point" in geometry_type.lower():
            gdf.plot(color=gdf['outlier_color'], ax=axs[0], marker=".", markersize=symbol_size)
            gdf.plot(color=gdf['class_color'], ax=axs[1], marker=".", markersize=symbol_size)

            handles1 = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                   markersize=symbol_size, label=label) for label, color in gb_l.outliers_dict.items()]
            handles2 = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                   markersize=symbol_size, label=label) for label, color in color_mapping.items()]

        elif "line" in geometry_type.lower():
            gdf.plot(color=gdf['outlier_color'], ax=axs[0], linewidth=symbol_size)
            gdf.plot(color=gdf['class_color'], ax=axs[1], linewidth=symbol_size)

            handles1 = [plt.Line2D([0], [0], color=color, linewidth=symbol_size, label=label) for label, color in
                        gb_l.outliers_dict.items()]
            handles2 = [plt.Line2D([0], [0], color=color, linewidth=symbol_size, label=label) for label, color in
                        color_mapping.items()]

        elif "polygon" in geometry_type.lower():
            gdf.plot(color=gdf['outlier_color'], ax=axs[0], edgecolor="grey", linewidth=symbol_size)
            gdf.plot(color=gdf['class_color'], ax=axs[1], linewidth=symbol_size)

            handles1 = [Patch(facecolor=color, edgecolor="grey", label=label) for label, color in
                        gb_l.outliers_dict.items()]
            handles2 = [Patch(facecolor=color, edgecolor="grey", label=label) for label, color in color_mapping.items()]

        axs[0].set_title(f"Outliers of the column {attribute_column}")
        axs[0].legend(handles=handles1, title=col_outlier, loc="center left", bbox_to_anchor=(1, 0.5))
        axs[1].set_title(f"Hot-Cold clusters of the column {attribute_column}")
        axs[1].legend(handles=handles2, title=col_hotspot_class, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()

        cm_l.plot_stat_numeric(gdf, col_zscore, exclude_zeros=False, figsize=(figsize_w, 2))
        gdf = gdf.drop(columns=['class_color', 'outlier_color'])
    else:
        print("Unsupported geometry type.")


def define_hot_spot_classes(gdf, attribute_column, analysis, significance_level=0.05):
    """Classify hotspots and outliers based on z-scores and significance levels.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame with z-scores and p-values.
    attribute_column : str
        The attribute column being analyzed.
    analysis : str
        The type of analysis (e.g., 'local', 'global', 'hotspot').
    significance_level : float, optional
        The p-value threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame with additional columns for hotspot classifications.

    """
    gdf = gdf.copy()
    col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow, col_mean, col_stdev, col_outlier = \
    define_hotspot_columns(analysis)

    # Identify statistically significant hotspots
    gdf[col_significant] = np.where(gdf[col_pvalue] < significance_level, True, False)

    gdf[col_hotspot_class] = "Not Significant"
    for z_class, (min_z, max_z, color) in gb_l.z_scores_dict.items():
        gdf.loc[((gdf[col_zscore] > min_z) & (gdf[col_zscore] <= max_z)), col_hotspot_class] = z_class

    # Classify observations into 'High-High', 'Low-Low', 'High-Low', 'Low-High' categories
    gdf[col_highlow] = "Not Significant"
    gdf.loc[(gdf[col_hotspot_class].str.startswith("Hot")) & (gdf[col_significant]) & (
                gdf[attribute_column] > gdf[col_mean]), col_highlow] = "HH cluster"
    gdf.loc[(gdf[col_hotspot_class].str.startswith("Hot")) & (gdf[col_significant]) & (
                gdf[attribute_column] < gdf[col_mean]), col_highlow] = "HL outlier"
    gdf.loc[(gdf[col_hotspot_class].str.startswith("Cold")) & (gdf[col_significant]) & (
                gdf[attribute_column] > gdf[col_mean]), col_highlow] = "LH outlier"
    gdf.loc[(gdf[col_hotspot_class].str.startswith("Cold")) & (gdf[col_significant]) & (
                gdf[attribute_column] < gdf[col_mean]), col_highlow] = "LL cluster"

    return gdf


def calc_zscore_and_pvalue(df, attribute_column, col_mean, col_stdev, col_zscore, col_pvalue):
    """Calculate z-scores and p-values for an attribute column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    attribute_column : str
        The column name for which to calculate z-scores and p-values.
    col_mean : str
        The column name for the mean.
    col_stdev : str
        The column name for the standard deviation.
    col_zscore : str
        The column name where z-scores will be stored.
    col_pvalue : str
        The column name where p-values will be stored.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with z-scores and p-values added.

    """
    df = df.copy()
    if col_mean not in list(df.columns):
        df[col_mean] = df[attribute_column].mean()
    if col_stdev not in list(df.columns):
        df[col_stdev] = df[attribute_column].std()

    cond = (df[col_stdev] > 0)
    df.loc[cond, col_zscore] = (df[cond][attribute_column] - df[cond][col_mean]) / df[cond][col_stdev]
    df[col_pvalue] = df.apply(lambda row: norm.sf(abs(row[col_zscore])) * 2, axis=1)

    return df


def spatial_outliers(gdf, attribute_column, analysis, search_distance_in_ft, outlier_z_threshold,
                     significance_level=0.05, exclude_zeros=True, plot_results=True, symbol_size=10,
                     figsize=(15, 15), verbose=True):
    """Detect spatial outliers using local or global analysis based on z-scores.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing the column to analyze.
    attribute_column : str
        The numeric column to check for outliers.
    analysis : str
        The type of analysis: 'local' or 'global'.
    search_distance_in_ft : float
        The search distance in feet to define the neighborhood for local analysis.
    outlier_z_threshold : float
        The z-score threshold above which a geometry is considered an outlier.
    significance_level : float, optional
        The p-value threshold for identifying statistically significant hotspots. Defaults to 0.05.
    exclude_zeros : bool, optional
        Whether to exclude zero values from calculations. Defaults to True.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.
    symbol_size : int, optional
        The size of symbols in plots. Defaults to 10.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 15).
    verbose : bool, optional
        Whether to print summary statistics. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the GeoDataFrame with outlier classifications, the z-score column name,
        and the maximum z-score value.

    """

    cm_l.check_needed_df_columns(gdf, [attribute_column])
    if not is_numeric_dtype(gdf[attribute_column]):
        raise ValueError(f"The {attribute_column} column is not numeric.")
    assert analysis in ['local', 'global'], "The variable <analysis> can be either 'local' or 'global'"

    gdf = gdf.copy()
    col_id = "tmp_id"
    col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow, col_mean, col_stdev, col_outlier = \
        define_hotspot_columns(analysis)

    cols_to_drop = ['neighbors_count', col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow,
                    col_mean, col_stdev, col_outlier]
    for col in cols_to_drop:
        if col in list(gdf.columns):
            gdf.drop(columns=[col], inplace=True)

    gdf[col_id] = gdf.index
    df = gdf.copy()

    if exclude_zeros:
        df[attribute_column] = df[attribute_column].fillna(0)
        df = df[df[attribute_column] > 0].copy()

    if analysis == "local":
        # Buffer geometries
        max_buffer = geom_l.convert_value_in_ft_to_df_units(gdf, search_distance_in_ft)
        df_buffer = df.copy()
        df_buffer['geometry'] = df_buffer['geometry'].buffer(max_buffer)
        col_id_left, col_id_right = f"{col_id}_left", f"{col_id}_right"

        # Perform spatial join
        df_join = gpd.sjoin(df_buffer[[col_id, 'geometry']], df[[col_id, attribute_column, 'geometry']])
        df_join = df_join[df_join[col_id_left] != df_join[col_id_right]].copy()

        # Group by ID and calculate statistics
        df_join = df_join.groupby(col_id_left).agg(
            neighbors_count=pd.NamedAgg(col_id_right, 'count'),
            mean=pd.NamedAgg(attribute_column, 'mean'),
            std=pd.NamedAgg(attribute_column, 'std'),
        ).reset_index()

        # Rename columns
        df_join.rename(columns={col_id_left: col_id, 'mean': col_mean, 'std': col_stdev}, inplace=True)

        # Merge statistics with original GeoDataFrame
        df = pd.merge(df, df_join, on=col_id)
    else:
        # Initialize columns for global analysis
        df['neighbors_count'] = len(df) - 1
        df[col_mean] = df[attribute_column].mean()
        df[col_stdev] = df[attribute_column].std()

    # Calculate z-score and p-value
    df = calc_zscore_and_pvalue(df, attribute_column, col_mean, col_stdev, col_zscore, col_pvalue)
    df[col_outlier] = abs(df[col_zscore]) > outlier_z_threshold
    df[col_outlier] = df[col_outlier].fillna(False)

    # Merge results with original GeoDataFrame
    gdf = pd.merge(gdf, df[[col_id, 'neighbors_count', col_mean, col_stdev, col_zscore, col_pvalue, col_outlier]],
                   how="left", on=col_id)
    gdf = gdf.fillna({'neighbors_count': 0, col_pvalue: 0, col_zscore: 0, col_mean: 0, col_stdev: 0, col_outlier: False})

    # Convert outlier column to 'yes'/'no'/'null' strings
    gdf[col_outlier] = gdf[col_outlier].fillna("null").astype(str).str.lower()
    gdf[col_outlier] = gdf[col_outlier].map({"true": "yes", "false": "no"})

    # Exclude zeros from outlier classification
    if exclude_zeros:
        gdf.loc[gdf[attribute_column] == 0, col_outlier] = "null"

    # Drop temporary ID column
    gdf.drop(columns=[col_id], inplace=True)
    max_z = round(gdf[col_zscore].max(), 2)
    if verbose:
        # Print summary statistics
        df_null = gdf[gdf[attribute_column] == 0].copy()
        df_outlier = gdf[gdf[col_outlier] == "yes"].copy()
        if analysis == "local":
            print(f"Cluster analysis of {attribute_column} with cluster size: {search_distance_in_ft}")
        print(f"Total zero/nulls values: {len(df_null)} ({round(100 * len(df_null) / len(gdf), 1)}%)")
        print(f"Total outliers: {len(df_outlier)} ({round(100 * len(df_outlier) / len(gdf), 1)}%)")
        print(f"Max z: {max_z}")

    # Create categories and plots
    gdf = define_hot_spot_classes(gdf, attribute_column, analysis=analysis, significance_level=significance_level)
    if plot_results:
        plot_outliers(gdf, col_outlier, attribute_column, col_hotspot_class, col_zscore, figsize, symbol_size)
    return gdf, col_zscore, max_z


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

    assert max_clusters is not None, "Variable <max_clusters> is None; please provide a valid integer number!"
    assert max_clusters > 0, "Variable <max_clusters> is zero; please provide a valid integer number!"
    assert type(max_clusters) == int, "Variable <max_clusters> is not integer; please provide a valid integer number!"
    assert max_clusters == int(
        max_clusters), "Variable <max_clusters> is not integer; please provide a valid integer number!"

    # Extract spatial coordinates from the geodataframe
    gdf = gdf.copy()
    gdf = geom_l.convert_geometries_to_points(gdf)
    coordinates = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    if num_clusters == "auto":
        # Calculate inertia for different numbers of clusters
        inertias = []
        for num_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20)
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
        deltas = np.diff(inertias, 2)
        optimal_num_clusters = deltas.argmax() + 2  # Add 2 because of zero-based indexing

        print(f"Optimal number of clusters based on Elbow Method: {optimal_num_clusters}")
    else:
        optimal_num_clusters = num_clusters
        print(f"Number of user defined clusters: {optimal_num_clusters}")

    # Perform K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42, n_init=20)
    gdf['cluster_label'] = kmeans.fit_predict(coordinates)

    # Plot clustering results
    if plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
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
    coordinates = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
    gdf[field_cluster] = clustering.fit_predict(coordinates)

    # Plot clustering results
    if plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        gdf.plot(column=field_cluster, cmap="viridis", legend=True, ax=ax)
        ax.set_title("DBSCAN Clustering Results")
        plt.show()

    return gdf


def create_voronoi_polygons(df_points, df_extend=None, col_id="pipe_id", max_distance_in_ft=1000, simplify_in_ft=80,
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
    max_distance_in_ft : float, optional
        The maximum distance in feet for outer polygon creation. Defaults to 1000.
    simplify_in_ft : float, optional
        The simplification distance in feet. Defaults to 80.
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
    df_extend = df_extend[['geometry']].copy()

    df_points = geom_l.convert_geometries_to_points(df_points)
    points = df_points.unary_union
    if isinstance(df_extend, gpd.geodataframe.GeoDataFrame):
        clip_polygon = geom_l.create_outer_polygons(df_extend, max_distance_in_ft, simplify_in_ft)
        clip_polygon = clip_polygon.unary_union
        regions = voronoi_diagram(points, envelope=clip_polygon)
        voronoi_pols = gpd.GeoDataFrame(geometry=[Polygon(region) for region in regions.geoms], crs=df_points.crs)
        voronoi_pols = gpd.clip(voronoi_pols, clip_polygon)
    else:
        regions = voronoi_diagram(points)
        voronoi_pols = gpd.GeoDataFrame(geometry=[Polygon(region) for region in regions.geoms], crs=df_points.crs)

    voronoi_pols = geom_l.explode_multigeometries(voronoi_pols)
    voronoi_pols = geom_l.return_valid_geometries(voronoi_pols)
    voronoi_pols = gpd.sjoin(voronoi_pols, df_points)
    if plot_graphs:
        voronoi_pols.plot()
    return voronoi_pols


def spatial_clustering_using_buffer(gdf, field_id, min_cluster_distance, simplify_in_ft=80, field_cluster="cluster",
                                    plot_results=True, figsize=(15, 15)):
    """Perform clustering using buffer-based spatial analysis.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    field_id : str
        The column name with unique identifiers.
    min_cluster_distance : float
        The minimum distance to define clusters.
    simplify_in_ft : float, optional
        The simplification distance in feet. Defaults to 80.
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
    gdf = gdf.copy()
    cluster = geom_l.create_outer_polygons(gdf, max_distance_in_ft=min_cluster_distance / 2,
                                           simplify_in_ft=simplify_in_ft)
    cluster['area'] = cluster.area
    cluster = cluster.sort_values(by="area", ascending=False).reset_index(drop=True)
    cluster[field_cluster] = cluster.index.to_series().apply(lambda x: f"Cluster {x}")
    cluster.loc[(cluster[field_cluster] == "Cluster 0"), field_cluster] = "Cluster Main"

    gdf_cluster_group = gpd.sjoin(cluster[[field_cluster, 'geometry']], gdf[[field_id, 'geometry']])
    gdf_cluster_group = gdf_cluster_group.groupby(field_cluster, as_index=False).agg({field_id: list})
    gdf_cluster_group['total'] = gdf_cluster_group[field_id].apply(lambda x: len(x))
    display(gdf_cluster_group)

    gdf_cluster = gpd.sjoin(gdf[[field_id, 'geometry']], cluster[[field_cluster, 'geometry']])
    # Plot clustering results
    if plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        gdf_cluster.plot(column=field_cluster, cmap="tab20c", legend=True, ax=ax)
        ax.set_title("Spatial Clustering Results")
        plt.show()
    return gdf


def spatial_clustering_using_network(df, field_id, min_cluster_distance=1000, field_cluster="cluster",
                                     plot_results=True, figsize=(15,15)):
    """Identify spatially isolated clusters based on network connectivity.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    field_id : str
        The column name with unique identifiers.
    min_cluster_distance : float, optional
        The minimum distance in feet between clusters for isolation. Defaults to 1000.
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
        raise Exception("geometry column is not included in the <field_id> dataframe!")
    if field_id not in df.columns:
        raise Exception(f"{field_id} column is not included in the <field_id> dataframe!")

    df = df.copy()
    G = geom_l.create_nearby_network(df, field_id, min_cluster_distance)

    components = geom_l.order_components_bylen(G)
    len_components = len(components)
    len_largest_component = components[0][0]
    len_smallest_component = components[-1][0]

    df[field_cluster] = None
    # create labels
    for i in tqdm(range(0, len_components), desc="Assign Cluster labels"):
        cluster_ids = list(components[i][1])
        cond = (df[field_id].isin(cluster_ids))
        df.loc[cond, field_cluster] = f"Cluster {i}"

    df.loc[(df[field_cluster] == "Cluster 0"), field_cluster] = "Cluster Main"
    len_clusters = len(df[df[field_cluster] != "Cluster Main"].groupby(field_cluster))
    print(f"The total number of isolated pipe clusters are: {len_components}")
    print(f"The largest cluster has: {len_largest_component} geometries")
    print(f"The smallest cluster has: {len_smallest_component} geometries")
    print(f"The disconnected clusters based on minimum distance of {min_cluster_distance} are: {len_clusters}")
    components = None
    G = None

    gdf_cluster_group = df.groupby(field_cluster, as_index=False).agg({field_id: list})
    gdf_cluster_group['total'] = gdf_cluster_group[field_id].apply(lambda x: len(x))
    display(gdf_cluster_group)

    # Plot clustering results
    if plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        show_legend = False if len(gdf_cluster_group)>10 else True
        df.plot(column=field_cluster, cmap="tab20c", legend=show_legend, ax=ax)
        ax.set_title("Spatial Clustering Results")
        plt.show()
    return df


def qq_plot_with_distribution(values, attribute_column, exclude_zeros=True, plot=True, figsize=(20, 8)):
    """Create a Q-Q plot with distribution analysis for normality assessment.

    Parameters
    ----------
    values : pandas.DataFrame or list
        The input DataFrame or list of values.
    attribute_column : str
        The name of the attribute column to analyze.
    exclude_zeros : bool, optional
        Whether to exclude zero values. Defaults to True.
    plot : bool, optional
        Whether to display the plot. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (20, 8).

    Returns
    -------
    bool
        True if data appears normally distributed, False otherwise.

    """
    # Check if the column is numerical
    if isinstance(values, pd.DataFrame):
        if not np.issubdtype(values[attribute_column].dtype, np.number):
            raise ValueError(f"Column {attribute_column} must be numerical.")
        df = values.copy()
    else:
        if not all(isinstance(x, (int, float)) for x in values):
            raise ValueError("All values must be numerical.")
        df = pd.DataFrame({attribute_column: values})

    df = df.copy()
    # Extract the non-null values
    df = df[df[attribute_column].notna()].copy()
    if exclude_zeros:
        df = df[df[attribute_column] != 0].copy()

     # Shapiro-Wilk test
    stat, p_value = shapiro(df[attribute_column])
    print(f"The p value is: {p_value}")
    if p_value > 0.05:
        print(f"The {attribute_column} does not follow a normal distribution.")
        is_normal = False
    elif p_value <= 0.05 and p_value > 0:
        print(f"The {attribute_column} follows a normal distribution.")
        is_normal = True
    elif p_value == 0:
        print("Cannot define normal distribution.")
        is_normal = False

    if plot:
        # Calculate the minimum & maximum value of the attribute column
        min_value = df[attribute_column].min()
        max_value = df[attribute_column].max()

        # Create subplots
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=False, sharey=False)

        # Create QQ plot comparing attribute values to a theoretical normal distribution
        sm.qqplot(df[attribute_column], line="45", fit=True, ax=ax[0])
        ax[0].plot([min_value, max_value], [min_value, max_value], color="gray", linestyle="--")
        ax[0].set_xlabel("Theoretical Quantiles")
        ax[0].set_ylabel("Sample Quantiles")
        ax[0].set_title(f"QQ Plot of '{attribute_column}'")

        # Overlay a distribution plot (histogram or KDE) on the second subplot
        sns.histplot(df[attribute_column], kde=True, ax=ax[1], color="skyblue", alpha=0.5)
        ax[1].set_xlabel(attribute_column)
        ax[1].set_ylabel("Density")
        ax[1].set_title(f"Distribution of '{attribute_column}'")
        plt.show()

    return is_normal


def calculate_spatial_weights_by_dist(gdf, search_distance_in_ft, standardization="R"):
    """Calculate spatial weights matrix based on distance threshold.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    search_distance_in_ft : float
        The distance threshold in feet for defining spatial neighbors.
    standardization : str, optional
        The type of spatial weights standardization ('R' or 'B'). Defaults to 'R'.

    Returns
    -------
    weights.DistanceBand
        The spatial weights matrix.

    """
    # Extract coordinates from GeoDataFrame
    gdf = gdf.copy()
    gdf = geom_l.convert_geometries_to_points(gdf)
    coordinates = np.column_stack((gdf.geometry.x, gdf.geometry.y))

    # Define neighbors based on distance threshold (e.g., distance less than a certain threshold)
    search_distance = geom_l.convert_value_in_ft_to_df_units(gdf, search_distance_in_ft)

    spatial_weights_matrix = DistanceBand(coordinates, threshold=search_distance, binary=False, alpha=-1.0,
                                          silence_warnings=True)
    spatial_weights_matrix.transform = standardization
    spatial_weights_matrix = fill_diagonal(spatial_weights_matrix, 0)
    return spatial_weights_matrix


def calculate_spatial_weights_for_polygons(gdf, standardization="R"):
    """Calculate spatial weights matrix based on polygon contiguity.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing polygon geometries.
    standardization : str, optional
        The type of spatial weights standardization ('R' or 'B'). Defaults to 'R'.

    Returns
    -------
    weights.Queen
        The spatial weights matrix based on Queen contiguity.

    """
    gdf = gdf.copy()
    spatial_weights_matrix = Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
    spatial_weights_matrix.transform = standardization
    spatial_weights_matrix = fill_diagonal(spatial_weights_matrix, 0)
    return spatial_weights_matrix


def get_spatial_weights(gdf, search_distance_in_ft, standardization):
    """Get spatial weights matrix based on geometry type.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    search_distance_in_ft : float
        The search distance in feet (used for point-based analyses).
    standardization : str
        The type of spatial weights standardization ('R' or 'B').

    Returns
    -------
    weights object
        The spatial weights matrix (Queen for polygons, DistanceBand for points).

    """
    if "polygon" in gdf.iloc[0]['geometry'].geom_type:
        print("Calculate spatial weights for polygons")
        if standardization.upper() == "B":
            print("The standardization is 'B'. Consider using 'R' for polygons.")
        spatial_weights_matrix = calculate_spatial_weights_for_polygons(gdf, standardization)
    else:
        print("Calculate spatial weights for points")
        spatial_weights_matrix = calculate_spatial_weights_by_dist(gdf, search_distance_in_ft, standardization)
    return spatial_weights_matrix


def calculate_global_morans_i(gdf, attribute_column, search_distance_in_ft, standardization="R",
                              significance_level=0.05, permutations=999, plot_results=True):
    """Calculate global Moran's I statistic for spatial autocorrelation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    attribute_column : str
        The column name containing the attribute values to analyze.
    search_distance_in_ft : float
        The search distance in feet for defining spatial neighbors.
    standardization : str, optional
        The type of spatial weights standardization ('R' or 'B'). Defaults to 'R'.
    significance_level : float, optional
        The significance level for statistical testing. Defaults to 0.05.
    permutations : int, optional
        The number of permutations for significance testing. Defaults to 999.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing (Moran's I, z-score, p-value).

    """
    cm_l.check_needed_df_columns(gdf, [attribute_column, 'geometry'])
    assert standardization.upper() in ['R', 'B'], "The <standardization> variable must be either 'R' or 'B'"
    spatial_weights_matrix = get_spatial_weights(gdf, search_distance_in_ft, standardization)
    observed_i = Moran(gdf[attribute_column], spatial_weights_matrix, permutations=permutations)
    moran_i = observed_i.I
    if permutations > 0:
        moran_z = observed_i.z_sim
        moran_p = observed_i.p_sim
    else:
        moran_z = observed_i.z_norm
        moran_p = observed_i.p_norm

    if plot_results:
        print(f"Number of observations: {spatial_weights_matrix.n}")
        print(f"Average number of neighbors: {spatial_weights_matrix.mean_neighbors}")
        print(f"Min number of neighbors: {spatial_weights_matrix.min_neighbors}")
        print(f"Max number of neighbors: {spatial_weights_matrix.max_neighbors}")
        print(f"Islands (observations disconnected): {spatial_weights_matrix.islands}")
        p = sns.histplot(pd.Series(spatial_weights_matrix.cardinalities), bins=10)
        p.set(xlabel="Number of Neigbors", ylabel="Count of geometries")

        print(f"Observed Moran's I: {moran_i}")
        print(f"Observed z-score: {moran_z}")
        print(f"Observed p-value: {moran_p}")
        if moran_p < significance_level:
            cm_l.print_formatted_txt(f"The {attribute_column} is clustered!", "RESULTS")
        else:
            cm_l.print_formatted_txt(f"The {attribute_column} is NOT clustered!", "RESULTS")

        display(moran_scatterplot(observed_i))
    return moran_i, moran_z, moran_p


def calculate_z_value(gdf, attribute_column, dist, significance_level, z_for_99_confidence):
    """Calculate z-value statistics for a given distance threshold.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    attribute_column : str
        The column name to analyze.
    dist : float
        The distance threshold in feet.
    significance_level : float
        The significance level for statistical testing.
    z_for_99_confidence : float
        The z-score threshold for 99% confidence.

    Returns
    -------
    dict
        A dictionary containing z-value statistics including distance, min/max/median z-scores,
        and neighbor counts.

    """
    df_result, col_zscore, _ = spatial_outliers(gdf, attribute_column=attribute_column, analysis="local",
                                                search_distance_in_ft=dist, outlier_z_threshold=z_for_99_confidence,
                                                significance_level=significance_level, exclude_zeros=True,
                                                plot_results=False, verbose=False)

    min_Q1, max_Q3 = cm_l.get_IQR_outlier_limits(df_result, field_name=col_zscore, quantiles=[0.25, 0.75],
                                                          max_range=1.5, verbose=False)
    df_result = df_result[(df_result[col_zscore] >= min_Q1) & (df_result[col_zscore] <= max_Q3)].copy()
    df_result[col_zscore] = abs(df_result[col_zscore])

    return {'distance': int(dist),
            'min_z': round(df_result[col_zscore].min(), 2), 'max_z': round(df_result[col_zscore].max(), 2),
            'median_z': round(df_result[col_zscore].median(), 2),
            'min_neighbors': int(df_result['neighbors_count'].min()),
            'max_neighbors': int(df_result['neighbors_count'].max()),
            'median_neighbors': int(df_result['neighbors_count'].median())}


def process_dist(dist, gdf, attribute_column, significance_level, z_for_99_confidence):
    """Process a single distance for parallel z-value calculation.

    Parameters
    ----------
    dist : float
        The distance threshold in feet.
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    attribute_column : str
        The column name to analyze.
    significance_level : float
        The significance level for statistical testing.
    z_for_99_confidence : float
        The z-score threshold for 99% confidence.

    Returns
    -------
    dict
        Z-value statistics for the given distance.

    """
    return calculate_z_value(gdf, attribute_column, dist, significance_level, z_for_99_confidence)


def calculate_optimum_cluster_size(gdf, attribute_column, min_dist_in_ft, max_dist_in_ft, step_dist_in_ft,
                                   significance_level=0.05, min_neighbors=2, plot_results=True, num_processes=2):
    """Calculate optimal cluster size based on z-value analysis across distance ranges.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    attribute_column : str
        The column name to analyze.
    min_dist_in_ft : float
        The minimum distance in feet.
    max_dist_in_ft : float
        The maximum distance in feet.
    step_dist_in_ft : float
        The step size in feet for distance intervals.
    significance_level : float, optional
        The significance level for statistical testing. Defaults to 0.05.
    min_neighbors : int, optional
        The minimum number of neighbors required. Defaults to 2.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.
    num_processes : int, optional
        The number of processes for parallel computation. Defaults to 2.

    Returns
    -------
    tuple
        A tuple containing (first_optimum_distance, global_optimum_distance, results_dataframe).

    """
    warnings.filterwarnings('ignore')
    if min_dist_in_ft >= max_dist_in_ft:
        raise Exception("Input distance range values are not correct.")

    z_for_99_confidence = list(gb_l.z_scores_dict.items())[-1][1][0]
    results_list = []
    search_dist = list(np.arange(min_dist_in_ft, max_dist_in_ft + step_dist_in_ft, step_dist_in_ft))

    # Prepare arguments for process_dist
    args = [(dist, gdf, attribute_column, significance_level, z_for_99_confidence) for dist in search_dist]

    # Execute the loop using parallel processing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        if num_processes is None:
            num_processes = 1
        for result in pool.starmap(process_dist, args, chunksize=num_processes):
            results_list.append(result)

    # Create DataFrame from the list of dictionaries
    df_results = pd.DataFrame(results_list)
    df_results.sort_values(by=['distance'], ascending=False)
    display(df_results)

    df_with_neighbors = df_results[df_results['min_neighbors'] >= min_neighbors].copy()
    if df_with_neighbors.empty:
        raise Exception(f"There aren't any results with min_neighbors >= {min_neighbors}!")

    first_max_z = np.nan
    df_with_neighbors = df_with_neighbors[df_with_neighbors['max_z'] >= z_for_99_confidence].sort_values(by="distance",
                                                                                                         ascending=True)
    optimum_dist1, optimum_dist2 = 0, 0
    global_max_z = np.nan
    if not df_with_neighbors.empty:
        optimum_dist_row = df_with_neighbors.iloc[0]
        first_max_z = optimum_dist_row['max_z']
        optimum_dist1 = optimum_dist_row['distance']
        optimum_dist1 = geom_l.convert_value_in_df_units_from_ft(gdf, optimum_dist1)
        print(f"\nFirst maximum z-value: {abs(first_max_z)} at distance {optimum_dist1}")

        optimum_dist_row = df_with_neighbors.loc[df_with_neighbors['max_z'].idxmax()]
        global_max_z = optimum_dist_row['max_z']
        optimum_dist2 = optimum_dist_row['distance']
        optimum_dist2 = geom_l.convert_value_in_df_units_from_ft(gdf, optimum_dist2)
        print(f"\nTotal maximum z-value: {abs(global_max_z)} at distance {optimum_dist2}")

    if plot_results:
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_results, x="distance", y="max_z", color="green", markers=False, dashes=True)

        for z_class, (min_z, max_z, color) in gb_l.z_scores_dict.items():
            z_data = df_results[(df_results['max_z'] >= min_z) & (df_results['max_z'] < max_z)]
            if not z_data.empty:
                plt.scatter(z_data['distance'], z_data['max_z'], color=color, label=z_class)

        plt.xlabel("Distance in source units")
        plt.ylabel("Z score")
        plt.title("Z score vs Distance")
        plt.grid(True)
        plt.legend()
        plt.show()

    return optimum_dist1, optimum_dist2, df_results


def high_low_colormap(value):
    """Map high-low classification values to colors for visualization.

    Parameters
    ----------
    value : str
        The high-low classification value (e.g., 'HH cluster', 'LL cluster', 'HL outlier', 'LH outlier').

    Returns
    -------
    str
        A hex color code corresponding to the classification.

    """
    if "HH " in value:
        return "#D7191C"
    elif "HL " in value:
        return "#FDAE61"
    elif "LH " in value:
        return "#ABD9E9"
    elif "LL " in value:
        return "#2C7BB6"
    else:
        return "#D3D3D3"


def hotspot_analysis(gdf, attribute_column, outlier_z_threshold, standardization="R", significance_level=0.05,
                     search_distance_in_ft=1000, exclude_zeros=True,
                     permutations=999, num_processes=None, seed_number=9999, plot_results=True):
    """Perform local hotspot analysis using Local Moran's I statistic.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    attribute_column : str
        The column name containing values for analysis.
    outlier_z_threshold : float
        The z-score threshold for outlier classification.
    standardization : str, optional
        The type of spatial weights standardization ('R' or 'B'). Defaults to 'R'.
    significance_level : float, optional
        The significance level for identifying significant hotspots. Defaults to 0.05.
    search_distance_in_ft : float, optional
        The search distance in feet for defining spatial relationships. Defaults to 1000.
    exclude_zeros : bool, optional
        Whether to exclude zero values. Defaults to True.
    permutations : int, optional
        The number of permutations for significance testing. Defaults to 999.
    num_processes : int, optional
        The number of processes for parallel computation. Defaults to None.
    seed_number : int, optional
        The random seed for reproducibility. Defaults to 9999.
    plot_results : bool, optional
        Whether to plot the results. Defaults to True.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame with hotspot analysis results.

    """
    cm_l.check_needed_df_columns(gdf, [attribute_column])
    if not is_numeric_dtype(gdf[attribute_column]):
        raise ValueError(f"The {attribute_column} column is not numeric.")

    gdf = gdf.copy()
    col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow, col_mean, col_stdev, col_outlier = define_hotspot_columns(
        "hotspot")
    spatial_weights_matrix = get_spatial_weights(gdf, search_distance_in_ft, standardization)
    num_processes = -1 if num_processes is None or num_processes == 0 else num_processes
    local_moran = Moran_Local(gdf[attribute_column], spatial_weights_matrix, permutations=permutations,
                              n_jobs=num_processes, seed=seed_number)

    # Dict to map local moran's classification codes
    local_moran_classification = {1: 'HH cluster', 2: 'LH outlier', 3: 'LL cluster', 4: 'HL outlier'}

    # Add results to GeoDataFrame
    gdf[col_hotspot_class] = local_moran.q
    if permutations > 0:
        gdf[col_zscore] = local_moran.z_sim
        gdf[col_pvalue] = local_moran.p_sim
    else:
        gdf[col_zscore] = local_moran.z_norm
        gdf[col_pvalue] = local_moran.p_norm
    gdf[col_zscore] = gdf[col_zscore].fillna(0)
    gdf[col_pvalue] = gdf[col_pvalue].fillna(0)

    gdf[col_highlow] = gdf[col_hotspot_class].map(local_moran_classification)
    gdf[col_significant] = np.where(gdf[col_pvalue] < significance_level, True, False)
    gdf.loc[~gdf[col_significant], col_highlow] = "Not Significant"

    gdf[col_outlier] = abs(gdf[col_zscore]) > outlier_z_threshold
    gdf[col_outlier] = gdf[col_outlier].fillna(False)
    gdf[col_outlier] = gdf[col_outlier].astype(str).str.lower()
    gdf[col_outlier] = gdf[col_outlier].map({"true": "yes", "false": "no"})

    if exclude_zeros:
        gdf.loc[gdf[attribute_column] == 0, col_outlier] = "null"

    if plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
        moran_scatterplot(local_moran, p=significance_level, ax=ax)
        plt.text(1.95, 1, "HH")
        plt.text(1.95, -1, "HL")
        plt.text(-1.5, 1, "LH")
        plt.text(-1.5, -1, "LL")
        plt.show()

        # Plotting Local Moran's I classification map of pop_count column
        display(gdf.explore(
            tiles="cartodbpositron",
            column=col_highlow,
            height="100%",
            width="100%",
            cmap=[high_low_colormap(x) for x in sorted(list(gdf[col_highlow].unique()))],
            style_kwds={
                'stroke': True,
                'edgecolor': 'k',
                'linewidth': 0.03,
                'radius': 3,
                'fillOpacity': 1
            },
        ))

    return gdf


def check_spatial_trends(df, attribute_column, exclude_zeros=True):
    """Check if spatial data exhibits directional trends.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        The input GeoDataFrame containing spatial data.
    attribute_column : str
        The column name to analyze for trends.
    exclude_zeros : bool, optional
        Whether to exclude zero values from analysis. Defaults to True.

    Returns
    -------
    bool
        True if data exhibits spatial trends, False otherwise.

    """

    cm_l.check_needed_df_columns(df, [attribute_column])
    if not is_numeric_dtype(df[attribute_column]):
        raise ValueError(f"The {attribute_column} column is not numeric.")

    df = df.copy()
    df = geom_l.convert_geometries_to_points(df)

    df = df[df[attribute_column].notna()].copy()
    if exclude_zeros:
        df = df[df[attribute_column] != 0].copy()

    x = df.geometry.x.values.reshape(-1, 1)
    y = df.geometry.y.values.reshape(-1, 1)
    values = df[attribute_column].values

    # Fit linear regression models
    model_x = LinearRegression().fit(x, values)
    model_y = LinearRegression().fit(y, values)

    # Plotting the scatter plot with regression lines
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(x, values)
    plt.plot(x, model_x.predict(x), color='red')
    plt.title('Trend in X direction')

    plt.subplot(1, 2, 2)
    plt.scatter(y, values)
    plt.plot(y, model_y.predict(y), color='red')
    plt.title('Trend in Y direction')

    plt.tight_layout()
    plt.show()

    # Check if slopes are significantly different from 0
    trend_x = np.abs(model_x.coef_[0]) > 1e-3
    trend_y = np.abs(model_y.coef_[0]) > 1e-3

    return trend_x or trend_y


def spatial_extent_similarity(gdf1, gdf2, buffer_radius):
    """Calculate spatial extent similarity between two GeoDataFrames.

    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        The first GeoDataFrame.
    gdf2 : geopandas.GeoDataFrame
        The second GeoDataFrame.
    buffer_radius : float
        The buffer radius to apply to the geometries.

    Returns
    -------
    None
        Prints the overlap percentage and displays the polygon difference.

    """
    gdf1 = gdf1.copy()
    gdf2 = gdf2.copy()
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)

    buffer_dist = geom_l.convert_value_in_df_units_from_ft(gdf1, buffer_radius)

    outer_polygon1 = gdf1.copy()
    outer_polygon1['geometry'] = outer_polygon1['geometry'].buffer(buffer_dist)
    outer_polygon1 = outer_polygon1.geometry.unary_union.convex_hull

    outer_polygon2 = gdf2.copy()
    outer_polygon2['geometry'] = outer_polygon2['geometry'].buffer(buffer_dist)
    outer_polygon2 = outer_polygon2.geometry.unary_union.convex_hull

    pol_diference = outer_polygon2.difference(outer_polygon1)
    pol_intersection = outer_polygon2.intersection(outer_polygon1)
    pol_union = outer_polygon2.union(outer_polygon1)

    overlap = round(100*pol_intersection.area/pol_union.area, 2)
    print(f"Overlap percentage: {overlap}%")
    display(pol_diference)


def spatial_jaccard_similarity(gdf1, gdf2, buffer_radius):
    """Calculate Jaccard similarity index between two GeoDataFrames.

    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        The first GeoDataFrame.
    gdf2 : geopandas.GeoDataFrame
        The second GeoDataFrame.
    buffer_radius : float
        The buffer radius to apply to the geometries.

    Returns
    -------
    float
        The Jaccard similarity index.

    """
    gdf1 = gdf1.copy()
    gdf2 = gdf2.copy()
    cm_l.reset_index(gdf1, gdf2)
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)

    buffer_dist = geom_l.convert_value_in_df_units_from_ft(gdf1, buffer_radius)
    gdf1['geometry'] = gdf1['geometry'].buffer(buffer_dist)

    intersection_count = len(gdf1.intersects(gdf2))
    union_count = len(gdf1) + len(gdf2) - intersection_count
    jaccard_index = intersection_count / union_count

    if jaccard_index == 0:
        print("The datasets have no spatial overlap.")
    elif jaccard_index == 1:
        print("The datasets are identical in terms of spatial extent.")
    else:
        print(f"The Jaccard similarity index is {jaccard_index:.2f}.")

    return jaccard_index


def plot_two_distributions(values1, values2, compare_col, color1="blue", color2="red", bins=20, figsize=(15, 6)):
    """Plot two distributions on the same histogram for comparison.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    compare_col : str
        The label for the values.
    color1 : str, optional
        The color for the first distribution. Defaults to 'blue'.
    color2 : str, optional
        The color for the second distribution. Defaults to 'red'.
    bins : int, optional
        The number of bins for the histogram. Defaults to 20.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 6).

    Returns
    -------
    None
        Displays the histogram plot.

    """
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=figsize)
    axs[0].hist(values1, bins=bins, alpha=0.5, label=compare_col, color=color1)
    axs[1].hist(values2, bins=bins, alpha=0.5, label=compare_col, color=color2)

    for ax in axs:
        ax.set_xlabel(compare_col)
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.suptitle(f"Compare distributions of {compare_col} between two datasets")
    plt.show()


def compare_distributions_ks(values1, values2, p_value_threshold=0.05):
    """Compare two distributions using the Kolmogorov-Smirnov test.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """

    sorted_values1 = np.sort(values1)
    sorted_values2 = np.sort(values2)

    ks_statistic, p_value = ks_2samp(sorted_values1, sorted_values2)
    print(f"\nks_statistic: {ks_statistic}")

    if p_value < p_value_threshold:
        print(f"The distributions are statistically different (p_value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The distributions are statistically similar (p_value: {p_value}>={p_value_threshold}).")


def compare_distributions_minkowski(values1, values2, p_value_threshold=0.05):
    """Compare two distributions using Minkowski distance and Mann-Whitney U test.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the distance and test results.

    """
    sorted_values1 = np.sort(values1)
    sorted_values2 = np.sort(values2)

    distance = minkowski(sorted_values1, sorted_values2, p=2)  # with p=2 for Euclidean distance
    # perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(sorted_values1, sorted_values2)

    print(f"\nThe Minkowski distance between the two lists is {distance:.2f}.")
    print("Mann-Whitney U test: Statistics=%.3f, p=%.3f" % (stat, p_value))

    if p_value < p_value_threshold:
        print(f"The distributions are statistically different (p_value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The distributions are statistically similar (p_value: {p_value}>={p_value_threshold}).")


def compare_distributions_ttest(values1, values2, p_value_threshold=0.05):
    """Compare two distributions using the two-sample t-test.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    sorted_values1 = np.sort(values1)
    sorted_values2 = np.sort(values2)

    t_statistic, p_value = stats.ttest_ind(sorted_values1, sorted_values2)
    print(f"\nT-statistic: {t_statistic}, P-value: {p_value}")

    if p_value < p_value_threshold:
        print(f"The distributions are statistically different (p_value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The distributions are statistically similar (p_value: {p_value}>={p_value_threshold}).")


def compare_distributions_expected(values1, values2, p_value_threshold=0.05):
    """Compare observed distribution against expected using Chi-Square Goodness of Fit test.

    Parameters
    ----------
    values1 : array-like
        The expected values.
    values2 : array-like
        The observed values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    expected = np.sort(values1)
    observed = np.sort(values2)

    # Perform Chi-Square Goodness of Fit test
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

    print("\nResults of the Chi-Square Goodness of Fit test with the expected:")
    print(f"Chi-square statistic: {chi2_stat}, P-value: {p_value}")

    # Decision based on p-value
    if p_value < p_value_threshold:
        print("Reject the null hypothesis: The observed distribution does not fit the expected distribution.")
    else:
        print("Fail to reject the null hypothesis: The observed distribution fits the expected distribution.")


def compare_distributions_after_event(values1, values2, p_value_threshold=0.05):
    """Compare distributions before and after an event using Wilcoxon signed-rank test.

    Parameters
    ----------
    values1 : array-like
        The values before the event.
    values2 : array-like
        The values after the event.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.

    Returns
    -------
    None
        Prints the test results.

    """
    before = np.sort(values1)
    after = np.sort(values2)

    # Perform Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(before, after)

    # Print results
    print("\nResults of the Wilcoxon Signed-Rank Test:")
    print(f"Statistic: {statistic}, P-value: {p_value}")

    # Decision based on p-value
    if p_value < p_value_threshold:
        print("Reject the null hypothesis: There is a significant difference between the before and after lists.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference between the before and after lists.")


def compare_two_distribution(values1, values2, compare_col, p_value_threshold=0.05, plot=True, figsize=(20, 8)):
    """Compare two distributions using multiple statistical methods.

    Parameters
    ----------
    values1 : array-like
        The first set of values.
    values2 : array-like
        The second set of values.
    compare_col : str
        The label for the values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.
    plot : bool, optional
        Whether to plot the distributions. Defaults to True.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (20, 8).

    Returns
    -------
    None
        Prints statistical comparison results.

    """
    compare_distributions_ks(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_minkowski(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_ttest(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_expected(values1, values2, p_value_threshold=p_value_threshold)
    compare_distributions_after_event(values1, values2, p_value_threshold=p_value_threshold)

    print(f"\nFirst  list: mean: {round(np.mean(values1),3)}, median: {round(np.median(values1),3)}, sdev: {round(np.std(values1),3)}, var: {round(np.var(values1),3)}")
    print(f"Second list: mean: {round(np.mean(values2),3)}, median: {round(np.median(values2),3)}, sdev: {round(np.std(values2),3)}, var: {round(np.var(values2),3)}")

    if plot:
        plot_two_distributions(values1, values2, compare_col, figsize=figsize)

def check_normality_distribution(values, label, p_value_threshold=0.05, exclude_zeros=False, plot=True, bins=50, color="skyblue", figsize=(20, 8)):
    """Check normality of distribution using multiple statistical tests.

    Parameters
    ----------
    values : array-like
        The set of values to test.
    label : str
        The label for the values.
    p_value_threshold : float, optional
        The threshold for statistical significance. Defaults to 0.05.
    exclude_zeros : bool, optional
        Whether to exclude zero values. Defaults to False.
    plot : bool, optional
        Whether to create diagnostic plots. Defaults to True.
    bins : int, optional
        The number of bins for histograms. Defaults to 50.
    color : str, optional
        The color for histogram. Defaults to 'skyblue'.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (20, 8).

    Returns
    -------
    None
        Prints normality test results.

    """

    values = [x for x in values if not np.isnan(x)]
    if exclude_zeros:
        values = [x for x in values if x != 0]

    # Perform Shapiro-Wilk test
    stat, p_value = shapiro(values)
    print(f"\nShapiro-Wilk test: Statistics={stat:.3f}, p={p_value:.3f}")
    # Decision based on p-value
    if p_value < p_value_threshold:
        print(f"The {label} does not follow a normal distribution (p-value: {p_value}<{p_value_threshold}).")
    else:
        print(f"The {label} follows a normal distribution (p-value: {p_value}>={p_value_threshold}).")

    # Perform D'Agostino's K^2 test
    stat, p_value = normaltest(values)
    print(f"\nD'Agostino's K^2 test: Statistics={stat:.3f}, p={p_value:.3f}")
    if p_value > p_value_threshold:
        print("The data is likely normally distributed (fail to reject H0)")
    else:
        print("The data is likely not normally distributed (reject H0)")

    stat, p_value = jarque_bera(values)
    print(f"\nJarque-Bera test statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value > p_value_threshold:
        print("The data is likely normally distributed (fail to reject H0)")
    else:
        print("The data is likely not normally distributed (reject H0)")

    # Perform Anderson-Darling test
    result = anderson(values)
    print(f"\nAnderson-Darling test statistic: {result.statistic:.4f}")
    print("Critical values:", result.critical_values)
    print("Significance levels:", result.significance_level)
    # Interpret the result
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f"At {sl}% significance level, the data is normally distributed (fail to reject H0)")
        else:
            print(f"At {sl}% significance level, the data is not normally distributed (reject H0)")

    skewness, kurt = skew(values), kurtosis(values)
    print(f"\nSkewness: {skewness:.4f}, Kurtosis: {kurt:.4f}")
    if abs(skewness) < 0.5 and abs(kurt) < 0.5:
        print("The data is approximately normally distributed based on skewness and kurtosis.")
    else:
        print("The data may not be normally distributed based on skewness and kurtosis.")

    _ = qq_plot_with_distribution(values, label, exclude_zeros=exclude_zeros, plot=plot, figsize=figsize)


def check_distribution_of_transformed_values(values, figsize=(15, 5)):
    """Analyze and compare transformed distributions (log and Box-Cox).

    Parameters
    ----------
    values : array-like
        The set of values to transform and analyze.
    figsize : tuple, optional
        The figure size as (width, height). Defaults to (15, 5).

    Returns
    -------
    None
        Displays transformation results and prints normality test results.

    """
    # Shift the data to make all values positive
    min_value = np.min(values)
    if min_value <= 0:
        shift = abs(min_value) + 1
        print(f"Negative values are detected: the data will be shifted (shift value: {shift}).")
    else:
        shift = 0

    shifted_values = values + shift
    # Apply log transformation
    log_values = np.log(shifted_values)

    # Apply Box-Cox transformation
    boxcox_values, lambda_param = boxcox(shifted_values)

    # Compare distributions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1.hist(values, bins=30, density=True, alpha=0.7)
    ax1.set_title("Original Data")

    ax2.hist(log_values, bins=30, density=True, alpha=0.7)
    ax2.set_title("Log Transformed")

    ax3.hist(boxcox_values, bins=30, density=True, alpha=0.7)
    ax3.set_title("Box-Cox Transformed")

    plt.tight_layout()
    plt.show()

    # Test normality of transformed data
    _, p_log = shapiro(log_values)
    _, p_boxcox = shapiro(boxcox_values)

    print(f"Log transform Shapiro-Wilk p-value: {p_log:.4f}")
    print(f"Box-Cox transform Shapiro-Wilk p-value: {p_boxcox:.4f}")


def time_sereis_cusum(data, target, threshold, k):
    """Detect mean deviation in time series using CUSUM algorithm.

    Parameters
    ----------
    data : list
        Time series data.
    target : float
        Target mean value.
    threshold : float
        The threshold for change detection.
    k : float
        The sensitivity parameter.

    Returns
    -------
    list
        Cumulative sum values indicating deviation.

    """
    cumulative_sum = [0]
    for i in range(1, len(data)):
        d_t = data[i] - target
        s_t = cumulative_sum[i-1] + d_t
        if s_t > 0:
            s_t = max(0, s_t - k)
        elif s_t < 0:
            s_t = min(0, s_t + k)
        cumulative_sum.append(s_t)
        if abs(s_t) > threshold:
            print("Change detected at time step", i)
    return cumulative_sum


def ruptures_change_point_detection(df, kpi, penalty=7):
    """Detect change points in time series using PELT algorithm.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time series data.
    kpi : str
        The column name to analyze for change points.
    penalty : float, optional
        The penalty parameter for the PELT algorithm. Defaults to 7.

    Returns
    -------
    list
        Indices of detected change points.

    """
    import ruptures as rpt
    # Assuming df has Date column
    df['Date'] = pd.to_datetime(df.index).date

    # Fit the ruptures algorithm
    algo = rpt.Pelt(model="rbf").fit(df[kpi].values)
    result = algo.predict(pen=penalty)

    return result
