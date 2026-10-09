"""Exploratory neighborhood-distance selection from local z-score profiles."""

import multiprocessing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

import shared.constants as gb_l
import shared.geometry as geom_l
import shared.tabular as cm_l

from .hotspots import spatial_outliers


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

    # Trim extreme signed scores before taking magnitudes, reducing the influence of isolated peaks on distance ranking.
    min_Q1, max_Q3 = cm_l.get_IQR_outlier_limits(df_result, field_name=col_zscore, quantiles=[0.25, 0.75],
                                                          max_range=1.5, verbose=False)
    df_result = df_result[(df_result[col_zscore] >= min_Q1) & (df_result[col_zscore] <= max_Q3)].copy()
    df_result[col_zscore] = abs(df_result[col_zscore])

    # Neighbor extrema describe the trimmed cohort, not necessarily every row in the original dataset.
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
    # Keep the pool worker at module scope so Windows spawn can import and pickle it.
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
        raise ValueError("Error: Input distance range values are not correct.")

    z_for_99_confidence = list(gb_l.z_scores_dict.items())[-1][1][0]
    results_list = []
    # Adding one step includes an aligned endpoint but can overshoot max_dist_in_ft for an unaligned range.
    search_dist = list(np.arange(min_dist_in_ft, max_dist_in_ft + step_dist_in_ft, step_dist_in_ft))

    # Prepare arguments for process_dist
    args = [(dist, gdf, attribute_column, significance_level, z_for_99_confidence) for dist in search_dist]

    # Execute the loop using parallel processing
    # The pool size follows CPU count; the num_processes argument below actually controls starmap chunk size.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        if num_processes is None:
            num_processes = 1
        results_list = list(pool.starmap(process_dist, args, chunksize=num_processes))

    # Create DataFrame from the list of dictionaries
    df_results = pd.DataFrame(results_list)
    df_results.sort_values(by=['distance'], ascending=False)
    display(df_results)

    # Require the minimum neighbor count across the retained cohort, not merely an adequate average count.
    df_with_neighbors = df_results[df_results['min_neighbors'] >= min_neighbors].copy()
    if df_with_neighbors.empty:
        raise ValueError(f"Error: There aren't any results with min_neighbors >= {min_neighbors}!")

    first_max_z = np.nan
    df_with_neighbors = df_with_neighbors[df_with_neighbors['max_z'] >= z_for_99_confidence].sort_values(by="distance",
                                                                                                         ascending=True)
    optimum_dist1, optimum_dist2 = 0, 0
    global_max_z = np.nan
    if not df_with_neighbors.empty:
        # The first optimum is the smallest qualifying radius, not necessarily a local maximum of the z-score curve.
        optimum_dist_row = df_with_neighbors.iloc[0]
        first_max_z = optimum_dist_row['max_z']
        optimum_dist1 = optimum_dist_row['distance']
        # The search distances are already in feet, but this legacy helper converts as if they were native CRS units.
        optimum_dist1 = geom_l.convert_value_in_df_units_from_ft(gdf, optimum_dist1)
        print(f"\nFirst maximum z-value: {abs(first_max_z)} at distance {optimum_dist1}")

        # The second optimum chooses the strongest retained absolute score among the qualifying radii.
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
