"""Attribute outliers, Moran autocorrelation, and hotspot visualization."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from esda.moran import Moran, Moran_Local
from IPython.display import display
from matplotlib.patches import Patch
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
from splot.esda import moran_scatterplot

import shared.constants as gb_l
import shared.geometry as geom_l
import shared.tabular as cm_l
from shared.assertions import expect_true

from .weights import get_spatial_weights


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
    # Prefix every output with the analysis type so local, global, and Moran results can coexist.
    # The return order is a shared contract used by both computation and plotting helpers.
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
    # Dispatch from the first geometry; callers should supply a nonempty, homogeneous geometry column.
    geometry_type = gdf.geometry.geom_type.iloc[0]

    if "point" in geometry_type.lower() or "line" in geometry_type.lower() or "polygon" in geometry_type.lower():
        color_mapping = {}
        gdf['class_color'] = "#9C9C9C"
        for z_class, (min_z, max_z, color) in gb_l.z_scores_dict.items():
            color_mapping[z_class] = color
            gdf.loc[gdf[col_hotspot_class] == z_class, 'class_color'] = color

        # This second assignment resets the hotspot colors to gray before plotting; retained legacy behavior.
        gdf['class_color'] = "#9C9C9C"
        for outlier, color in gb_l.outliers_dict.items():
            gdf.loc[gdf[col_outlier] == outlier, 'outlier_color'] = color

        figsize_w, figsize_h = figsize
        _fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figsize_w, figsize_h - 4), sharex=False, sharey=True)

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

            # Polygon legends need filled patches, whereas line and point legends use line artists.
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
    col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow, col_mean, _col_stdev, _col_outlier = \
    define_hotspot_columns(analysis)

    # Identify statistically significant hotspots
    gdf[col_significant] = np.where(gdf[col_pvalue] < significance_level, True, False)

    # Z-score bins use an open lower boundary and a closed upper boundary.
    # The bin label itself is not gated by p-value; significance is applied to the high/low labels below.
    gdf[col_hotspot_class] = "Not Significant"
    for z_class, (min_z, max_z, color) in gb_l.z_scores_dict.items():
        gdf.loc[((gdf[col_zscore] > min_z) & (gdf[col_zscore] <= max_z)), col_hotspot_class] = z_class

    # Classify observations into 'High-High', 'Low-Low', 'High-Low', 'Low-High' categories
    # These labels compare the observation with its supplied local/global mean; this is not Moran quadrant coding.
    # Values exactly equal to the mean keep the default label because both comparisons are strict.
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
    # Honor supplied per-row neighborhood moments; compute whole-column moments only when absent.
    if col_mean not in list(df.columns):
        df[col_mean] = df[attribute_column].mean()
    if col_stdev not in list(df.columns):
        df[col_stdev] = df[attribute_column].std()

    # Only positive standard deviations support division; zero or missing spread leaves the score unassigned.
    cond = (df[col_stdev] > 0)
    df.loc[cond, col_zscore] = (df[cond][attribute_column] - df[cond][col_mean]) / df[cond][col_stdev]
    # Two normal tails turn the absolute standardized deviation into an unadjusted two-sided p-value.
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
        raise ValueError(f"Error: The {attribute_column} column is not numeric.")
    expect_true(analysis in ['local', 'global'], "The variable <analysis> can be either 'local' or 'global'")

    gdf = gdf.copy()
    col_id = "tmp_id"
    col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow, col_mean, col_stdev, col_outlier = \
        define_hotspot_columns(analysis)

    cols_to_drop = ['neighbors_count', col_significant, col_pvalue, col_zscore, col_hotspot_class, col_highlow,
                    col_mean, col_stdev, col_outlier]
    for col in cols_to_drop:
        if col in list(gdf.columns):
            gdf.drop(columns=[col], inplace=True)

    # Use the original index to join results back; a unique input index is needed for one-to-one row recovery.
    gdf[col_id] = gdf.index
    df = gdf.copy()

    if exclude_zeros:
        df[attribute_column] = df[attribute_column].fillna(0)
        # Despite the option name, this retains only positive observations, excluding negative values as well as zero.
        df = df[df[attribute_column] > 0].copy()

    if analysis == "local":
        # Buffer geometries
        max_buffer = geom_l.convert_value_in_ft_to_df_units(gdf, search_distance_in_ft)
        df_buffer = df.copy()
        df_buffer['geometry'] = df_buffer['geometry'].buffer(max_buffer)
        col_id_left, col_id_right = f"{col_id}_left", f"{col_id}_right"

        # Perform spatial join
        df_join = gpd.sjoin(df_buffer[[col_id, 'geometry']], df[[col_id, attribute_column, 'geometry']])
        # Remove self-matches so local moments describe other features intersecting the focal buffer.
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
        # This inner join drops rows with no neighbors; the final left join restores them to the output.
        # A single neighbor has undefined sample standard deviation and cannot yield a local z-score.
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
    # Legacy placeholders include p=0 for unscored rows; that value must not be read as measured significance.
    gdf = gdf.fillna({'neighbors_count': 0, col_pvalue: 0, col_zscore: 0, col_mean: 0, col_stdev: 0, col_outlier: False})

    # Convert outlier column to 'yes'/'no'/'null' strings
    gdf[col_outlier] = gdf[col_outlier].fillna("null").astype(str).str.lower()
    gdf[col_outlier] = gdf[col_outlier].map({"true": "yes", "false": "no"})

    # Exclude zeros from outlier classification
    if exclude_zeros:
        gdf.loc[gdf[attribute_column] == 0, col_outlier] = "null"

    # Drop temporary ID column
    gdf.drop(columns=[col_id], inplace=True)
    # The returned maximum is signed, even though outlier detection above uses absolute deviations.
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
    expect_true(standardization.upper() in ['R', 'B'], "The <standardization> variable must be either 'R' or 'B'")
    spatial_weights_matrix = get_spatial_weights(gdf, search_distance_in_ft, standardization)
    # Values and weight rows must stay aligned; this helper does not filter missing attributes before fitting.
    observed_i = Moran(gdf[attribute_column], spatial_weights_matrix, permutations=permutations)
    moran_i = observed_i.I
    # Use permutation-reference diagnostics when requested; otherwise report the normal-approximation diagnostics.
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
        # This message tests significance only; the sign of Moran's I is still needed to distinguish clustering from dispersion.
        if moran_p < significance_level:
            cm_l.print_formatted_txt(f"The {attribute_column} is clustered!", "RESULTS")
        else:
            cm_l.print_formatted_txt(f"The {attribute_column} is NOT clustered!", "RESULTS")

        display(moran_scatterplot(observed_i))
    return moran_i, moran_z, moran_p


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
    # The trailing space distinguishes the quadrant prefix while allowing descriptive suffixes in labels.
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
        raise ValueError(f"Error: The {attribute_column} column is not numeric.")

    gdf = gdf.copy()
    (
        col_significant, col_pvalue, col_zscore, col_hotspot_class,
        col_highlow, _col_mean, _col_stdev, col_outlier,
    ) = define_hotspot_columns("hotspot")
    spatial_weights_matrix = get_spatial_weights(gdf, search_distance_in_ft, standardization)
    num_processes = -1 if num_processes is None or num_processes == 0 else num_processes
    # The seed is passed into the permutation estimator so repeated local analyses can reproduce the reference draws.
    local_moran = Moran_Local(gdf[attribute_column], spatial_weights_matrix, permutations=permutations,
                              n_jobs=num_processes, seed=seed_number)

    # Dict to map local moran's classification codes
    # Quadrants describe the observation and its spatial lag: like-valued clusters versus contrasting neighbors.
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
    # A missing p-value is currently replaced with zero and can therefore pass the significance gate below.
    gdf[col_pvalue] = gdf[col_pvalue].fillna(0)

    gdf[col_highlow] = gdf[col_hotspot_class].map(local_moran_classification)
    # Apply the raw per-feature p-value threshold; no multiple-testing correction is performed here.
    gdf[col_significant] = np.where(gdf[col_pvalue] < significance_level, True, False)
    gdf.loc[~gdf[col_significant], col_highlow] = "Not Significant"

    gdf[col_outlier] = abs(gdf[col_zscore]) > outlier_z_threshold
    gdf[col_outlier] = gdf[col_outlier].fillna(False)
    gdf[col_outlier] = gdf[col_outlier].astype(str).str.lower()
    gdf[col_outlier] = gdf[col_outlier].map({"true": "yes", "false": "no"})

    # Zero values participated in Moran estimation; this option only changes their displayed outlier label.
    if exclude_zeros:
        gdf.loc[gdf[attribute_column] == 0, col_outlier] = "null"

    if plot_results:
        _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
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
            cmap=[high_low_colormap(x) for x in sorted(gdf[col_highlow].unique())],
            style_kwds={
                'stroke': True,
                'edgecolor': 'k',
                'linewidth': 0.03,
                'radius': 3,
                'fillOpacity': 1
            },
        ))

    return gdf
