"""Data analysis, plotting, outlier detection, and confidence utilities."""

import math
import os
import statistics
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.ticker import MaxNLocator
from numpy import arange, array
from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.stats import norm
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

import shared.constants as gb_l
import shared.geometry as geom_l

from .core import check_needed_df_columns
from .transforms import get_column_types


def plot_histogram_with_mean_std_lines(df, column_name, ax, x_label=""):
    """Plot histogram with mean std lines."""
    series = df[column_name]
    mean_val = series.mean()
    std_val = series.std()

    sns.histplot(series, kde=True, color="skyblue", bins=100, ax=ax)

    for value, color, label in [
        (mean_val, "red", f"Mean: {mean_val:.2f}"),
        (mean_val - std_val, "green", f"Mean - Std: {mean_val - std_val:.2f}"),
        (mean_val + std_val, "green", f"Mean + Std: {mean_val + std_val:.2f}"),
    ]:
        ax.axvline(value, color=color, linestyle="--", label=label)

    min_val, max_val = series.min(), series.max()
    range_val = max_val - min_val

    if range_val == 0:
        # Handle the case where all values are the same
        ax.set_xticks([min_val])
        ax.set_xticklabels([f"{min_val:.2f}"])
    else:
        step = range_val / 10

        if range_val > 10:
            ticks = np.arange(int(min_val), int(max_val) + 2, max(1, int(step)))
        elif range_val > 0.1:
            ticks = np.arange(round(min_val, 2), round(max_val, 2) + step, round(step, 2))
        else:
            ticks = np.linspace(min_val, max_val, num=11)

        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])

    ax.set_xlabel(x_label)
    ax.legend()


def _show_figure_if_interactive():
    """Avoid noisy display warnings when Matplotlib is configured for a non-interactive backend."""
    backend = plt.get_backend().lower()
    if backend.endswith("agg"):
        return
    plt.show()


def plot_box_plot(df, column_name, ax):
    """Plot box plot."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="vert: bool will be deprecated.*",
            category=PendingDeprecationWarning,
        )
        sns.boxplot(
            x=df[column_name],
            ax=ax,
            color="skyblue",
            orientation="horizontal",
            flierprops={"markerfacecolor": "r", "marker": "D"},
        )
    ax.set_title(f"Distribution of {column_name}")


def plot_stat_numeric(df, column_name, exclude_zeros=True, figsize=(18, 3), save_folder=""):
    """Plot stat numeric."""
    df = df.copy()
    if exclude_zeros:
        df = df[df[column_name] > 0].copy()

    _fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize)
    plot_box_plot(df, column_name, axs[0])
    plot_histogram_with_mean_std_lines(df, column_name, axs[1])

    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    # Show when save_folder is falsy (None or empty string)
    if not save_folder:
        _show_figure_if_interactive()
    else:
        plt.savefig(os.path.join(save_folder, f"{column_name}_barplot.png"))
        plt.close()


def plot_stat_object(df, column_name, show_common=10, palette="pastel", figsize=(30, 6), save_folder=""):
    """Plot stat object."""
    df = df.copy()
    top_common_values = df[column_name].value_counts().head(show_common).reset_index()
    filtered_df = df[df[column_name].isin(top_common_values[column_name])]
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=filtered_df, y=column_name, order=top_common_values[column_name], hue=column_name, palette=palette, legend=False)
    plt.title(f"Distribution of {column_name}")
    annotate_bars(ax)
    plt.tight_layout()
    # Show when save_folder is falsy (None or empty string)
    if not save_folder:
        _show_figure_if_interactive()
    else:
        plt.savefig(os.path.join(save_folder, f"{column_name}_barplot.png"))
        plt.close()


def annotate_bars(ax, precision=0):
    """Annotate bars."""
    for p in ax.patches:
        ax.annotate(
            f"{p.get_width():.{precision}f}",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )


def preprocess_number_column(df, field_to_check, null_values, min_value, max_value, plot_graphs, exclude_zeros_in_plot=True):
    """Preprocess number column."""
    df = df.copy()
    if is_numeric_dtype(df[field_to_check]):
        print(f"{field_to_check} is numeric")
        df[field_to_check] = df[field_to_check].fillna(0).astype(float)
        df.loc[df[field_to_check].isin(null_values), field_to_check] = 0
    else:
        print(f"{field_to_check} is object")
        df.loc[df[field_to_check].isin(null_values), field_to_check] = "0"
        df[field_to_check] = df[field_to_check].fillna("0").astype(str)

        if any(~df[field_to_check].astype(str).str.match(r"^-?\d+(.\d+)?$")):
            display(df[field_to_check].value_counts(dropna=False))
            raise ValueError(f"Error: Check column {field_to_check} values")
        df[field_to_check] = df[field_to_check].astype(float)

    df_nulls = df[df[field_to_check] == 0]
    print(f"\nNull {field_to_check}: {len(df_nulls)} ({round(100 * len(df_nulls) / len(df), 2)}%)")

    cond_invalid = (df[field_to_check] < min_value) | (df[field_to_check] > max_value)
    df_issues = df[(df[field_to_check] != 0) & cond_invalid].copy()
    if len(df_issues) > 0:
        print(f"Invalid {field_to_check} values: {len(df_issues)} (diameter<{min_value} or diameter>{max_value})")
        display(df_issues.head())
        df.loc[cond_invalid, field_to_check] = 0

    if plot_graphs:
        plot_stat_numeric(df, field_to_check, exclude_zeros=exclude_zeros_in_plot)
    return df, df_issues


def process_date(df, null_values, col_date, min_year, max_year, col_year="year", col_month="month", col_day="day", plot_graphs=False):
    """Process date field and separate year-month-day."""
    df = df.copy()
    if is_numeric_dtype(df[col_date]):
        print(f"{col_date} is numeric")
        df[col_year] = df[col_date].fillna(0).astype(int)
        df[col_year] = df[col_year].apply(lambda x: 0 if x in null_values else x)
    else:
        print(f"{col_date} is not numeric")
        df[col_date] = df[col_date].astype(str)

        def parse_date(date_string):
            date_string = date_string.split(" |T")[0].replace("/", "-")
            vals = date_string.split("-")
            year, month, day = 0, 0, 0

            if len(vals) == 3:
                if len(vals[0]) == 4:
                    year, month, day = vals
                elif len(vals[2]) == 4:
                    year, day, month = vals[2], vals[1], vals[0]
            elif len(vals) == 2:
                if len(vals[0]) == 4:
                    year, month = vals
                elif len(vals[1]) == 4:
                    year, month = vals[1], vals[0]
            elif len(vals) == 1 and len(vals[0]) == 4:
                year = vals[0]

            return pd.Series({col_year: year, col_month: month, col_day: day})

        df[[col_year, col_month, col_day]] = df[col_date].apply(parse_date)
        df[[col_year, col_month, col_day]] = df[[col_year, col_month, col_day]].apply(pd.to_numeric, errors="coerce")

        # Swap month and day if month > 12 and day <= 12
        cond_invalid_values = (df[col_month] > 12) & (df[col_day] <= 12)
        if not df[cond_invalid_values].empty:
            df[[col_month, col_day]] = df[[col_day, col_month]]

        # Convert to Int64, NaN will become null
        for col in [col_year, col_month, col_day]:
            df[col] = df[col].astype("Int64")

    df_nulls = df[df[col_year] == 0]
    print(f"\nNull {col_year}: {len(df_nulls)} ({round(100 * len(df_nulls) / len(df), 2)}%)")

    cond_invalid = (df[col_year] < min_year) | (df[col_year] > max_year)
    df_issues = df[(df[col_year] != 0) & cond_invalid].copy()
    if len(df_issues) > 0:
        print(f"Invalid {col_year} values: {len(df_issues)} ({col_year}<{min_year} or {col_year}>{max_year})")
        display(df_issues.head(10))
        df.loc[cond_invalid, col_year] = 0

    if plot_graphs:
        plot_stat_numeric(df, col_year)

    return df, df_issues


def get_geometries_info(gdf, figsize=(18, 3), plot_graphs=True, verbose=True, save_folder=None):
    """Get geometries info."""
    gdf = gdf.copy()
    gdf.reset_index(inplace=True)
    unit = geom_l.get_unit_of_length(gdf)
    len_invalid_geoms = len(geom_l.return_invalid_geometries(gdf))
    gdf = geom_l.return_valid_geometries(gdf)

    col_id = "gid"
    gdf[col_id] = gdf.index + 1
    _, geom_outliers = geom_l.get_coordinate_outliers(
        gdf,
        col_id=col_id,
        plot_graphs=plot_graphs,
        figsize=figsize,
        verbose=verbose,
        save_folder=save_folder,
    )
    geom_doubles = geom_l.get_duplicate_geometries(gdf, col_id=col_id, verbose=verbose)

    gdf["geom_type"] = gdf["geometry"].geom_type.astype(str)
    geom_types = gdf["geom_type"].value_counts(dropna=False).reset_index()

    # Create the formatted string
    geom_types_str = ", ".join([f"{row['geom_type']}: {row['count']}" for _, row in geom_types.iterrows()])

    result = [
        {
            "Column": "geometry",
            "Invalid Geometries": len_invalid_geoms,
            "Outliers Geometries": len(geom_outliers),
            "Double Geometries": len(geom_doubles),
            "Coordinates projected": gdf.crs.is_projected,
            "Coordinates Unit": unit,
            "Geometry Types": geom_types_str,
        }
    ]
    return pd.DataFrame(result)


def df_info_per_column(df, object_cols, numerical_cols, verbose=True):
    """Get information per column of a dataframe.

    Args:
        df ([dataframe]): Input dataframe
        object_cols ([list]): Columns of object type
        numerical_cols ([list]): Columns of numerical type

    Returns:
        [dataframes]: two dataframes with info got object and numerical columns,
        respectively

    """
    df = df.copy()
    object_cols = set(object_cols)
    numerical_cols = set(numerical_cols)
    all_cols = list(object_cols.union(numerical_cols))

    total_rows = len(df)

    def format_count(count):
        return f"{count} ({100 * count / total_rows:.2f}%)"

    def column_info(col):
        series = df[col]
        null_count = series.isna().sum() + series.isin(["None", "Null", "nan", "NaN"]).sum()
        non_null_count = total_rows - null_count

        if col == "geometry":
            geom_nulls = geom_l.return_invalid_geometries(df)
            null_count = len(geom_nulls)
            non_null_count = total_rows - null_count
            zero_count = 0
            no_zero_count = non_null_count
            object_cols.add(col)
        else:
            zero_count = (series == 0).sum()
            no_zero_count = non_null_count - zero_count

        return {
            "Column": col,
            "Data type": str(series.dtype),
            "Nulls": format_count(null_count),
            "Non-nulls": format_count(non_null_count),
            "Zeros": format_count(zero_count),
            "Non-Zeros": format_count(no_zero_count),
            "Duplicates": df.duplicated(subset=[col]).sum(),
            "Unique": series.nunique(),
        }

    info = [column_info(col) for col in df.columns if col in all_cols]
    info_df = pd.DataFrame(info).sort_values("Column")

    info_df_obj = info_df[info_df["Column"].isin(object_cols)][["Column", "Data type", "Nulls", "Non-nulls", "Duplicates", "Unique"]]
    info_df_num = info_df[info_df["Column"].isin(numerical_cols)][
        ["Column", "Data type", "Nulls", "Non-nulls", "Zeros", "Non-Zeros", "Duplicates", "Unique"]
    ]

    if verbose:
        display(info_df_obj)
        display(info_df_num)
    return info_df_obj, info_df_num


def analyze_dataframe(gdf, hue_column=None, exclude_zeros_in_plot=True, plot=True, figsize=(15, 3)):
    """Analyze dataframe."""
    gdf, object_cols, numerical_cols = get_column_types(gdf)

    # Print info per column
    print("Info per column:")
    _, _ = df_info_per_column(gdf, object_cols, numerical_cols)

    if plot:
        # Plot bar plot for object columns
        for col in object_cols:
            plot_stat_object(gdf, col, show_common=10, figsize=figsize)

        # Plot box plot for numerical columns
        for col in numerical_cols:
            plot_stat_numeric(gdf, col, exclude_zeros=exclude_zeros_in_plot, figsize=figsize)

        display(gdf.describe())

        if exclude_zeros_in_plot:
            for col in numerical_cols:
                df = gdf[gdf[col] != 0].copy()

        if len(list(df.columns)) > 0:
            sns.pairplot(data=df, corner=True) if hue_column is None else sns.pairplot(data=df, hue=hue_column, corner=True)

    return gdf


def plot_correlation_matrix(df, figsize=(10, 5), fontsize=6):
    # Calculate the correlation matrix
    """Plot correlation matrix."""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    _fig, _ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        cbar_kws={"shrink": 0.5},
        annot_kws={"fontsize": fontsize},
    )
    plt.title("Correlation Matrix with Heatmap")
    plt.show()


def get_IQR_outlier_limits(df, field_name, quantiles, max_range=1.5, verbose=True, plot=True):
    """Get IQR outlier limits."""
    df = df.copy()
    # calculate Q1 and Q3: determine the 1st quartile(quantiles[0]) and 3rd quartile(quantiles[1]) of the data.
    Q1 = df[field_name].quantile(quantiles[0])
    Q3 = df[field_name].quantile(quantiles[1])
    # calculate the Interquartile Range (IQR) by subtracting Q1 from Q3.
    IQR = Q3 - Q1

    # find the lower and upper bounds for detecting outliers; these are typically calculated as
    # Q1-max_range*IQR (for the lower bound) and
    # Q3+max_range* IQR (for the upper bound)
    # any data points falling outside of these bounds are considered outliers
    min_Q1 = Q1 - max_range * IQR
    max_Q3 = Q3 + max_range * IQR
    if verbose:
        print("----------------------------------")
        print(f"Q1-Q3 analysis for {field_name}")
        print(f"Q1: {round(Q1, 5)} Q3: {round(Q3, 5)}")
        print(f"IQR: {round(IQR, 5)}")
        print(f"Q1-{max_range}*IQR: {round(min_Q1, 5)}")
        print(f"Q3+{max_range}*IQR: {round(max_Q3, 5)}")
        print("----------------------------------")

    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return min_Q1, max_Q3


def get_dataframe_outliers_with_IQR(df, field_name, quantiles, max_range=1.5, verbose=True, plot=True):
    """Get dataframe outliers with IQR."""
    min_Q1, max_Q3 = get_IQR_outlier_limits(df, field_name, quantiles, max_range=max_range, verbose=verbose, plot=plot)

    filter_outlier = (df[field_name] < min_Q1) | (df[field_name] > max_Q3)
    if verbose:
        print(f"There are {len(df[filter_outlier])} outliers!")

    df["is_outlier"] = False
    df.loc[filter_outlier, "is_outlier"] = True
    return df


def get_dataframe_outliers_with_zscore(df, field_name, zscore=3, verbose=True, plot=True):
    """Get dataframe outliers with zscore."""
    df = df.copy()
    # Calculate z-scores
    df["z_scores"] = np.abs(stats.zscore(df[field_name]))

    # Identify anomalies (z-score threshold > zscore)
    filter_outlier = df["z_scores"] > zscore
    if verbose:
        print(f"There are {len(df[filter_outlier])}!")
    df.loc[filter_outlier, "is_outlier"] = True
    df.drop(columns=["z_scores"], inplace=True)
    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return df


def get_dataframe_outliers_with_isolation_forest(df, field_name, contamination=0.01, verbose=True, plot=True):
    """Get dataframe outliers with isolation forest."""
    df = df.copy()
    # Fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=gb_l.SEED_NUMBER)

    iso_forest.fit(df[[field_name]])
    df["anomalies"] = iso_forest.predict(df[[field_name]])

    filter_outlier = df["anomalies"] == 0
    if verbose:
        print(f"There are {len(df[filter_outlier])}!")

    df.loc[filter_outlier, "is_outlier"] = True
    df.drop(columns=["anomalies"], inplace=True)
    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return df


def get_dataframe_outliers_with_DBSCAN(df, field_name, eps=0.01, min_samples=10, verbose=True, plot=True):
    """Get dataframe outliers with DBSCAN."""
    df = df.copy()
    # Initialize the DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the DBSCAN model and predict anomalies
    df["anomalies"] = dbscan.fit_predict(df[[field_name]])

    filter_outlier = df["anomalies"] == -1
    if verbose:
        print(f"There are {len(df[filter_outlier])}!")

    df.loc[filter_outlier, "is_outlier"] = True
    df.drop(columns=["anomalies"], inplace=True)
    if plot:
        plot_stat_numeric(df, field_name, exclude_zeros=False)
    return df


def get_statistics_per_group(df, field_group, field_to_check, exclude_zeros=True, digits=2):
    """Get statistics per group."""
    check_needed_df_columns(df, [field_group, field_to_check])
    if field_to_check == "geometry" or str(df[field_to_check].dtype) == "object":
        raise ValueError("Error: Geometry and Object columns cannot be checked!")

    df = df.copy()
    if exclude_zeros:
        df = df[df[field_to_check] != 0].copy()

    results = []
    for group_name, grouped_df in df.groupby(field_group):
        cond = grouped_df[field_to_check].isna()
        df_no_nulls = grouped_df[~cond].copy()
        total_df = len(grouped_df)
        perc_null = round(100 * cond.sum() / total_df, 2)

        if len(df_no_nulls):
            stats_values = {
                f"{field_to_check}_{k}": round(v(df_no_nulls[field_to_check]), digits)
                for k, v in zip(
                    ["mode", "median", "mean", "min", "max", "std"],
                    [lambda x: x.mode()[0], lambda x: x.median(), lambda x: x.mean(), lambda x: x.min(), lambda x: x.max(), lambda x: x.std()],
                )
            }
        else:
            stats_values = {f"{field_to_check}_{k}": None for k in ["mode", "median", "mean", "min", "max", "std"]}

        results.append({field_group: group_name, "total_df": total_df, f"{field_to_check}_perc_null": perc_null, **stats_values})

    df_stats = pd.DataFrame(results).sort_values(by="total_df", ascending=False)
    style_format = "{:." + str(digits) + "f}"
    for col in df_stats.columns.drop([field_group, "total_df"]):
        df_stats[col] = df_stats[col].astype(float).round(digits).map(style_format.format)

    display(df_stats)

    _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    sns.histplot(data=df, x=field_to_check, hue=field_group, multiple="stack", ax=ax)


def xicor(X, Y, ties=True):
    """Compute the Xi correlation statistic."""
    rng = np.random.default_rng(42)
    n = len(X)
    order = array([i[0] for i in sorted(enumerate(X), key=lambda x: x[1])])
    if ties:
        l = array([sum(y >= Y[order]) for y in Y[order]])
        r = l.copy()
        for j in range(n):
            if sum([r[j] == r[i] for i in range(n)]) > 1:
                tie_index = array([r[j] == r[i] for i in range(n)])
                tie_count = int(sum([r[j] == r[i] for i in range(n)]))
                r[tie_index] = rng.choice(
                    r[tie_index] - arange(0, tie_count),
                    tie_count,
                    replace=False,
                )
        return 1 - n * sum(abs(r[1:] - r[: n - 1])) / (2 * sum(l * (n - l)))
    r = array([sum(y >= Y[order]) for y in Y[order]])
    return 1 - 3 * sum(abs(r[1:] - r[: n - 1])) / (n**2 - 1)


def calc_confidence_level(zeta):
    """Calculate confidence level."""
    return norm.cdf(zeta, loc=0, scale=1) - norm.cdf(-zeta, loc=0, scale=1)


def calc_confidence_level_for_sample(n_population, n_sample):
    """Calculate confidence level for sample."""
    if n_population > n_sample:
        e = 0.05
        p = 0.5
        a = n_sample * (n_population - 1) / (n_population - n_sample)
        z = math.sqrt((a * e * e) / (p * (1 - p)))
        return round(calc_confidence_level(z) * 100, 1)
    if n_population == n_sample:
        return 100
    raise ValueError("Error: The sample size exceeds the population.")


def zeta_score(conf_level):
    """Zeta score."""
    z = abs(norm.interval(conf_level, loc=0, scale=1)[1])
    return z


def get_population_mean_and_margins(n_population, n_sample, lead_in_sample, conf_level=95):
    """Get population mean and margins."""
    p = lead_in_sample / n_sample
    z = zeta_score(conf_level / 100)
    total_mean = round(n_population * p)
    total_variance = n_population * p * (1 - p)
    total_sdev = math.sqrt(total_variance)
    margin = round(z * total_sdev)
    p = round(100 * p, 2)
    return p, total_mean, margin


def calculate_confidence_interval_from_list(data, conf_level, test, verbose=True, plot=True):
    """Calculate confidence interval from list."""
    conf_level_perc = conf_level / 100
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    z = zeta_score(conf_level_perc)
    conf_interval = (mean - z * (std_dev / np.sqrt(n)), mean + z * (std_dev / np.sqrt(n)))
    output_text = f"{conf_level}%: {int(conf_interval[0])} - {int(conf_interval[1] + 0.5)}"

    if verbose:
        print(f"\nConfidence interval for {output_text}")

    if plot:
        _fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].plot(data, "o")
        axs[0].axhline(y=mean, color="r", linestyle="-")  # average line
        axs[0].axhline(y=conf_interval[0], color="g", linestyle="--")  # lower bound of confidence interval
        axs[0].axhline(y=conf_interval[1], color="b", linestyle="--")  # upper bound of confidence interval
        axs[0].fill_between(range(n), conf_interval[0], conf_interval[1], color="b", alpha=0.1)  # fill between lines
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].set_title(f"Test {test}: Mean and Conf. Intervals for {conf_level}% conf. level", fontsize=10)

        # plot distribution
        axs[1].hist(data, bins=10, density=True, alpha=0.6, color="g")
        xmin, xmax = axs[1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std_dev)
        axs[1].plot(x, p, "k", linewidth=2)
        axs[1].set_title(f"Test {test}: Distribution of Data", fontsize=10)

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axs[2])
        axs[2].set_title(f"Test {test}: Q-Q Plot", fontsize=10)
        plt.show()
    return conf_interval, output_text


def calculate_confidence_level_from_list(data, verbose):
    """Calculate confidence level from list."""
    n = len(data)
    mean_value, min_value, max_value = np.mean(data), min(data), max(data)
    std_dev = np.std(data)
    var_value = statistics.variance(data)

    if verbose:
        print(
            f"\nMean: {round(mean_value, 2)}, Min: {round(min_value)}, Max: {round(max_value)}, "
            f"SDev: {round(std_dev, 2)}, Variance: {round(var_value, 2)}"
        )

    # use variance as margin of error
    margin_of_error = var_value
    z_score = margin_of_error * np.sqrt(n) / std_dev
    confidence_level = 100.0 * norm.cdf(z_score)
    min_value = int(mean_value - margin_of_error)
    max_value = min(int(mean_value + margin_of_error + 0.5), 100)
    if confidence_level <= 99:
        output_text = f"{round(confidence_level, 1)}%: {min_value} - {max_value}"
    else:
        output_text = f">99%: {min_value} - {max_value}"

    if verbose:
        print(f"Confidence Level: {output_text}")

    return confidence_level, output_text


def create_bar_graph(df, col_name, labels, graph_type, color_dict):
    """Create bar graph."""
    category_counts = df[col_name].value_counts()

    plt.figure(figsize=(15, 8))
    colors = [color_dict.get(cat, "#000000") for cat in category_counts.index]
    if graph_type == "bar":
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                textcoords="offset points",
                weight="bold",
            )

    elif graph_type == "pie":
        _wedges, _texts, autotexts = plt.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        plt.setp(autotexts, size=10, weight="bold", color="black")
        center_circle = plt.Circle((0, 0), 0.40, fc="white")
        plt.gca().add_artist(center_circle)

    else:
        print("Not valid graph type")
        return

    plt.title(labels["title"])
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])

    plt.xticks(rotation=45)
    plt.show()
