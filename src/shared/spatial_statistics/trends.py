"""Exploratory directional trends against spatial coordinates."""

import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression

import shared.geometry as geom_l
import shared.tabular as cm_l


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
        raise ValueError(f"Error: The {attribute_column} column is not numeric.")

    df = df.copy()
    df = geom_l.convert_geometries_to_points(df)

    df = df[df[attribute_column].notna()].copy()
    if exclude_zeros:
        df = df[df[attribute_column] != 0].copy()

    x = df.geometry.x.values.reshape(-1, 1)
    y = df.geometry.y.values.reshape(-1, 1)
    values = df[attribute_column].values

    # Fit linear regression models
    # Fit separate one-coordinate regressions; neither slope controls for the other coordinate.
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
    # The fixed slope cutoff is unit-dependent and is not a statistical significance test.
    trend_x = np.abs(model_x.coef_[0]) > 1e-3
    trend_y = np.abs(model_y.coef_[0]) > 1e-3

    return trend_x or trend_y
