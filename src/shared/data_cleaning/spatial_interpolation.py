"""Nearest, IDW, and neighborhood filling for GeoDataFrames."""

from __future__ import annotations

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.interpolate import NearestNDInterpolator

from ._validation import projected_distance_units, representative_points, require_columns


def fill_nulls_nearest_neighbor(nulls: gpd.GeoDataFrame, non_nulls: gpd.GeoDataFrame, fill_col_name: str) -> np.ndarray:
    """Interpolate target rows from the closest non-null point."""
    if non_nulls.empty:
        return np.full(len(nulls), np.nan, dtype=float)
    # Build coordinate arrays for the donor and target points and use a
    # nearest-neighbour interpolator to assign the donor value to each
    # target point. This is deterministic and cheap but does not smooth.
    donor_coordinates = np.column_stack((non_nulls.geometry.x.to_numpy(), non_nulls.geometry.y.to_numpy()))
    target_coordinates = np.column_stack((nulls.geometry.x.to_numpy(), nulls.geometry.y.to_numpy()))
    interpolator = NearestNDInterpolator(donor_coordinates, non_nulls[fill_col_name].to_numpy(dtype=float))
    return np.asarray(interpolator(target_coordinates), dtype=float)


def fill_nulls_inverse_distance_weighting(
    nulls: gpd.GeoDataFrame, non_nulls: gpd.GeoDataFrame, fill_col_name: str, max_distance: float | None = None
) -> np.ndarray:
    """Interpolate target rows using inverse-distance weights."""
    # Compute pairwise Euclidean distances between targets and donors.
    # If `max_distance` is provided, ignore donors beyond that range.
    if non_nulls.empty:
        return np.full(len(nulls), np.nan, dtype=float)
    distances = np.hypot(
        nulls.geometry.x.to_numpy()[:, np.newaxis] - non_nulls.geometry.x.to_numpy(),
        nulls.geometry.y.to_numpy()[:, np.newaxis] - non_nulls.geometry.y.to_numpy(),
    )
    within_distance = np.ones(distances.shape, dtype=bool) if max_distance is None else distances <= max_distance
    # Avoid division by zero by capping the minimum distance; donors outside
    # the search distance contribute zero weight.
    inverse_distances = np.where(within_distance, 1.0 / np.maximum(distances, 1e-6), 0.0)
    numerator = np.sum(non_nulls[fill_col_name].to_numpy(dtype=float)[np.newaxis, :] * inverse_distances, axis=1)
    denominator = inverse_distances.sum(axis=1)
    result = np.full(len(nulls), np.nan, dtype=float)
    # Use numpy.divide with `where` to leave NaN where no donors contribute.
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _aggregate_neighbors(values: pd.Series, method: str):
    """Apply one supported neighborhood reducer."""
    reducers: dict[str, Callable] = {
        "min": pd.Series.min,
        "max": pd.Series.max,
        "median": pd.Series.median,
        "mean": pd.Series.mean,
        "sum": pd.Series.sum,
        "mode": lambda series: series.mode().iloc[0] if not series.mode().empty else np.nan,
    }
    if method not in reducers:
        raise ValueError(f"Error: Unsupported neighborhood method {method!r}; choose from {sorted(reducers)}.")
    return reducers[method](values)


def fill_nulls_using_neighborhood_values(
    gdf: gpd.GeoDataFrame,
    field_name: str,
    allow_zeros: bool = False,
    method: str = "median",
    distance_in_feet: float = 500,
    min_neighbors: int = 5,
    expand: bool = True,
    select_closest: bool = False,
) -> gpd.GeoDataFrame:
    """Fill missing values from nearby valid geometries without mutating the input."""
    # Validate inputs and prepare a working copy with valid geometries.
    require_columns(gdf, [field_name, "geometry"])
    if min_neighbors < 1:
        raise ValueError("Error: min_neighbors must be at least 1.")
    result = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    result.geometry = result.geometry.make_valid()
    distance = projected_distance_units(result, float(distance_in_feet) * 0.3048)
    missing = result[field_name].isna() | (~allow_zeros & result[field_name].eq(0))
    donors = result.loc[~missing].copy()
    if not missing.any() or donors.empty:
        return result

    # Use a spatial index to efficiently find candidate donors within an
    # expanding search radius. The loop doubles the search distance until
    # either enough neighbors are found or all donors have been reached.
    spatial_index = donors.sindex
    for index, geometry in result.loc[missing].geometry.items():
        search_distance = distance
        while True:
            search_area = geometry.buffer(search_distance)
            candidates = donors.iloc[list(spatial_index.intersection(search_area.bounds))]
            neighbors = candidates.loc[candidates.intersects(search_area)].copy()
            all_donors_reached = len(neighbors) == len(donors)
            if len(neighbors) >= min_neighbors or (all_donors_reached and not neighbors.empty):
                if select_closest:
                    # Optionally trim to the closest N neighbors by geometric distance.
                    neighbors["_distance"] = neighbors.geometry.distance(geometry)
                    neighbors = neighbors.nsmallest(min_neighbors, "_distance")
                result.at[index, field_name] = _aggregate_neighbors(neighbors[field_name], method)
                break
            if not expand or all_donors_reached:
                break
            search_distance *= 2
    return result


def fill_null_values_using_interpolation(
    gdf: gpd.GeoDataFrame,
    method: str,
    fill_col_name: str,
    null_value=np.nan,
    max_distance_in_meters: float | None = None,
    variogram_lags: int = 15,
    variogram_lags_max_dist_in_meters: float | None = None,
    plot: bool = False,
) -> gpd.GeoDataFrame:
    """Fill a numeric GeoDataFrame column using nearest, IDW, or kriging interpolation."""
    # Validate method selection and inputs, then dispatch to the chosen
    # interpolation routine. The function returns a copy of the GeoDataFrame
    # with the target column filled where possible.
    valid_methods = {"nearest", "idw", "kriging"}
    if method not in valid_methods:
        raise ValueError(f"Error: Invalid interpolation method {method!r}; choose from {sorted(valid_methods)}.")
    require_columns(gdf, [fill_col_name, "geometry"])
    if not is_numeric_dtype(gdf[fill_col_name]):
        raise ValueError(f"Error: The {fill_col_name} column is not numeric.")

    # Use representative points so polygon centroids or complex geometries
    # are represented by single point coordinates for interpolation.
    result = representative_points(gdf)
    result[fill_col_name] = result[fill_col_name].astype(float)
    missing = result[fill_col_name].isna()
    if null_value is not None and not pd.isna(null_value):
        missing |= result[fill_col_name].eq(null_value)
    targets = result.loc[missing].copy()
    donors = result.loc[~missing].copy()
    if targets.empty or donors.empty:
        return result

    if method == "nearest":
        filled_values = fill_nulls_nearest_neighbor(targets, donors, fill_col_name)
    elif method == "idw":
        # Inverse-distance weighting requires a maximum search radius.
        if max_distance_in_meters is None:
            raise ValueError("Error: max_distance_in_meters is required for idw interpolation.")
        max_distance = projected_distance_units(result, max_distance_in_meters)
        filled_values = fill_nulls_inverse_distance_weighting(targets, donors, fill_col_name, max_distance)
    else:
        # Kriging needs a fitted variogram model; ensure relevant distance
        # parameters are provided and compute the best-fit model.
        if variogram_lags_max_dist_in_meters is None:
            raise ValueError("Error: variogram_lags_max_dist_in_meters is required for kriging interpolation.")
        from .variograms import calculate_variogram, fill_nulls_kriging

        max_distance = projected_distance_units(result, variogram_lags_max_dist_in_meters)
        best_model, _ = calculate_variogram(donors, fill_col_name, variogram_lags, max_distance, plot=plot)
        filled_values = fill_nulls_kriging(targets, donors, fill_col_name, best_model)

    result.loc[targets.index, fill_col_name] = np.asarray(filled_values, dtype=float)
    if plot:
        # Optional plotting to help visually inspect filled values.
        import matplotlib.pyplot as plt

        _, axis = plt.subplots(figsize=(15, 10))
        result.plot(column=fill_col_name, ax=axis, cmap="coolwarm_r", legend=True, markersize=5)
        plt.show()
    return result
