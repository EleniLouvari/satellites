"""Variogram fitting and ordinary-kriging helpers."""

from __future__ import annotations

from collections.abc import Callable

import geopandas as gpd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from ._validation import representative_points, require_columns


def linear_model(lags, slope, nugget):
    """Calculate semivariance with a linear variogram model."""
    # Simple linear variogram: semivariance increases linearly with lag.
    return slope * lags + nugget


def power_model(lags, scale, exponent, nugget):
    """Calculate semivariance with a power variogram model."""
    # Power-law model allowing non-linear growth with lag distance.
    return scale * np.power(lags, exponent) + nugget


def gaussian_model(lags, sill, range_, nugget):
    """Calculate semivariance with a Gaussian variogram model."""
    # Gaussian model that rises smoothly to the sill with squared-lag decay.
    return sill * (1.0 - np.exp(-(lags**2) / range_**2)) + nugget


def exponential_model(lags, sill, range_, nugget):
    """Calculate semivariance with an exponential variogram model."""
    # Exponential model with range parameter controlling decay speed.
    return sill * (1.0 - np.exp(-lags / range_)) + nugget


def spherical_model(lags, sill, range_, nugget):
    """Calculate semivariance with a spherical variogram model."""
    # Spherical model: rises with lag and flattens to the sill at `range_`.
    return np.where(lags <= range_, sill * (1.5 * lags / range_ - 0.5 * (lags / range_) ** 3) + nugget, sill + nugget)


def compute_variogram(x, y, values, n_lags: int = 15, max_distance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Calculate an experimental semivariogram."""
    if n_lags < 1:
        raise ValueError("Error: n_lags must be at least 1.")
    # Compute pairwise distances and semivariances for all donor points.
    coordinates = np.column_stack([x, y])
    distances = pdist(coordinates)
    semivariances = pdist(np.asarray(values).reshape(-1, 1), metric="sqeuclidean") / 2.0
    if max_distance is not None:
        selected = distances <= max_distance
        distances, semivariances = distances[selected], semivariances[selected]
    if distances.size == 0:
        # Not enough donor pairs exist to estimate semivariance.
        raise ValueError("Error: At least two donor points within max_distance are required for a variogram.")
    bins = np.linspace(0, distances.max(), n_lags + 1)
    indices = np.clip(np.digitize(distances, bins) - 1, 0, n_lags - 1)
    lag_values, semivariance_values = [], []
    # Aggregate distances into lag bins and compute the experimental
    # semivariance per bin for model fitting.
    for index in range(n_lags):
        selected = indices == index
        if selected.any():
            lag_values.append(distances[selected].mean())
            semivariance_values.append(semivariances[selected].mean())
    return np.asarray(lag_values), np.asarray(semivariance_values)


def fit_variogram_model(lags, semivariance, model: Callable, initial_parameters: list[float]):
    """Fit one candidate variogram model and return parameters plus mean squared error."""
    # Use non-linear least squares to fit model parameters and return a
    # simple mean-squared-error score for model selection.
    parameters, _ = curve_fit(model, lags, semivariance, p0=initial_parameters, maxfev=10_000)
    fitted = model(lags, *parameters)
    return parameters, float(np.mean((semivariance - fitted) ** 2))


def calculate_variogram(
    gdf: gpd.GeoDataFrame, attribute_column: str, variogram_lags: int, max_distance: float | None, plot: bool = False
) -> tuple[str, np.ndarray]:
    """Fit supported variogram models and return the lowest-error candidate."""
    # Validate inputs, extract donor points, and compute the experimental
    # variogram which will be used to fit candidate theoretical models.
    require_columns(gdf, [attribute_column, "geometry"])
    if not is_numeric_dtype(gdf[attribute_column]):
        raise ValueError(f"Error: The {attribute_column} column is not numeric.")
    donors = representative_points(gdf).dropna(subset=[attribute_column])
    if len(donors) < 2:
        raise ValueError("Error: Kriging requires at least two non-null donor points.")
    lags, semivariance = compute_variogram(
        donors.geometry.x.to_numpy(),
        donors.geometry.y.to_numpy(),
        donors[attribute_column].to_numpy(),
        variogram_lags,
        max_distance,
    )
    models = {
        "linear": (linear_model, [1.0, 0.1]),
        "power": (power_model, [1.0, 1.0, 0.1]),
        "gaussian": (gaussian_model, [1.0, 1.0, 0.1]),
        "exponential": (exponential_model, [1.0, 1.0, 0.1]),
        "spherical": (spherical_model, [1.0, 1.0, 0.1]),
    }
    # Fit each supported theoretical model and collect those that succeed.
    candidates = []
    for name, (model, initial_parameters) in models.items():
        try:
            parameters, score = fit_variogram_model(lags, semivariance, model, initial_parameters)
            candidates.append((score, name, parameters, model))
        except (RuntimeError, ValueError, TypeError):
            continue
    if not candidates:
        raise ValueError("Error: No supported variogram model could be fitted to the donor points.")
    _, best_name, best_parameters, _ = min(candidates, key=lambda candidate: candidate[0])
    if plot:
        # Optional diagnostic plot showing experimental variogram and
        # fitted candidate curves to aid visual model selection.
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(lags, semivariance, "o", label="Experimental variogram")
        for score, name, parameters, model in candidates:
            plt.plot(lags, model(lags, *parameters), label=f"{name} ({score:.4f})")
        plt.xlabel("Lag")
        plt.ylabel("Semivariance")
        plt.legend()
        plt.show()
    return best_name, best_parameters


def fill_nulls_kriging(
    gdf_nulls: gpd.GeoDataFrame,
    gdf_no_nulls: gpd.GeoDataFrame,
    fill_col_name: str,
    variogram_model_name: str,
    variogram_lags: int = 15,
    plot: bool = False,
) -> np.ndarray:
    """Predict target points with ordinary kriging."""
    # Ordinary kriging using pykrige. Validate the requested model and then
    # predict the value at each target point. If there are no donors return
    # NaNs for all targets.
    del variogram_lags, plot
    valid_models = {"linear", "power", "gaussian", "exponential", "spherical"}
    if variogram_model_name not in valid_models:
        raise ValueError(f"Error: Invalid variogram model {variogram_model_name!r}; choose from {sorted(valid_models)}.")
    if gdf_no_nulls.empty:
        return np.full(len(gdf_nulls), np.nan, dtype=float)
    from pykrige.ok import OrdinaryKriging

    model = OrdinaryKriging(
        gdf_no_nulls.geometry.x,
        gdf_no_nulls.geometry.y,
        gdf_no_nulls[fill_col_name],
        variogram_model=variogram_model_name,
        verbose=False,
        enable_plotting=False,
    )
    predicted, _ = model.execute("points", gdf_nulls.geometry.x, gdf_nulls.geometry.y)
    return np.asarray(predicted, dtype=float)
