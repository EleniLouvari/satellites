"""Shared tabular and geospatial data-cleaning utilities.

This module re-exports commonly-used helpers from the package so callers
can import a single namespace. It intentionally does not implement logic
itself; the values below are thin aliases that make the public API tidy.
"""

from .outliers import OutlierAnalysis
from .spatial_interpolation import (
    fill_null_values_using_interpolation,
    fill_nulls_inverse_distance_weighting,
    fill_nulls_nearest_neighbor,
    fill_nulls_using_neighborhood_values,
)
from .tabular_imputation import (
    fill_categorical_with_KNN,
    fill_missing_values_in_text_fields,
    fill_numerical_with_KNN,
)
from .variograms import calculate_variogram, compute_variogram, fill_nulls_kriging, fit_variogram_model

# Public symbols exported by the package. Keep in alphabetical order where
# practical so the list is easy to scan in documentation and REPL introspection.
__all__ = [
    "OutlierAnalysis",
    "calculate_variogram",
    "compute_variogram",
    "fill_categorical_with_KNN",
    "fill_missing_values_in_text_fields",
    "fill_null_values_using_interpolation",
    "fill_nulls_inverse_distance_weighting",
    "fill_nulls_kriging",
    "fill_nulls_nearest_neighbor",
    "fill_nulls_using_neighborhood_values",
    "fill_numerical_with_KNN",
    "fit_variogram_model",
]
