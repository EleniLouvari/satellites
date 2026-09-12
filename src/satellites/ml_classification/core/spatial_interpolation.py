"""Spatial interpolation of missing modeling features."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd

from satellites.shared.data_cleaning import fill_null_values_using_interpolation

from .config import ClassificationPipelineConfig


def interpolate_spatial_nulls(
    dataset: pd.DataFrame, feature_columns: list[str], config: ClassificationPipelineConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Report and optionally spatially fill nulls in modeling features."""
    # Work on a copy so the caller's dataset is not mutated.
    result = dataset.copy()
    # Record per-column null counts prior to any filling so summaries are informative.
    null_counts_before = {column: int(result[column].isna().sum()) for column in feature_columns if result[column].isna().any()}
    method = config.spatial_interpolation_method
    skipped: dict[str, str] = {}

    if method:
        # Require a GeoDataFrame and a geometry column for spatial methods.
        if not isinstance(result, gpd.GeoDataFrame) or "geometry" not in result.columns:
            raise ValueError("Spatial interpolation requires a GeoDataFrame with a geometry column.")
        for column in null_counts_before:
            # Only numeric columns can be interpolated by the shared helpers.
            if not pd.api.types.is_numeric_dtype(result[column]):
                skipped[column] = "non_numeric"
                continue
            # If no donor values exist for a column skip it.
            if result[column].notna().sum() == 0:
                skipped[column] = "all_null"
                continue
            interpolated = fill_null_values_using_interpolation(
                result,
                method=method,
                fill_col_name=column,
                null_value=float("nan"),
                max_distance_in_meters=config.spatial_interpolation_max_distance_in_meters,
                variogram_lags=config.spatial_interpolation_variogram_lags,
                variogram_lags_max_dist_in_meters=(config.spatial_interpolation_variogram_max_distance_in_meters),
                plot=False,
            )
            # The shared helper converts geometries to points internally. Copy
            # back only values so the input geometries remain unchanged.
            result[column] = interpolated[column]

    null_counts_after = {column: int(result[column].isna().sum()) for column in feature_columns if result[column].isna().any()}
    summary = {
        "interpolation_method": method,
        "null_counts_before_interpolation": null_counts_before,
        "null_cells_before_interpolation": int(sum(null_counts_before.values())),
        "interpolation_skipped": skipped,
        "null_counts_after_interpolation": null_counts_after,
        "null_cells_after_interpolation": int(sum(null_counts_after.values())),
    }
    return result, summary
