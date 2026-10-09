"""Reduction of monthly parcel statistics into compact annual ML features.

These helpers convert wide, dated monthly parcel statistics into a compact
annual feature table suitable for ML workflows. The main reducer expects a
GeoDataFrame where monthly median columns follow the ``<SOURCE>_median__YYYYMMDD``
pattern and preserves geometry and non-dated attributes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd

from shared.io import read_data, write_data

DEFAULT_TEMPORAL_SOURCES = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B08",
    "B11",
    "B12",
    "EVI",
    "NDMI",
    "NDRE",
    "NDVI",
    "NDWI",
    "SAVI",
)
DEFAULT_REDUCED_FILENAME = "satellite_parcel_annual_features.geoparquet"
_DATED_COLUMN_PATTERN = re.compile(r"__(\d{8})$")


def _validate_reduction_inputs(parcels: gpd.GeoDataFrame, temporal_sources: Sequence[str], expected_periods: int | None) -> None:
    """Validate reduction inputs before feature extraction starts."""
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError("Error: parcels must be a GeoDataFrame.")
    if parcels.crs is None:
        raise ValueError("Error: Parcel dataset has no CRS.")
    if expected_periods is not None and expected_periods < 1:
        raise ValueError("Error: expected_periods must be at least 1.")
    if not temporal_sources or len(set(temporal_sources)) != len(temporal_sources):
        raise ValueError("Error: temporal_sources must contain unique source names.")


def _split_static_and_dated_columns(parcels: gpd.GeoDataFrame) -> tuple[list[str], list[str]]:
    """Split static from dated columns so all dated sources are dropped from output."""
    dated_columns = [column for column in parcels.columns if _DATED_COLUMN_PATTERN.search(str(column))]
    static_columns = [column for column in parcels.columns if column not in dated_columns]
    return static_columns, dated_columns


def _source_monthly_columns(parcels: gpd.GeoDataFrame, source: str) -> list[str]:
    """Return sorted monthly median column names for one temporal source."""
    prefix = f"{source}_median__"
    return sorted(
        column
        for column in parcels.columns
        if str(column).startswith(prefix) and _DATED_COLUMN_PATTERN.search(str(column))
    )


def _validate_source_periods(
    source: str,
    monthly_columns: Sequence[str],
    expected_periods: int | None,
    reference_periods: tuple[str, ...] | None,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """Validate monthly count and period alignment for one source."""
    periods = tuple(str(column).rsplit("__", 1)[-1] for column in monthly_columns)
    if expected_periods is not None and len(monthly_columns) != expected_periods:
        raise ValueError(
            f"Error: Expected {expected_periods} monthly median columns for {source}; found {len(monthly_columns)}: {list(monthly_columns)}"
        )
    if reference_periods is None:
        return periods, periods
    if periods != reference_periods:
        raise ValueError(
            f"Error: Monthly periods for {source} do not match the other sources. Expected {reference_periods}; found {periods}."
        )
    return reference_periods, periods


def _annual_feature_names(source: str) -> dict[str, str]:
    """Return annual feature names for one temporal source."""
    return {
        "mean": f"{source}_median_annual_mean",
        "min": f"{source}_median_annual_min",
        "max": f"{source}_median_annual_max",
        "std": f"{source}_median_annual_std",
    }


def _assign_annual_feature_columns(
    reduced: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    source: str,
    monthly_columns: Sequence[str],
) -> list[str]:
    """Compute and assign annual feature columns for one source."""
    monthly_values = parcels[list(monthly_columns)].apply(pd.to_numeric, errors="coerce")
    feature_names = _annual_feature_names(source)
    collisions = sorted(set(feature_names.values()).intersection(reduced.columns))
    if collisions:
        raise ValueError(f"Error: Generated annual features already exist: {collisions}")

    reduced[feature_names["mean"]] = monthly_values.mean(axis=1)
    reduced[feature_names["min"]] = monthly_values.min(axis=1)
    reduced[feature_names["max"]] = monthly_values.max(axis=1)
    reduced[feature_names["std"]] = monthly_values.std(axis=1, ddof=0)
    return list(feature_names.values())


def reduce_annual_median_features(
    parcels: gpd.GeoDataFrame,
    *,
    temporal_sources: Sequence[str] = DEFAULT_TEMPORAL_SOURCES,
    expected_periods: int | None = 12,
) -> gpd.GeoDataFrame:
    """Aggregate each source's monthly parcel medians over the full period.

    All dated columns are removed. Non-dated parcel attributes, contextual
    features, geometry, row order, index, and CRS are retained. Four features
    are created per source: annual mean, minimum, maximum, and population
    standard deviation of the monthly median values.
    """
    _validate_reduction_inputs(parcels, temporal_sources, expected_periods)

    original_crs = parcels.crs
    original_geometry = parcels.geometry.copy()
    static_columns, _dated_columns = _split_static_and_dated_columns(parcels)
    reduced = parcels[static_columns].copy()

    reference_periods: tuple[str, ...] | None = None
    generated_features: list[str] = []
    for source in temporal_sources:
        monthly_columns = _source_monthly_columns(parcels, source)
        reference_periods, _periods = _validate_source_periods(source, monthly_columns, expected_periods, reference_periods)
        generated_features.extend(_assign_annual_feature_columns(reduced, parcels, source, monthly_columns))

    reduced = gpd.GeoDataFrame(reduced, geometry=parcels.geometry.name, crs=original_crs)
    if reduced.crs != original_crs or not reduced.geometry.equals(original_geometry):
        raise RuntimeError("Error: Annual feature reduction unexpectedly changed parcel geometry or CRS.")
    # Store feature lineage to support downstream report generation/debugging.
    reduced.attrs["annual_feature_columns"] = generated_features
    reduced.attrs["annual_periods"] = list(reference_periods or ())
    return reduced


def save_reduced_annual_parcel_features(
    input_path: Path,
    output_path: Path | None = None,
    *,
    temporal_sources: Sequence[str] = DEFAULT_TEMPORAL_SOURCES,
    expected_periods: int | None = 12,
    overwrite: bool = False,
) -> tuple[gpd.GeoDataFrame, Path]:
    """Reduce a wide parcel GeoParquet and atomically save its sibling output."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Error: Parcel GeoParquet does not exist: {input_path}")
    output_path = Path(output_path) if output_path is not None else input_path.with_name(DEFAULT_REDUCED_FILENAME)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Error: Reduced output_path must differ from input_path.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Error: Reduced output already exists: {output_path}. Set overwrite=True to replace it.")

    parcels = read_data(str(input_path))
    reduced = reduce_annual_median_features(
        parcels,
        temporal_sources=temporal_sources,
        expected_periods=expected_periods,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        write_data(reduced, str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(
        f"Saved {len(reduced):,} parcels with {len(temporal_sources) * 4} annual satellite features: {output_path}"
    )
    return reduced, output_path
