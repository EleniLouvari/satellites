"""Reduction of monthly parcel statistics into compact annual ML features.

These helpers convert wide, dated monthly parcel statistics into a compact
annual feature table suitable for ML workflows. The main reducer expects a
GeoDataFrame where monthly median columns follow the ``<SOURCE>_median__YYYYMMDD``
pattern and preserves geometry and non-dated attributes.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import geopandas as gpd
import pandas as pd

from common_libraries.io_library import read_data, write_data


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
    # Validate inputs early to provide meaningful errors to callers.
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError("parcels must be a GeoDataFrame.")
    if parcels.crs is None:
        raise ValueError("Parcel dataset has no CRS.")
    if expected_periods is not None and expected_periods < 1:
        raise ValueError("expected_periods must be at least 1.")
    if not temporal_sources or len(set(temporal_sources)) != len(temporal_sources):
        raise ValueError("temporal_sources must contain unique source names.")

    original_crs = parcels.crs
    original_geometry = parcels.geometry.copy()
    dated_columns = [column for column in parcels.columns if _DATED_COLUMN_PATTERN.search(str(column))]
    static_columns = [column for column in parcels.columns if column not in dated_columns]
    reduced = parcels[static_columns].copy()

    reference_periods: tuple[str, ...] | None = None
    generated_features: list[str] = []
    for source in temporal_sources:
        prefix = f"{source}_median__"
        monthly_columns = sorted(
            column
            for column in parcels.columns
            if str(column).startswith(prefix) and _DATED_COLUMN_PATTERN.search(str(column))
        )
        periods = tuple(str(column).rsplit("__", 1)[-1] for column in monthly_columns)
        if expected_periods is not None and len(monthly_columns) != expected_periods:
            raise ValueError(
                f"Expected {expected_periods} monthly median columns for {source}; "
                f"found {len(monthly_columns)}: {monthly_columns}"
            )
        if reference_periods is None:
            reference_periods = periods
        elif periods != reference_periods:
            raise ValueError(
                f"Monthly periods for {source} do not match the other sources. "
                f"Expected {reference_periods}; found {periods}."
            )

        monthly_values = parcels[monthly_columns].apply(pd.to_numeric, errors="coerce")
        feature_names = {
            "mean": f"{source}_median_annual_mean",
            "min": f"{source}_median_annual_min",
            "max": f"{source}_median_annual_max",
            "std": f"{source}_median_annual_std",
        }
        collisions = sorted(set(feature_names.values()).intersection(reduced.columns))
        if collisions:
            raise ValueError(f"Generated annual features already exist: {collisions}")

        reduced[feature_names["mean"]] = monthly_values.mean(axis=1)
        reduced[feature_names["min"]] = monthly_values.min(axis=1)
        reduced[feature_names["max"]] = monthly_values.max(axis=1)
        reduced[feature_names["std"]] = monthly_values.std(axis=1, ddof=0)
        generated_features.extend(feature_names.values())

    reduced = gpd.GeoDataFrame(reduced, geometry=parcels.geometry.name, crs=original_crs)
    if reduced.crs != original_crs or not reduced.geometry.equals(original_geometry):
        raise RuntimeError("Annual feature reduction unexpectedly changed parcel geometry or CRS.")
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
        raise FileNotFoundError(f"Parcel GeoParquet does not exist: {input_path}")
    output_path = Path(output_path) if output_path is not None else input_path.with_name(DEFAULT_REDUCED_FILENAME)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Reduced output_path must differ from input_path.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Reduced output already exists: {output_path}. Set overwrite=True to replace it.")

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
