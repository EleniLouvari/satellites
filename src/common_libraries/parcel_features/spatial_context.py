"""Reusable functions for adding vector-based spatial context to parcels."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from common_libraries.io_library import read_data, write_data


LAKE_DISTANCE_COLUMN = "distance_to_nearest_lake_m"
LAKE_ROW_ID_COLUMN = "nearest_lake_row_id"
SOIL_ATTRIBUTE_COLUMN = "GAIOIKANOT"


def _resolve_layer(path: Path, layer: str | None) -> str:
    """Return an explicit layer name and reject ambiguous multi-layer inputs."""
    layers = gpd.list_layers(path)["name"].tolist()
    if layer is not None:
        if layer not in layers:
            raise ValueError(f"Layer '{layer}' is not present in {path}. Available layers: {layers}")
        return layer
    if len(layers) != 1:
        raise ValueError(f"Specify a layer for {path}. Available layers: {layers}")
    return str(layers[0])


def validate_projected_metric_crs(crs, *, dataset_name: str = "Dataset") -> CRS:
    """Validate and return a projected CRS whose horizontal units are metres."""
    if crs is None:
        raise ValueError(f"{dataset_name} has no CRS.")

    try:
        parsed = CRS.from_user_input(crs)
    except Exception as e:
        raise ValueError(
            f"{dataset_name} CRS could not be parsed. Received: {crs!r}. Error: {e}"
        ) from e

    if not parsed.is_projected:
        raise ValueError(
            f"{dataset_name} must use a projected CRS, not a geographic CRS such as EPSG:4326; "
            f"received {parsed.name}. Reproject and save the source dataset before creating features."
        )

    # Collect all unit names from axes, filtering out None values
    units = {axis.unit_name.lower() for axis in parsed.axis_info if axis.unit_name}

    if not units:
        raise ValueError(
            f"{dataset_name} CRS has no usable axis units. CRS name: {parsed.name}. "
            f"Axes: {[axis.name for axis in parsed.axis_info]}"
        )

    # Check that all units are either 'metre' or 'meter' (American spelling)
    valid_units = {"metre", "meter"}
    invalid_units = units - valid_units
    if invalid_units:
        raise ValueError(
            f"{dataset_name} CRS must use metres; received invalid units {sorted(invalid_units)} "
            f"in {parsed.name}. Valid units: {sorted(valid_units)}"
        )

    return parsed


def load_lakes(lakes_path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Load lake geometries and retain their stable GeoPackage feature IDs."""
    if not lakes_path.is_file():
        raise FileNotFoundError(f"Lake dataset does not exist: {lakes_path}")
    selected_layer = _resolve_layer(lakes_path, layer)
    lakes = read_data(str(lakes_path), layer=selected_layer, fid_as_index=True)
    if lakes.empty:
        raise ValueError(f"Lake layer is empty: {lakes_path}, layer={selected_layer}")
    validate_projected_metric_crs(lakes.crs, dataset_name="Lake layer")

    lakes = lakes.loc[lakes.geometry.notna() & ~lakes.geometry.is_empty].copy()
    if lakes.empty:
        raise ValueError("Lake layer contains no usable geometries.")
    if not lakes.geometry.geom_type.isin({"Polygon", "MultiPolygon"}).all():
        raise ValueError("Lake layer must contain only Polygon or MultiPolygon geometries.")

    # GeoPackage FIDs are stable row identifiers and are distinct from any
    # domain-specific attribute such as LAKES_ID.
    lakes[LAKE_ROW_ID_COLUMN] = pd.array(lakes.index, dtype="Int64")
    return lakes


def calculate_nearest_lake_features(
    parcels: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Return nearest-lake distance and GeoPackage row ID for every parcel."""
    original_crs = parcels.crs
    validate_projected_metric_crs(original_crs, dataset_name="Parcel dataset")
    if parcels.geometry.isna().any() or parcels.geometry.is_empty.any():
        raise ValueError("Every parcel must have a non-empty geometry.")
    validate_projected_metric_crs(lakes.crs, dataset_name="Lake layer")

    parcel_metric = parcels.to_crs(lakes.crs)
    parcel_geometry = gpd.GeoDataFrame(
        {"_parcel_position": np.arange(len(parcel_metric), dtype=np.int64)},
        geometry=parcel_metric.geometry,
        crs=lakes.crs,
    )
    lake_geometry = lakes[[LAKE_ROW_ID_COLUMN, "geometry"]].copy()

    nearest = gpd.sjoin_nearest(
        parcel_geometry,
        lake_geometry,
        how="left",
        distance_col=LAKE_DISTANCE_COLUMN,
    )
    # Equidistant lakes can generate multiple matches. Resolve ties by the
    # smallest stable GeoPackage row ID so output remains one row per parcel.
    nearest = nearest.sort_values(
        ["_parcel_position", LAKE_DISTANCE_COLUMN, LAKE_ROW_ID_COLUMN],
        kind="stable",
        na_position="last",
    ).drop_duplicates("_parcel_position", keep="first")
    nearest = nearest.set_index("_parcel_position").reindex(range(len(parcels)))

    return pd.DataFrame(
        {
            LAKE_DISTANCE_COLUMN: pd.to_numeric(nearest[LAKE_DISTANCE_COLUMN], errors="coerce").to_numpy(),
            LAKE_ROW_ID_COLUMN: pd.array(nearest[LAKE_ROW_ID_COLUMN], dtype="Int64"),
        },
        index=parcels.index,
    )


def append_nearest_lake_features(
    parcels_path: Path,
    lakes_path: Path,
    output_path: Path,
    *,
    lakes_layer: str | None = None,
    overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """Append nearest-lake features and atomically save the parcel GeoParquet."""
    if not parcels_path.is_file():
        raise FileNotFoundError(f"Parcel GeoParquet does not exist: {parcels_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Set overwrite=True to replace it.")

    parcels = read_data(str(parcels_path))
    lakes = load_lakes(lakes_path, layer=lakes_layer)
    features = calculate_nearest_lake_features(parcels, lakes)

    result = parcels.copy()
    result[LAKE_DISTANCE_COLUMN] = features[LAKE_DISTANCE_COLUMN]
    result[LAKE_ROW_ID_COLUMN] = "lake_" + features[LAKE_ROW_ID_COLUMN].astype(str)
    if result.crs != parcels.crs or not result.geometry.equals(parcels.geometry):
        raise RuntimeError("Nearest-lake feature creation unexpectedly changed the original parcel geometry or CRS.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        write_data(result, str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Saved parcels with nearest-lake features: {output_path}")
    return result


def load_soil_polygons(
    soil_path: Path,
    *,
    layer: str | None = None,
    attribute_column: str = SOIL_ATTRIBUTE_COLUMN,
) -> gpd.GeoDataFrame:
    """Load, validate, and repair soil polygons used for centroid lookup."""
    if not soil_path.is_file():
        raise FileNotFoundError(f"Soil dataset does not exist: {soil_path}")
    selected_layer = _resolve_layer(soil_path, layer)
    soil = read_data(str(soil_path), layer=selected_layer, fid_as_index=True)
    if soil.empty:
        raise ValueError(f"Soil layer is empty: {soil_path}, layer={selected_layer}")
    if attribute_column not in soil.columns:
        raise ValueError(
            f"Soil attribute '{attribute_column}' is absent from layer '{selected_layer}'. "
            f"Available columns: {soil.columns.tolist()}"
        )
    validate_projected_metric_crs(soil.crs, dataset_name="Soil layer")

    soil = soil.loc[soil.geometry.notna() & ~soil.geometry.is_empty].copy()
    if not soil.geometry.geom_type.isin({"Polygon", "MultiPolygon"}).all():
        raise ValueError("Soil layer must contain only Polygon or MultiPolygon geometries.")
    if (~soil.geometry.is_valid).any():
        soil.geometry = soil.geometry.make_valid()
    if (~soil.geometry.is_valid).any():
        raise ValueError("Some soil geometries remain invalid after repair.")

    soil["_soil_row_id"] = pd.array(soil.index, dtype="Int64")
    return soil[[attribute_column, "_soil_row_id", "geometry"]]


def calculate_centroid_soil_feature(
    parcels: gpd.GeoDataFrame,
    soil: gpd.GeoDataFrame,
    *,
    attribute_column: str = SOIL_ATTRIBUTE_COLUMN,
) -> pd.Series:
    """Assign the soil polygon attribute intersecting each parcel centroid."""
    validate_projected_metric_crs(parcels.crs, dataset_name="Parcel dataset")
    if parcels.geometry.isna().any() or parcels.geometry.is_empty.any():
        raise ValueError("Every parcel must have a non-empty geometry.")
    validate_projected_metric_crs(soil.crs, dataset_name="Soil layer")
    if attribute_column not in soil.columns or "_soil_row_id" not in soil.columns:
        raise ValueError("Soil data must be loaded with load_soil_polygons().")

    centroids = gpd.GeoDataFrame(
        {"_parcel_position": np.arange(len(parcels), dtype=np.int64)},
        geometry=parcels.geometry.centroid,
        crs=parcels.crs,
    ).to_crs(soil.crs)
    matches = gpd.sjoin(
        centroids,
        soil[[attribute_column, "_soil_row_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    # A centroid on a shared boundary can intersect multiple polygons. Resolve
    # ties by the smallest stable GeoPackage row ID.
    matches = matches.sort_values(
        ["_parcel_position", "_soil_row_id"], kind="stable", na_position="last"
    ).drop_duplicates("_parcel_position", keep="first")
    matches = matches.set_index("_parcel_position").reindex(range(len(parcels)))
    return pd.Series(matches[attribute_column].to_numpy(), index=parcels.index, name=attribute_column)


def append_soil_feature(
    parcels_path: Path,
    soil_path: Path,
    output_path: Path,
    *,
    soil_layer: str | None = None,
    attribute_column: str = SOIL_ATTRIBUTE_COLUMN,
    overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """Append a centroid-matched soil attribute and atomically save parcels."""
    if not parcels_path.is_file():
        raise FileNotFoundError(f"Parcel GeoParquet does not exist: {parcels_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Set overwrite=True to replace it.")

    parcels = read_data(str(parcels_path))
    original_crs = parcels.crs
    validate_projected_metric_crs(original_crs, dataset_name="Parcel dataset")
    original_geometry = parcels.geometry.copy()
    soil = load_soil_polygons(soil_path, layer=soil_layer, attribute_column=attribute_column)

    result = parcels.copy()
    result[attribute_column] = calculate_centroid_soil_feature(
        parcels, soil, attribute_column=attribute_column
    )
    if result.crs != original_crs or not result.geometry.equals(original_geometry):
        raise RuntimeError("Soil feature creation unexpectedly changed the original parcel geometry or CRS.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        write_data(result, str(temporary_path))
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    missing = int(result[attribute_column].isna().sum())
    print(f"Saved parcels with soil feature '{attribute_column}': {output_path}")
    print(f"Parcels whose centroid did not intersect a soil polygon: {missing:,}/{len(result):,}")
    return result
