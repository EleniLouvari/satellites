"""Parcel geometry preparation and deterministic spatial batching.

This module normalizes parcel geometries to polygons, enforces a stable
string identifier column, and groups parcels into fixed-size spatial grid
cells so remote and local processing operate on deterministic batches.
"""

from __future__ import annotations

from collections.abc import Iterator

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


class ParcelBatchPlanner:
    """Prepare polygon inputs and split them into bounded spatial batches."""

    def _extract_polygonal_geometry(self, geometry):
        """Repair one geometry and retain only polygonal components."""
        if geometry is None or geometry.is_empty:
            return None
        geometry = make_valid(geometry)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry
        if isinstance(geometry, GeometryCollection):
            polygon_parts = []
            for part in geometry.geoms:
                if isinstance(part, Polygon):
                    polygon_parts.append(part)
                elif isinstance(part, MultiPolygon):
                    polygon_parts.extend(part.geoms)
            if polygon_parts:
                return unary_union(polygon_parts)
        return None

    def _prepare_parcels(self, parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate, repair and normalize parcel identifiers and geometries."""
        # Defensive validation protects later spatial operations from
        # ambiguous inputs and provides clearer error messages early.
        if not isinstance(parcels, gpd.GeoDataFrame):
            raise TypeError("parcels must be a geopandas.GeoDataFrame.")
        if parcels.crs is None:
            raise ValueError("The parcels GeoDataFrame must have a CRS.")
        # Keep every source attribute so labels and other parcel metadata reach the final ML GeoParquet.
        prepared = parcels.copy()
        if self.PARCEL_ID_FIELD not in prepared.columns:
            # A stable string ID is required even when callers only provide a meaningful dataframe index.
            prepared.insert(0, self.PARCEL_ID_FIELD, parcels.index.map(str))
        prepared[self.PARCEL_ID_FIELD] = prepared[self.PARCEL_ID_FIELD].astype(str)
        # Repair geometries once here so all later batching and masking stages receive polygons only.
        prepared.geometry = prepared.geometry.map(self._extract_polygonal_geometry)
        valid = prepared.geometry.notna() & ~prepared.geometry.is_empty
        prepared = prepared.loc[valid].copy()
        if prepared.empty:
            raise ValueError("No valid Polygon or MultiPolygon parcels were found.")
        duplicated = prepared[self.PARCEL_ID_FIELD].duplicated(keep=False)
        if duplicated.any():
            examples = prepared.loc[duplicated, self.PARCEL_ID_FIELD].unique()[:10]
            raise ValueError(f"{self.PARCEL_ID_FIELD} values must be unique. Example duplicates: {examples.tolist()}")
        prepared = prepared.to_crs("EPSG:4326").reset_index(drop=True)
        self.logger.info(f"Prepared {len(prepared)} valid parcels.")
        return prepared

    def _iter_spatial_batches(self) -> Iterator[gpd.GeoDataFrame]:
        """Yield geographically compact, parcel-count-bounded WGS84 batches."""
        metric = self.parcels.to_crs(epsg=self.working_epsg)
        centroids = metric.geometry.centroid
        grouping = pd.DataFrame(
            {
                "grid_x": (centroids.x // self.GRID_SIZE_METRES).astype("int64"),
                "grid_y": (centroids.y // self.GRID_SIZE_METRES).astype("int64"),
                "centroid_x": centroids.x,
                "centroid_y": centroids.y,
            },
            index=self.parcels.index,
        )
        for indices in grouping.groupby(["grid_x", "grid_y"]).groups.values():
            order = grouping.loc[list(indices)].sort_values(["centroid_y", "centroid_x"]).index
            cell = self.parcels.loc[order].copy()
            for start in range(0, len(cell), self.MAX_FEATURES_PER_JOB):
                yield cell.iloc[start : start + self.MAX_FEATURES_PER_JOB].reset_index(drop=True)
