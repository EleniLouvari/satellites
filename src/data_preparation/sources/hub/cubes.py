"""Build openEO-compatible raw parcel-batch cubes from HUB Sentinel ARD assets.

``HubCubePipeline`` is a mixin for :class:`ParcelStatsBase`. It discovers HUB
ARD items, reads only the windows around each parcel batch, and writes one
``batch_XXXXX_monthly.nc`` per batch with the layout the shared local engine
expects: dimensions ``t, y, x``, one float32 variable per physical band, NaN as
nodata, and EPSG:2100 grid-mapping metadata.

Pixel rules
-----------
* Sentinel-2: SCL classes ``SENTINEL2_INVALID_SCL_CLASSES`` and SCL nodata
  (255) are invalid; ``SENTINEL2_BUFFERED_SCL_CLASSES`` are dilated by one
  pixel (3x3), as in the openEO graph; ARD ``MASK == 0`` is an extra nodata guard.
  Values are already physical reflectance, so no scale factor is applied.
* Sentinel-1: ARD ``MASK == 1`` only (0 nodata, 2 layover/shadow); linear
  gamma0 values ``<= 0`` (thermal-noise artefacts) are invalid.
* A pixel is valid for an observation only when every band of that sensor is
  valid, so indices never mix bands from different scenes.
* The batch grid uses the 10 m lattice of the dominant Sentinel-2 tile, so most
  optical pixels are copied without resampling. Other assets are masked first
  and then reprojected with NaN-aware bilinear resampling.
* Items of one observation are mosaicked whole-pixel, first valid wins, in the
  priority order from :func:`ard_items.group_observations`. Observations within
  one reporting interval are then reduced with the configured temporal reducer.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from pyproj import CRS
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
from shapely.geometry import box, mapping
from shapely.prepared import prep

from data_preparation.sources.hub import ard_items
from data_preparation.sources.hub.assets import HubAssetResolver, read_asset
from data_preparation.sources.hub.catalog_library import USERS, CatalogSearchUtils

HUB_GRID_EPSG = 2100
HUB_RESOLUTION_METRES = 10.0
# ARD float rasters use the float32 maximum as nodata; compression can perturb it slightly.
INVALID_FLOAT_THRESHOLD = 1e38
SCL_NODATA = 255
# Source pixels read beyond each block edge: one for SCL dilation, one for bilinear support.
READ_HALO_PIXELS = 2
# HUB batches are split on this centroid grid so sparse parcels never span a long, mostly empty raster.
HUB_BATCH_CELL_METRES = 10_000


@dataclass(frozen=True)
class BlockGrid:
    """A target-grid block: row/column offsets and shape within the batch grid."""

    row: int
    column: int
    height: int
    width: int


def snap_grid(
    bounds: tuple[float, float, float, float], lattice: tuple[float, float], resolution: float
) -> tuple[Affine, int, int]:
    """Return a north-up transform and shape covering ``bounds`` on the lattice through ``lattice``.

    Parameters
    ----------
    bounds:
        ``(minx, miny, maxx, maxy)`` in the grid CRS.
    lattice:
        Any pixel corner ``(x, y)`` of the lattice to align with.
    resolution:
        Pixel size in CRS units.

    """
    minx, miny, maxx, maxy = bounds
    if not (maxx > minx and maxy > miny):
        raise ValueError("Error: Grid bounds must have positive width and height.")
    x0 = lattice[0] + math.floor((minx - lattice[0]) / resolution) * resolution
    y0 = lattice[1] + math.ceil((maxy - lattice[1]) / resolution) * resolution
    width = max(1, math.ceil(round((maxx - x0) / resolution, 6)))
    height = max(1, math.ceil(round((y0 - miny) / resolution, 6)))
    return Affine(resolution, 0.0, x0, 0.0, -resolution, y0), width, height


def iter_blocks(height: int, width: int, block_size: int):
    """Yield row-major :class:`BlockGrid` tiles covering a ``height`` x ``width`` grid."""
    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            yield BlockGrid(row, column, min(block_size, height - row), min(block_size, width - column))


def sentinel2_invalid(scl: np.ndarray, mask: np.ndarray, invalid_classes, buffered_classes) -> np.ndarray:
    """Return invalid Sentinel-2 pixels: SCL invalid/nodata, dilated cloud classes, or ARD nodata."""
    invalid = np.isin(scl, invalid_classes) | (scl == SCL_NODATA) | (mask == 0)
    cloud = binary_dilation(np.isin(scl, buffered_classes), structure=np.ones((3, 3), dtype=bool))
    return invalid | cloud


def sentinel1_invalid(mask: np.ndarray) -> np.ndarray:
    """Return invalid Sentinel-1 pixels: ARD nodata (0) and layover/shadow (2)."""
    return mask != 1


def clean_values(values: np.ndarray, nodata: float | None, invalid: np.ndarray, *, positive_only: bool) -> np.ndarray:
    """Return float32 values with nodata, invalid and non-finite pixels set to NaN."""
    result = values.astype("float32", copy=True)
    bad = invalid | ~np.isfinite(result) | (np.abs(result) >= INVALID_FLOAT_THRESHOLD)
    if nodata is not None and np.isfinite(nodata):
        bad |= result == np.float32(nodata)
    if positive_only:
        bad |= result <= 0
    result[bad] = np.nan
    return result


def require_all_bands(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Invalidate pixels where any band is NaN so one observation stays spectrally consistent."""
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays.values()])
    for array in arrays.values():
        array[~valid] = np.nan
    return arrays


def mosaic_first_valid(layers: list[dict[str, np.ndarray]], variables, shape) -> dict[str, np.ndarray]:
    """Fill each pixel from the first layer valid for all variables (whole-pixel priority mosaic)."""
    result = {name: np.full(shape, np.nan, dtype="float32") for name in variables}
    filled = np.zeros(shape, dtype=bool)
    for layer in layers:
        take = ~filled & np.logical_and.reduce([np.isfinite(layer[name]) for name in variables])
        for name in variables:
            result[name][take] = layer[name][take]
        filled |= take
    return result


def reduce_observations(stack: list[np.ndarray], reducer: str, shape) -> np.ndarray:
    """Reduce observation arrays with a NaN-skipping reducer; all-NaN pixels stay NaN."""
    if not stack:
        return np.full(shape, np.nan, dtype="float32")
    if len(stack) == 1:
        return stack[0]
    values = np.stack(stack)
    functions = {"median": np.nanmedian, "mean": np.nanmean, "min": np.nanmin, "max": np.nanmax, "sum": np.nansum}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = functions[reducer](values, axis=0).astype("float32")
    reduced[~np.isfinite(values).any(axis=0)] = np.nan
    return reduced


def _source_window(dataset, bounds, halo):
    """Return the integer source window around ``bounds`` (with halo) or ``None`` if disjoint."""
    left, bottom, right, top = bounds
    inverse = ~dataset.transform
    col0, row0 = inverse @ (left, top)
    col1, row1 = inverse @ (right, bottom)
    col_off, row_off = math.floor(min(col0, col1)) - halo, math.floor(min(row0, row1)) - halo
    col_end, row_end = math.ceil(max(col0, col1)) + halo, math.ceil(max(row0, row1)) + halo
    if col_off >= dataset.width or row_off >= dataset.height or col_end <= 0 or row_end <= 0:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _metadata_grid(asset):
    """Return a transform/width/height stand-in from STAC ``proj:*`` metadata, or ``None``."""
    if asset.transform is None or asset.shape is None:
        return None
    return SimpleNamespace(transform=Affine(*asset.transform), height=asset.shape[0], width=asset.shape[1])


def _read_window(dataset, window):
    """Read one band window, filling outside pixels with the asset nodata."""
    fill = dataset.nodata if dataset.nodata is not None else 0
    return dataset.window_transform(window), dataset.crs, dataset.read(1, window=window, boundless=True, fill_value=fill)


def _block_bounds(block_transform: Affine, shape) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` of a north-up block."""
    left, top = block_transform.c, block_transform.f
    return left, top + block_transform.e * shape[0], left + block_transform.a * shape[1], top


def _align(values, source_transform, source_crs, block_transform, shape) -> np.ndarray:
    """Copy aligned pixels, or reproject masked values NaN-aware bilinear, onto the block grid."""
    target = np.full(shape, np.nan, dtype="float32")
    offsets = _aligned_offsets(source_transform, block_transform)
    if offsets is not None:
        row, column = offsets
        piece = values[row : row + shape[0], column : column + shape[1]]
        target[: piece.shape[0], : piece.shape[1]] = piece
        return target
    reproject(
        source=values,
        destination=target,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=block_transform,
        dst_crs=CRS.from_epsg(HUB_GRID_EPSG),
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return target


def _aligned_offsets(source: Affine, target: Affine) -> tuple[int, int] | None:
    """Return integer (row, column) offsets of ``target`` inside ``source`` when both share one lattice."""
    if not (math.isclose(source.a, target.a) and math.isclose(source.e, target.e) and source.b == target.b == 0):
        return None
    column = (target.c - source.c) / source.a
    row = (target.f - source.f) / source.e
    if abs(column - round(column)) > 1e-6 or abs(row - round(row)) > 1e-6:
        return None
    return round(row), round(column)


def split_by_cell(parcels: gpd.GeoDataFrame, cell_metres: float) -> list[gpd.GeoDataFrame]:
    """Split parcels into deterministic groups by the EPSG:2100 grid cell of their centroid."""
    centroids = parcels.to_crs(epsg=HUB_GRID_EPSG).geometry.centroid
    keys = pd.DataFrame(
        {"x": (centroids.x // cell_metres).astype("int64"), "y": (centroids.y // cell_metres).astype("int64")},
        index=parcels.index,
    )
    groups = sorted(keys.groupby(["y", "x"]).groups.items())
    return [parcels.loc[index].reset_index(drop=True) for _, index in groups]


def project_footprints(observations) -> dict:
    """Return prepared EPSG:2100 footprints keyed by item id; items without a footprint are absent."""
    items = {item.item_id: item.footprint for o in observations for item in o.items if item.footprint is not None}
    if not items:
        return {}
    projected = gpd.GeoSeries(list(items.values()), crs=4326).to_crs(epsg=HUB_GRID_EPSG)
    return {item_id: prep(geometry) for item_id, geometry in zip(items, projected, strict=True)}


def block_items(observation, block_geometry, footprints) -> tuple:
    """Return the observation's items whose footprint may cover ``block_geometry``."""
    return tuple(
        item
        for item in observation.items
        if item.item_id not in footprints or footprints[item.item_id].intersects(block_geometry)
    )


def block_box(transform: Affine, block: BlockGrid):
    """Return the EPSG:2100 polygon of one target-grid block."""
    left, top = transform @ (block.column, block.row)
    right, bottom = transform @ (block.column + block.width, block.row + block.height)
    return box(left, bottom, right, top)


class HubCubePipeline:
    """Acquire HUB Sentinel ARD windows as raw parcel-batch NetCDF cubes."""

    SENTINEL2_ARD_COLLECTION = ard_items.SENTINEL2_ARD_COLLECTION
    SENTINEL1_ARD_COLLECTION = ard_items.SENTINEL1_ARD_COLLECTION
    HUB_PROCESSING_STRATEGY = "hub_ard_local_v2"
    HUB_SENTINEL1_BACKSCATTER = "gamma0-terrain-flattened-linear"

    def _configure_hub_source(
        self, *, catalog_url: str | None = None, client_id: str | None = None, read_workers: int = 8, block_size: int = 512
    ) -> None:
        """Validate and store HUB settings without network access.

        ``catalog_url`` and ``client_id`` default to ``CATALOG_URL`` and
        ``CATALOG_CLIENT_ID`` (else ``internal``). The client must be a
        configured catalog role with a secret, so load ``.env`` before importing
        the catalog modules.
        """
        url = catalog_url or os.getenv("CATALOG_URL")
        if not url:
            raise ValueError("Error: HUB catalog URL is missing; pass hub_catalog_url or set CATALOG_URL.")
        client = client_id or os.getenv("CATALOG_CLIENT_ID", "internal")
        if not USERS.get(client):
            raise ValueError(
                f"Error: HUB client_id {client!r} has no configured secret. Load .env before importing the catalog modules."
            )
        for name, value, minimum in (("read_workers", read_workers, 1), ("block_size", block_size, 64)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Error: hub {name} must be an integer >= {minimum}.")
        self.hub_catalog_url = url
        self.hub_client_id = client
        self.hub_read_workers = read_workers
        self.hub_block_size = block_size
        self._hub_manifest = None
        self._hub_manifest_digest = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _hub_sensor_variables(self) -> dict[str, tuple[str, ...]]:
        """Return requested physical bands per HUB collection, omitting unused sensors."""
        variables = {
            self.SENTINEL2_ARD_COLLECTION: self._required_sentinel2_bands(),
            self.SENTINEL1_ARD_COLLECTION: self._required_sentinel1_bands(),
        }
        return {collection: bands for collection, bands in variables.items() if bands}

    def _hub_asset_keys(self) -> dict[str, tuple[str, ...]]:
        """Return every asset read per collection: bands plus the masks they depend on."""
        auxiliary = {self.SENTINEL2_ARD_COLLECTION: ("SCL", "MASK"), self.SENTINEL1_ARD_COLLECTION: ("MASK",)}
        return {collection: (*bands, *auxiliary[collection]) for collection, bands in self._hub_sensor_variables().items()}

    def _hub_context_metres(self) -> float | None:
        """Return the context needed around parcels for local filling, or ``None`` when unbounded."""
        if not self.fill_nulls:
            # No spatial filling or interpolation runs, so parcels need only the read halo.
            return READ_HALO_PIXELS * HUB_RESOLUTION_METRES
        halo = (max(self.spatial_fill_window_sizes) // 2 + READ_HALO_PIXELS) * HUB_RESOLUTION_METRES
        if self.interpolation_method == "kriging":
            distance = self.interpolation_variogram_max_distance_in_meters
        else:
            distance = self.interpolation_max_distance_in_meters
        return None if distance is None else halo + distance

    def _hub_margin_metres(self) -> float:
        """Return the margin around parcels: fill context, or the fill halo when context is unbounded."""
        context = self._hub_context_metres()
        if context is not None:
            return context
        return (max(self.spatial_fill_window_sizes) // 2 + READ_HALO_PIXELS) * HUB_RESOLUTION_METRES

    def _hub_batch_extent(self, parcels: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
        """Return the EPSG:2100 batch extent: parcel bounds plus the margin."""
        margin = self._hub_margin_metres()
        minx, miny, maxx, maxy = parcels.to_crs(epsg=HUB_GRID_EPSG).total_bounds
        return (minx - margin, miny - margin, maxx + margin, maxy + margin)

    def _hub_iter_batches(self):
        """Yield HUB batches: shared spatial batches split further on the HUB centroid grid."""
        for batch in self._iter_spatial_batches():
            yield from split_by_cell(batch, HUB_BATCH_CELL_METRES)

    def _hub_search_area(self, parcels: gpd.GeoDataFrame):
        """Return the WGS84 search rectangle (GeoJSON) and the buffered parcel region used for pruning."""
        buffered = parcels.to_crs(epsg=HUB_GRID_EPSG).geometry.buffer(self._hub_margin_metres()).union_all()
        region = gpd.GeoSeries([buffered], crs=HUB_GRID_EPSG).to_crs(4326).iloc[0]
        return mapping(box(*region.bounds)), region

    def _hub_collection_observations(self, catalog, collection, keys, geometry, region) -> list:
        """Search, filter, group and footprint-prune one collection for one batch."""
        sentinel2 = collection == self.SENTINEL2_ARD_COLLECTION
        max_cloud = self.sentinel2_max_scene_cloud_cover if sentinel2 else None
        items = ard_items.search_ard_items(
            catalog, collection, geometry, self.start_date, self.end_date, max_cloud_cover=max_cloud
        )
        items = ard_items.filter_items(
            items,
            required_assets=keys,
            max_cloud_cover=max_cloud,
            orbit_direction="BOTH" if sentinel2 else self.sentinel1_orbit_direction,
        )
        return ard_items.intersecting_observations(ard_items.group_observations(items), region)

    def _resolve_hub_manifest(self) -> pd.DataFrame:
        """Discover and group HUB items for every batch; store and return the manifest.

        Sets ``self._hub_batches`` (``[(batch_number, parcels), ...]``),
        ``self._hub_observations`` (batch number → observations),
        ``self._hub_manifest`` and ``self._hub_manifest_digest``. Raises when a
        requested sensor has no usable observation in some batch.
        """
        catalog = CatalogSearchUtils(catalog_endpoint=self.hub_catalog_url, client_id=self.hub_client_id, verbose=False)
        asset_keys = self._hub_asset_keys()
        self._hub_batches = list(enumerate(self._hub_iter_batches(), start=1))
        if not self._hub_batches:
            raise RuntimeError("Error: No processing batches were produced.")
        self._hub_observations, frames, empty = {}, [], []
        for batch_number, parcels in self._hub_batches:
            geometry, region = self._hub_search_area(parcels)
            observations = []
            for collection, keys in asset_keys.items():
                grouped = self._hub_collection_observations(catalog, collection, keys, geometry, region)
                if not grouped:
                    empty.append(f"batch {batch_number}: {collection}")
                observations.extend(grouped)
            self._hub_observations[batch_number] = observations
            frame = ard_items.manifest_frame(observations, asset_keys)
            frame.insert(0, "batch_number", batch_number)
            frames.append(frame)
            self.source_logger.info(f"HUB batch {batch_number}: {len(observations)} observations for {len(parcels)} parcels.")
        if empty:
            raise ValueError(f"Error: No usable HUB ARD observations for the requested period: {', '.join(empty)}.")
        manifest = pd.concat(frames, ignore_index=True)
        self._hub_manifest = manifest
        self._hub_manifest_digest = ard_items.manifest_digest(
            manifest, search_date=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        )
        return manifest

    def _hub_source_signature(self) -> dict:
        """Return HUB-specific cache-signature entries; requires a resolved manifest."""
        if self._hub_manifest_digest is None:
            raise RuntimeError("Error: Resolve the HUB manifest before building the cache signature.")
        return {
            "processing_strategy": self.HUB_PROCESSING_STRATEGY,
            "hub_collections": sorted(self._hub_asset_keys()),
            "hub_asset_keys": {collection: list(keys) for collection, keys in sorted(self._hub_asset_keys().items())},
            "hub_manifest_digest": self._hub_manifest_digest,
            "hub_grid": f"EPSG:{HUB_GRID_EPSG}/{HUB_RESOLUTION_METRES:g}m/dominant-sentinel2-lattice",
            "hub_resampling": "copy-aligned/bilinear-nan-aware-after-masking",
            "hub_sentinel2_mask": "scl-invalid+scl255+dilated-buffered-3x3+ard-mask0",
            "hub_sentinel1_mask": "ard-mask==1;values>0",
            "hub_sentinel1_backscatter": self.HUB_SENTINEL1_BACKSCATTER,
            "hub_mosaic": "datatake-or-pass;whole-pixel-first-valid;nodata,cloud,newest-processing,id",
            "hub_batch_cell_metres": HUB_BATCH_CELL_METRES,
            "hub_context": "fill-aware;footprint-pruned",
        }

    # ------------------------------------------------------------------
    # Cube construction
    # ------------------------------------------------------------------
    def _hub_lattice(self, observations) -> tuple[float, float]:
        """Return the pixel-corner lattice of the most common S2 (else S1) asset grid."""
        for collection in (self.SENTINEL2_ARD_COLLECTION, self.SENTINEL1_ARD_COLLECTION):
            origins = Counter()
            for observation in observations:
                if observation.collection != collection:
                    continue
                for item in observation.items:
                    transform = next((asset.transform for asset in item.assets.values() if asset.transform), None)
                    if transform:
                        origins[
                            (round(transform[2] % HUB_RESOLUTION_METRES, 6), round(transform[5] % HUB_RESOLUTION_METRES, 6))
                        ] += 1
            if origins:
                return origins.most_common(1)[0][0]
        return (0.0, 0.0)

    def _hub_time_layout(self, observations) -> tuple[list[pd.Timestamp], list[list]]:
        """Return cube time labels and the observations contributing to each label."""
        if self.temporal_reducer == "none":
            times = sorted({observation.datetime for observation in observations})
            return times, [[o for o in observations if o.datetime == when] for when in times]
        intervals, labels = self._temporal_intervals()
        groups = []
        for start, end in intervals:
            lower, upper = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
            groups.append([o for o in observations if lower <= o.datetime < upper])
        return [pd.Timestamp(label, tz="UTC") for label in labels], groups

    def _hub_active_blocks(self, parcels, transform, height, width) -> list[BlockGrid]:
        """Return blocks within fill context of a parcel; every block when the context is unbounded."""
        blocks = list(iter_blocks(height, width, self.hub_block_size))
        context = self._hub_context_metres()
        if context is None:
            return blocks
        region = prep(parcels.to_crs(epsg=HUB_GRID_EPSG).geometry.buffer(context).union_all())
        return [block for block in blocks if region.intersects(block_box(transform, block))]

    def _item_window(self, resolver, item, key, bounds, asset_pool):
        """Locate the source window from metadata; open the asset only when metadata is incomplete.

        The fallback open runs on the asset pool, so it counts against the same
        ``hub_read_workers`` remote-concurrency cap as every band read.
        """
        grid = _metadata_grid(item.assets[key])
        if grid is not None:
            return _source_window(grid, bounds, READ_HALO_PIXELS)
        href = item.assets[key].href
        return asset_pool.submit(read_asset, resolver, href, lambda d: _source_window(d, bounds, READ_HALO_PIXELS)).result()

    def _read_item_assets(self, resolver, item, keys, window, asset_pool):
        """Read one window from every asset concurrently; return arrays plus the shared grid."""
        reads = list(
            asset_pool.map(lambda key: read_asset(resolver, item.assets[key].href, lambda d: _read_window(d, window)), keys)
        )
        transforms = {transform for transform, _crs, _array in reads}
        if len(transforms) != 1:
            raise ValueError(f"Error: Assets of HUB item {item.item_id} do not share one pixel grid.")
        source_transform, source_crs, _ = reads[0]
        return {key: array for key, (_t, _c, array) in zip(keys, reads, strict=True)}, source_transform, source_crs

    def _mask_item(self, item, raw, variables) -> dict[str, np.ndarray]:
        """Apply the sensor validity rule and require every band to be valid at a pixel."""
        sentinel2 = item.collection == self.SENTINEL2_ARD_COLLECTION
        if sentinel2:
            invalid = sentinel2_invalid(
                raw["SCL"], raw["MASK"], self.SENTINEL2_INVALID_SCL_CLASSES, self.SENTINEL2_BUFFERED_SCL_CLASSES
            )
        else:
            invalid = sentinel1_invalid(raw["MASK"])
        return require_all_bands(
            {name: clean_values(raw[name], item.assets[name].nodata, invalid, positive_only=not sentinel2) for name in variables}
        )

    def _read_item_block(self, resolver, item, variables, block_transform, shape, asset_pool) -> dict[str, np.ndarray] | None:
        """Read, mask and align one item's bands onto a target block; ``None`` when disjoint."""
        auxiliary = ("SCL", "MASK") if item.collection == self.SENTINEL2_ARD_COLLECTION else ("MASK",)
        keys = (*variables, *auxiliary)
        window = self._item_window(resolver, item, keys[0], _block_bounds(block_transform, shape), asset_pool)
        if window is None:
            return None
        raw, source_transform, source_crs = self._read_item_assets(resolver, item, keys, window, asset_pool)
        cleaned = self._mask_item(item, raw, variables)
        aligned = {name: _align(values, source_transform, source_crs, block_transform, shape) for name, values in cleaned.items()}
        return require_all_bands(aligned)

    def _read_observation_block(
        self, resolver, items, variables, block_transform, shape, asset_pool
    ) -> dict[str, np.ndarray] | None:
        """Return the whole-pixel mosaic of one observation's items on a block, or ``None`` when none overlap."""
        layers = [
            layer
            for item in items
            if (layer := self._read_item_block(resolver, item, variables, block_transform, shape, asset_pool)) is not None
        ]
        return mosaic_first_valid(layers, variables, shape) if layers else None

    def _reduce_block(self, observation_layers, sensor_variables, shape) -> dict[str, np.ndarray]:
        """Reduce per-observation block mosaics (collection → list of layers) for one time label."""
        result = {}
        for collection, variables in sensor_variables.items():
            layers = observation_layers.get(collection, [])
            if self.temporal_reducer == "none" and layers:
                # Several observations at one exact timestamp are mosaicked rather than reduced.
                layers = [mosaic_first_valid(layers, variables, shape)]
            reducer = "median" if self.temporal_reducer == "none" else self.temporal_reducer
            for name in variables:
                result[name] = reduce_observations([layer[name] for layer in layers], reducer, shape)
        return result

    def _hub_work_chunks(self, groups, blocks):
        """Yield lists of (time index, block) whose observation reads fit about ``2 * hub_read_workers``."""
        budget = 2 * self.hub_read_workers
        chunk, reads = [], 0
        for time_index, group in enumerate(groups):
            for block in blocks if group else ():
                if chunk and reads + len(group) > budget:
                    yield chunk
                    chunk, reads = [], 0
                chunk.append((time_index, block))
                reads += len(group)
        if chunk:
            yield chunk

    def _submit_block_reads(self, pools, resolver, transform, block, observations, sensor_variables, footprints):
        """Submit one read per observation that may cover ``block``; return ``[(collection, future), ...]``."""
        block_transform = transform @ Affine.translation(block.column, block.row)
        shape, geometry = (block.height, block.width), block_box(transform, block)
        pending = []
        for observation in observations:
            if observation.collection not in sensor_variables:
                continue
            items = block_items(observation, geometry, footprints)
            if items:
                variables = sensor_variables[observation.collection]
                observation_pool, asset_pool = pools
                future = observation_pool.submit(
                    self._read_observation_block, resolver, items, variables, block_transform, shape, asset_pool
                )
                pending.append((observation.collection, future))
        return pending

    def _write_block(self, variables, time_index, block, pending, sensor_variables) -> None:
        """Reduce finished observation reads for one (time, block) and write them; caller thread only."""
        layers: dict[str, list] = {}
        for collection, future in pending:
            if (layer := future.result()) is not None:
                layers.setdefault(collection, []).append(layer)
        reduced = self._reduce_block(layers, sensor_variables, (block.height, block.width))
        rows, columns = slice(block.row, block.row + block.height), slice(block.column, block.column + block.width)
        for name, values in reduced.items():
            variables[name][time_index, rows, columns] = values

    def _fill_hub_cube(self, resolver, variables, transform, groups, blocks, sensor_variables, footprints) -> None:
        """Read observation blocks in parallel and write each reduced (time, block) in this thread.

        Observation tasks only wait on a separate asset pool of ``hub_read_workers``
        threads, so remote concurrency is bounded by that setting and the pools
        cannot deadlock. netCDF-C is not thread-safe, so only this thread writes.
        Chunking keeps at most a few block mosaics per worker in memory.
        """
        workers = self.hub_read_workers
        with ThreadPoolExecutor(max_workers=workers) as observation_pool, ThreadPoolExecutor(max_workers=workers) as asset_pool:
            pools = (observation_pool, asset_pool)
            for chunk in self._hub_work_chunks(groups, blocks):
                submitted = [
                    (
                        time_index,
                        block,
                        self._submit_block_reads(
                            pools, resolver, transform, block, groups[time_index], sensor_variables, footprints
                        ),
                    )
                    for time_index, block in chunk
                ]
                for time_index, block, pending in submitted:
                    self._write_block(variables, time_index, block, pending, sensor_variables)

    def _write_hub_cube(self, resolver, batch_number, parcels, observations, netcdf_path: Path) -> list[dict]:
        """Build one batch cube atomically and return its coverage rows."""
        from netCDF4 import Dataset, date2num

        sensor_variables = self._hub_sensor_variables()
        transform, width, height = snap_grid(
            self._hub_batch_extent(parcels), self._hub_lattice(observations), HUB_RESOLUTION_METRES
        )
        times, groups = self._hub_time_layout(observations)
        blocks = self._hub_active_blocks(parcels, transform, height, width)
        crs = CRS.from_epsg(HUB_GRID_EPSG)
        temporary = netcdf_path.with_suffix(".partial.nc")
        temporary.unlink(missing_ok=True)
        self.source_logger.info(
            f"Building HUB batch {batch_number}: {height}x{width} px, {len(blocks)} active blocks, {len(times)} time labels."
        )
        try:
            with Dataset(temporary, "w", format="NETCDF4") as cube:
                cube.createDimension("t", len(times))
                cube.createDimension("y", height)
                cube.createDimension("x", width)
                time_variable = cube.createVariable("t", "f8", ("t",))
                time_variable.units = "seconds since 1970-01-01 00:00:00"
                time_variable.calendar = "standard"
                time_variable[:] = date2num(
                    [when.tz_convert(None).to_pydatetime() for when in times], time_variable.units, time_variable.calendar
                )
                x_variable = cube.createVariable("x", "f8", ("x",))
                x_variable[:] = transform.c + transform.a * (np.arange(width) + 0.5)
                y_variable = cube.createVariable("y", "f8", ("y",))
                y_variable[:] = transform.f + transform.e * (np.arange(height) + 0.5)
                spatial_ref = cube.createVariable("spatial_ref", "i4")
                spatial_ref.setncatts(
                    {**crs.to_cf(), "spatial_ref": crs.to_wkt(), "GeoTransform": " ".join(str(v) for v in transform.to_gdal())}
                )
                chunks = (1, min(512, height), min(512, width))
                variables = {}
                for name in self._cube_sensor_variables():
                    variable = cube.createVariable(
                        name, "f4", ("t", "y", "x"), zlib=True, complevel=1, chunksizes=chunks, fill_value=np.float32(np.nan)
                    )
                    variable.grid_mapping = "spatial_ref"
                    variables[name] = variable
                cube.source = f"HUB ARD {', '.join(sensor_variables)} via {self.HUB_PROCESSING_STRATEGY}"
                footprints = project_footprints(observations)
                self._fill_hub_cube(resolver, variables, transform, groups, blocks, sensor_variables, footprints)
            temporary.replace(netcdf_path)
        finally:
            temporary.unlink(missing_ok=True)
        return [
            {
                "batch_number": batch_number,
                "time_label": when.isoformat(),
                "collection": collection,
                "observations": sum(1 for o in group if o.collection == collection),
                "items": sum(len(o.items) for o in group if o.collection == collection),
                "parcels": len(parcels),
                "active_blocks": len(blocks),
            }
            for when, group in zip(times, groups)
            for collection in sensor_variables
        ]

    def _run_hub_batches(self, cube_dir: Path):
        """Yield each committed ``(batch_number, parcels, netcdf_path)`` cube as it becomes available.

        Requires :meth:`_resolve_hub_manifest`. Existing final cubes or saved
        batch statistics are reused through ``_batch_available``. The manifest
        and a coverage report are written to ``cube_dir``.
        """
        if self._hub_manifest is None:
            raise RuntimeError("Error: Resolve the HUB manifest before building cubes.")
        cube_dir = Path(cube_dir)
        cube_dir.mkdir(parents=True, exist_ok=True)
        self._hub_manifest.to_parquet(cube_dir / "hub_item_manifest.parquet", index=False)
        resolver = HubAssetResolver(self.hub_client_id)
        coverage = []
        for batch_number, parcels in self._hub_batches:
            netcdf_path = cube_dir / f"batch_{batch_number:05d}_monthly.nc"
            if self._batch_available(netcdf_path):
                self.source_logger.info(f"Using existing HUB batch {batch_number} cube {netcdf_path}.")
            else:
                coverage.extend(
                    self._write_hub_cube(resolver, batch_number, parcels, self._hub_observations[batch_number], netcdf_path)
                )
                self.source_logger.info(f"Cached HUB batch {batch_number} cube at {netcdf_path}.")
                # Persist coverage before yielding so later acquisition or local
                # failures cannot hide the diagnostics for completed cubes.
                pd.DataFrame(coverage).to_csv(cube_dir / "hub_coverage_report.csv", index=False)
            yield batch_number, parcels, netcdf_path
