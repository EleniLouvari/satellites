"""Parcel display grid, HUB window reads and pixel layers for inspection previews.

The display grid depends only on the parcel: its EPSG:2100 bounds plus a fixed buffer, as
a north-up Web Mercator (EPSG:3857) grid with pixels four times finer than the 10 m source.
Every source item is warped onto that grid with nearest-neighbour resampling, so each source
pixel keeps its true position and shape. Leaflet places a Web Mercator image exactly from
its corner coordinates. Oversampling adds no detail; it only keeps source pixel edges sharp.
Planet basemap quads (about 4.8 m, EPSG:3857) are warped onto the same grid in the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from affine import Affine

from data_preparation.parcel_stats.core.base import ParcelStatsBase
from data_preparation.parcel_stats.core.parcel_statistics.checkpoints import ParcelStatisticsCheckpointMixin
from data_preparation.sources.hub import ard_items, cubes
from data_preparation.sources.hub.assets import read_asset

SOURCE_RESOLUTION_METRES = cubes.HUB_RESOLUTION_METRES
BUFFER_METRES = 500.0
DISPLAY_OVERSAMPLING = 4
MAX_SOURCE_PIXELS = 512
DISPLAY_EPSG = 3857
SENTINEL2_RGB_BANDS = ("B04", "B03", "B02")
SENTINEL1_BANDS = ("VV", "VH")
# Planet monthly basemaps: one multi-band asset; preview name -> 1-based band index of ``quad_bandmath``.
PLANET_BASEMAP_ASSET = "quad_bandmath"
PLANET_BASEMAP_BANDS = {"RED": 6, "GREEN": 4, "BLUE": 2, "NDVI": 9}
PLANET_BASEMAP_ALPHA_BAND = 10  # 0 outside the mosaic's data area, 255 inside
PLANET_RGB_BANDS = ("RED", "GREEN", "BLUE")


@dataclass(frozen=True)
class DisplayGrid:
    """North-up EPSG:3857 grid of one parcel preview."""

    transform: Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)``."""
        return self.height, self.width

    def mercator_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(left, bottom, right, top)`` in EPSG:3857."""
        left, top = self.transform.c, self.transform.f
        return left, top + self.transform.e * self.height, left + self.transform.a * self.width, top


@dataclass(frozen=True)
class ItemWindow:
    """Bands of one item read around the parcel, with nodata as NaN, and its statistics validity mask."""

    bands: dict[str, np.ndarray]
    valid: np.ndarray
    transform: Affine
    crs: object

    @property
    def nbytes(self) -> int:
        """Return the memory held by the arrays."""
        return sum(array.nbytes for array in self.bands.values()) + self.valid.nbytes


class _IndexAlgebra(ParcelStatisticsCheckpointMixin):
    """Expose the statistics index formulas without building a statistics pipeline."""

    OPTICAL_BAND_ROLES = ParcelStatsBase.OPTICAL_BAND_ROLES


INDEX_ALGEBRA = _IndexAlgebra()


def parcel_source_bounds(geometry) -> tuple[float, float, float, float]:
    """Return the buffered EPSG:2100 bounds of a WGS84 parcel geometry, refusing an oversized parcel."""
    import geopandas as gpd

    minx, miny, maxx, maxy = gpd.GeoSeries([geometry], crs=4326).to_crs(cubes.HUB_GRID_EPSG).total_bounds
    bounds = (minx - BUFFER_METRES, miny - BUFFER_METRES, maxx + BUFFER_METRES, maxy + BUFFER_METRES)
    limit = MAX_SOURCE_PIXELS * SOURCE_RESOLUTION_METRES
    if bounds[2] - bounds[0] > limit or bounds[3] - bounds[1] > limit:
        raise ValueError(f"Error: The parcel is too large for an image preview (more than {limit / 1000:g} km across).")
    return bounds


def display_grid(geometry) -> DisplayGrid:
    """Return the parcel's preview grid; bounds and rendering both use this single calculation."""
    from pyproj import Transformer

    source = parcel_source_bounds(geometry)
    to_mercator = Transformer.from_crs(cubes.HUB_GRID_EPSG, DISPLAY_EPSG, always_xy=True)
    left, bottom, right, top = to_mercator.transform_bounds(*source, densify_pts=21)
    # Web Mercator metres are ground metres divided by cos(latitude).
    latitude = math.radians(Transformer.from_crs(DISPLAY_EPSG, 4326, always_xy=True).transform(0, (bottom + top) / 2)[1])
    pixel = SOURCE_RESOLUTION_METRES / DISPLAY_OVERSAMPLING / math.cos(latitude)
    x0, y0 = math.floor(left / pixel) * pixel, math.ceil(top / pixel) * pixel
    width, height = math.ceil((right - x0) / pixel), math.ceil((y0 - bottom) / pixel)
    return DisplayGrid(Affine(pixel, 0.0, x0, 0.0, -pixel, y0), width, height)


def leaflet_bounds(grid: DisplayGrid) -> list[list[float]]:
    """Return ``[[south, west], [north, east]]`` of the grid edges for ``L.imageOverlay``."""
    from pyproj import Transformer

    left, bottom, right, top = grid.mercator_bounds()
    to_wgs84 = Transformer.from_crs(DISPLAY_EPSG, 4326, always_xy=True)
    west, south = to_wgs84.transform(left, bottom)
    east, north = to_wgs84.transform(right, top)
    return [[south, west], [north, east]]


def item_keys(collection: str, bands: tuple[str, ...]) -> tuple[str, ...]:
    """Return every asset read for an item: the bands plus the masks of its sensor."""
    masks = ("SCL", "MASK") if collection == ard_items.SENTINEL2_ARD_COLLECTION else ("MASK",)
    return (*bands, *masks)


def _read_assets(resolver, item, keys, window, pool) -> tuple[dict[str, np.ndarray], Affine, object]:
    """Read one window of every asset concurrently and require one shared pixel grid."""
    reads = list(pool.map(lambda key: read_asset(resolver, item.assets[key].href, lambda d: cubes._read_window(d, window)), keys))
    _require_shared_native_grid(item.item_id, reads)
    transform, crs, _ = reads[0]
    return {key: array for key, (_t, _c, array) in zip(keys, reads, strict=True)}, transform, crs


def _require_shared_native_grid(item_id: str, reads) -> None:
    """Refuse assets whose transform, CRS or shape differ, or whose pixels are not north-up 10 m."""
    grids = {(transform, str(crs), array.shape) for transform, crs, array in reads}
    if len(grids) != 1:
        raise ValueError(f"Error: Assets of HUB item {item_id} do not share one pixel grid.")
    transform = reads[0][0]
    native = math.isclose(transform.a, SOURCE_RESOLUTION_METRES) and math.isclose(-transform.e, SOURCE_RESOLUTION_METRES)
    if not native or transform.b != 0 or transform.d != 0:
        raise ValueError(f"Error: HUB item {item_id} is not on a north-up {SOURCE_RESOLUTION_METRES:g} m grid.")


def _validity(collection: str, raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(no_data, invalid)``: missing source pixels, and pixels the statistics would exclude."""
    if collection == ard_items.SENTINEL2_ARD_COLLECTION:
        no_data = (raw["MASK"] == 0) | (raw["SCL"] == cubes.SCL_NODATA)
        invalid = cubes.sentinel2_invalid(
            raw["SCL"], raw["MASK"], ParcelStatsBase.SENTINEL2_INVALID_SCL_CLASSES, ParcelStatsBase.SENTINEL2_BUFFERED_SCL_CLASSES
        )
        return no_data, invalid
    invalid = cubes.sentinel1_invalid(raw["MASK"])
    return invalid, invalid


def read_item_window(resolver, item, bands: tuple[str, ...], bounds, pool) -> ItemWindow | None:
    """Read an item's bands around EPSG:2100 ``bounds``; ``None`` when the item misses them.

    Band values keep clouds (only source nodata becomes NaN); ``valid`` holds the statistics
    cloud/shadow or layover rule, applied later to index layers only.
    """
    keys = item_keys(item.collection, bands)
    grid = cubes._metadata_grid(item.assets[keys[0]])
    if grid is None:
        # Without STAC grid metadata, open the asset once for its grid (the dataset closes afterwards).
        grid = read_asset(
            resolver,
            item.assets[keys[0]].href,
            lambda dataset: SimpleNamespace(transform=dataset.transform, width=dataset.width, height=dataset.height),
        )
    window = cubes._source_window(grid, bounds, cubes.READ_HALO_PIXELS)
    if window is None:
        return None
    raw, transform, crs = _read_assets(resolver, item, keys, window, pool)
    no_data, invalid = _validity(item.collection, raw)
    sentinel1 = item.collection == ard_items.SENTINEL1_ARD_COLLECTION
    cleaned = {name: cubes.clean_values(raw[name], item.assets[name].nodata, no_data, positive_only=sentinel1) for name in bands}
    return ItemWindow(cleaned, ~invalid, transform, crs)


def _read_basemap_bands(dataset, bounds) -> ItemWindow | None:
    """Read the preview bands of an open basemap quad around EPSG:3857 ``bounds``; pixels without data are NaN."""
    from rasterio.warp import transform_bounds

    window = cubes._source_window(dataset, transform_bounds(DISPLAY_EPSG, dataset.crs, *bounds), 1)
    if window is None:
        return None
    indexes = [*PLANET_BASEMAP_BANDS.values(), PLANET_BASEMAP_ALPHA_BAND]
    # Pixels beyond the quad edge read as alpha 0, so they become NaN like the mosaic's own gaps.
    values = dataset.read(indexes, window=window, boundless=True, fill_value=0).astype("float32")
    *values, alpha = values
    no_data = alpha == 0
    bands = {
        name: np.where(no_data | ~np.isfinite(band), np.nan, band)
        for name, band in zip(PLANET_BASEMAP_BANDS, values, strict=True)
    }
    # Basemaps are cloud-reduced monthly mosaics without a pixel quality mask.
    return ItemWindow(bands, np.ones(alpha.shape, dtype=bool), dataset.window_transform(window), dataset.crs)


def read_basemap_window(resolver, item, bounds) -> ItemWindow | None:
    """Read a Planet basemap quad around EPSG:3857 ``bounds``; ``None`` when the quad misses them.

    The quad is opened for its own grid and CRS, because item-level ``proj:code`` metadata
    can describe the footprint rather than the raster.
    """
    href = item.assets[PLANET_BASEMAP_ASSET].href
    return read_asset(resolver, href, lambda dataset: _read_basemap_bands(dataset, bounds))


def _warp(values: np.ndarray, window: ItemWindow, grid: DisplayGrid) -> np.ndarray:
    """Warp one source array onto the display grid with nearest-neighbour resampling."""
    from pyproj import CRS
    from rasterio.warp import Resampling, reproject

    target = np.full(grid.shape, np.nan, dtype="float32")
    reproject(
        source=values.astype("float32", copy=False), destination=target,
        src_transform=window.transform, src_crs=window.crs, src_nodata=np.nan,
        dst_transform=grid.transform, dst_crs=CRS.from_epsg(DISPLAY_EPSG), dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )  # fmt: skip
    return target


def mosaic(windows: list[ItemWindow], bands: tuple[str, ...], grid: DisplayGrid, *, masked: bool) -> dict[str, np.ndarray]:
    """Warp and mosaic one acquisition's items, first valid whole pixel wins, in mosaic priority order.

    ``masked`` applies the statistics validity rule (index layers); otherwise clouds are kept.
    """
    layers = []
    for window in windows:
        bands_on_grid = {}
        for name in bands:
            values = np.where(window.valid, window.bands[name], np.nan) if masked else window.bands[name]
            bands_on_grid[name] = _warp(values, window, grid)
        layers.append(bands_on_grid)
    return cubes.mosaic_first_valid(layers, bands, grid.shape)


def index_values(name: str, bands: dict[str, np.ndarray]) -> np.ndarray:
    """Calculate one configured index with the same formulas as the parcel statistics."""
    if name in ParcelStatsBase.SUPPORTED_SENTINEL1_INDICES:
        return INDEX_ALGEBRA._calculate_local_sentinel1_index(bands, name)
    return INDEX_ALGEBRA._calculate_local_optical_index(bands, name)
