"""HUB imagery previews of one parcel for the ML inspection map.

Previews are read on demand from the HUB Sentinel ARD COGs: only a small window around the
parcel, and only acquisitions recorded in the parcel statistics run's ``hub_item_manifest``
files, so the map shows the same items (and processing versions) as the statistics. Nothing is
written to disk: a byte-bounded memory cache keeps recent band windows, so switching layer or
returning to a date does not read HUB again, and the browser caches each image by its revision.

The previews show observed ARD pixels. They do not reproduce outlier removal, gap filling or
temporal reduction applied to the parcel statistics. RGB and the Sentinel-1 false colour keep
clouds (only source nodata is transparent); index layers apply the statistics validity rule.

Planet monthly basemaps (``PL_BSM``) are extra reference layers of every HUB analysis. They
are not a statistics source, so their quads are searched over the run's months rather than
taken from the manifests: true colour from ``quad_bandmath`` bands 6, 4, 2 and the delivered
NDVI from band 9, at the basemap's own resolution.

Error messages raised as ``ValueError`` are safe to show in the browser. Catalog, network and
asset errors propagate unchanged for the caller to report without their text.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from . import palettes
from .credentials import require_configuration

# Change when the pixels of an unchanged acquisition would render differently (browser cache key).
RENDERER_VERSION = "2"  # bump when the image content or extent changes, e.g. the preview buffer
SENTINEL2 = "ARD_S2"
SENTINEL1 = "ARD_S1"
PLANET = "PL_BSM"
SENSORS = {SENTINEL2: "S2", SENTINEL1: "S1", PLANET: "PL"}
RGB_LAYER = "RGB"
S1_FALSE_COLOUR_LAYER = "S1_FALSE_COLOUR"
PLANET_RGB_LAYER = "PL_RGB"
PLANET_NDVI_LAYER = "PL_NDVI"
MANIFEST_NAME = "hub_item_manifest.parquet"
READ_WORKERS = 8
WINDOW_CACHE_BYTES = 256 * 1024**2
WINDOW_CACHE_ENTRIES = 128


@dataclass(frozen=True)
class PreviewLayer:
    """One selectable preview layer; ``legend`` is ``None`` for colour composites."""

    name: str
    label: str
    sensor: Literal["S2", "S1", "PL"]
    kind: Literal["rgb", "index"]
    legend: dict | None

    def to_dict(self) -> dict:
        """Return the JSON form of the layer."""
        return asdict(self)


@dataclass(frozen=True)
class PreviewAcquisition:
    """One acquisition covering the parcel; ``revision`` changes whenever its image would."""

    id: str
    datetime: str
    sensor: Literal["S2", "S1", "PL"]
    cloud_cover: float | None
    revision: str

    def to_dict(self) -> dict:
        """Return the JSON form of the acquisition."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Layers from the run configuration
# ---------------------------------------------------------------------------
def _names(processing: dict, key: str) -> tuple[str, ...]:
    """Return a configured name list upper-cased and de-duplicated in order."""
    return tuple(dict.fromkeys(str(name).strip().upper() for name in processing.get(key) or []))


def _uses_sentinel2(processing: dict) -> bool:
    """Return whether the run read Sentinel-2; omitted ``sentinel2_bands`` means the default bands."""
    bands = processing.get("sentinel2_bands")
    return bands is None or bool(bands) or bool(_names(processing, "sentinel2_indices"))


def _uses_sentinel1(processing: dict) -> bool:
    """Return whether the run read Sentinel-1."""
    return bool(processing.get("sentinel1_bands")) or bool(_names(processing, "sentinel1_indices"))


def preview_layers(processing: dict) -> list[PreviewLayer]:
    """Return the layers of a HUB run: a composite per sensor, then the run's configured indices."""
    layers = []
    if _uses_sentinel2(processing):
        layers.append(PreviewLayer(RGB_LAYER, "True colour (RGB)", "S2", "rgb", None))
        layers.extend(
            PreviewLayer(name, name, "S2", "index", palettes.legend(name)) for name in _names(processing, "sentinel2_indices")
        )
    if _uses_sentinel1(processing):
        layers.append(PreviewLayer(S1_FALSE_COLOUR_LAYER, "Sentinel-1 false colour (VV, VH, VV/VH)", "S1", "rgb", None))
        layers.extend(
            PreviewLayer(name, name, "S1", "index", palettes.legend(name)) for name in _names(processing, "sentinel1_indices")
        )
    # Planet basemaps are reference imagery, offered whatever sensors and indices the run used.
    layers.append(PreviewLayer(PLANET_RGB_LAYER, "Planet true colour (bands 6, 4, 2)", "PL", "rgb", None))
    layers.append(PreviewLayer(PLANET_NDVI_LAYER, "Planet NDVI (band 9)", "PL", "index", palettes.legend("NDVI")))
    return layers


def _layer(processing: dict, name: str) -> PreviewLayer:
    """Return one configured layer or refuse an unknown name."""
    layer = next((layer for layer in preview_layers(processing) if layer.name == name), None)
    if layer is None:
        raise ValueError(f"Error: {name} is not a preview layer of this parcel statistics run.")
    return layer


def _sensor_bands(processing: dict) -> dict[str, tuple[str, ...]]:
    """Return the bands read per collection: the composite bands plus every configured index's bands."""
    from data_preparation.parcel_stats.core.base import ParcelStatsBase

    from .preview_raster import PLANET_BASEMAP_BANDS, SENTINEL1_BANDS, SENTINEL2_RGB_BANDS

    bands = {}
    if _uses_sentinel2(processing):
        needed = [*SENTINEL2_RGB_BANDS]
        for name in _names(processing, "sentinel2_indices"):
            needed.extend(ParcelStatsBase.SUPPORTED_SENTINEL2_INDICES[name])
        bands[SENTINEL2] = tuple(dict.fromkeys(needed))
    if _uses_sentinel1(processing):
        bands[SENTINEL1] = SENTINEL1_BANDS
    bands[PLANET] = tuple(PLANET_BASEMAP_BANDS)
    return bands


# ---------------------------------------------------------------------------
# Acquisitions: catalog items limited to the run's manifest
# ---------------------------------------------------------------------------
def _manifest_paths(output_dir: Path) -> list[Path]:
    """Return the run's partition manifests, refusing a run without any."""
    paths = sorted(Path(output_dir).glob(f"partition_*/{MANIFEST_NAME}"))
    if not paths:
        raise ValueError("Error: This parcel statistics run has no HUB item inventory.")
    return paths


def _manifest_signature(output_dir: Path) -> tuple:
    """Return a cheap identity of the manifest files, so a re-run invalidates cached acquisitions."""
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in _manifest_paths(output_dir))


def _text(value) -> str | None:
    """Return a manifest or item value as text, with missing values as ``None``."""
    import pandas as pd

    return None if value is None or pd.isna(value) else str(value)


def _used_items(output_dir: Path) -> dict[tuple[str, str], dict]:
    """Return (collection, item ID) → identity recorded by the run, across all partitions and batches.

    The identity holds the processing datetime/version and every recorded asset checksum.
    """
    import pandas as pd

    columns = ["collection", "item_id", "processing_datetime", "processing_version", "asset_key", "asset_checksum"]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in _manifest_paths(output_dir)], ignore_index=True)
    used: dict[tuple[str, str], dict] = {}
    for row in frame.drop_duplicates(["collection", "item_id", "asset_key"]).itertuples():
        identity = used.setdefault(
            (row.collection, row.item_id),
            {"processing": (_text(row.processing_datetime), _text(row.processing_version)), "checksums": {}},
        )
        if _text(row.asset_checksum) is not None:
            identity["checksums"][row.asset_key] = _text(row.asset_checksum)
    return used


def _unchanged(item, used: dict) -> bool:
    """Return whether the run recorded this item and its processing and recorded asset checksums still match."""
    identity = used.get((item.collection, item.item_id))
    if identity is None or identity["processing"] != (item.processing_datetime, item.processing_version):
        return False
    return all(key in item.assets and item.assets[key].checksum == value for key, value in identity["checksums"].items())


def _criteria(processing: dict) -> tuple:
    """Return the run's hashable search criteria: dates, Sentinel-2 cloud limit and Sentinel-1 orbit."""
    from data_preparation.parcel_stats.core.base import ParcelStatsBase

    cloud = processing.get("sentinel2_max_scene_cloud_cover", ParcelStatsBase.SENTINEL2_MAX_SCENE_CLOUD_COVER)
    orbit = str(processing.get("sentinel1_orbit_direction") or "BOTH").upper()
    return processing["start_date"], processing["end_date"], cloud, orbit


def _catalog():
    """Return a catalog client configured only from the server environment."""
    require_configuration()
    from data_preparation.sources.hub.catalog_library import CatalogSearchUtils

    client = os.getenv("CATALOG_CLIENT_ID", "internal")
    return CatalogSearchUtils(catalog_endpoint=os.environ["CATALOG_URL"], client_id=client, verbose=False)


def _basemap_items(catalog, search, start, end) -> list:
    """Search the basemap quads of every month the run overlaps; basemaps are dated by their first day."""
    import pandas as pd

    from data_preparation.sources.hub import ard_items

    from .preview_raster import PLANET_BASEMAP_ASSET

    first_month = pd.Timestamp(start).to_period("M").to_timestamp()
    items = ard_items.search_ard_items(catalog, PLANET, search, first_month, end)
    return ard_items.filter_items(items, required_assets=(PLANET_BASEMAP_ASSET,))


def _collection_items(catalog, collection, bands, search, criteria, used):
    """Search one collection around the parcel and keep the run's items with unchanged processing.

    Planet basemaps are not a statistics source and so are absent from the manifests: every
    quad of the run's months is kept.
    """
    from data_preparation.sources.hub import ard_items

    from .preview_raster import item_keys

    start, end, cloud, orbit = criteria
    if collection == PLANET:
        return _basemap_items(catalog, search, start, end)
    max_cloud = cloud if collection == SENTINEL2 else None
    items = ard_items.search_ard_items(catalog, collection, search, start, end, max_cloud_cover=max_cloud)
    items = ard_items.filter_items(
        items, required_assets=item_keys(collection, bands), max_cloud_cover=max_cloud,
        orbit_direction=orbit if collection == SENTINEL1 else "BOTH",
    )  # fmt: skip
    return [item for item in items if _unchanged(item, used)]


@lru_cache(maxsize=64)
def _observations(output_dir: str, signature: tuple, criteria: tuple, bands: tuple, geometry_wkb: bytes) -> tuple:
    """Return the run's grouped observations whose footprints meet the parcel window (cached)."""
    import geopandas as gpd
    from shapely import wkb
    from shapely.geometry import box, mapping

    from data_preparation.sources.hub import ard_items
    from data_preparation.sources.hub.cubes import HUB_GRID_EPSG

    from .preview_raster import parcel_source_bounds

    window = box(*parcel_source_bounds(wkb.loads(geometry_wkb)))
    region = gpd.GeoSeries([window], crs=HUB_GRID_EPSG).to_crs(4326).iloc[0]
    catalog, used = _catalog(), _used_items(Path(output_dir))
    items = [
        item
        for collection, names in bands
        for item in _collection_items(catalog, collection, names, mapping(box(*region.bounds)), criteria, used)
    ]
    return tuple(ard_items.intersecting_observations(ard_items.group_observations(items), region))


def _parcel_geometry(parcel: dict):
    """Return the shapely geometry of the WGS84 GeoJSON parcel feature."""
    from shapely.geometry import shape

    return shape(parcel["geometry"])


def _parcel_observations(output_dir: Path, processing: dict, geometry) -> tuple:
    """Return the cached observations of one parcel for the run's configuration."""
    bands = tuple(sorted(_sensor_bands(processing).items()))
    return _observations(str(output_dir), _manifest_signature(output_dir), _criteria(processing), bands, geometry.wkb)


def _revision(observation, geometry) -> str:
    """Return a public digest of renderer, parcel geometry and the observation's item versions and asset checksums."""
    items = [
        [item.collection, item.item_id, item.processing_datetime, item.processing_version,
         sorted((key, asset.checksum) for key, asset in item.assets.items())]
        for item in observation.items
    ]  # fmt: skip
    content = json.dumps([RENDERER_VERSION, observation.key, items, geometry.wkb.hex()])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _acquisition(observation, geometry) -> PreviewAcquisition:
    """Describe one observation; the cloud cover is that of its highest-priority item."""
    return PreviewAcquisition(
        id=observation.key,
        datetime=observation.datetime.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        sensor=SENSORS[observation.collection],
        cloud_cover=observation.items[0].cloud_cover,
        revision=_revision(observation, geometry),
    )


def parcel_acquisitions(output_dir: Path, processing: dict, parcel: dict) -> list[PreviewAcquisition]:
    """List the run's acquisitions covering the parcel, oldest first."""
    geometry = _parcel_geometry(parcel)
    return [_acquisition(observation, geometry) for observation in _parcel_observations(output_dir, processing, geometry)]


def parcel_preview_bounds(parcel: dict) -> list[list[float]]:
    """Return ``[[south, west], [north, east]]`` of every preview image of the parcel."""
    from .preview_raster import display_grid, leaflet_bounds

    return leaflet_bounds(display_grid(_parcel_geometry(parcel)))


# ---------------------------------------------------------------------------
# Band windows: remote reads with a byte-bounded memory cache
# ---------------------------------------------------------------------------
class WindowCache:
    """Least-recently-used cache bounded by total array bytes and by entry count."""

    def __init__(self, max_bytes: int, max_entries: int) -> None:
        """Create an empty cache."""
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._entries: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        """Return a cached value and mark it recently used, else ``None``."""
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key][0]

    def put(self, key, value, size: int) -> None:
        """Store a value; values larger than the whole cache are not kept."""
        with self._lock:
            if key in self._entries:
                self._bytes -= self._entries.pop(key)[1]
            if size > self.max_bytes:
                return
            self._entries[key] = (value, size)
            self._bytes += size
            while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
                self._bytes -= self._entries.popitem(last=False)[1][1]


_WINDOWS = WindowCache(WINDOW_CACHE_BYTES, WINDOW_CACHE_ENTRIES)
_POOL = ThreadPoolExecutor(max_workers=READ_WORKERS, thread_name_prefix="hub-preview")
_RESOLVERS: dict = {}
_RESOLVER_LOCK = threading.Lock()


def _resolver():
    """Return the shared asset resolver, which keeps the bearer token and signed URLs in memory only."""
    from data_preparation.sources.hub.assets import HubAssetResolver

    client = os.getenv("CATALOG_CLIENT_ID", "internal")
    with _RESOLVER_LOCK:
        if client not in _RESOLVERS:
            _RESOLVERS[client] = HubAssetResolver(client)
        return _RESOLVERS[client]


def _windows(observation, bands: tuple[str, ...], geometry, revision: str) -> list:
    """Return the observation's item windows around the parcel, reading HUB only on a cache miss."""
    from .preview_raster import display_grid, parcel_source_bounds, read_basemap_window, read_item_window

    key = (revision, bands)
    cached = _WINDOWS.get(key)
    if cached is not None:
        return cached
    resolver = _resolver()
    if observation.collection == PLANET:
        # Basemap quads are Web Mercator rasters: read exactly the display extent, one asset per quad.
        bounds = display_grid(geometry).mercator_bounds()
        windows = [window for item in observation.items if (window := read_basemap_window(resolver, item, bounds))]
    else:
        bounds = parcel_source_bounds(geometry)
        # Items are read one after another; their assets share the bounded pool of remote reads.
        windows = [window for item in observation.items if (window := read_item_window(resolver, item, bands, bounds, _POOL))]
    _WINDOWS.put(key, windows, sum(window.nbytes for window in windows))
    return windows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _image(layer: PreviewLayer, windows: list, grid):
    """Return the RGBA image of one layer from an observation's item windows."""
    from data_preparation.parcel_stats.core.base import ParcelStatsBase

    from .preview_raster import PLANET_RGB_BANDS, SENTINEL1_BANDS, SENTINEL2_RGB_BANDS, index_values, mosaic

    if layer.name == PLANET_RGB_LAYER:
        bands = mosaic(windows, PLANET_RGB_BANDS, grid, masked=False)
        return palettes.basemap_true_colour(*(bands[name] for name in PLANET_RGB_BANDS))
    if layer.name == PLANET_NDVI_LAYER:
        # The basemap's own NDVI band is shown as delivered, on the Sentinel-2 NDVI colour scale.
        return palettes.colour_index(mosaic(windows, ("NDVI",), grid, masked=False)["NDVI"], "NDVI")
    if layer.name == RGB_LAYER:
        bands = mosaic(windows, SENTINEL2_RGB_BANDS, grid, masked=False)
        return palettes.true_colour(*(bands[name] for name in SENTINEL2_RGB_BANDS))
    if layer.name == S1_FALSE_COLOUR_LAYER:
        bands = mosaic(windows, SENTINEL1_BANDS, grid, masked=False)
        return palettes.sentinel1_false_colour(bands["VV"], bands["VH"])
    sources = {**ParcelStatsBase.SUPPORTED_SENTINEL2_INDICES, **ParcelStatsBase.SUPPORTED_SENTINEL1_INDICES}[layer.name]
    bands = mosaic(windows, tuple(dict.fromkeys(sources)), grid, masked=True)
    return palettes.colour_index(index_values(layer.name, bands), layer.name)


def _png(image) -> bytes:
    """Encode an RGBA ``uint8`` array as PNG."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def _observation(observations: tuple, acquisition_id: str, layer: PreviewLayer):
    """Return the requested observation, refusing unknown IDs and layers of the other sensor."""
    observation = next((item for item in observations if item.key == acquisition_id), None)
    if observation is None:
        raise ValueError("Error: This acquisition does not cover the parcel in this parcel statistics run.")
    if SENSORS[observation.collection] != layer.sensor:
        raise ValueError(f"Error: {layer.label} is not available for a {SENSORS[observation.collection]} acquisition.")
    return observation


def render_preview(output_dir: Path, processing: dict, parcel: dict, acquisition_id: str, layer: str) -> bytes:
    """Return the PNG of one layer of one acquisition, on the grid of :func:`parcel_preview_bounds`."""
    from .preview_raster import display_grid

    chosen = _layer(processing, layer)
    geometry = _parcel_geometry(parcel)
    observation = _observation(_parcel_observations(output_dir, processing, geometry), acquisition_id, chosen)
    bands = dict(_sensor_bands(processing))[observation.collection]
    windows = _windows(observation, bands, geometry, _revision(observation, geometry))
    image = _image(chosen, windows, display_grid(geometry))
    if not image[..., 3].any():
        raise ValueError(f"Error: {chosen.label} has no valid pixels over this parcel on this date.")
    return _png(image)
