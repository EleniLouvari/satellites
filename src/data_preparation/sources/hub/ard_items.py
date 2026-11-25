"""Discover HUB Sentinel ARD items and group them into sensor observations.

Catalog items are normalised into flat records so that later stages never depend
on the inconsistent spellings found in deployed metadata (for example
``proj:code`` as ``"EPSG:2100"`` or ``2100`` and lowercase/uppercase platforms).
Items that belong to one acquisition are grouped into an :class:`Observation`:
Sentinel-2 by datatake, Sentinel-1 by orbit pass. The manifest built from the
records has a digest that changes whenever an item or asset is reprocessed; it
records only query-free asset locations, never tokens or signed URLs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from shapely.errors import ShapelyError
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

SENTINEL2_ARD_COLLECTION = "ARD_S2"
SENTINEL1_ARD_COLLECTION = "ARD_S1"
SUPPORTED_COLLECTIONS = (SENTINEL2_ARD_COLLECTION, SENTINEL1_ARD_COLLECTION)
# Planet monthly basemap quads: searchable for inspection previews, not a parcel statistics source.
PLANET_BASEMAP_COLLECTION = "PL_BSM"
SEARCHABLE_COLLECTIONS = (*SUPPORTED_COLLECTIONS, PLANET_BASEMAP_COLLECTION)
MISSING = "missing"


@dataclass(frozen=True)
class ArdAsset:
    """Stable identity and raster metadata of one item asset."""

    key: str
    href: str
    checksum: str | None
    nodata: float | None
    # Affine coefficients (a, b, c, d, e, f) from ``proj:transform``; used only to pick a grid lattice.
    transform: tuple[float, ...] | None = None
    # Raster ``(rows, columns)`` from ``proj:shape``; with ``transform`` it locates windows without opening.
    shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class ArdItem:
    """Normalised catalog item used for filtering, grouping and the manifest."""

    collection: str
    item_id: str
    datetime: pd.Timestamp
    platform: str | None
    orbit_state: str | None
    relative_orbit: int | None
    absolute_orbit: int | None
    datatake_id: str | None
    cloud_cover: float | None
    nodata_fraction: float | None
    processing_datetime: str | None
    processing_version: str | None
    epsg: int | None
    assets: dict[str, ArdAsset] = field(default_factory=dict, compare=False)
    # WGS84 data footprint from the STAC geometry; ``None`` when absent or invalid.
    footprint: BaseGeometry | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Observation:
    """Items of one acquisition, ordered by mosaic priority (best first)."""

    collection: str
    key: str
    datetime: pd.Timestamp
    items: tuple[ArdItem, ...]


def _optional_int(value: Any) -> int | None:
    """Return an integer for numeric metadata, or ``None`` when absent/invalid."""
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    """Return a float for numeric metadata, or ``None`` when absent/invalid."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any, *, case: str | None = None) -> str | None:
    """Return stripped text with optional case normalisation."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    return text.upper() if case == "upper" else text.lower() if case == "lower" else text


def _parse_epsg(value: Any) -> int | None:
    """Accept ``2100``, ``"2100"`` and ``"EPSG:2100"`` projection codes."""
    if value is None:
        return None
    text = str(value).strip().upper().removeprefix("EPSG:")
    return _optional_int(text)


def _numbers(raw: Any, length: int, convert) -> tuple | None:
    """Return the first ``length`` converted numbers of a list, or ``None`` if any is missing."""
    if not isinstance(raw, list) or len(raw) < length:
        return None
    values = tuple(convert(value) for value in raw[:length])
    return None if None in values else values


def _band_nodata(asset: dict[str, Any]) -> float | None:
    """Return the first ``raster:bands`` nodata value, if declared."""
    bands = asset.get("raster:bands") or [{}]
    return _optional_float(bands[0].get("nodata")) if isinstance(bands[0], dict) else None


def _parse_asset(key: str, asset: dict[str, Any]) -> ArdAsset | None:
    """Return a data asset with an href, or ``None`` for unusable entries."""
    href = asset.get("href")
    if not isinstance(href, str) or not href.strip():
        return None
    return ArdAsset(
        key=key,
        href=href.strip(),
        checksum=_optional_text(asset.get("hub:checksum") or asset.get("file:checksum")),
        nodata=_band_nodata(asset),
        transform=_numbers(asset.get("proj:transform"), 6, _optional_float),
        shape=_numbers(asset.get("proj:shape"), 2, _optional_int),
    )


def normalise_item(feature: dict[str, Any], collection: str) -> ArdItem:
    """Convert one STAC feature into an :class:`ArdItem`.

    Raises
    ------
    ValueError
        If the feature has no id or no parseable ``datetime``.

    """
    item_id = feature.get("id")
    properties = feature.get("properties") or {}
    timestamp = pd.to_datetime(properties.get("datetime"), utc=True, errors="coerce")
    if not item_id or pd.isna(timestamp):
        raise ValueError(f"Error: HUB item without id or datetime in {collection}: {item_id!r}.")
    assets = {}
    for key, asset in (feature.get("assets") or {}).items():
        parsed = _parse_asset(key, asset) if isinstance(asset, dict) else None
        if parsed is not None:
            assets[key] = parsed
    return ArdItem(
        collection=collection,
        item_id=str(item_id),
        datetime=timestamp,
        platform=_optional_text(properties.get("platform"), case="lower"),
        orbit_state=_optional_text(properties.get("sat:orbit_state"), case="upper"),
        relative_orbit=_optional_int(properties.get("sat:relative_orbit")),
        absolute_orbit=_optional_int(properties.get("sat:absolute_orbit")),
        datatake_id=_optional_text(properties.get("s2:datatake_id")),
        cloud_cover=_optional_float(properties.get("eo:cloud_cover")),
        nodata_fraction=_optional_float(properties.get("statistics:nodata_pixel")),
        processing_datetime=_optional_text(properties.get("processing:datetime")),
        processing_version=_optional_text(properties.get("processing:version")),
        epsg=_parse_epsg(properties.get("proj:code") or properties.get("proj:epsg")),
        assets=assets,
        footprint=_parse_footprint(feature.get("geometry")),
    )


def build_search_body(
    collection: str, geometry: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, *, max_cloud_cover: float | None = None
) -> dict[str, Any]:
    """Return a CQL2 POST search body for one collection, WGS84 geometry and inclusive day range.

    Every predicate is a HUB *queryable* (``datetime``, ``geometry``,
    ``eo:cloud_cover``), so the HUB stac-auth-proxy applies the whole filter in
    the catalog database. Proxy-side properties such as ``sat:orbit_state`` are
    deliberately left to :func:`filter_items`: mixing one into this ``and``
    would move the entire expression to slower response-side evaluation.
    ``max_cloud_cover`` excludes items without ``eo:cloud_cover`` server-side.
    """
    if collection not in SEARCHABLE_COLLECTIONS:
        raise ValueError(f"Error: Unsupported HUB collection {collection!r}; expected one of {SEARCHABLE_COLLECTIONS}.")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if last < first:
        raise ValueError("Error: The HUB search end date precedes the start date.")
    predicates = [
        {"op": ">=", "args": [{"property": "datetime"}, first.strftime("%Y-%m-%dT00:00:00Z")]},
        # The configured end date is inclusive: use a half-open bound at the start of the next day.
        {"op": "<", "args": [{"property": "datetime"}, (last.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")]},
        {"op": "s_intersects", "args": [{"property": "geometry"}, geometry]},
    ]
    if max_cloud_cover is not None:
        predicates.append({"op": "<=", "args": [{"property": "eo:cloud_cover"}, float(max_cloud_cover)]})
    return {"collections": [collection], "filter-lang": "cql2-json", "filter": {"op": "and", "args": predicates}}


def search_ard_items(
    catalog,
    collection: str,
    geometry: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    page_size: int = 200,
    max_cloud_cover: float | None = None,
) -> list[ArdItem]:
    """Search one collection with ``CatalogSearchUtils.fetch_features`` and normalise the results.

    Duplicate features returned by overlapping pages are removed by item id.
    """
    body = build_search_body(collection, geometry, start, end, max_cloud_cover=max_cloud_cover)
    features = catalog.fetch_features(search_body=body, limit=page_size, method="POST", page_retry_attempts=3)
    unique: dict[str, ArdItem] = {}
    for feature in features:
        item = normalise_item(feature, collection)
        unique.setdefault(item.item_id, item)
    return sorted(unique.values(), key=lambda item: (item.datetime, item.item_id))


def filter_items(
    items: Iterable[ArdItem],
    *,
    required_assets: Sequence[str],
    max_cloud_cover: float | None = None,
    orbit_direction: str = "BOTH",
) -> list[ArdItem]:
    """Keep items that carry every required asset and pass cloud/orbit filters.

    ``max_cloud_cover`` applies only to items that report ``eo:cloud_cover``;
    ``orbit_direction`` (``ASCENDING``, ``DESCENDING`` or ``BOTH``) only to
    items that report ``sat:orbit_state``. Items lacking the property are kept.
    """
    direction = orbit_direction.strip().upper()
    if direction not in {"ASCENDING", "DESCENDING", "BOTH"}:
        raise ValueError("Error: orbit_direction must be ASCENDING, DESCENDING, or BOTH.")
    return [item for item in items if _keep_item(item, required_assets, max_cloud_cover, direction)]


def _keep_item(item: ArdItem, required_assets: Sequence[str], max_cloud_cover: float | None, direction: str) -> bool:
    """Return whether one item has every asset and passes the cloud and orbit filters."""
    if any(key not in item.assets for key in required_assets):
        return False
    too_cloudy = max_cloud_cover is not None and item.cloud_cover is not None and item.cloud_cover > max_cloud_cover
    wrong_orbit = direction != "BOTH" and item.orbit_state is not None and item.orbit_state != direction
    return not (too_cloudy or wrong_orbit)


def _parse_footprint(geometry: Any) -> BaseGeometry | None:
    """Return the item footprint as a shapely geometry, or ``None`` when unusable."""
    if not isinstance(geometry, dict):
        return None
    try:
        footprint = shape(geometry)
    except (ShapelyError, KeyError, TypeError, ValueError):
        return None
    return footprint if not footprint.is_empty else None


def intersecting_observations(observations: Iterable[Observation], region: BaseGeometry) -> list[Observation]:
    """Keep only items whose footprint intersects ``region`` (WGS84); drop observations left empty.

    Items without a footprint are kept, because their coverage is unknown.
    Pruning saves remote reads for neighbouring tiles and orbits whose raster
    extent overlaps a batch while their valid data misses every parcel.
    """
    kept = []
    for observation in observations:
        items = tuple(item for item in observation.items if item.footprint is None or item.footprint.intersects(region))
        if items:
            kept.append(Observation(observation.collection, observation.key, observation.datetime, items))
    return kept


def observation_key(item: ArdItem) -> str:
    """Return the acquisition key that groups items into one mosaic.

    Sentinel-2 uses the datatake identifier, which also unites duplicate ESA
    product generations and neighbouring MGRS tiles of one pass. Sentinel-1
    uses platform and absolute orbit, uniting consecutive slices of one pass.
    Without positive evidence of a shared acquisition (missing datatake, or a
    missing platform or absolute orbit) each item stays its own observation.
    Planet basemap quads are monthly mosaics: every quad of one month forms one observation.
    """
    if item.collection == PLANET_BASEMAP_COLLECTION:
        return f"pl|{item.datetime.tz_convert('UTC'):%Y-%m}"
    if item.collection == SENTINEL2_ARD_COLLECTION and item.datatake_id:
        return f"s2|{item.datatake_id}"
    if item.collection == SENTINEL1_ARD_COLLECTION and item.platform and item.absolute_orbit is not None:
        return f"s1|{item.platform}|{item.absolute_orbit}"
    prefix = "s2" if item.collection == SENTINEL2_ARD_COLLECTION else "s1"
    return f"{prefix}|item|{item.item_id}"


def _mosaic_priority(item: ArdItem) -> tuple:
    """Sort key: least nodata, least cloud, newest processing, then item id."""
    missing_high = float("inf")
    processing = pd.to_datetime(item.processing_datetime, utc=True, errors="coerce")
    newest_first = -processing.value if not pd.isna(processing) else 0
    return (
        item.nodata_fraction if item.nodata_fraction is not None else missing_high,
        item.cloud_cover if item.cloud_cover is not None else missing_high,
        newest_first,
        item.item_id,
    )


def group_observations(items: Iterable[ArdItem]) -> list[Observation]:
    """Group items into observations ordered by time; items within one by mosaic priority.

    The observation timestamp is the earliest item datetime of the group, so
    one pass gets one time label even when its slices differ by seconds.
    """
    groups: dict[tuple[str, str], list[ArdItem]] = {}
    for item in items:
        groups.setdefault((item.collection, observation_key(item)), []).append(item)
    observations = [
        Observation(
            collection=collection,
            key=key,
            datetime=min(member.datetime for member in members),
            items=tuple(sorted(members, key=_mosaic_priority)),
        )
        for (collection, key), members in groups.items()
    ]
    return sorted(observations, key=lambda observation: (observation.datetime, observation.collection, observation.key))


def asset_location(href: str) -> str:
    """Return a stable asset location: scheme, host and path, without any query or fragment."""
    parsed = urlparse(href)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" if parsed.scheme else parsed.path


def manifest_frame(observations: Iterable[Observation], asset_keys: dict[str, Sequence[str]]) -> pd.DataFrame:
    """Return one manifest row per used (item, asset) with query-free asset locations and no credentials.

    Parameters
    ----------
    observations:
        Grouped observations to record.
    asset_keys:
        Collection id → asset keys that the cube builder reads.

    """
    rows = []
    for observation in observations:
        for priority, item in enumerate(observation.items):
            for key in asset_keys.get(item.collection, ()):
                asset = item.assets.get(key)
                rows.append(
                    {
                        "collection": item.collection,
                        "observation_key": observation.key,
                        "observation_datetime": observation.datetime,
                        "mosaic_priority": priority,
                        "item_id": item.item_id,
                        "item_datetime": item.datetime,
                        "platform": item.platform,
                        "orbit_state": item.orbit_state,
                        "relative_orbit": item.relative_orbit,
                        "absolute_orbit": item.absolute_orbit,
                        "cloud_cover": item.cloud_cover,
                        "nodata_fraction": item.nodata_fraction,
                        "processing_datetime": item.processing_datetime,
                        "processing_version": item.processing_version,
                        "asset_key": key,
                        "asset_checksum": asset.checksum if asset else None,
                        "asset_nodata": asset.nodata if asset else None,
                        "asset_location": asset_location(asset.href) if asset else None,
                    }
                )
    columns = [
        "collection", "observation_key", "observation_datetime", "mosaic_priority", "item_id", "item_datetime",
        "platform", "orbit_state", "relative_orbit", "absolute_orbit", "cloud_cover", "nodata_fraction",
        "processing_datetime", "processing_version", "asset_key", "asset_checksum", "asset_nodata", "asset_location",
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=columns)


def unversioned_items(manifest: pd.DataFrame) -> list[str]:
    """Return item ids that have neither an asset checksum nor a processing datetime."""
    if manifest.empty:
        return []
    unversioned = manifest["asset_checksum"].isna() & manifest["processing_datetime"].isna()
    return sorted(manifest.loc[unversioned, "item_id"].unique().tolist())


def manifest_digest(manifest: pd.DataFrame, *, search_date: str | None = None) -> str:
    """Return a 16-character digest of item identities, versions, checksums, nodata, locations and grouping.

    When any item is unversioned, ``search_date`` is mixed into the digest so
    cached cubes for that inventory expire instead of becoming silently stale.
    """
    identity_columns = [
        "collection", "observation_key", "observation_datetime", "mosaic_priority", "item_id", "processing_datetime",
        "processing_version", "asset_key", "asset_checksum", "asset_nodata", "asset_location",
    ]  # fmt: skip
    records = (
        manifest[identity_columns].astype(object).where(manifest[identity_columns].notna(), MISSING).astype(str).values.tolist()
        if not manifest.empty
        else []
    )
    digest = hashlib.sha256(json.dumps(sorted(records)).encode("utf-8"))
    if unversioned_items(manifest):
        if not search_date:
            raise ValueError("Error: search_date is required when the HUB manifest contains unversioned items.")
        digest.update(f"unversioned:{search_date}".encode())
    return digest.hexdigest()[:16]
