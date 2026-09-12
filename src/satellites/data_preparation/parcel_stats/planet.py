"""Adapt HUB monthly Planet basemaps to the shared parcel-statistics engine.

``run`` consumes retained local rasters; ``run_catalog`` trades repeat downloads
for bounded temporary storage. See README.md for inputs, scaling and restart rules.
"""

from __future__ import annotations

import calendar
from contextlib import contextmanager, nullcontext
from copy import copy
import fnmatch
import shutil
from uuid import uuid4
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import Polygon, box

from .core.base import ParcelStatsBase
from satellites.data_preparation.features.temporal import reduce_annual_median_features


@contextmanager
def _temporary_workspace(root, *, delete=True):
    """Create an owned scratch directory and optionally remove it on exit."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"planet-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        # Verify the absolute deletion target immediately before recursive cleanup.
        if path.is_symlink() or path.resolve().parent != root:
            raise RuntimeError(f"Refusing cleanup outside owned workspace: {path}")
        if delete and path.exists():
            shutil.rmtree(path)


def _safe_component(value, name):
    """Keep catalog identifiers and filenames inside their assigned directory."""
    if not isinstance(value, str) or not value.strip() or value in {".", ".."} or any(c in value for c in "/\\:"):
        raise ValueError(f"{name} must be a nonempty path component")
    return value


def _raster_footprints(manifest, working_epsg):
    """Validate rasters and project their densified outer pixel boundaries once."""
    footprints = []
    for row in manifest.itertuples():
        with rasterio.open(row.path) as src:
            if src.crs is None or src.count < 8:
                raise ValueError(f"Expected a georeferenced raster with at least eight bands: {row.path}")
            # Actual pixel corners also handle rotated rasters; a bounds box would
            # incorrectly claim coverage in the empty corners of their envelope.
            corners = [(0, 0), (src.width, 0), (src.width, src.height), (0, src.height), (0, 0)]
            points = []
            for start, end in zip(corners, corners[1:]):
                for fraction in np.linspace(0, 1, 22, endpoint=False):
                    points.append(src.transform * tuple(a + fraction * (b - a) for a, b in zip(start, end)))
            footprint = gpd.GeoSeries([Polygon(points)], crs=src.crs).to_crs(working_epsg).iloc[0]
            footprints.append(footprint)
    return gpd.GeoDataFrame(manifest.copy(), geometry=footprints, crs=working_epsg)


def _require_coverage(sources, metric_parcels, label):
    """Reject missing geometric coverage; nodata inside a footprint remains valid input for cleaning."""
    # Reprojected adjoining quad edges can differ at floating-point precision:
    # GEOS may report an uncovered seam even when its geometric difference is
    # empty. A one-micrometre tolerance fixes that without accepting pixel gaps.
    coverage = sources.geometry.union_all().buffer(1e-6) if not sources.empty else None
    if sources.empty or not metric_parcels.geometry.covered_by(coverage).all():
        raise ValueError(f"Raster footprints do not cover every parcel for {label}")


def _write_parquet(frame, target):
    """Publish a table only after its temporary Parquet file has been written."""
    partial = target.with_suffix(".partial.parquet")
    frame.to_parquet(partial, index=False)
    partial.replace(target)


def _download_state(item, assets, folder, analysis_pattern, analysis_only):
    """Describe both selected asset metadata and the local files a completion marker certifies."""
    from satellites.data_preparation.sources.hub.catalog_s3_checks import get_asset_filename

    files = []
    for key, asset in assets.items():
        path = folder / get_asset_filename(key, asset)
        if not path.is_file():
            return None
        stat = path.stat()
        if not stat.st_size:
            return None
        files.append([path.name, stat.st_size, stat.st_mtime_ns])
    # Store a digest rather than asset URLs, which may contain signed credentials.
    digest = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    return {
        "version": 2,
        "item_id": item["id"],
        "assets": digest,
        "files": files,
        "analysis_pattern": analysis_pattern,
        "analysis_only": analysis_only,
    }


def download_monthly_tiles(
    catalog,
    tiles,
    start_date,
    end_date,
    download_dir,
    collection_id="PL_BSM",
    quad_id_field="quad_id",
    analysis_pattern="*Analysis*.tif",
    analysis_only=False,
):
    """Resolve exactly one item per quad/month and return a local raster manifest.

    Uses the catalog's existing identifier convention. Missing or ambiguous items
    fail explicitly. Existing readable analysis files are reused; failed downloads
    never become manifest entries. Credentials stay in the existing catalog helper.
    """
    from satellites.data_preparation.sources.hub.catalog_s3_checks import download_all_assets_for_item, get_asset_filename

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("start_date and end_date must be valid ordered dates")
    if quad_id_field not in tiles or tiles.empty or tiles[quad_id_field].isna().any():
        raise ValueError("Tiles require nonempty, non-null quad identifiers")
    quad_ids = sorted(tiles[quad_id_field].astype(str).unique())
    if any(not quad.strip() for quad in quad_ids):
        raise ValueError("Quad identifiers must not be blank")
    _safe_component(collection_id, "collection_id")
    root = Path(download_dir).resolve()
    records = []
    for month in pd.period_range(start, end, freq="M"):
        for quad_id in quad_ids:
            # HUB names monthly products with an English month, year and quad ID.
            identifier = f"{calendar.month_name[month.month].lower()}_{month.year}_{quad_id}_"
            body = {
                "collections": [collection_id],
                "filter-lang": "cql2-json",
                "filter": {"op": "like", "args": [{"property": "identifier"}, f"%{identifier}%"]},
            }
            items = catalog.fetch_features(search_body=body, max_items=2)
            if len(items) != 1:
                raise ValueError(f"Expected one {collection_id} item for {month}/{quad_id}; found {len(items)}")
            item = items[0]
            item_id = _safe_component(item["id"], "item_id")
            folder = (root / collection_id / item_id).resolve()
            if not folder.is_relative_to(root):
                raise ValueError("Catalog item path escapes download directory")
            assets, analysis = {}, []
            for key, asset in item.get("assets", {}).items():
                filename = _safe_component(get_asset_filename(key, asset), "asset filename")
                matches = fnmatch.fnmatchcase(filename.lower(), analysis_pattern.lower())
                if matches:
                    analysis.append(filename)
                if matches or not analysis_only:
                    assets[key] = asset
            if len(analysis) != 1:
                raise ValueError(f"Expected one analysis asset for {item_id}; found {len(analysis)}")
            names = [get_asset_filename(key, asset) for key, asset in assets.items()]
            if len(names) != len(set(name.casefold() for name in names)):
                raise ValueError(f"Asset filenames collide for {item_id}")

            # Reuse the resolved STAC item instead of making a second identical search.
            def fetch_download_item(**kwargs):
                return [{**item, "assets": assets}]

            completion = folder / "planet_download_complete.json"
            current = _download_state(item, assets, folder, analysis_pattern, analysis_only)
            try:
                saved = json.loads(completion.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError):
                saved = None
            if current is None or saved != current:
                # A failed refresh must never leave a previous completion marker valid.
                if completion.exists():
                    completion.unlink()
                result = download_all_assets_for_item(
                    collection_id, item_id, root, fetch_features=fetch_download_item, client_id=catalog.client_id
                )
                if result["failed"]:
                    raise RuntimeError(f"Asset download failed for {item_id}; retry before extraction")
            analysis_path = folder / analysis[0]
            with rasterio.open(analysis_path) as src:
                if src.count < 8 or src.crs is None:
                    raise ValueError(f"Expected a georeferenced raster with at least eight bands: {analysis_path}")
                src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            state = _download_state(item, assets, folder, analysis_pattern, analysis_only)
            if state is None:
                raise RuntimeError(f"Asset download incomplete for {item_id}")
            partial = completion.with_suffix(".partial.json")
            partial.write_text(json.dumps(state), encoding="utf-8")
            partial.replace(completion)
            records.append(
                {"period_start": str(month.start_time.date()), "quad_id": quad_id, "item_id": item_id, "path": str(analysis_path)}
            )
    return pd.DataFrame(records)


class PlanetBasemapZonalStats(ParcelStatsBase):
    """Reuse local cleaning, quality metrics and output schemas without openEO jobs.

    Bands B1..B8 follow TIFF order. NDVI is recomputed from cleaned B8 and B6;
    extra bands (e.g. HUB NDVI or a Planet alpha band) are not spectral inputs.
    Physical bands use reflectance units after TIFF scale
    and offset metadata, or the configured fallback reflectance scale. Planet
    band roles are mapped explicitly before reusing the shared index algebra.
    """

    SOURCE_NAME = "Planet Basemaps"

    SUPPORTED_OPTICAL_BANDS = tuple(f"B{i}" for i in range(1, 9))
    DEFAULT_OPTICAL_BANDS = SUPPORTED_OPTICAL_BANDS
    SUPPORTED_OPTICAL_INDICES = {
        "EVI": ("B8", "B6", "B2"),
        "NDRE": ("B8", "B7"),
        "NDVI": ("B8", "B6"),
        "NDWI": ("B4", "B8"),
        "SAVI": ("B8", "B6"),
    }

    OPTICAL_BAND_ROLES = {"blue": "B2", "green": "B4", "red": "B6", "red_edge": "B7", "nir": "B8"}

    # Compatibility bridge for the existing constructor/validation/cache schema.
    # Planet code and shared index algebra use the optical names above.
    SUPPORTED_SENTINEL2_BANDS = SUPPORTED_OPTICAL_BANDS
    DEFAULT_SENTINEL2_BANDS = DEFAULT_OPTICAL_BANDS
    SUPPORTED_SENTINEL2_INDICES = SUPPORTED_OPTICAL_INDICES

    def __init__(
        self,
        parcels,
        start_date,
        end_date,
        output_dir,
        working_epsg,
        *,
        parcel_id_field="parcel_id",
        resolution_metres=5.0,
        batch_size_metres=2000.0,
        buffer_metres=100.0,
        max_cube_bytes=512_000_000,
        indices=None,
        reflectance_scale=0.0001,
        **cleaning_options,
    ):
        indices = list(self.SUPPORTED_OPTICAL_INDICES) if indices is None else indices
        if any(str(name).strip().upper() == "NDMI" for name in indices):
            raise ValueError("NDMI requires NIR and SWIR; eight-band Planet basemaps have no SWIR band")
        if not np.isfinite(reflectance_scale) or reflectance_scale <= 0:
            raise ValueError("reflectance_scale must be finite and positive")
        self.reflectance_scale = float(reflectance_scale)
        for value, name in [
            (resolution_metres, "resolution_metres"),
            (batch_size_metres, "batch_size_metres"),
            (max_cube_bytes, "max_cube_bytes"),
        ]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(buffer_metres) or buffer_metres < 0:
            raise ValueError("buffer_metres must be finite and non-negative")
        if pd.Timestamp(start_date).day != 1:
            raise ValueError("Monthly basemaps require start_date on the first day of a month")
        if not pd.Timestamp(end_date).is_month_end:
            raise ValueError("Monthly basemaps require end_date on the last day of a month")
        # The shared validator accepts any projected CRS, including feet. Planet
        # grids and their memory estimates explicitly assume metres on both axes.
        crs = CRS.from_epsg(working_epsg)
        if not crs.is_projected or any(not np.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info[:2]):
            raise ValueError("working_epsg must identify a projected CRS with metre units")
        if any(
            cleaning_options.get(name)
            for name in ("sentinel1_bands", "sentinel1_indices", "sentinel2_bands", "sentinel2_indices")
        ):
            raise ValueError("Use indices to select Planet indices; Sentinel sensor options are not supported")
        for name in ("sentinel1_bands", "sentinel1_indices", "sentinel2_bands", "sentinel2_indices"):
            cleaning_options.pop(name, None)
        if cleaning_options.get("batch_workers", 1) != 1:
            raise ValueError("Planet processing is sequential; batch_workers must be 1")
        self.resolution_metres = float(resolution_metres)
        self.GRID_SIZE_METRES = float(batch_size_metres)
        self.buffer_metres = float(buffer_metres)
        self.max_cube_bytes = int(max_cube_bytes)
        cleaning_options.setdefault("spatial_statistics", ["mean", "median", "sd", "min", "max", "range"])
        # Keep the final shared checkpoint; retaining the preceding physical-band
        # cube as well is optional and otherwise doubles intermediate disk use.
        cleaning_options.setdefault("keep_cleaned_checkpoint", False)
        super().__init__(
            parcels,
            start_date,
            end_date,
            output_dir,
            working_epsg,
            parcel_id_field=parcel_id_field,
            sentinel2_bands=list(self.DEFAULT_OPTICAL_BANDS),
            sentinel2_indices=indices,
            sentinel1_bands=[],
            **cleaning_options,
        )
        if self.temporal_period != "1M":
            raise ValueError("Planet basemap extraction supports monthly periods only")
        if "median" not in self.spatial_statistics:
            raise ValueError("Include median for the annual feature output")
        if self.PARCEL_ID_FIELD in parcels:
            identifiers = parcels[self.PARCEL_ID_FIELD]
            if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
                raise ValueError("Parcel identifiers must not be null or blank")

    def _calculate_local_optical_index(self, values_by_band, index_name):
        """Allow only indices supported by Planet's eight physical bands."""
        if index_name not in self.SUPPORTED_OPTICAL_INDICES:
            raise ValueError(f"Unsupported Planet index: {index_name}")
        return super()._calculate_local_optical_index(values_by_band, index_name)

    def _cleaning_checkpoint_signature(self):
        # Index selection/formulas also affect the shared final checkpoint.
        payload = [super()._cleaning_checkpoint_signature(), "planet_indices_v2", self.optical_indices]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]

    def _batch_grid(self, batch):
        """Snap the buffered parcel extent to the common metre grid."""
        metric = batch.to_crs(self.working_epsg)
        left, bottom, right, top = metric.total_bounds
        res, pad = self.resolution_metres, self.buffer_metres
        left, bottom = math.floor((left - pad) / res) * res, math.floor((bottom - pad) / res) * res
        right, top = math.ceil((right + pad) / res) * res, math.ceil((top + pad) / res) * res
        width, height = round((right - left) / res), round((top - bottom) / res)
        return (left, bottom, right, top), width, height

    def _build_tile_cube(self, batch, manifest, number, month_loader=None):
        """Mosaic each month onto one bounded grid, then pass its cube to shared cleaning."""
        (left, bottom, right, top), width, height = self._batch_grid(batch)
        res = self.resolution_metres
        transform = from_origin(left, top, res, res)
        _, labels = self._temporal_intervals()
        size = len(labels) * 8 * height * width * 4
        if size > self.max_cube_bytes:
            raise MemoryError(
                f"Batch {number} raw cube needs {size:,} bytes. Reduce batch_size_metres or buffer_metres. Cleaning requires several times the raw cube memory."
            )
        # Use the spatial index instead of scanning every quad/month for each batch.
        selected = (
            manifest.iloc[sorted(manifest.sindex.query(box(left, bottom, right, top), predicate="intersects"))]
            if manifest is not None
            else pd.DataFrame()
        )
        fingerprint = {
            "format": 3,
            "reflectance_scale": self.reflectance_scale,
            "bounds": [left, bottom, right, top],
            "epsg": self.working_epsg,
            "resolution": res,
            "periods": labels,
            "sources": [],
        }
        for row in selected.itertuples():
            path = Path(row.path).resolve()
            # GDAL sidecars can change masks or scaling while the TIFF itself
            # remains untouched. Include every file used by the raster dataset.
            with rasterio.open(path) as src:
                source_files = sorted(str(Path(filename).resolve()) for filename in src.files)
            for filename in source_files:
                stat = Path(filename).stat()
                fingerprint["sources"].append([row.period_start, filename, stat.st_size, stat.st_mtime_ns])
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
        folder = self.output_dir / "monthly_cubes" / key
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"batch_{number}.nc"
        if target.exists():
            return target
        values = np.full((len(labels), 8, height, width), np.nan, dtype="float32")
        for t, label in enumerate(labels):
            context = month_loader(label) if month_loader else nullcontext(selected[selected.period_start == label])
            with context as sources:
                if sources.empty:
                    raise ValueError(f"No raster coverage for batch {number}, month {label}")
                for row in sources.itertuples():
                    with rasterio.open(row.path) as src:
                        with WarpedVRT(
                            src,
                            crs=f"EPSG:{self.working_epsg}",
                            transform=transform,
                            width=width,
                            height=height,
                            dtype="float32",
                            nodata=np.nan,
                            resampling=Resampling.nearest,
                        ) as vrt:
                            layer = vrt.read(indexes=list(range(1, 9)), masked=True).filled(np.nan)
                        scales = np.asarray(src.scales[:8], dtype="float32")
                        offsets = np.asarray(src.offsets[:8], dtype="float32")
                        if np.all(scales == 1) and np.all(offsets == 0):
                            scales = np.full(8, self.reflectance_scale, dtype="float32")
                        # Work in place: full band/month arrays are large. Metadata
                        # scaling takes precedence over the fallback reflectance scale.
                        layer *= scales[:, None, None]
                        layer += offsets[:, None, None]
                        valid = np.isfinite(layer) & ~np.isfinite(values[t])
                        np.copyto(values[t], layer, where=valid)
        dataset = xr.Dataset(
            {f"B{b + 1}": (("time", "y", "x"), values[:, b]) for b in range(8)},
            coords={
                "time": pd.to_datetime(labels),
                "x": left + (np.arange(width) + 0.5) * res,
                "y": top - (np.arange(height) + 0.5) * res,
            },
        )
        # Metadata writes otherwise deep-copy the complete time cube twice.
        dataset.rio.write_crs(self.working_epsg, inplace=True)
        dataset.rio.write_transform(transform, inplace=True)
        partial = target.with_suffix(".partial.nc")
        try:
            dataset.to_netcdf(partial)
        finally:
            dataset.close()
        partial.replace(target)
        return target

    def run(self, manifest):
        """Extract a manifest with period_start, quad_id and local path columns.

        Each parcel belongs to one processing batch, but all overlapping source
        rasters feed its mosaic. First valid pixels win in sorted quad/path order,
        so tile overlap never duplicates observations. Missing quad/months fail.
        """
        required = {"period_start", "quad_id", "path"}
        if not required.issubset(manifest.columns) or manifest.empty:
            raise ValueError(f"Nonempty manifest requires {sorted(required)}")
        manifest = manifest.copy()
        if manifest[list(required)].isna().any().any():
            raise ValueError("Manifest period_start, quad_id and path must not be null")
        manifest["path"] = manifest.path.map(lambda path: str(Path(path).resolve()))
        manifest["period_start"] = pd.to_datetime(manifest.period_start).dt.strftime("%Y-%m-%d")
        manifest["quad_id"] = manifest.quad_id.astype(str)
        if manifest.quad_id.str.strip().eq("").any():
            raise ValueError("Manifest quad_id must not be blank")
        if manifest.duplicated(["period_start", "quad_id"]).any():
            raise ValueError("Duplicate quad/month entries in manifest")
        _, labels = self._temporal_intervals()
        manifest = manifest[manifest.period_start.isin(labels)].sort_values(["period_start", "quad_id", "path"])
        for quad, rows in manifest.groupby("quad_id"):
            if set(rows.period_start) != set(labels):
                raise ValueError(f"Missing months for quad {quad}")
        if manifest.empty:
            raise ValueError("Manifest has no requested months")
        manifest = _raster_footprints(manifest, self.working_epsg)
        metric_parcels = self.parcels.to_crs(self.working_epsg)
        for label in labels:
            _require_coverage(manifest[manifest.period_start == label], metric_parcels, label)
        results, reports = [], []
        for number, batch in enumerate(self._iter_spatial_batches(), 1):
            self.logger.info("Processing Planet batch %s (%s parcels)", number, len(batch))
            path = self._build_tile_cube(batch, manifest, number)
            result, report = self._calculate_local_statistics(path, batch, number)
            result["batch_number"] = number
            results.append(result)
            reports.append(report)
        return self._save_outputs(results, reports, manifest.drop(columns="geometry"))

    def run_local(self, source):
        """Process an existing manifest CSV or directory of raw Planet deliveries.

        This entry point never connects to HUB or S3 and never removes input
        imagery. Metadata sidecars identify each raw delivery's quad and month.
        """
        from satellites.data_preparation.sources.planet.delivery import build_local_planet_manifest

        manifest = build_local_planet_manifest(source, start_date=self.start_date, end_date=self.end_date)
        return self.run(manifest)

    def _save_outputs(self, results, reports, manifest):
        """Persist the shared output tables for either local or streaming input."""
        final = self._reshape_time_series_for_ml(self._add_derived_stats(pd.concat(results, ignore_index=True)))
        reduced = reduce_annual_median_features(final, temporal_sources=self._output_sensor_variables(), expected_periods=None)
        for frame, name in [(final, self.PARCEL_OUTPUT_FILE_NAME), (reduced, self.REDUCED_PARCEL_OUTPUT_FILE_NAME)]:
            _write_parquet(frame, self.output_dir / name)
        self.cleaning_report = pd.concat(reports, ignore_index=True)
        self.cleaning_report.to_csv(self.output_dir / "satellite_pixel_cleaning_report.csv", index=False)
        manifest.to_csv(self.output_dir / "planet_tile_manifest.csv", index=False)
        return final

    def _prepare_tiles(self, tiles, quad_id_field):
        """Normalize the quad grid for assignment, buffered reads and cache identity."""
        if tiles.crs is None or quad_id_field not in tiles:
            raise ValueError("Tile grid requires a CRS and a quad identifier column")
        tiles = tiles.to_crs(self.working_epsg).copy()
        if tiles[quad_id_field].isna().any():
            raise ValueError("Quad identifiers must not be null")
        tiles[quad_id_field] = tiles[quad_id_field].astype(str)
        if tiles[quad_id_field].str.strip().eq("").any():
            raise ValueError("Quad identifiers must not be blank")
        if tiles[quad_id_field].duplicated().any():
            raise ValueError("Quad identifiers must be unique")
        if tiles.empty or tiles.geometry.isna().any() or tiles.geometry.is_empty.any() or not tiles.is_valid.all():
            raise ValueError("Tile grid must contain valid nonempty geometries")
        return tiles.sort_values(quad_id_field).reset_index(drop=True)

    def _iter_tile_groups(self, tiles, quad_id_field):
        """Assign every parcel once to its containing quad or intersecting quads."""
        tiles = self._prepare_tiles(tiles, quad_id_field)
        metric = self.parcels.to_crs(self.working_epsg)
        groups = {}
        for index, parcel in metric.iterrows():
            candidates = tiles.iloc[tiles.sindex.query(parcel.geometry, predicate="intersects")]
            containing = candidates[candidates.geometry.covers(parcel.geometry)]
            if not containing.empty:
                quad_ids = (min(containing[quad_id_field]),)
            else:
                # Mere boundary touches do not require another source tile.
                candidates = candidates[candidates.geometry.intersection(parcel.geometry).area > 0]
                if candidates.empty or not candidates.geometry.union_all().covers(parcel.geometry):
                    raise ValueError(f"Tile grid does not fully cover parcel {parcel[self.PARCEL_ID_FIELD]}")
                quad_ids = tuple(sorted(candidates[quad_id_field]))
            groups.setdefault(quad_ids, []).append(index)
        for quad_ids, indices in sorted(groups.items(), key=lambda pair: (len(pair[0]), pair[0])):
            yield tiles[tiles[quad_id_field].isin(quad_ids)], self.parcels.loc[indices].copy()

    def run_catalog(
        self,
        catalog,
        tiles,
        scratch_dir=None,
        *,
        collection_id="PL_BSM",
        quad_id_field="quad_id",
        analysis_pattern="*Analysis*.tif",
        resume=True,
        delete_temporary_files=True,
    ):
        """Download and mosaic tile groups, optionally discarding temporary files.

        Single-quad parcels are processed first. Seam parcels are grouped by the
        exact set of intersecting quads, then split into bounded spatial batches.
        Neighboring quads intersecting each batch's buffer also supply pixels.
        With cleanup enabled, only one month's analysis TIFFs coexist on disk.
        Their cropped pixels are
        retained in memory across months for the shared temporal cleaning logic.
        Raw/cleaned/final batch cubes live only in an owned temporary directory.

        Small result/report checkpoints are committed before moving to the next
        batch. With resume=True they are reused without contacting the catalog;
        set resume=False to refresh results after catalog imagery changes.
        Pre-existing downloads and external source assets are never removed.
        Set delete_temporary_files=False to retain owned scratch directories for
        inspection, including on errors. Retained scratch files are not a download
        cache; use run_local to reuse an existing raster directory.
        """
        if not isinstance(delete_temporary_files, bool):
            raise TypeError("delete_temporary_files must be a bool")
        scratch_root = Path(scratch_dir) if scratch_dir is not None else self.output_dir / "scratch"
        # Plan before any downloads, so uncovered parcels fail without consuming space.
        tiles = self._prepare_tiles(tiles, quad_id_field)
        groups = list(self._iter_tile_groups(tiles, quad_id_field))
        signature = {
            "version": "planet_streaming_v2",
            "run": self._run_signature(),
            "parcel_id_field": self.PARCEL_ID_FIELD,
            "tile_crs": tiles.crs.to_wkt(),
            "max_features": self.MAX_FEATURES_PER_JOB,
            "cleaning": self._cleaning_checkpoint_signature(),
            "statistics": self.spatial_statistics,
            "resolution": self.resolution_metres,
            "buffer": self.buffer_metres,
            "batch_size": self.GRID_SIZE_METRES,
            "scale": self.reflectance_scale,
            "collection": collection_id,
            "analysis_pattern": analysis_pattern,
            "endpoint": str(getattr(catalog, "base_url", getattr(catalog, "catalog_endpoint", ""))),
            "tiles": [(str(row[quad_id_field]), row.geometry.wkb_hex) for _, row in tiles.sort_values(quad_id_field).iterrows()],
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        checkpoint_dir = self.output_dir / "streaming_results" / key
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results, reports, audit = [], [], []
        number = 0
        for _, group_parcels in groups:
            planner = copy(self)
            planner.parcels = group_parcels
            for batch in planner._iter_spatial_batches():
                number += 1
                result_path = checkpoint_dir / f"batch_{number}.parquet"
                report_path = checkpoint_dir / f"batch_{number}_report.parquet"
                commit_path = checkpoint_dir / f"batch_{number}.json"
                if resume and all(path.exists() for path in (result_path, report_path, commit_path)):
                    results.append(pd.read_parquet(result_path))
                    reports.append(pd.read_parquet(report_path))
                    audit.extend(json.loads(commit_path.read_text(encoding="utf-8")))
                    self.logger.info("Reusing completed Planet batch %s", number)
                    continue
                # A failed forced refresh must not leave an old commit marker valid.
                if commit_path.exists():
                    commit_path.unlink()
                # Parcel ownership defines output rows, but pixel sources must
                # include every quad intersecting the buffered processing grid.
                bounds, _, _ = self._batch_grid(batch)
                extent = box(*bounds)
                source_tiles = tiles.iloc[sorted(tiles.sindex.query(extent, predicate="intersects"))]
                source_tiles = source_tiles[source_tiles.geometry.intersection(extent).area > 0]
                metric_batch = batch.to_crs(self.working_epsg)
                self.logger.info(
                    "Planet batch %s: %s parcels, quads %s", number, len(batch), source_tiles[quad_id_field].tolist()
                )
                batch_audit = []
                with _temporary_workspace(scratch_root, delete=delete_temporary_files) as work:
                    worker = copy(self)
                    worker.output_dir = work
                    worker.parcels = batch

                    @contextmanager
                    def load_month(label):
                        with _temporary_workspace(work / "downloads", delete=delete_temporary_files) as download_root:
                            self.logger.info("Downloading batch %s month %s", number, label)
                            manifest = download_monthly_tiles(
                                catalog,
                                source_tiles,
                                label,
                                label,
                                download_root,
                                collection_id=collection_id,
                                quad_id_field=quad_id_field,
                                analysis_pattern=analysis_pattern,
                                analysis_only=True,
                            )
                            footprints = _raster_footprints(manifest, self.working_epsg)
                            _require_coverage(footprints, metric_batch, label)
                            yield manifest.sort_values(["quad_id", "path"])
                            entries = manifest.assign(batch_number=number, local_files_deleted=delete_temporary_files)
                            batch_audit.extend(entries.to_dict("records"))
                        self.logger.info(
                            "%s batch %s month %s downloads", "Removed" if delete_temporary_files else "Retained", number, label
                        )

                    path = worker._build_tile_cube(batch, None, number, month_loader=load_month)
                    result, report = worker._calculate_local_statistics(path, batch, number)
                    result["batch_number"] = number
                    for frame, target in ((result, result_path), (report, report_path)):
                        _write_parquet(frame, target)
                    partial_commit = commit_path.with_suffix(".partial.json")
                    partial_commit.write_text(json.dumps(batch_audit), encoding="utf-8")
                    partial_commit.replace(commit_path)
                # Apply the same retention policy to intermediate cubes and downloaded TIFFs.
                results.append(result)
                reports.append(report)
                audit.extend(batch_audit)
        return self._save_outputs(results, reports, pd.DataFrame(audit))
