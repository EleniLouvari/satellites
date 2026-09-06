"""Monthly Planet basemap tiles feeding the shared parcel-statistics engine."""
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
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import box

from openeo_parcel_stats_pipeline.zonal_stats import SatelliteZonalStats
from openeo_parcel_stats_pipeline.feature_reduction import reduce_annual_median_features


@contextmanager
def _temporary_workspace(root):
    """Remove only the unique directory created by this invocation."""
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
        if path.exists():
            shutil.rmtree(path)


def download_monthly_tiles(catalog, tiles, start_date, end_date, download_dir,
                           collection_id="PL_BSM", quad_id_field="quad_id",
                           analysis_pattern="*Analysis*.tif", analysis_only=False):
    """Resolve exactly one item per quad/month and return a local raster manifest.

    Uses the catalog's existing identifier convention. Missing or ambiguous items
    fail explicitly. Existing readable analysis files are reused; failed downloads
    never become manifest entries. Credentials stay in the existing catalog helper.
    """
    from common_libraries.catalog_s3_checks import download_all_assets_for_item, get_asset_filename

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    records = []
    for month in pd.period_range(start, end, freq="M"):
        for quad_id in sorted(tiles[quad_id_field].astype(str).unique()):
            identifier = f"{calendar.month_name[month.month].lower()}_{month.year}_{quad_id}_"
            body = {"collections": [collection_id], "filter-lang": "cql2-json",
                    "filter": {"op": "like", "args": [{"property": "identifier"}, f"%{identifier}%"]}}
            items = catalog.fetch_features(search_body=body, max_items=2)
            if len(items) != 1:
                raise ValueError(f"Expected one {collection_id} item for {month}/{quad_id}; found {len(items)}")
            item_id = items[0]["id"]
            root = Path(download_dir).resolve()
            folder = (root / collection_id / item_id).resolve()
            if not folder.is_relative_to(root):
                raise ValueError("Catalog item path escapes download directory")
            def fetch_download_item(**kwargs):
                features = catalog.fetch_features(**kwargs)
                if not analysis_only:
                    return features
                selected_items = []
                for feature in features:
                    assets = {}
                    for key, asset in feature.get("assets", {}).items():
                        filename = get_asset_filename(key, asset)
                        if fnmatch.fnmatch(filename.lower(), analysis_pattern.lower()):
                            if Path(filename).name != filename:
                                raise ValueError("Analysis asset filename must not contain directories")
                            assets[key] = asset
                    if len(assets) != 1:
                        raise ValueError(f"Expected one analysis asset for {feature['id']}; found {len(assets)}")
                    selected_items.append({**feature, "assets": assets})
                return selected_items

            candidates = sorted(folder.glob(analysis_pattern))
            completion = folder / "planet_download_complete.json"
            if not candidates or not completion.exists():
                result = download_all_assets_for_item(
                    collection_id, item_id, root, fetch_features=fetch_download_item,
                    client_id=catalog.client_id,
                )
                candidates = sorted(folder.glob(analysis_pattern))
                if result["failed"]:
                    raise RuntimeError(f"Asset download failed for {item_id}; retry before extraction")
            if len(candidates) != 1:
                raise ValueError(f"Expected one analysis raster in {folder}; found {len(candidates)}")
            with rasterio.open(candidates[0]) as src:
                if src.count < 8 or src.crs is None:
                    raise ValueError(f"Expected a georeferenced raster with at least eight bands: {candidates[0]}")
                src.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
            completion.write_text(json.dumps({"item_id": item_id, "path": str(candidates[0])}), encoding="utf-8")
            records.append({"period_start": str(month.start_time.date()), "quad_id": quad_id,
                            "item_id": item_id, "path": str(candidates[0])})
    return pd.DataFrame(records)


class PlanetBasemapZonalStats(SatelliteZonalStats):
    """Reuse local cleaning, quality metrics and output schemas without openEO jobs.

    Bands B1..B8 follow TIFF order. NDVI is recomputed from cleaned B8 and B6;
    the supplied ninth band is deliberately replaced to match shared pipeline
    processing order. Physical bands use reflectance units after TIFF scale
    and offset metadata, or the configured fallback reflectance scale. No Sentinel spectral mappings are used.
    """
    SUPPORTED_SENTINEL2_BANDS = tuple(f"B{i}" for i in range(1, 9))
    DEFAULT_SENTINEL2_BANDS = SUPPORTED_SENTINEL2_BANDS
    SUPPORTED_SENTINEL2_INDICES = {
        "EVI": ("B8", "B6", "B2"), "NDRE": ("B8", "B7"),
        "NDVI": ("B8", "B6"), "NDWI": ("B4", "B8"), "SAVI": ("B8", "B6"),
    }

    def __init__(self, parcels, start_date, end_date, output_dir, working_epsg,
                 *, parcel_id_field="parcel_id", resolution_metres=5.0,
                 batch_size_metres=2000.0, buffer_metres=100.0,
                 max_cube_bytes=512_000_000, indices=None, reflectance_scale=0.0001, **cleaning_options):
        indices = list(self.SUPPORTED_SENTINEL2_INDICES) if indices is None else indices
        if any(str(name).strip().upper() == "NDMI" for name in indices):
            raise ValueError("NDMI requires NIR and SWIR; eight-band Planet basemaps have no SWIR band")
        if not np.isfinite(reflectance_scale) or reflectance_scale <= 0:
            raise ValueError("reflectance_scale must be finite and positive")
        self.reflectance_scale = float(reflectance_scale)
        for value, name in [(resolution_metres, "resolution_metres"),
                            (batch_size_metres, "batch_size_metres"),
                            (max_cube_bytes, "max_cube_bytes")]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(buffer_metres) or buffer_metres < 0:
            raise ValueError("buffer_metres must be finite and non-negative")
        if pd.Timestamp(start_date).day != 1:
            raise ValueError("Monthly basemaps require start_date on the first day of a month")
        self.resolution_metres = float(resolution_metres)
        self.GRID_SIZE_METRES = float(batch_size_metres)
        self.buffer_metres = float(buffer_metres)
        self.max_cube_bytes = int(max_cube_bytes)
        cleaning_options.setdefault("spatial_statistics", ["mean", "median", "sd", "min", "max", "range"])
        super().__init__(parcels, start_date, end_date, output_dir, working_epsg,
                         parcel_id_field=parcel_id_field,
                         sentinel2_bands=list(self.DEFAULT_SENTINEL2_BANDS),
                         sentinel2_indices=indices, sentinel1_bands=[],
                         keep_cleaned_checkpoint=True, **cleaning_options)
        if self.temporal_period != "1M":
            raise ValueError("Planet basemap extraction supports monthly periods only")
        if "median" not in self.spatial_statistics:
            raise ValueError("Include median for the annual feature output")
        if parcel_id_field in parcels and parcels[parcel_id_field].isna().any():
            raise ValueError("Parcel identifiers must not be null")

    def _calculate_local_sentinel2_index(self, values_by_band, index_name):
        nir, red = values_by_band["B8"], values_by_band["B6"]
        if index_name == "NDVI":
            return self._safe_ratio(nir - red, nir + red)
        if index_name == "NDRE":
            edge = values_by_band["B7"]
            return self._safe_ratio(nir - edge, nir + edge)
        if index_name == "NDWI":
            green = values_by_band["B4"]
            return self._safe_ratio(green - nir, green + nir)
        if index_name == "EVI":
            return self._safe_ratio(2.5 * (nir - red), nir + 6 * red - 7.5 * values_by_band["B2"] + 1)
        if index_name == "SAVI":
            return self._safe_ratio(1.5 * (nir - red), nir + red + 0.5)
        raise ValueError(f"Unsupported Planet index: {index_name}")

    def _cleaning_checkpoint_signature(self):
        # Index selection/formulas also affect the shared final checkpoint.
        payload = [super()._cleaning_checkpoint_signature(), "planet_indices_v2", self.sentinel2_indices]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]

    def _build_tile_cube(self, batch, manifest, number, month_loader=None):
        metric = batch.to_crs(self.working_epsg)
        left, bottom, right, top = metric.total_bounds
        res, pad = self.resolution_metres, self.buffer_metres
        left, bottom = math.floor((left-pad)/res)*res, math.floor((bottom-pad)/res)*res
        right, top = math.ceil((right+pad)/res)*res, math.ceil((top+pad)/res)*res
        width, height = round((right-left)/res), round((top-bottom)/res)
        transform = from_origin(left, top, res, res)
        _, labels = self._temporal_intervals()
        size = len(labels)*8*height*width*4
        if size > self.max_cube_bytes:
            raise MemoryError(f"Batch {number} raw cube needs {size:,} bytes. Reduce batch_size_metres or buffer_metres. Cleaning requires several times the raw cube memory.")
        selected = manifest[manifest.geometry.intersects(box(left, bottom, right, top))] if manifest is not None else pd.DataFrame()
        fingerprint = {"format": 2, "reflectance_scale": self.reflectance_scale, "bounds": [left, bottom, right, top], "epsg": self.working_epsg,
                       "resolution": res, "periods": labels, "sources": []}
        for row in selected.itertuples():
            path = Path(row.path).resolve()
            stat = path.stat()
            fingerprint["sources"].append([row.period_start, str(path), stat.st_size, stat.st_mtime_ns])
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
                        with WarpedVRT(src, crs=f"EPSG:{self.working_epsg}", transform=transform,
                                       width=width, height=height, dtype="float32", nodata=np.nan,
                                       resampling=Resampling.nearest) as vrt:
                            layer = vrt.read(indexes=list(range(1, 9)), masked=True).filled(np.nan)
                        scales = np.asarray(src.scales[:8], dtype="float32")
                        offsets = np.asarray(src.offsets[:8], dtype="float32")
                        if np.all(scales == 1) and np.all(offsets == 0):
                            scales = np.full(8, self.reflectance_scale, dtype="float32")
                        layer = layer * scales[:, None, None]
                        layer += np.asarray(src.offsets[:8], dtype="float32")[:, None, None]
                        valid = np.isfinite(layer) & ~np.isfinite(values[t])
                        values[t][valid] = layer[valid]
        dataset = xr.Dataset({f"B{b+1}": (("time", "y", "x"), values[:, b]) for b in range(8)},
                             coords={"time": pd.to_datetime(labels),
                                     "x": left+(np.arange(width)+0.5)*res,
                                     "y": top-(np.arange(height)+0.5)*res})
        dataset = dataset.rio.write_crs(self.working_epsg).rio.write_transform(transform)
        partial = target.with_suffix(".partial.nc")
        dataset.to_netcdf(partial)
        partial.replace(target)
        return target

    def run(self, manifest):
        """Extract a manifest with period_start, quad_id and local path columns.

        Each parcel belongs to one processing batch, but all overlapping source
        rasters feed its mosaic. First valid pixels win in sorted quad/path order,
        so tile overlap never duplicates observations. Missing quad/months fail.
        """
        import geopandas as gpd
        from rasterio.warp import transform_bounds

        required = {"period_start", "quad_id", "path"}
        if not required.issubset(manifest.columns) or manifest.empty:
            raise ValueError(f"Nonempty manifest requires {sorted(required)}")
        manifest = manifest.copy()
        manifest["period_start"] = pd.to_datetime(manifest.period_start).dt.strftime("%Y-%m-%d")
        manifest["quad_id"] = manifest.quad_id.astype(str)
        if manifest.duplicated(["period_start", "quad_id"]).any():
            raise ValueError("Duplicate quad/month entries in manifest")
        _, labels = self._temporal_intervals()
        manifest = manifest[manifest.period_start.isin(labels)].sort_values(["period_start", "quad_id", "path"])
        for quad, rows in manifest.groupby("quad_id"):
            if set(rows.period_start) != set(labels):
                raise ValueError(f"Missing months for quad {quad}")
        if manifest.empty:
            raise ValueError("Manifest has no requested months")
        footprints = []
        for row in manifest.itertuples():
            with rasterio.open(row.path) as src:
                if src.crs is None or src.count < 8:
                    raise ValueError(f"Invalid Planet raster: {row.path}")
                footprints.append(box(*transform_bounds(src.crs, self.working_epsg, *src.bounds)))
        manifest = gpd.GeoDataFrame(manifest, geometry=footprints, crs=self.working_epsg)
        for label in labels:
            coverage = manifest[manifest.period_start == label].geometry.union_all()
            if not self.parcels.to_crs(self.working_epsg).geometry.covered_by(coverage).all():
                raise ValueError(f"Raster footprints do not cover every parcel for {label}")
        results, reports = [], []
        for number, batch in enumerate(self._iter_spatial_batches(), 1):
            self.logger.info("Processing Planet batch %s (%s parcels)", number, len(batch))
            path = self._build_tile_cube(batch, manifest, number)
            result, report = self._calculate_local_statistics(path, batch, number)
            result["batch_number"] = number
            results.append(result)
            reports.append(report)
        return self._save_outputs(results, reports, manifest.drop(columns="geometry"))

    def _save_outputs(self, results, reports, manifest):
        """Persist the shared output tables for either local or streaming input."""
        final = self._reshape_time_series_for_ml(self._add_derived_stats(pd.concat(results, ignore_index=True)))
        reduced = reduce_annual_median_features(final, temporal_sources=self._output_sensor_variables(), expected_periods=None)
        for frame, name in [(final, self.PARCEL_OUTPUT_FILE_NAME), (reduced, self.REDUCED_PARCEL_OUTPUT_FILE_NAME)]:
            target = self.output_dir / name
            partial = target.with_name(f".{target.stem}.partial.parquet")
            frame.to_parquet(partial, index=False)
            partial.replace(target)
        self.cleaning_report = pd.concat(reports, ignore_index=True)
        self.cleaning_report.to_csv(self.output_dir / "satellite_pixel_cleaning_report.csv", index=False)
        manifest.to_csv(self.output_dir / "planet_tile_manifest.csv", index=False)
        return final

    def _iter_tile_groups(self, tiles, quad_id_field):
        """Assign every parcel once to its containing quad or intersecting quads."""
        if tiles.crs is None or quad_id_field not in tiles:
            raise ValueError("Tile grid requires a CRS and a quad identifier column")
        tiles = tiles.to_crs(self.working_epsg).copy()
        if tiles[quad_id_field].isna().any():
            raise ValueError("Quad identifiers must not be null")
        tiles[quad_id_field] = tiles[quad_id_field].astype(str)
        if tiles[quad_id_field].duplicated().any():
            raise ValueError("Quad identifiers must be unique")
        if tiles.empty or tiles.geometry.isna().any() or tiles.geometry.is_empty.any() or not tiles.is_valid.all():
            raise ValueError("Tile grid must contain valid nonempty geometries")
        tiles = tiles.sort_values(quad_id_field).reset_index(drop=True)
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

    def run_catalog(self, catalog, tiles, scratch_dir=None, *, collection_id="PL_BSM",
                    quad_id_field="quad_id", analysis_pattern="*Analysis*.tif", resume=True):
        """Download, mosaic and discard one tile group/month at a time.

        Single-quad parcels are processed first. Seam parcels are grouped by the
        exact set of intersecting quads, then split into bounded spatial batches.
        Only one month's analysis TIFFs coexist on disk. Their cropped pixels are
        retained in memory across months for the shared temporal cleaning logic.
        Raw/cleaned/final batch cubes live only in an owned temporary directory.

        Small result/report checkpoints are committed before moving to the next
        batch. With resume=True they are reused without contacting the catalog;
        set resume=False to refresh results after catalog imagery changes.
        Pre-existing downloads and external source assets are never removed.
        """
        from rasterio.warp import transform_bounds

        scratch_root = Path(scratch_dir) if scratch_dir is not None else self.output_dir / "scratch"
        # Plan before any downloads, so uncovered parcels fail without consuming space.
        groups = list(self._iter_tile_groups(tiles, quad_id_field))
        signature = {
            "version": "planet_streaming_v1", "run": self._run_signature(),
            "cleaning": self._cleaning_checkpoint_signature(), "statistics": self.spatial_statistics,
            "resolution": self.resolution_metres, "buffer": self.buffer_metres,
            "batch_size": self.GRID_SIZE_METRES, "scale": self.reflectance_scale,
            "collection": collection_id, "analysis_pattern": analysis_pattern,
            "endpoint": str(getattr(catalog, "base_url", getattr(catalog, "catalog_endpoint", ""))),
            "tiles": [(str(row[quad_id_field]), row.geometry.wkb_hex)
                      for _, row in tiles.sort_values(quad_id_field).iterrows()],
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        checkpoint_dir = self.output_dir / "streaming_results" / key
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results, reports, audit = [], [], []
        number = 0
        for group_tiles, group_parcels in groups:
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
                self.logger.info("Planet batch %s: %s parcels, quads %s", number, len(batch),
                                 group_tiles[quad_id_field].tolist())
                batch_audit = []
                with _temporary_workspace(scratch_root) as work:
                    worker = copy(self)
                    worker.output_dir = work
                    worker.parcels = batch

                    @contextmanager
                    def load_month(label):
                        with _temporary_workspace(work / "downloads") as download_root:
                            self.logger.info("Downloading batch %s month %s", number, label)
                            manifest = download_monthly_tiles(
                                catalog, group_tiles, label, label, download_root,
                                collection_id=collection_id, quad_id_field=quad_id_field,
                                analysis_pattern=analysis_pattern, analysis_only=True,
                            )
                            footprints = []
                            for row in manifest.itertuples():
                                with rasterio.open(row.path) as src:
                                    footprints.append(box(*transform_bounds(src.crs, self.working_epsg, *src.bounds)))
                            from shapely.ops import unary_union
                            coverage = unary_union(footprints)
                            if not batch.to_crs(self.working_epsg).geometry.covered_by(coverage).all():
                                raise ValueError(f"Raster footprints do not cover batch {number} for {label}")
                            yield manifest.sort_values(["quad_id", "path"])
                            entries = manifest.assign(batch_number=number, local_files_deleted=True)
                            batch_audit.extend(entries.to_dict("records"))
                        self.logger.info("Removed batch %s month %s downloads", number, label)

                    path = worker._build_tile_cube(batch, None, number, month_loader=load_month)
                    result, report = worker._calculate_local_statistics(path, batch, number)
                    result["batch_number"] = number
                    for frame, target in ((result, result_path), (report, report_path)):
                        partial = target.with_suffix(".partial.parquet")
                        frame.to_parquet(partial, index=False)
                        partial.replace(target)
                    partial_commit = commit_path.with_suffix(".partial.json")
                    partial_commit.write_text(json.dumps(batch_audit), encoding="utf-8")
                    partial_commit.replace(commit_path)
                # Exiting the owned workspace also removes all intermediate NetCDFs.
                results.append(result)
                reports.append(report)
                audit.extend(batch_audit)
        return self._save_outputs(results, reports, pd.DataFrame(audit))
