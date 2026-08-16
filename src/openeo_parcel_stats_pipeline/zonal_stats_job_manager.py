"""Tile-bounded openEO zonal statistics managed by MultiBackendJobManager.

This is an opt-in implementation for benchmarking the openEO Python client's
``split_area`` and ``MultiBackendJobManager`` workflow.  The established
``SatelliteZonalStats`` implementation is not modified.

Each parcel is owned by the core tile containing its representative point. The
remote load extent is buffered so boundary-crossing parcels remain complete,
while ownership prevents duplicate parcel results in overlapping tile rasters.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
from openeo.extra.job_management import MultiBackendJobManager, create_job_db, get_job_db, split_area

from common_libraries.io_library import write_data

from .zonal_stats import SatelliteZonalStats


def compute_tile_buffer_metres(
    parcels: gpd.GeoDataFrame, working_epsg: int, safety_factor: float = 2.5, round_up_to_metres: int = 50
) -> int:
    """Calculate a conservative tile buffer from the largest parcel radius.

    Each parcel is measured from the representative point used for tile
    ownership to the farthest corner of its axis-aligned bounding box.  This is
    robust for long, narrow parcels, unlike deriving a length from parcel area.
    The maximum radius is multiplied by ``safety_factor`` and rounded upward.
    """

    # Validate inputs and compute a conservative buffer that ensures every
    # parcel remains fully contained in the buffered remote tile used for
    # downloading raster extents.
    if not isinstance(parcels, gpd.GeoDataFrame):
        raise TypeError("parcels must be a geopandas.GeoDataFrame.")
    if parcels.empty:
        raise ValueError("parcels must contain at least one geometry.")
    if parcels.crs is None:
        raise ValueError("parcels must have a CRS.")
    factor = float(safety_factor)
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("safety_factor must be a finite number of at least 1.")
    if isinstance(round_up_to_metres, bool):
        raise TypeError("round_up_to_metres must be a positive integer.")
    rounding = int(round_up_to_metres)
    if rounding <= 0:
        raise ValueError("round_up_to_metres must be positive.")

    metric = parcels.to_crs(epsg=int(working_epsg))
    valid = metric.geometry.notna() & ~metric.geometry.is_empty
    metric = metric.loc[valid]
    if metric.empty:
        raise ValueError("parcels contain no non-empty geometries.")
    if not metric.crs.is_projected:
        raise ValueError("working_epsg must be a projected CRS with metre units.")

    points = metric.geometry.representative_point()
    bounds = metric.geometry.bounds
    # Compute the maximum distance from each representative point to the farthest
    # corner of its axis-aligned bounding box.  This is robust for long, narrow
    dx = np.maximum(points.x - bounds.minx, bounds.maxx - points.x)
    dy = np.maximum(points.y - bounds.miny, bounds.maxy - points.y)
    maximum_radius = float(np.hypot(dx, dy).max())
    # Round up to the nearest multiple of ``round_up_to_metres`` after applying the safety factor.
    tile_buffer_metres = int(math.ceil(maximum_radius * factor / rounding) * rounding)
    print(f"Recommended tile buffer: {tile_buffer_metres:,} m")
    return tile_buffer_metres


def compute_tile_width(parcels: gpd.GeoDataFrame, parcel_id_col: str, working_epsg: int) -> int:
    """Test candidate tile sizes and return the smallest that avoids crossing parcels."""
    # Evaluate candidate square tile sizes and return the smallest size that
    # avoids splitting parcels across tiles (which would create duplicate work).
    from openeo.extra.job_management import split_area

    west, south, east, north = parcels.total_bounds
    aoi_extent = {
        "west": float(west),
        "south": float(south),
        "east": float(east),
        "north": float(north),
        "crs": f"EPSG:{working_epsg}",
    }
    # Test candidate tile sizes and report the number of non-empty jobs, parcels
    comparison_rows = []
    for candidate_metres in (5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000):
        tiles = (
            split_area(aoi=aoi_extent, tile_size=candidate_metres, projection=f"EPSG:{working_epsg}")
            .to_crs(parcels.crs)
            .reset_index(drop=True)
        )
        # Assign parcels to tiles by checking which tile contains the parcel geometry.
        joined = gpd.sjoin(parcels[[parcel_id_col, "geometry"]], tiles[["geometry"]], how="left", predicate="within")
        assigned = joined.loc[joined["index_right"].notna()].drop_duplicates(parcel_id_col)
        counts = assigned.groupby("index_right").size()
        used_tiles = counts.index.astype(int).tolist()
        comparison_rows.append(
            {
                "tile_size_km": candidate_metres // 1_000,
                "non_empty_jobs": len(counts),
                "crossing_parcels": len(parcels) - assigned[parcel_id_col].nunique(),
                "median_parcels_per_job": float(counts.median()) if len(counts) else 0,
                "max_parcels_per_job": int(counts.max()) if len(counts) else 0,
                "loaded_area_km2": float(tiles.loc[used_tiles].geometry.area.sum() / 1_000_000),
            }
        )
    # Display the comparison table and return the smallest tile size that avoids crossing parcels.
    tile_comparison = pd.DataFrame(comparison_rows)
    safe = tile_comparison.loc[tile_comparison["crossing_parcels"] == 0]
    tile_size_meters = int(safe.iloc[0]["tile_size_km"] * 1_000) if not safe.empty else 50_000
    print(f"AOI extent: {(east - west) / 1_000:.2f} x {(north - south) / 1_000:.2f} km; parcels: {len(parcels):,}")
    print(tile_comparison.to_string(index=False))
    print(
        f"Recommended tile size: {tile_size_meters / 1_000:g} km"
        if not safe.empty
        else "No tested tile size avoids crossing parcels; use a custom grid."
    )
    return tile_size_meters


class _NetCDFTileJobManager(MultiBackendJobManager):
    """Download each completed single-asset job to its deterministic cache path."""

    def on_job_done(self, job, row) -> None:
        target = Path(row["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial.nc")
        if temporary.exists():
            temporary.unlink()
        job.get_results().download_file(target=temporary)
        temporary.replace(target)


class JobManagerSatelliteZonalStats(SatelliteZonalStats):
    """Run one tile-bounded CDSE batch job per ``split_area`` tile.

    Parameters are identical to :class:`SatelliteZonalStats`, with these
    additions:

    ``tile_size_metres``
        Square tile size in the projected ``working_epsg`` CRS.
    ``tile_buffer_metres``
        Buffer added around every remote tile extent. Parcels are still owned
        by one unbuffered core tile, so overlapping rasters do not duplicate
        parcel results.
    ``openeo_parallel_jobs``
        Jobs kept active on CDSE.  The general-user limit is currently two.
    ``job_poll_seconds``
        Delay between manager status polls.
    ``run_identifier``
        Short label included in every openEO job title for this run. When
        omitted, a local timestamp such as ``run-20260811-143025`` is created
        once and shared by all tile jobs from the instance.

    ``batch_workers`` continues to control local statistics processes only;
    unlike the original implementation, it does not control remote concurrency.
    """

    DEFAULT_TILE_SIZE_METRES = 50_000
    DEFAULT_OPENEO_PARALLEL_JOBS = 2

    def __init__(
        self,
        *args,
        tile_size_metres: int = DEFAULT_TILE_SIZE_METRES,
        tile_buffer_metres: int = 500,
        openeo_parallel_jobs: int = DEFAULT_OPENEO_PARALLEL_JOBS,
        job_poll_seconds: int = 30,
        run_identifier: str | None = None,
        **kwargs,
    ) -> None:
        self.tile_size_metres = self._positive_integer(tile_size_metres, "tile_size_metres")
        self.tile_buffer_metres = self._non_negative_integer(tile_buffer_metres, "tile_buffer_metres")
        self.openeo_parallel_jobs = self._positive_integer(openeo_parallel_jobs, "openeo_parallel_jobs")
        self.job_poll_seconds = self._positive_integer(job_poll_seconds, "job_poll_seconds")
        self.run_identifier = self._validate_run_identifier(run_identifier)
        super().__init__(*args, **kwargs)

    def _validate_run_identifier(self, value: str | None) -> str:
        """Return a compact user label or generate one for this run."""

        if value is None:
            return datetime.now().astimezone().strftime("run-%Y%m%d-%H%M%S")
        if not isinstance(value, str):
            raise TypeError("run_identifier must be a string or None.")
        # Collapse whitespace so the identifier remains readable in job lists.
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("run_identifier must not be empty.")
        if len(normalized) > 80:
            raise ValueError("run_identifier must contain at most 80 characters.")
        return normalized

    def _job_title(self, plan: dict) -> str:
        """Build the web-editor title shared by one run and one tile job."""
        # The run identifier is included so multiple runs can be distinguished in the openEO web editor.
        return f"Zonal statistics [{self.run_identifier}] {plan['tile_id']} batch {plan['batch_number']}"

    def _positive_integer(self, value, name: str) -> int:
        """Check that a value is a positive integer and return it as an int."""

        if isinstance(value, bool):
            raise TypeError(f"{name} must be a positive integer.")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a positive integer.") from exc
        if result <= 0:
            raise ValueError(f"{name} must be positive.")
        return result

    def _non_negative_integer(self, value, name: str) -> int:
        """Check that a value is a non-negative integer and return it as an int."""

        if isinstance(value, bool):
            raise TypeError(f"{name} must be a non-negative integer.")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a non-negative integer.") from exc
        if result < 0:
            raise ValueError(f"{name} must not be negative.")
        return result

    def _run_signature(self) -> str:
        """Keep tile caches separate from original and differently sized runs."""

        # The base signature is derived from the original parameters, so that any change
        # in the original parameters will invalidate the cache. The suffix is derived
        # from the tile parameters, so that different tile sizes or buffers will also
        # invalidate the cache.

        base = super()._run_signature()
        suffix = hashlib.sha256(
            (f"job-manager-v7-buffered-tiles:{self.tile_size_metres}:buffer:{self.tile_buffer_metres}").encode("utf-8")
        ).hexdigest()[:8]
        return f"{base}-tiles-{suffix}"

    def _build_tile_plan(self, cube_dir: Path):
        """Return tile rows and whole-parcel batches, rejecting split parcels."""

        metric = self.parcels.to_crs(epsg=self.working_epsg)
        west, south, east, north = metric.total_bounds
        tiles = split_area(
            # Use the overall extent, not the union of parcel polygons. Passing
            # a disjoint polygon union can yield parcel-shaped fragments and
            # effectively create one tiny job per isolated parcel instead of
            # grouping nearby parcels in a regular tile.
            aoi={
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north),
                "crs": f"EPSG:{self.working_epsg}",
            },
            tile_size=self.tile_size_metres,
            projection=f"EPSG:{self.working_epsg}",
        ).to_crs(epsg=self.working_epsg)
        if tiles.empty:
            raise RuntimeError("split_area produced no tiles for the parcel area.")
        tiles = tiles.reset_index(drop=True)
        tiles["tile_id"] = [f"tile_{index:05d}" for index in tiles.index]

        # Assign ownership with one interior point. Core tiles do not overlap;
        # deterministic de-duplication handles the rare point on a shared edge.
        ownership_points = metric[[self.PARCEL_ID_FIELD, "geometry"]].copy()
        ownership_points.geometry = metric.geometry.representative_point()
        assignments = gpd.sjoin(ownership_points, tiles[["tile_id", "geometry"]], how="left", predicate="intersects")
        assignments = assignments.loc[assignments["tile_id"].notna()]
        assignments = assignments.sort_values("tile_id").drop_duplicates(self.PARCEL_ID_FIELD, keep="first")
        assigned_ids = set(assignments[self.PARCEL_ID_FIELD])
        missing_owner = metric.loc[~metric[self.PARCEL_ID_FIELD].isin(assigned_ids)]
        if not missing_owner.empty:
            raise RuntimeError(f"No core tile owns {len(missing_owner)} parcel representative points.")

        buffered_tiles = tiles.copy()
        buffered_tiles.geometry = buffered_tiles.geometry.buffer(self.tile_buffer_metres)
        tile_geometry = buffered_tiles.set_index("tile_id").geometry
        parcel_geometry = metric.set_index(self.PARCEL_ID_FIELD).geometry
        uncovered_ids = [
            parcel_id
            for parcel_id, tile_id in assignments[[self.PARCEL_ID_FIELD, "tile_id"]].itertuples(index=False, name=None)
            if not tile_geometry.loc[tile_id].covers(parcel_geometry.loc[parcel_id])
        ]
        if uncovered_ids:
            report = cube_dir / "uncovered_buffered_tile_parcels.csv"
            uncovered_report = pd.DataFrame(
                {
                    self.PARCEL_ID_FIELD: uncovered_ids,
                    "reason": (f"parcel is not covered by its owner tile with a {self.tile_buffer_metres} m buffer"),
                }
            )
            write_data(uncovered_report, str(report), plain_csv=True)
            raise ValueError(
                f"{len(uncovered_ids)} parcels exceed the {self.tile_buffer_metres} m tile "
                f"buffer. Increase tile_buffer_metres; details: {report}"
            )

        parcel_lookup = self.parcels.set_index(self.PARCEL_ID_FIELD, drop=False)
        plans = []
        batch_number = 0
        # Build a plan of tile jobs, each with its assigned parcels and buffered extent.
        for tile_index, tile in tiles.iterrows():
            parcel_ids = assignments.loc[assignments["tile_id"] == tile["tile_id"], self.PARCEL_ID_FIELD].tolist()
            if not parcel_ids:
                continue
            batch_number += 1
            parcels = parcel_lookup.loc[parcel_ids].reset_index(drop=True)
            buffered_geometry = tile_geometry.loc[tile["tile_id"]]
            west, south, east, north = buffered_geometry.bounds
            spatial_extent = {
                "west": float(west),
                "south": float(south),
                "east": float(east),
                "north": float(north),
                "crs": f"EPSG:{self.working_epsg}",
            }
            # The target path is deterministic and unique for each tile job, so that
            # the MultiBackendJobManager can skip already-cached NetCDF files.
            target = cube_dir / f"batch_{batch_number:05d}_monthly.nc"
            plans.append(
                {
                    "batch_number": batch_number,
                    "tile_id": str(tile["tile_id"]),
                    "spatial_extent": spatial_extent,
                    "target_path": str(target.resolve()),
                    "parcels": parcels,
                }
            )
        if not plans:
            raise RuntimeError("No non-empty tile jobs were produced.")
        return plans

    def _build_tile_cube(self, connection, spatial_extent):
        """Build a tile-bounded graph without redundant remote parcel masking.

        Parcel masks are applied during local statistics. Sending hundreds of
        detailed polygons through ``filter_spatial`` duplicates that work and
        can fail in the CDSE GeoTrellis/JTS precision reducer.
        """
        sentinel2_temporal = None
        if self.sentinel2_bands or self.sentinel2_indices:
            # Load Sentinel-2 only when bands or indices are requested.
            required_sentinel2_bands = self._required_sentinel2_bands()
            sentinel2 = connection.load_collection(
                self.SENTINEL2_COLLECTION,
                spatial_extent=spatial_extent,
                temporal_extent=[self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")],
                bands=[*required_sentinel2_bands, self.SENTINEL2_SCENE_CLASSIFICATION_BAND],
                max_cloud_cover=self.SENTINEL2_MAX_SCENE_CLOUD_COVER,
            )
            reflectance = sentinel2.filter_bands(list(required_sentinel2_bands)).resample_spatial(
                resolution=self.TARGET_RESOLUTION_METRES, projection=f"EPSG:{self.working_epsg}", method="bilinear"
            )
            reflectance = reflectance * self.SENTINEL2_REFLECTANCE_SCALE_FACTOR
            scl = sentinel2.band(self.SENTINEL2_SCENE_CLASSIFICATION_BAND).resample_cube_spatial(reflectance, method="near")
            invalid = self._combine_class_conditions(scl, self.SENTINEL2_INVALID_SCL_CLASSES)
            cloud = self._combine_class_conditions(scl, self.SENTINEL2_BUFFERED_SCL_CLASSES)
            cloud_buffer = cloud.apply_kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) > 0
            sentinel2_temporal = self._aggregate_temporal_cube(
                self._build_sentinel2_output_cube(reflectance.mask(invalid | cloud_buffer))
            )
        if not self.sentinel1_bands:
            return sentinel2_temporal

        # 2. SENTINEL1: load, filter bands, resample, apply backscatter coefficient
        sentinel1 = connection.load_collection(
            self.SENTINEL1_COLLECTION,
            spatial_extent=spatial_extent,
            temporal_extent=[self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")],
            bands=list(self.sentinel1_bands),
        )
        sentinel1 = sentinel1.sar_backscatter(
            coefficient=self.SENTINEL1_BACKSCATTER_COEFFICIENT,
            local_incidence_angle=False,
            elevation_model=self.SENTINEL1_ELEVATION_MODEL,
        )
        sentinel1_temporal = self._aggregate_temporal_cube(sentinel1)
        if sentinel2_temporal is None:
            # Sentinel-1-only jobs no longer depend on a Sentinel-2 target cube.
            return sentinel1_temporal.resample_spatial(
                resolution=self.TARGET_RESOLUTION_METRES, projection=f"EPSG:{self.working_epsg}", method="bilinear"
            )
        sentinel1_temporal = sentinel1_temporal.resample_cube_spatial(sentinel2_temporal, method="bilinear")
        return sentinel2_temporal.merge_cubes(sentinel1_temporal)

    def _run_openeo_jobs(self, cube_dir):
        """Persist, submit, monitor and download true tile jobs."""

        plans = self._build_tile_plan(cube_dir)
        parcel_counts = [len(plan["parcels"]) for plan in plans]
        singleton_jobs = sum(count == 1 for count in parcel_counts)
        self.openeo_logger.info(
            "Tile plan: %s parcels in %s remote jobs; parcels/job min=%s, median=%.1f, max=%s; singleton jobs=%s.",
            sum(parcel_counts),
            len(plans),
            min(parcel_counts),
            float(pd.Series(parcel_counts).median()),
            max(parcel_counts),
            singleton_jobs,
        )
        if len(plans) >= 10 and singleton_jobs == len(plans):
            raise RuntimeError(
                "Refusing to submit a pathological tile plan: every remote job contains "
                "one parcel. Check the split_area AOI and tile grouping before retrying."
            )
        # Keep the tile intact for both remote processing and local statistics:
        # all parcels assigned to a tile are handled together.
        pending_batches = [(plan["batch_number"], plan["parcels"], Path(plan["target_path"])) for plan in plans]
        uncached = [plan for plan in plans if not Path(plan["target_path"]).exists()]
        if not uncached:
            self.openeo_logger.info("All %s tile cubes are already cached.", len(plans))
            return pending_batches

        self.openeo_logger.info("Connecting to openEO backend %s.", self.OPENEO_URL)
        connection = openeo.connect(self.OPENEO_URL, auto_validate=False)
        self._authenticate_openeo_connection(connection)
        plan_lookup = {plan["batch_number"]: plan for plan in plans}

        job_rows = pd.DataFrame(
            [
                {
                    "batch_number": plan["batch_number"],
                    "run_identifier": self.run_identifier,
                    "tile_id": plan["tile_id"],
                    "west": plan["spatial_extent"]["west"],
                    "south": plan["spatial_extent"]["south"],
                    "east": plan["spatial_extent"]["east"],
                    "north": plan["spatial_extent"]["north"],
                    "crs": plan["spatial_extent"]["crs"],
                    "target_path": plan["target_path"],
                }
                for plan in uncached
            ]
        )
        database_path = cube_dir / "tile_jobs.parquet"
        if database_path.exists():
            job_db = get_job_db(database_path)
        else:
            job_db = create_job_db(database_path, df=job_rows)

        def start_job(row, connection, **_kwargs):
            plan = plan_lookup[int(row["batch_number"])]
            cube = self._build_tile_cube(connection, plan["spatial_extent"])
            job = cube.create_job(out_format="netCDF", title=self._job_title(plan))
            message = (
                f"Created openEO job {job.job_id} for run {self.run_identifier!r}, "
                f"batch {plan['batch_number']} "
                f"({plan['tile_id']}, {len(plan['parcels'])} parcels)."
            )
            print(message, flush=True)
            self.openeo_logger.info(message)
            return job

        manager = _NetCDFTileJobManager(
            poll_sleep=self.job_poll_seconds, root_dir=cube_dir / "job_manager", download_results=False
        )
        manager.add_backend("cdse", connection=connection, parallel_jobs=self.openeo_parallel_jobs)
        manager.run_jobs(job_db=job_db, start_job=start_job)

        missing = [Path(plan["target_path"]) for plan in plans if not Path(plan["target_path"]).exists()]
        if missing:
            raise RuntimeError(
                f"MultiBackendJobManager finished without {len(missing)} expected NetCDF files. "
                f"Inspect {database_path} and {cube_dir / 'job_manager'} for statuses and error logs."
            )
        return pending_batches
