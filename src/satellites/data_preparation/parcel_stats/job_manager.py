"""Tile-bounded openEO zonal statistics managed by MultiBackendJobManager.

This is an opt-in implementation for benchmarking the openEO Python client's
``split_area`` and ``MultiBackendJobManager`` workflow.  The established
``SatelliteZonalStats`` implementation is not modified.

Each parcel is owned by the core tile containing its representative point. The
remote load extent is buffered so boundary-crossing parcels remain complete,
while ownership prevents duplicate parcel results in overlapping tile rasters.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
from openeo.extra.job_management import MultiBackendJobManager, create_job_db, get_job_db, split_area

from satellites.shared.io import write_data

from satellites.data_preparation.parcel_stats.openeo import SatelliteZonalStats


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


def _non_empty_tile_job_count(partition: gpd.GeoDataFrame, tile_width: int, working_epsg: int) -> tuple[int, int]:
    """Count representative-point-owned jobs and their maximum parcel load."""
    metric = partition.to_crs(epsg=working_epsg)
    west, south, east, north = metric.total_bounds
    tiles = split_area(
        aoi={
            "west": float(west),
            "south": float(south),
            "east": float(east),
            "north": float(north),
            "crs": f"EPSG:{working_epsg}",
        },
        tile_size=tile_width,
        projection=f"EPSG:{working_epsg}",
    ).to_crs(epsg=working_epsg)
    points = metric[["geometry"]].reset_index(drop=True)
    points["parcel_row"] = np.arange(len(points))
    points.geometry = metric.geometry.representative_point().reset_index(drop=True)
    assignments = gpd.sjoin(points, tiles[["geometry"]], how="left", predicate="intersects")
    assignments = assignments.loc[assignments["index_right"].notna()]
    # A point on a shared edge may intersect two tiles; deterministically own only one.
    assignments = assignments.sort_values("index_right").drop_duplicates("parcel_row")
    if len(assignments) != len(metric):
        raise RuntimeError(f"Tile planning assigned {len(assignments)} of {len(metric)} parcels.")
    counts = assignments.groupby("index_right").size()
    return len(counts), int(counts.max())


def compute_tile_width(
    parcels: gpd.GeoDataFrame,
    parcel_id_col: str,
    working_epsg: int,
    *,
    spatial_parts: Sequence[gpd.GeoDataFrame] | None = None,
    user_count: int = 1,
    jobs_per_user: int = 2,
    candidate_widths: Sequence[int] = (5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000),
) -> int:
    """Choose the largest tile width that fills every active user's job slots.

    Counts are calculated independently inside each spatial partition because
    each partition creates its own job manager. Core-tile ownership follows the
    real planner's representative-point rule; boundary-crossing geometries are
    handled by ``tile_buffer_metres`` rather than forcing oversized core tiles.
    """
    if not isinstance(parcels, gpd.GeoDataFrame) or parcels.empty:
        raise ValueError("parcels must be a non-empty GeoDataFrame.")
    if parcels.crs is None:
        raise ValueError("parcels must have a CRS.")
    if parcel_id_col not in parcels.columns:
        raise ValueError(f"Parcel ID column does not exist: {parcel_id_col!r}.")
    if parcels[parcel_id_col].isna().any() or parcels[parcel_id_col].duplicated().any():
        raise ValueError(f"{parcel_id_col!r} must contain unique non-null values.")
    if isinstance(user_count, bool) or int(user_count) < 1:
        raise ValueError("user_count must be a positive integer.")
    if isinstance(jobs_per_user, bool) or int(jobs_per_user) < 1:
        raise ValueError("jobs_per_user must be a positive integer.")
    users, slots = int(user_count), int(jobs_per_user)

    parts = list(spatial_parts) if spatial_parts is not None else [parcels]
    if not parts or any(not isinstance(part, gpd.GeoDataFrame) or part.empty or part.crs is None for part in parts):
        raise ValueError("spatial_parts must contain non-empty GeoDataFrames.")
    combined_ids = [parcel_id for part in parts for parcel_id in part[parcel_id_col].tolist()]
    if len(combined_ids) != len(set(combined_ids)) or set(combined_ids) != set(parcels[parcel_id_col]):
        raise ValueError("spatial_parts must contain every input parcel exactly once.")

    widths = tuple(int(width) for width in candidate_widths)
    if not widths or any(width <= 0 for width in widths) or len(set(widths)) != len(widths):
        raise ValueError("candidate_widths must contain unique positive integers.")
    widths = tuple(sorted(widths))

    comparison_rows = []
    for width in widths:
        job_counts, maximum_loads = zip(*(_non_empty_tile_job_count(part, width, int(working_epsg)) for part in parts))
        comparison_rows.append(
            {
                "tile_size_km": width / 1_000,
                "total_jobs": sum(job_counts),
                "min_jobs_per_partition": min(job_counts),
                "median_jobs_per_partition": float(np.median(job_counts)),
                "max_jobs_per_partition": max(job_counts),
                "max_parcels_per_job": max(maximum_loads),
                "fills_user_slots": min(job_counts) >= slots,
            }
        )

    tile_comparison = pd.DataFrame(comparison_rows)
    eligible = tile_comparison.loc[tile_comparison["fills_user_slots"]]
    selected = int((eligible.iloc[-1] if not eligible.empty else tile_comparison.iloc[0])["tile_size_km"] * 1_000)
    active_users = min(users, len(parts))
    print(
        f"Concurrency target: {users} users x {slots} jobs = {users * slots} remote slots; "
        f"{len(parts)} spatial partitions ({active_users} initially active users)."
    )
    print(tile_comparison.to_string(index=False))
    if len(parts) < users:
        print(f"Warning: {users - len(parts)} users will be idle because there are fewer non-empty partitions than users.")
    if eligible.empty:
        print(
            f"No candidate provides {slots} jobs in every partition; using the smallest tile width "
            f"({selected / 1_000:g} km) for maximum available concurrency."
        )
    else:
        print(f"Recommended tile size: {selected / 1_000:g} km (largest candidate that fills every user's slots).")
    return selected


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
    ``max_job_retries``
        Maximum retries after a tile reaches ``error`` or its start request
        fails. Retry counts are stored in the job database, so the limit is
        preserved across notebook restarts.
    ``job_retry_delay_seconds``
        Cooldown before retrying a failed job. This delay is useful for
        transient CDSE Kubernetes/API failures and is separate from polling.
    ``run_identifier``
        Short label included in every openEO job title for this run. When
        omitted, a local timestamp such as ``run-20260811-143025`` is created
        once and shared by all tile jobs from the instance.

    ``batch_workers`` continues to control local statistics processes only;
    unlike the original implementation, it does not control remote concurrency.
    """

    DEFAULT_TILE_SIZE_METRES = 50_000
    DEFAULT_OPENEO_PARALLEL_JOBS = 2
    DEFAULT_MAX_JOB_RETRIES = 3
    DEFAULT_JOB_RETRY_DELAY_SECONDS = 60
    RETRYABLE_JOB_FAILURE_STATUSES = ("error", "start_failed", "queued_for_start_failed")

    def __init__(
        self,
        *args,
        tile_size_metres: int = DEFAULT_TILE_SIZE_METRES,
        tile_buffer_metres: int = 500,
        openeo_parallel_jobs: int = DEFAULT_OPENEO_PARALLEL_JOBS,
        job_poll_seconds: int = 30,
        max_job_retries: int = DEFAULT_MAX_JOB_RETRIES,
        job_retry_delay_seconds: int = DEFAULT_JOB_RETRY_DELAY_SECONDS,
        run_identifier: str | None = None,
        **kwargs,
    ) -> None:
        self.tile_size_metres = self._positive_integer(tile_size_metres, "tile_size_metres")
        self.tile_buffer_metres = self._non_negative_integer(tile_buffer_metres, "tile_buffer_metres")
        self.openeo_parallel_jobs = self._positive_integer(openeo_parallel_jobs, "openeo_parallel_jobs")
        self.job_poll_seconds = self._positive_integer(job_poll_seconds, "job_poll_seconds")
        self.max_job_retries = self._non_negative_integer(max_job_retries, "max_job_retries")
        self.job_retry_delay_seconds = self._non_negative_integer(job_retry_delay_seconds, "job_retry_delay_seconds")
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

    def _ensure_job_retry_tracking(self, job_db) -> None:
        """Add persistent retry metadata to new or pre-retry job databases."""

        jobs = job_db.df
        changed = False
        if "retry_count" not in jobs.columns:
            jobs["retry_count"] = 0
            changed = True
        else:
            normalized = pd.to_numeric(jobs["retry_count"], errors="coerce").fillna(0).astype(int)
            if not normalized.equals(jobs["retry_count"]):
                jobs["retry_count"] = normalized
                changed = True
        if "failed_job_ids" not in jobs.columns:
            jobs["failed_job_ids"] = ""
            changed = True
        else:
            normalized = jobs["failed_job_ids"].fillna("").astype(str)
            if not normalized.equals(jobs["failed_job_ids"]):
                jobs["failed_job_ids"] = normalized
                changed = True
        if "retry_mode" not in jobs.columns:
            jobs["retry_mode"] = ""
            changed = True
        else:
            normalized = jobs["retry_mode"].fillna("").astype(str)
            if not normalized.equals(jobs["retry_mode"]):
                jobs["retry_mode"] = normalized
                changed = True
        if changed:
            job_db.persist(jobs.copy())

    @staticmethod
    def _clean_job_id(value) -> str:
        """Return a usable persisted job ID, excluding null-like values."""

        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    def _reset_retryable_failed_jobs(self, job_db) -> int:
        """Reset eligible terminal failures for a bounded retry."""

        self._ensure_job_retry_tracking(job_db)
        failed = job_db.get_by_status(statuses=self.RETRYABLE_JOB_FAILURE_STATUSES).copy()
        if failed.empty:
            return 0

        retry_counts = pd.to_numeric(failed["retry_count"], errors="coerce").fillna(0).astype(int)
        retryable = failed.loc[retry_counts < self.max_job_retries].copy()
        exhausted = failed.loc[retry_counts >= self.max_job_retries]
        if not exhausted.empty:
            self.source_logger.error(
                "%s tile job(s) exhausted the limit of %s retries: batches %s.",
                len(exhausted),
                self.max_job_retries,
                exhausted["batch_number"].astype(int).tolist(),
            )
        if retryable.empty:
            return 0

        for index, row in retryable.iterrows():
            failure_status = str(row["status"])
            failed_id = self._clean_job_id(row.get("id"))
            history = [job_id for job_id in str(row.get("failed_job_ids") or "").split(";") if job_id]
            restart_existing = failure_status in {"start_failed", "queued_for_start_failed"} and bool(failed_id)
            if not restart_existing and failed_id and failed_id not in history:
                history.append(failed_id)
            next_retry = int(retry_counts.loc[index]) + 1
            retryable.at[index, "failed_job_ids"] = ";".join(history)
            retryable.at[index, "retry_count"] = next_retry
            retryable.at[index, "retry_mode"] = "restart" if restart_existing else "replace"
            retryable.at[index, "status"] = "not_started"
            self.source_logger.warning(
                "Retrying batch %s after status %s for openEO job %s (retry %s/%s; mode=%s).",
                int(row["batch_number"]),
                failure_status,
                failed_id or "<unknown>",
                next_retry,
                self.max_job_retries,
                retryable.at[index, "retry_mode"],
            )

        job_db.persist(retryable)
        return len(retryable)

    def _wait_before_job_retry(self, retry_count: int) -> None:
        """Apply the configured backend cooldown before a retry wave."""

        if retry_count > 0 and self.job_retry_delay_seconds > 0:
            self.source_logger.warning(
                "Waiting %s seconds before retrying %s failed tile job(s).", self.job_retry_delay_seconds, retry_count
            )
            time.sleep(self.job_retry_delay_seconds)

    def _build_or_reuse_tile_job(self, row, connection, plan):
        """Reuse a job whose start failed, or build a replacement job."""

        existing_job_id = self._clean_job_id(row.get("id"))
        if row.get("retry_mode") == "restart" and existing_job_id:
            return connection.job(existing_job_id), True
        cube = self._build_tile_cube(connection, plan["spatial_extent"])
        return cube.create_job(out_format="netCDF", title=self._job_title(plan)), False

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
        required_sentinel1_bands = self._required_sentinel1_bands()
        if not required_sentinel1_bands:
            return sentinel2_temporal

        # 2. SENTINEL1: load, filter bands, resample, apply backscatter coefficient
        sentinel1 = connection.load_collection(
            self.SENTINEL1_COLLECTION,
            spatial_extent=spatial_extent,
            temporal_extent=[self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d")],
            bands=list(required_sentinel1_bands),
            **self._sentinel1_load_options(),
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
        self.source_logger.info(
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
            self.source_logger.info("All %s tile cubes are already cached.", len(plans))
            return pending_batches

        self.source_logger.info("Connecting to openEO backend %s.", self.OPENEO_URL)
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
                    "retry_count": 0,
                    "failed_job_ids": "",
                    "retry_mode": "",
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
            job, reused = self._build_or_reuse_tile_job(row, connection, plan)
            action = "Retrying start of existing" if reused else "Created"
            message = (
                f"{action} openEO job {job.job_id} for run {self.run_identifier!r}, "
                f"batch {plan['batch_number']} ({plan['tile_id']}, {len(plan['parcels'])} parcels)."
            )
            print(message, flush=True)
            self.source_logger.info(message)
            return job

        manager = _NetCDFTileJobManager(
            poll_sleep=self.job_poll_seconds, root_dir=cube_dir / "job_manager", download_results=False
        )
        manager.add_backend("cdse", connection=connection, parallel_jobs=self.openeo_parallel_jobs)
        retry_count = self._reset_retryable_failed_jobs(job_db)
        self._wait_before_job_retry(retry_count)
        while True:
            manager.run_jobs(job_db=job_db, start_job=start_job)
            retry_count = self._reset_retryable_failed_jobs(job_db)
            if retry_count == 0:
                break
            self._wait_before_job_retry(retry_count)

        missing = [Path(plan["target_path"]) for plan in plans if not Path(plan["target_path"]).exists()]
        if missing:
            failed = job_db.get_by_status(statuses=self.RETRYABLE_JOB_FAILURE_STATUSES)
            failure_details = ""
            if not failed.empty:
                descriptions = [
                    f"batch {int(row['batch_number'])} job {row['id']} with status {row['status']} "
                    f"after {int(row['retry_count'])} retries"
                    for _, row in failed.iterrows()
                ]
                failure_details = f" Failed jobs: {', '.join(descriptions)}."
            raise RuntimeError(
                f"MultiBackendJobManager finished without {len(missing)} expected NetCDF files. "
                f"Inspect {database_path} and {cube_dir / 'job_manager'} for statuses and error logs."
                f"{failure_details}"
            )
        return pending_batches


# Source-explicit public name, retaining the original class identity.
OpenEOJobManagerZonalStats = JobManagerSatelliteZonalStats
