"""openEO cube construction, remote job execution, and NetCDF downloads.

Helpers in this module build sensor cubes, submit them as openEO jobs and
download NetCDF artifacts into a deterministic cache. Authentication may use
explicit credentials (suitable for unattended runs) or the default OIDC flow.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import openeo
from shapely.geometry import mapping


class OpenEOCubePipeline:
    """Build and download temporal Sentinel raster cubes through openEO."""

    def _authenticate_openeo_connection(self, connection: openeo.Connection) -> openeo.Connection:
        """Authenticate with explicit credentials or the default OIDC flow."""
        # Prefer explicit resource-owner credentials when provided so headless
        # execution on workers or CI does not require interactive OIDC flows.
        if self.openeo_username is not None:
            # Explicit credentials support unattended notebook and worker execution.
            connection.authenticate_oidc_resource_owner_password_credentials(
                username=self.openeo_username, password=self.openeo_password, client_id=self.OPENEO_OIDC_PASSWORD_CLIENT_ID
            )
        else:
            connection.authenticate_oidc()
        return connection

    def _to_feature_collection(self, parcels) -> dict:
        """Convert one WGS84 parcel batch to an openEO GeoJSON collection."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": str(row[self.PARCEL_ID_FIELD]),
                    "properties": {self.PARCEL_ID_FIELD: str(row[self.PARCEL_ID_FIELD])},
                    "geometry": mapping(row.geometry),
                }
                for _, row in parcels.iterrows()
            ],
        }

    def _combine_class_conditions(self, scl_cube, classes: Sequence[int]):
        """Build an openEO Boolean cube matching any requested SCL class."""
        condition = scl_cube == classes[0]
        # openEO process graphs combine Boolean cubes explicitly instead of using Python membership tests.
        for class_id in classes[1:]:
            condition = condition | (scl_cube == class_id)
        return condition

    def _build_sentinel2_output_cube(self, masked_sentinel2_reflectance):
        """Select physical Sentinel-2 bands for cleaning before local index calculation."""
        # Keep every index source band in the NetCDF; indices are derived only after these bands are filled locally.
        return masked_sentinel2_reflectance.filter_bands(list(self._required_sentinel2_bands()))

    def _aggregate_temporal_cube(self, cube):
        """Optionally aggregate acquisitions using the configured shared intervals."""
        if self.temporal_reducer == "none":
            return cube
        if self.temporal_period == "1M" and self.start_date.is_month_start:
            # Use the backend's native monthly operator only when its boundaries match the requested range.
            return cube.aggregate_temporal_period(period="month", reducer=self.temporal_reducer)
        intervals, labels = self._temporal_intervals()
        return cube.aggregate_temporal(intervals=intervals, labels=labels, reducer=self.temporal_reducer)

    def _openeo_temporal_extent(self) -> list[str]:
        """Include the final calendar day when downloading individual acquisitions."""
        end = self.end_date
        if self.temporal_reducer == "none":
            end = end + timedelta(days=1)
        return [self.start_date.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]

    def _build_multisensor_temporal_cube(self, connection: openeo.Connection, feature_collection: dict):
        """Build the Sentinel-1/Sentinel-2 temporal openEO cube."""
        sentinel2_temporal = None
        if self.sentinel2_bands or self.sentinel2_indices:
            # Cloud masking is performed at acquisition resolution before temporal aggregation.
            required_bands = self._required_sentinel2_bands()
            sentinel2 = connection.load_collection(
                self.SENTINEL2_COLLECTION,
                temporal_extent=self._openeo_temporal_extent(),
                bands=[*required_bands, self.SENTINEL2_SCENE_CLASSIFICATION_BAND],
                max_cloud_cover=self.SENTINEL2_MAX_SCENE_CLOUD_COVER,
            ).filter_spatial(feature_collection)
            reflectance = sentinel2.filter_bands(list(required_bands)).resample_spatial(
                resolution=self.TARGET_RESOLUTION_METRES, projection=f"EPSG:{self.working_epsg}", method="bilinear"
            )
            reflectance = reflectance * self.SENTINEL2_REFLECTANCE_SCALE_FACTOR
            scl = sentinel2.band(self.SENTINEL2_SCENE_CLASSIFICATION_BAND).resample_cube_spatial(reflectance, method="near")
            invalid = self._combine_class_conditions(scl, self.SENTINEL2_INVALID_SCL_CLASSES)
            cloud = self._combine_class_conditions(scl, self.SENTINEL2_BUFFERED_SCL_CLASSES)
            # Dilate cloud-related classes by one pixel to suppress contaminated edge pixels.
            cloud_buffer = cloud.apply_kernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) > 0
            sentinel2_temporal = self._aggregate_temporal_cube(
                self._build_sentinel2_output_cube(reflectance.mask(invalid | cloud_buffer))
            )

        required_sentinel1_bands = self._required_sentinel1_bands()
        if not required_sentinel1_bands:
            return sentinel2_temporal

        sentinel1 = connection.load_collection(
            self.SENTINEL1_COLLECTION,
            temporal_extent=self._openeo_temporal_extent(),
            bands=list(required_sentinel1_bands),
            **self._sentinel1_load_options(),
        ).filter_spatial(feature_collection)
        sentinel1 = sentinel1.sar_backscatter(
            coefficient=self.SENTINEL1_BACKSCATTER_COEFFICIENT,
            local_incidence_angle=False,
            elevation_model=self.SENTINEL1_ELEVATION_MODEL,
        )
        sentinel1_temporal = self._aggregate_temporal_cube(sentinel1)
        # Align radar pixels to the optical grid before merging the two sensor cubes.
        if sentinel2_temporal is None:
            return sentinel1_temporal.resample_spatial(
                resolution=self.TARGET_RESOLUTION_METRES, projection=f"EPSG:{self.working_epsg}", method="bilinear"
            )
        sentinel1_temporal = sentinel1_temporal.resample_cube_spatial(sentinel2_temporal, method="bilinear")
        return sentinel2_temporal.merge_cubes(sentinel1_temporal)

    def _run_openeo_batch_job(self, job, temporary_path, netcdf_path, batch_number):
        """Run and download one already-created openEO batch job."""
        job.start_and_wait(print=lambda _message: None)
        job.get_results().download_file(target=temporary_path)
        # Atomic rename makes the final path a reliable completion marker after notebook restarts.
        temporary_path.replace(netcdf_path)
        return batch_number, netcdf_path

    def _run_openeo_job_wave(self, remote_jobs):
        """Run and download one bounded wave of already-created openEO jobs."""
        # Download completed remote jobs in parallel using a thread pool.
        # Each job writes a temporary partial file and is atomically renamed
        # into place to avoid exposing incomplete downloads as cache entries.
        if not remote_jobs:
            return
        batch_numbers = [number for number, *_rest in remote_jobs]
        self.source_logger.info(f"Starting openEO job wave for batches {batch_numbers}.")
        workers = min(self.batch_workers, len(remote_jobs))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Network-bound downloads can share threads without copying the full pipeline into processes.
            futures = {
                executor.submit(self._run_openeo_batch_job, job, temporary, path, number): number
                for number, job, temporary, path in remote_jobs
            }
            for future in as_completed(futures):
                expected = futures[future]
                number, path = future.result()
                if number != expected:
                    raise RuntimeError(f"openEO job returned batch {number}; expected {expected}.")
                self.source_logger.info(f"Cached completed batch {number} temporal cube at {path}.")

    def _run_openeo_jobs(self, cube_dir):
        """Create, run and download uncached batches in bounded job waves."""
        connection = None
        pending_batches, remote_jobs = [], []
        for batch_number, batch in enumerate(self._iter_spatial_batches(), start=1):
            netcdf_path = cube_dir / f"batch_{batch_number:05d}_monthly.nc"
            # Existing final files are complete cache entries; partial files are never reused.
            if netcdf_path.exists():
                self.source_logger.info(f"Using existing batch {batch_number} temporal cube {netcdf_path}.")
            else:
                if connection is None:
                    self.source_logger.info(f"Connecting to openEO backend {self.OPENEO_URL}.")
                    connection = openeo.connect(self.OPENEO_URL, auto_validate=False)
                    self._authenticate_openeo_connection(connection)
                print(f"Creating openEO job for batch {batch_number} with {len(batch)} parcels.")
                feature_collection = self._to_feature_collection(batch)
                cube = self._build_multisensor_temporal_cube(connection, feature_collection)
                temporary_path = netcdf_path.with_suffix(".partial.nc")
                if temporary_path.exists():
                    temporary_path.unlink()
                job = cube.create_job(out_format="netCDF", title=f"Sentinel-1/Sentinel-2 temporal cube batch {batch_number}")
                remote_jobs.append((batch_number, job, temporary_path, netcdf_path))
                if len(remote_jobs) >= self.batch_workers:
                    self._run_openeo_job_wave(remote_jobs)
                    remote_jobs.clear()
            pending_batches.append((batch_number, batch, netcdf_path))
        if not pending_batches:
            raise RuntimeError("No processing batches were produced.")
        self._run_openeo_job_wave(remote_jobs)
        return pending_batches
