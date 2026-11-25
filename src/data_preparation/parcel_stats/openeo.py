"""openEO acquisition composed with the shared local parcel-statistics engine."""
from __future__ import annotations

import geopandas as gpd

from data_preparation.sources.openeo.cubes import OpenEOCubePipeline

from .core.base import ParcelStatsBase


class OpenEOZonalStats(ParcelStatsBase, OpenEOCubePipeline):
    """Acquire Sentinel cubes with openEO and calculate local parcel statistics.

    Constructor options are inherited from ParcelStatsBase without changes.
    """

    def run(self) -> gpd.GeoDataFrame:
        """Run extraction, local cleaning, zonal statistics and persistence."""
        completed = self._load_completed_partition()
        if completed is not None:
            return completed
        # High-level orchestration:
        # 1) create/download cached monthly NetCDF cubes per batch
        # 2) compute local parcel statistics in parallel
        # 3) concatenate, derive additional features and persist outputs
        cube_dir = self.output_dir / "monthly_cubes" / self._run_signature()
        cube_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Temporal cube cache: {cube_dir}.")
        # Run openEO jobs for each uncached parcel batch, downloading the monthly NetCDF cubes.
        batches = self._run_openeo_jobs(cube_dir)
        return self.run_from_batches(batches)
