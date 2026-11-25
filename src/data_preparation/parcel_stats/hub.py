"""HUB Sentinel ARD acquisition composed with shared local parcel processing."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

import geopandas as gpd
import pandas as pd

from data_preparation.sources.hub.cubes import HubCubePipeline

from .core.base import ParcelStatsBase
from .core.batch_execution import run_streamed_batch_statistics


class HubZonalStats(ParcelStatsBase, HubCubePipeline):
    """Read HUB Sentinel ARD windows and calculate local parcel statistics.

    Parcel, date, sensor, cleaning and output arguments follow
    :class:`ParcelStatsBase`. Load environment credentials before importing HUB
    modules. Construction validates configuration without making network calls.

    Parameters
    ----------
    working_epsg:
        Projected CRS for parcel batching and output geometry, default 2100
        (Greek Grid). HUB raster acquisition uses EPSG:2100 independently.
    hub_catalog_url:
        STAC API base URL; defaults to ``CATALOG_URL``.
    hub_client_id:
        Configured catalog client role; defaults to ``CATALOG_CLIENT_ID``.
    hub_read_workers:
        Concurrent COG block-reading workers, default 8. NetCDF writes remain
        on the acquisition thread. Independent of local ``batch_workers``.
    hub_block_size:
        Target-grid block size in pixels, default 512 (minimum 64).
    batch_workers:
        Maximum concurrent local processing processes. Completed cubes start
        processing while subsequent cubes are acquired, even with one worker.

    """

    SOURCE_NAME = "HUB"
    SENTINEL2_COLLECTION = "ARD_S2"
    SENTINEL1_COLLECTION = "ARD_S1"
    SENTINEL2_REFLECTANCE_SCALE_FACTOR = 1.0
    SENTINEL1_BACKSCATTER_COEFFICIENT = "gamma0-terrain-flattened"
    SENTINEL1_BACKSCATTER_SCALE = "linear_power"
    SENTINEL1_ELEVATION_MODEL = "provider-defined"

    def __init__(
        self,
        *args,
        hub_catalog_url: str | None = None,
        hub_client_id: str | None = None,
        hub_read_workers: int = 8,
        hub_block_size: int = 512,
        **kwargs,
    ) -> None:
        """Validate HUB settings before shared initialization creates output files."""
        self._configure_hub_source(
            catalog_url=hub_catalog_url,
            client_id=hub_client_id,
            read_workers=hub_read_workers,
            block_size=hub_block_size,
        )
        # The fifth inherited positional argument already supplies working_epsg.
        if len(args) < 5:
            kwargs.setdefault("working_epsg", 2100)
        super().__init__(*args, **kwargs)

    def __getstate__(self) -> dict:
        """Send local workers processing identity without duplicating acquisition inventories."""
        state = super().__getstate__()
        for name in ("_hub_batches", "_hub_observations", "_hub_manifest"):
            state.pop(name, None)
        return state

    def _run_signature(self) -> str:
        """Separate HUB caches by configuration, catalog and fresh item inventory."""
        source = self._hub_source_signature()
        endpoint = urlsplit(self.hub_catalog_url)
        source["catalog"] = {
            "scheme": endpoint.scheme, "host": endpoint.hostname,
            "port": endpoint.port, "path": endpoint.path.rstrip("/"),
        }
        # Block selection can alter the raster context available to local filling.
        source["block_size"] = self.hub_block_size
        source["spatial_fill_window_sizes"] = self.spatial_fill_window_sizes
        source["interpolation_variogram_max_distance_in_meters"] = self.interpolation_variogram_max_distance_in_meters
        source["configuration"] = super()._run_signature()
        return hashlib.sha256(json.dumps(source, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _parcel_attributes_digest(self) -> str:
        """Fingerprint output attributes without invalidating reusable raster statistics."""
        attributes = pd.DataFrame(self.parcels.drop(columns=self.parcels.geometry.name))
        attributes = attributes.sort_values(self.PARCEL_ID_FIELD).set_index(self.PARCEL_ID_FIELD)
        content = {
            "values": attributes.to_json(
                orient="split", date_format="iso", date_unit="ns", double_precision=15, default_handler=str,
            ),
            "dtypes": [str(dtype) for dtype in attributes.dtypes],
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _partition_completion_record(self, status):
        """Keep inventory and calibration provenance in the validated completion record."""
        return super()._partition_completion_record(status) | {
            "provenance": {
                "source": self.SOURCE_NAME, **self._hub_source_signature(),
                "parcel_attributes_digest": self._parcel_attributes_digest(),
            },
        }

    def _load_completed_partition(self):
        """Require HUB provenance; markerless legacy outputs are an openEO compatibility path."""
        if not self._partition_completion_path().is_file():
            return None
        return super()._load_completed_partition()

    def _run_batch_statistics(self, batches):
        """Overlap incremental HUB acquisition with bounded local worker processes."""
        return run_streamed_batch_statistics(self, batches)

    def run(self) -> gpd.GeoDataFrame:
        """Refresh inventory, reuse compatible outputs, or acquire and process HUB cubes."""
        manifest = self._resolve_hub_manifest()
        completed = self._load_completed_partition()
        if completed is None:
            # Invalidate old completion before publishing a changed inventory or acquiring its rasters.
            self._write_partition_completion("processing")
        # Persist the current inventory even when compatible final outputs are reused.
        manifest_path = self.output_dir / "hub_item_manifest.parquet"
        temporary = manifest_path.with_suffix(".partial.parquet")
        try:
            manifest.to_parquet(temporary, index=False)
            temporary.replace(manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
        if completed is not None:
            return completed
        cube_dir = self.output_dir / "monthly_cubes" / self._run_signature()
        cube_dir.mkdir(parents=True, exist_ok=True)
        self.source_logger.info("HUB temporal cube cache: %s.", cube_dir)
        return self.run_from_batches(self._run_hub_batches(cube_dir))
