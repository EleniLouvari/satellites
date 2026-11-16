"""Streamed checkpoint execution and period-context orchestration."""

from __future__ import annotations

import gc
import traceback

import numpy as np
import pandas as pd
import xarray as xr

from shared.io import read_data

from ..streamed_raster import CHECKPOINT_CHUNK_PIXELS, LOCAL_RASTER_LOCK


class ParcelStatisticsStreamingMixin:
    """Methods for streaming cube checkpointing and parcel-stat row assembly."""

    # The local worker serializes access to raster data so only one cube is processed at a time while
    # still allowing each parcel batch to run independently once it has acquired the lock.
    def _calculate_local_statistics(self, netcdf_path, parcels, batch_number):
        """Bound local concurrency and release resources after successful or failed cubes."""
        with LOCAL_RASTER_LOCK:
            try:
                return self._calculate_streamed_statistics(netcdf_path, parcels, batch_number)
            except Exception as error:
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                gc.collect()

    def _calculate_streamed_statistics(self, netcdf_path, parcels, batch_number):
        """Clean on disk, then read only parcel windows from the final checkpoint."""
        self.parcel_logger.info(f"Reading temporal pixel cube {netcdf_path}.")
        with xr.open_dataset(netcdf_path, decode_coords="all", mask_and_scale=True, cache=False) as dataset:
            stream_context = self._build_stream_statistics_context(dataset, parcels)
            final_path = self._ensure_streamed_checkpoint(
                netcdf_path,
                dataset,
                stream_context["dimensions"],
                stream_context["order"],
                stream_context["periods"],
                stream_context["cube_variables"],
                stream_context["variables"],
                stream_context["spatial_shape"],
                stream_context["transform"],
                stream_context["cube_crs"],
                batch_number,
            )
            _, report_path = self._cleaning_checkpoint_paths(netcdf_path)
            raster_report = read_data(str(report_path))

        with (
            self._parcel_read_chunk_cache(len(stream_context["periods"]), len(stream_context["variables"])),
            xr.open_dataset(final_path, cache=False) as final_checkpoint,
        ):
            rows = self._build_streamed_stat_rows(final_checkpoint, parcels, stream_context)

        result = pd.DataFrame(rows)
        self.parcel_logger.info(
            f"Calculated {list(self.spatial_statistics)} locally for {len(parcels)} parcels from {netcdf_path.name}."
        )
        return result, raster_report

    # The stream context is the reusable metadata bundle that keeps parcel windows, timestamps, and CRS
    # aligned across the checkpoint-writing and final row-building stages. It is intentionally small but
    # complete enough that each parcel batch can be reproduced without re-reading the source NetCDF.
    def _build_stream_statistics_context(self, dataset, parcels):
        """Prepare temporal/raster context required for streamed parcel statistics."""
        time_dimension = self._find_cube_dimension(dataset, ("t", "time", "temporal"), "temporal")
        x_dimension = self._find_cube_dimension(dataset, ("x", "longitude", "lon"), "x")
        y_dimension = self._find_cube_dimension(dataset, ("y", "latitude", "lat"), "y")
        cube_variables = list(self._cube_sensor_variables())
        cube_crs, transform = self._resolve_stream_cube_crs(dataset, time_dimension, x_dimension, y_dimension, cube_variables)
        periods_context = self._resolve_stream_periods(dataset, time_dimension)
        variables = list(self._output_sensor_variables())
        return {
            "dimensions": (time_dimension, x_dimension, y_dimension),
            "cube_variables": cube_variables,
            "cube_crs": cube_crs,
            "transform": transform,
            "order": periods_context["order"],
            "periods": periods_context["periods"],
            "result_periods": periods_context["result_periods"],
            "period_groups": periods_context["period_groups"],
            "period_end_lookup": periods_context["period_end_lookup"],
            "variables": variables,
            "projected": parcels.to_crs(cube_crs),
            "spatial_shape": (dataset.sizes[y_dimension], dataset.sizes[x_dimension]),
            "pool_acquisitions": periods_context["pool_acquisitions"],
        }

    def _resolve_stream_cube_crs(self, dataset, time_dimension, x_dimension, y_dimension, cube_variables):
        """Resolve CRS and transform used for streamed checkpoint generation."""
        monthly_values = self._select_monthly_variables(
            dataset.isel({time_dimension: slice(0, 0)}),
            time_dimension,
            x_dimension,
            y_dimension,
            expected_variables=cube_variables,
        )
        monthly_values = monthly_values.rio.set_spatial_dims(x_dim=x_dimension, y_dim=y_dimension, inplace=False)
        cube_crs = monthly_values.rio.crs or dataset.rio.crs
        if cube_crs is None:
            raise ValueError(
                "Error: The native-CRS NetCDF does not contain usable CRS metadata. Parcel geometries cannot be aligned safely with the raster."
            )
        monthly_values = monthly_values.rio.write_crs(cube_crs, inplace=False)
        return cube_crs, monthly_values.rio.transform(recalc=False)

    # Period resolution decides whether acquisitions are pooled into reporting intervals or kept as single
    # time steps, which changes both the checkpoint layout and the parcel statistics that are returned.
    def _resolve_stream_periods(self, dataset, time_dimension):
        """Resolve checkpoint periods and parcel aggregation groups."""
        pool_acquisitions = getattr(self, "temporal_reducer", "median") == "none"
        time_values = pd.to_datetime(dataset[time_dimension].values, utc=True, errors="coerce")
        if pd.isna(time_values).any():
            raise ValueError(
                f"Error: The NetCDF temporal coordinate {time_dimension!r} contains values that cannot be parsed as dates."
            )
        order = np.argsort(time_values, kind="stable") if pool_acquisitions else np.arange(len(time_values))
        ordered_time_values = time_values[order]
        intervals, labels = self._temporal_intervals()
        period_end_lookup = {label: interval[1] for label, interval in zip(labels, intervals)}
        if pool_acquisitions:
            periods = [timestamp.isoformat() for timestamp in ordered_time_values]
            period_groups = [
                np.flatnonzero((ordered_time_values >= pd.Timestamp(start, tz="UTC")) & (ordered_time_values < pd.Timestamp(end, tz="UTC")))
                for start, end in intervals
            ]
            if sum(len(indices) for indices in period_groups) != len(ordered_time_values):
                raise ValueError("Error: The acquisition NetCDF contains timestamps outside the configured temporal intervals.")
            result_periods = labels
        else:
            periods = ordered_time_values.strftime("%Y-%m-%d").tolist()
            if len(periods) != len(set(periods)):
                raise ValueError("Error: The temporal NetCDF contains duplicate period-start labels.")
            unknown = sorted(set(periods).difference(period_end_lookup))
            if unknown:
                raise ValueError(f"Error: The temporal NetCDF contains unexpected period labels: {unknown}")
            period_groups = [np.array([index]) for index in range(len(periods))]
            result_periods = periods
        return {
            "pool_acquisitions": pool_acquisitions,
            "order": order,
            "periods": periods,
            "period_groups": period_groups,
            "result_periods": result_periods,
            "period_end_lookup": period_end_lookup,
        }

    # Each parcel is processed one at a time with its own local raster window, then its rows are appended to
    # the final tabular output for downstream reshape and modeling.
    def _build_streamed_stat_rows(self, final_checkpoint, parcels, stream_context):
        """Build parcel-period statistics rows from a streamed checkpoint."""
        transform = stream_context["transform"]
        period_end_lookup = stream_context["period_end_lookup"]
        pixel_cell_area = abs(float(transform.a * transform.e - transform.b * transform.d))
        parcel_ids = parcels[self.PARCEL_ID_FIELD].astype(str).to_list()
        geometries = stream_context["projected"].geometry.to_list()
        parcel_rows_by_position = [[] for _ in parcel_ids]
        # Visit parcels chunk by chunk so cached checkpoint chunks serve neighbours, then restore input order.
        for position in self._spatial_parcel_order(stream_context["projected"], transform):
            parcel_rows, fill_metrics = self._stream_parcel_statistics(
                final_checkpoint,
                geometries[position],
                transform,
                stream_context["result_periods"],
                stream_context["period_groups"],
                stream_context["variables"],
            )
            parcel_rows_by_position[position] = self._enrich_streamed_rows(
                parcel_rows, parcel_ids[position], fill_metrics, pixel_cell_area, period_end_lookup
            )
        return [row for parcel_rows in parcel_rows_by_position for row in parcel_rows]

    @staticmethod
    def _spatial_parcel_order(projected, transform):
        """Return parcel positions ordered by checkpoint chunk, then by pixel row and column."""
        bounds = projected.geometry.bounds
        x, y = bounds["minx"].to_numpy(), bounds["maxy"].to_numpy()
        # The upper-left bound locates each parcel window; empty geometries sort first.
        inverse = ~transform
        columns = inverse.a * x + inverse.b * y + inverse.c
        rows = inverse.d * x + inverse.e * y + inverse.f
        rows = np.floor(np.nan_to_num(rows, nan=0.0))
        columns = np.floor(np.nan_to_num(columns, nan=0.0))
        # np.lexsort uses the last key as the primary key: chunk row, chunk column, row, column.
        return np.lexsort((columns, rows, columns // CHECKPOINT_CHUNK_PIXELS, rows // CHECKPOINT_CHUNK_PIXELS))

    def _enrich_streamed_rows(self, parcel_rows, parcel_id, fill_metrics, pixel_cell_area, period_end_lookup):
        """Attach parcel identifiers and quality metrics to streamed period rows."""
        pixel_count = fill_metrics["intersected_pixel_count"]
        for row in parcel_rows:
            row.update(
                {
                    self.PARCEL_ID_FIELD: parcel_id,
                    "period_end": period_end_lookup[row["period_start"]],
                    "meets_minimum_pixel_count": pixel_count >= self.minimum_parcel_pixels,
                    **fill_metrics,
                    "pixel_area": pixel_count * pixel_cell_area,
                }
            )
        return parcel_rows
