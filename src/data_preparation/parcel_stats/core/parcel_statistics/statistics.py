"""Parcel-window masking and statistic reducers."""

from __future__ import annotations

import math
import warnings

import numpy as np
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import mapping


class ParcelStatisticsReducerMixin:
    """Methods for parcel masking and reducer operations."""

    def _reduce_local_pixels(self, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        """Calculate requested statistics across the final pixel axis."""
        pixel_values = np.where(np.isfinite(pixel_values), pixel_values, np.nan)
        valid_count = np.isfinite(pixel_values).sum(axis=-1)
        if pixel_values.shape[-1] == 0:
            empty = np.full(valid_count.shape, np.nan, dtype="float64")
            return {s: valid_count.copy() if s == "count" else empty.copy() for s in self.spatial_statistics}
        totals = np.nansum(pixel_values, axis=-1)
        mean = np.full(valid_count.shape, np.nan, dtype="float64")
        np.divide(totals, valid_count, out=mean, where=valid_count > 0)
        reduced = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for statistic in self.spatial_statistics:
                if statistic == "mean":
                    values = mean
                elif statistic == "median":
                    values = np.nanmedian(pixel_values, axis=-1)
                elif statistic == "sd":
                    squares = np.nansum((pixel_values - mean[..., np.newaxis]) ** 2, axis=-1)
                    values = np.full(valid_count.shape, np.nan, dtype="float64")
                    np.divide(squares, valid_count - 1, out=values, where=valid_count > 1)
                    values = np.sqrt(values)
                elif statistic == "min":
                    values = np.nanmin(pixel_values, axis=-1)
                elif statistic == "max":
                    values = np.nanmax(pixel_values, axis=-1)
                elif statistic == "count":
                    values = valid_count
                elif statistic == "range":
                    values = np.nanmax(pixel_values, axis=-1) - np.nanmin(pixel_values, axis=-1)
                else:
                    values = np.nanquantile(pixel_values, self.QUANTILE_PROBABILITIES[statistic], axis=-1, method="linear")
                reduced[statistic] = values
        return reduced

    def _reduce_period_pixels(self, pixel_values: np.ndarray, period_groups: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Pool acquisition and spatial samples within each period, separately per variable."""
        reductions = []
        for indices in period_groups:
            selected = pixel_values[indices].transpose(1, 0, 2)
            pooled = selected.reshape(pixel_values.shape[1], len(indices) * pixel_values.shape[2])
            reductions.append(self._reduce_local_pixels(pooled))
        return {statistic: np.stack([result[statistic] for result in reductions]) for statistic in self.spatial_statistics}

    @staticmethod
    def _summarize_parcel_filling(
        observed: np.ndarray, temporal_filled_mask: np.ndarray, final: np.ndarray, intersected_pixel_count: int
    ) -> dict[str, int | float]:
        """Aggregate fill provenance over every requested period and band."""
        expected_pixel_count = intersected_pixel_count * observed.shape[0] * observed.shape[1]
        observed_mask = np.isfinite(observed)
        temporal_mask = temporal_filled_mask.astype(bool, copy=False)
        final_mask = np.isfinite(final)
        observed_count = int(observed_mask.sum())
        temporal_count = int(temporal_mask.sum())
        spatial_count = int((~observed_mask & ~temporal_mask & final_mask).sum())
        return {
            "expected_pixel_count": expected_pixel_count,
            "temporal_filled_pixel_count": temporal_count,
            "spatial_filled_pixel_count": spatial_count,
            "temporal_filled_ratio": temporal_count / expected_pixel_count if expected_pixel_count else np.nan,
            "spatial_filled_ratio": spatial_count / expected_pixel_count if expected_pixel_count else np.nan,
            "data_reliability_score": observed_count / expected_pixel_count if expected_pixel_count else np.nan,
        }

    def _crop_values_to_parcel_mask(
        self, values: np.ndarray, parcel_mask: np.ndarray, transform
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Crop a cube to the bounding window of an eligible-pixel mask."""
        rows, columns = np.nonzero(parcel_mask)
        if rows.size == 0:
            empty_window = Window(col_off=0, row_off=0, width=1, height=1)
            return (values[:, :, :1, :1].copy(), np.zeros((1, 1), dtype=bool), window_transform(empty_window, transform))

        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        column_start, column_stop = int(columns.min()), int(columns.max()) + 1
        window = Window(col_off=column_start, row_off=row_start, width=column_stop - column_start, height=row_stop - row_start)
        return (
            values[:, :, row_start:row_stop, column_start:column_stop].copy(),
            parcel_mask[row_start:row_stop, column_start:column_stop].copy(),
            window_transform(window, transform),
        )

    def _mask_parcel_window(self, values: np.ndarray, geometry, transform):
        """Rasterize one parcel only within its clipped pixel bounding window."""
        raster_height, raster_width = values.shape[-2:]
        raw_window = from_bounds(*geometry.bounds, transform=transform)
        column_start = max(0, math.floor(raw_window.col_off))
        row_start = max(0, math.floor(raw_window.row_off))
        column_stop = min(raster_width, math.ceil(raw_window.col_off + raw_window.width))
        row_stop = min(raster_height, math.ceil(raw_window.row_off + raw_window.height))

        if column_start >= column_stop or row_start >= row_stop:
            empty_window = Window(col_off=0, row_off=0, width=1, height=1)
            return (values[:, :, :1, :1].copy(), np.zeros((1, 1), dtype=bool), window_transform(empty_window, transform))

        window = Window(col_off=column_start, row_off=row_start, width=column_stop - column_start, height=row_stop - row_start)
        parcel_transform = window_transform(window, transform)
        parcel_mask = geometry_mask(
            [mapping(geometry)],
            out_shape=(int(window.height), int(window.width)),
            transform=parcel_transform,
            invert=True,
            all_touched=True,
        )
        return (values[:, :, row_start:row_stop, column_start:column_stop].copy(), parcel_mask, parcel_transform)
