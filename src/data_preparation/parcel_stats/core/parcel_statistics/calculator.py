"""Public parcel statistics calculator assembled from focused mixins."""

from __future__ import annotations

import rioxarray  # noqa: F401 - registers the xarray ``rio`` accessor.

from ..streamed_raster import StreamedRasterProcessing
from .checkpoints import ParcelStatisticsCheckpointMixin
from .reshape import ParcelStatisticsReshapeMixin
from .statistics import ParcelStatisticsReducerMixin
from .streaming import ParcelStatisticsStreamingMixin


class ParcelStatisticsCalculator(
    ParcelStatisticsCheckpointMixin,
    ParcelStatisticsReducerMixin,
    ParcelStatisticsStreamingMixin,
    ParcelStatisticsReshapeMixin,
    StreamedRasterProcessing,
):
    """Aggregate an already-cleaned raster into parcel-level statistics."""
