"""Expose scientific configuration metadata without trusting server paths from clients."""

import inspect
import json

from fastapi import APIRouter, Depends

from .security import current_user
from .settings import get_settings

router = APIRouter()

GROUPS = [
    {"id": "basic", "label": "Basics", "description": "Dates, reporting intervals, projection and parcel statistics."},
    {"id": "imagery", "label": "Sentinel imagery", "description": "Choose optical and radar bands, indices and orbit direction."},
    {"id": "outliers", "label": "Outlier removal", "description": "Filter extreme raster values before filling gaps."},
    {"id": "filling", "label": "Gap filling", "description": "Control eligibility, temporal filling and spatial neighborhoods."},
    {
        "id": "interpolation",
        "label": "Interpolation",
        "description": "Fill remaining gaps after temporal and neighborhood filling.",
    },
    {"id": "execution", "label": "Tiles and execution", "description": "Tile planning, concurrency, polling and retries."},
    {"id": "storage", "label": "Storage", "description": "Choose which intermediate raster files to retain."},
]
GROUP_FIELDS = {
    "basic": [
        "start_date",
        "end_date",
        "temporal_period",
        "temporal_reducer",
        "working_epsg",
        "spatial_statistics",
        "minimum_parcel_pixels",
    ],
    "imagery": ["sentinel2_bands", "sentinel2_indices", "sentinel1_bands", "sentinel1_indices", "sentinel1_orbit_direction"],
    "outliers": ["remove_outliers", "iqr_quantiles", "iqr_multiplier", "iqr_min_valid_pixels"],
    "filling": [
        "fill_nulls",
        "minimum_observed_fraction_for_fill",
        "temporal_fill_mode",
        "temporal_fill_window_sizes",
        "temporal_fill_block_rows",
        "spatial_fill_window_sizes",
    ],
    "interpolation": [
        "interpolation_method",
        "interpolation_max_distance_in_meters",
        "interpolation_variogram_lags",
        "interpolation_variogram_max_distance_in_meters",
    ],
    "execution": [
        "tile_size_metres",
        "tile_buffer_metres",
        "fit_tiles_to_parcels",
        "batch_workers",
        "openeo_parallel_jobs",
        "openeo_job_options",
        "openeo_auto_memory_retry",
        "openeo_max_executor_memory",
        "job_poll_seconds",
        "max_job_retries",
        "job_retry_delay_seconds",
    ],
    "storage": ["keep_cleaned_checkpoint", "remove_nc_after_completion"],
}


def field_dependencies(name):
    """Describe when a setting actually participates in local processing."""
    if name.startswith("iqr_"):
        return {"remove_outliers": [True]}
    if name in GROUP_FIELDS["filling"] and name != "fill_nulls":
        return {"fill_nulls": [True]}
    if name.startswith("interpolation_"):
        conditions = {"fill_nulls": [True]}
        if name == "interpolation_max_distance_in_meters":
            conditions["interpolation_method"] = ["idw"]
        elif name.startswith("interpolation_variogram_"):
            conditions["interpolation_method"] = ["kriging"]
        return conditions
    return {}


def enabled_field(name, values):
    """Evaluate dependencies with the same canonical values as the form."""
    for parent, allowed in field_dependencies(name).items():
        value = values.get(parent, "nearest" if parent == "interpolation_method" else False)
        if isinstance(value, str):
            value = value.strip().lower()
            value = {"nearest_neighbor": "nearest", "inverse_distance_weighting": "idw"}.get(value, value)
        if value not in allowed:
            return False
    return True


def check_processing(value: dict) -> dict:
    """Apply the runner's request contract and operator concurrency ceiling."""
    from data_preparation.parcel_stats.run_config import _processing, _processing_parameters

    if set(value).difference(_processing_parameters()):
        raise ValueError("Error: Processing configuration contains unsupported fields.")
    # Inactive inputs are excluded from the saved effective configuration. In
    # particular, the engine validates IDW/Kriging distances even when filling is off.
    value = {name: item for name, item in value.items() if enabled_field(name, value)}
    if value.get("working_epsg") is None:
        value.pop("working_epsg", None)
    options = _processing(value)
    workers = options.get("batch_workers", 1)
    if type(workers) is not int or not 1 <= workers <= get_settings().max_batch_workers:
        raise ValueError("Error: Local worker count exceeds the server limit.")
    if len(json.dumps(options, allow_nan=False)) > 32000:
        raise ValueError("Error: Processing configuration is too large.")
    return options


@router.get("/config")
def configuration(user: dict = Depends(current_user)):
    """Return constructor-derived defaults and descriptive form metadata."""
    from data_preparation.parcel_stats.run_config import _processing_parameters

    from data_preparation.parcel_stats.core.base import ParcelStatsBase

    overrides = {
        "working_epsg": None,
        "sentinel2_bands": ["B04", "B08"],
        "sentinel2_indices": ["NDVI"],
        "spatial_statistics": ["mean", "median"],
        "batch_workers": 2,
        "fit_tiles_to_parcels": True,
        "remove_nc_after_completion": True,
        "temporal_fill_mode": "bidirectional",
        "temporal_reducer": "none",
        "spatial_fill_window_sizes": [3, 5, 7],
        "temporal_fill_window_sizes": [3],
        "temporal_fill_block_rows": 512,
    }
    choices = {
        "openeo_parallel_jobs": [1, 2],
        "temporal_reducer": ["median", "mean", "min", "max", "sum", "none"],
        "temporal_fill_mode": ["past_only", "bidirectional"],
        "interpolation_method": ["nearest", "idw", "kriging"],
        "sentinel1_orbit_direction": ["BOTH", "ASCENDING", "DESCENDING"],
    }
    descriptions = {
        "openeo_parallel_jobs": "Parallel remote jobs per Copernicus account. Maximum 2; default 2.",
        "openeo_job_options": (
            'Remote resources as a JSON object, e.g. {"executor-memory":"2G","max-executors":20}. '
            "Blank uses a 2G executor heap; omitted options use backend defaults."
        ),
        "openeo_auto_memory_retry": "Increase remote executor memory only after confirmed JVM out-of-memory failures.",
        "openeo_max_executor_memory": "Maximum executor heap for memory retries, e.g. 8G. Higher memory may use more credits.",
        "temporal_period": "Reporting interval, e.g. 1M or 15D. Also applies when temporal reducer is none.",
        "temporal_reducer": "None retains individual acquisitions; other reducers composite observations per interval.",
        "iqr_quantiles": "Lower and upper quantiles as fractions, e.g. [0.25, 0.75].",
        "temporal_fill_mode": "Bidirectional uses previous and next observations; past_only uses earlier observations.",
        "temporal_fill_window_sizes": "Search windows in time steps, e.g. [3, 5]. Leave blank for an unlimited search.",
        "temporal_fill_block_rows": (
            "Maximum rows processed together across dates, e.g. 256; aligned to storage chunks where possible. "
            "Larger values can use more RAM per worker. Leave blank for automatic sizing."
        ),
        "spatial_fill_window_sizes": "Neighborhood widths in pixels, e.g. [3, 5]. Applied in sequence.",
        "minimum_observed_fraction_for_fill": "Minimum observed coverage from 0 to 1 for a layer to receive or supply fills.",
        "interpolation_max_distance_in_meters": "Required for IDW: maximum search distance in metres.",
        "interpolation_variogram_lags": "Number of lag bins used to fit the kriging variogram.",
        "interpolation_variogram_max_distance_in_meters": "Required for kriging: maximum variogram distance in metres.",
    }
    for name, constant in {
        "sentinel2_bands": "SUPPORTED_SENTINEL2_BANDS",
        "sentinel2_indices": "SUPPORTED_SENTINEL2_INDICES",
        "sentinel1_bands": "SUPPORTED_SENTINEL1_BANDS",
        "sentinel1_indices": "SUPPORTED_SENTINEL1_INDICES",
        "spatial_statistics": "SUPPORTED_SPATIAL_STATISTICS",
    }.items():
        choices[name] = list(getattr(ParcelStatsBase, constant))
    defaults, fields = {}, []
    for name, parameter in _processing_parameters().items():
        value = parameter.default
        if value is inspect.Parameter.empty:
            value = None
        value = overrides.get(name, value)
        value = json.loads(json.dumps(value))
        kind = (
            "boolean"
            if isinstance(value, bool)
            else "number"
            if isinstance(value, (int, float))
            else "array"
            if isinstance(value, list)
            else "string"
        )
        annotation = str(parameter.annotation)
        if isinstance(value, dict) or "dict" in annotation:
            kind = "object"
        if value is None:
            kind = (
                "array"
                if "tuple" in annotation or "list" in annotation
                else "number"
                if "float" in annotation or "int" in annotation
                else kind
            )
        if name in {"start_date", "end_date"}:
            kind = "date"
        defaults[name] = value
        fields.append(
            {
                "name": name,
                "type": kind,
                "default": value,
                "label": name.replace("_", " ").capitalize(),
                "choices": choices.get(name),
                "nullable": value is None or name == "temporal_fill_block_rows",
                "group": next((group for group, names in GROUP_FIELDS.items() if name in names), "execution"),
                "enabled_when": field_dependencies(name),
                "description": descriptions[name]
                if name in descriptions
                else "Leave blank for automatic CRS selection."
                if name == "working_epsg"
                else "Leave blank to use the pipeline default."
                if value is None
                else "",
            }
        )
    # Stable panel order puts each parent switch before its dependent controls.
    ordering = [name for names in GROUP_FIELDS.values() for name in names]
    fields.sort(key=lambda field: ordering.index(field["name"]) if field["name"] in ordering else len(ordering))
    return {
        "groups": GROUPS,
        "defaults": defaults,
        "fields": fields,
        "limits": {
            "max_batch_workers": get_settings().max_batch_workers,
            "upload_limit_bytes": get_settings().upload_limit_bytes,
        },
    }
