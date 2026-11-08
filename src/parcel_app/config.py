"""Expose scientific configuration metadata without trusting server paths from clients."""

import inspect
import json

from fastapi import APIRouter, Depends

from .security import current_user
from .settings import get_settings

router = APIRouter()


def check_processing(value: dict) -> dict:
    """Apply the runner's request contract and operator concurrency ceiling."""
    from data_preparation.parcel_stats.run_config import _processing

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

    overrides = {"sentinel2_bands": ["B04", "B08"], "sentinel2_indices": ["NDVI"],
                 "spatial_statistics": ["mean", "median"], "batch_workers": 1,
                 "fit_tiles_to_parcels": True, "remove_nc_after_completion": True}
    choices = {"temporal_reducer": ["median", "mean", "none"], "temporal_fill_mode": ["past_only", "bidirectional"],
               "interpolation_method": ["nearest", "linear", "cubic", "kriging"],
               "sentinel1_orbit_direction": ["BOTH", "ASCENDING", "DESCENDING"]}
    defaults, fields = {}, []
    for name, parameter in _processing_parameters().items():
        value = parameter.default
        if value is inspect.Parameter.empty:
            value = None
        value = overrides.get(name, value)
        value = json.loads(json.dumps(value))
        kind = "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "array" if isinstance(value, list) else "string"
        if name in {"start_date", "end_date"}:
            kind = "date"
        defaults[name] = value
        fields.append({"name": name, "type": kind, "default": value, "label": name.replace("_", " ").capitalize(),
                       "choices": choices.get(name), "nullable": value is None})
    return {"defaults": defaults, "fields": fields, "limits": {"max_batch_workers": get_settings().max_batch_workers,
            "upload_limit_bytes": get_settings().upload_limit_bytes}}
