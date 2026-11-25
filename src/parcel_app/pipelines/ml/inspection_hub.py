"""Owner-authorized HUB imagery for the existing ML inspection map."""

import logging
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from parcel_app.core.security import current_user
from parcel_app.sources.hub import credentials

from . import inspection

router = APIRouter()
logger = logging.getLogger(__name__)


def hub_context(ml_id, user_id):
    """Resolve an owned best scenario and its HUB source before any cache access."""
    row = inspection.owned_ml(ml_id, user_id)
    run = inspection.owned_run(row["run_id"], user_id)
    if run.get("source", "openeo") != "hub":
        raise HTTPException(404, "Error: This analysis does not use HUB imagery.")
    inspection.best_predictions(row)
    processing = {**(run["processing"] or {}), **((run["config"] or {}).get("processing") or {})}
    return row, Path(run["output_dir"]), processing


def provider_call(name, *args):
    """Load the provider on demand, preserving safe validation errors only."""
    try:
        provider = import_module("parcel_app.sources.hub.previews")
        return getattr(provider, name)(*args)
    except ValueError as error:
        # Only explicit provider validation uses this prefix; parsers can raise
        # ValueError too, and their raw messages can include private URLs.
        if str(error).startswith("Error: "):
            raise HTTPException(409, str(error)) from error
        logger.warning(f"Error: HUB preview failed ({type(error).__name__}).")
        raise HTTPException(502, "Error: HUB imagery could not be loaded. Please try again.") from error
    except Exception as error:
        logger.warning(f"Error: HUB preview failed ({type(error).__name__}).")
        raise HTTPException(502, "Error: HUB imagery could not be loaded. Please try again.") from error


def require_hub():
    """Keep configuration failures readable without exposing environment values."""
    if not credentials.configured():
        raise HTTPException(409, "Error: HUB catalog access is not configured on this server.")


@router.get("/ml/{ml_id}/inspection/hub")
def hub_metadata(ml_id: UUID, user: dict = Depends(current_user)):
    """Describe the HUB layers configured for this scenario's source analysis."""
    _, _, processing = hub_context(ml_id, user["app_user_id"])
    available = credentials.configured()
    layers = provider_call("preview_layers", processing) if available else []
    return {
        "available": available,
        "layers": [asdict(layer) for layer in layers],
        "start_date": processing.get("start_date"),
        "end_date": processing.get("end_date"),
    }


@router.get("/ml/{ml_id}/inspection/hub/acquisitions")
def hub_acquisitions(
    ml_id: UUID, parcel_id: str = Query(min_length=1, max_length=500), user: dict = Depends(current_user)
):
    """Return one inspected parcel's exact display bounds and saved acquisitions."""
    row, output, processing = hub_context(ml_id, user["app_user_id"])
    require_hub()
    parcel = inspection.inspected_parcel(row, parcel_id)
    bounds = provider_call("parcel_preview_bounds", parcel)
    acquisitions = provider_call("parcel_acquisitions", output, processing, parcel)
    return {"bounds": bounds, "acquisitions": [asdict(item) for item in acquisitions]}


@router.get("/ml/{ml_id}/inspection/hub/statistics")
def hub_statistics(
    ml_id: UUID,
    parcel_id: str = Query(min_length=1, max_length=500),
    layer: str = Query(min_length=1, max_length=100),
    user: dict = Depends(current_user),
):
    """Return an inspected parcel's saved statistics of one index layer, per statistics period."""
    row, _, processing = hub_context(ml_id, user["app_user_id"])
    selected = next((item for item in provider_call("preview_layers", processing) if item.name == layer), None)
    if selected is None:
        raise HTTPException(404, "Error: This imagery layer was not configured in the analysis.")
    inspection.require_parcel(row, parcel_id)
    # Colour composites have no single value, and Planet basemaps are reference imagery rather than inputs.
    analysed = selected.kind == "index" and selected.sensor in {"S2", "S1"}
    periods = (
        inspection.parcel_statistics(inspection.parcel_result(row), row["config"]["id_column"], parcel_id, layer)
        if analysed
        else []
    )
    return {"variable": layer, "temporal_period": processing.get("temporal_period"), "periods": periods}


@router.get("/ml/{ml_id}/inspection/hub/image.png")
def hub_image(
    ml_id: UUID,
    parcel_id: str = Query(min_length=1, max_length=500),
    acquisition: str = Query(min_length=1, max_length=1000),
    layer: str = Query(min_length=1, max_length=100),
    rev: str | None = Query(default=None, max_length=128),
    user: dict = Depends(current_user),
):
    """Render only a configured layer of an owned parcel's recorded acquisition."""
    row, output, processing = hub_context(ml_id, user["app_user_id"])
    require_hub()
    parcel = inspection.inspected_parcel(row, parcel_id)
    layers = provider_call("preview_layers", processing)
    selected_layer = next((item for item in layers if item.name == layer), None)
    if selected_layer is None:
        raise HTTPException(404, "Error: This imagery layer was not configured in the analysis.")
    acquisitions = provider_call("parcel_acquisitions", output, processing, parcel)
    selected = next((item for item in acquisitions if item.id == acquisition and item.sensor == selected_layer.sensor), None)
    if selected is None:
        raise HTTPException(404, "Error: This acquisition is unavailable for the selected layer and parcel.")
    if rev is not None and rev != selected.revision:
        raise HTTPException(409, "Error: Imagery has changed. Reopen the map to refresh the acquisitions.")
    png = provider_call("render_preview", output, processing, parcel, acquisition, layer)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=86400", "Vary": "Cookie"})
