"""Inspection of the best ML scenario: every parcel's predictions in inspection order, with user reviews."""

import csv
import math
import re
from datetime import date as date_type
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from parcel_app.core import db
from parcel_app.core.security import current_user
from parcel_app.core.storage import contained
from parcel_app.pipelines.eda.routes import completed_run, source_artifact, source_columns
from parcel_app.pipelines.parcel_stats.routes import owned_run
from parcel_app.settings import get_settings
from parcel_app.sources.openeo import sentinel2_tiles

from .routes import owned_ml, public_ml

router = APIRouter()
PREDICTIONS = "05_predict/final_predictions_preview.csv"
# Inspection need bands, most urgent first, as ordered by the Predict step's inspection queue.
NEED_ORDER = ("VERY_HIGH", "HIGH", "MEDIUM_HIGH", "MEDIUM", "LOW_MEDIUM", "LOW", "VERY_LOW")
PAGE_LIMIT = 500
# Parcel statistics shown beside an index on the inspection map, when the analysis calculated them.
PARCEL_STATISTICS = ("mean", "median", "sd", "range")


def need_level(need):
    """Map an inspection need to 1 (most urgent) … 0 (least urgent); unknown needs have no level."""
    if need not in NEED_ORDER:
        return None
    return 1 - NEED_ORDER.index(need) / (len(NEED_ORDER) - 1)


def queue_key(need, score):
    """Sort by need band, then by inspection score (highest first); unknown bands and scores go last."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = math.nan
    rank = NEED_ORDER.index(need) if need in NEED_ORDER else len(NEED_ORDER)
    return (rank, math.isnan(value), -value if not math.isnan(value) else 0.0)


@lru_cache(maxsize=4)
def ordered_predictions(path, sha256):
    """Read every parcel's prediction row once and order it for inspection; ties keep the file order.

    ``sha256`` is unused in the body: it keys the cache to one published version of the file.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        rows = list(reader)
    if "inspection_need" not in columns:
        return columns, rows
    need, score = columns.index("inspection_need"), columns.index("inspection_score") if "inspection_score" in columns else None
    rows.sort(key=lambda row: queue_key(row[need], row[score] if score is not None else None))
    return columns, rows


def best_predictions(row):
    """Return the ordered predictions of a finished best scenario, or explain why there are none."""
    if not row["is_best"]:
        raise HTTPException(409, "Error: Mark this scenario as the best one to inspect its parcels.")
    if row["status"] != "complete":
        raise HTTPException(409, "Error: Wait for the scenario to finish all steps.")
    artifact = next((item for item in row["artifacts"] if item["name"] == PREDICTIONS), None)
    path = contained(row["output_dir"], Path(row["output_dir"]) / PREDICTIONS)
    if artifact is None or not path.is_file():
        raise HTTPException(409, "Error: This scenario has no parcel predictions.")
    columns, rows = ordered_predictions(str(path), artifact["sha256"])
    identifier = row["config"].get("id_column")
    if identifier not in columns or "inspection_need" not in columns:
        raise HTTPException(409, "Error: This scenario has no inspection priority; re-run Predict with it turned on.")
    return columns, rows, columns.index(identifier)


class BestChoice(BaseModel):
    """Mark or unmark a scenario as the best of its EDA analysis."""

    model_config = ConfigDict(extra="forbid")
    best: bool


@router.put("/ml/{ml_id}/best")
def mark_best(ml_id: UUID, body: BestChoice, user: dict = Depends(current_user)):
    """Make a finished scenario the single best one of its EDA analysis; reviews of others are kept."""
    user_id = user["app_user_id"]
    with db.connection() as conn:
        row = db.one("SELECT eda_id FROM ml_runs WHERE ml_id=%s AND app_user_id=%s", (str(ml_id), user_id), conn=conn)
        if not row:
            raise HTTPException(404, "Error: ML run not found.")
        # Serialise choices within one EDA analysis so only one scenario is ever the best.
        db.one("SELECT 1 FROM eda_runs WHERE eda_id=%s FOR UPDATE", (str(row["eda_id"]),), conn=conn)
        row = db.one("SELECT * FROM ml_runs WHERE ml_id=%s AND app_user_id=%s FOR UPDATE", (str(ml_id), user_id), conn=conn)
        if row["status"] == "deleting":
            raise HTTPException(409, "Error: This scenario is being deleted.")
        if body.best and row["status"] != "complete":
            raise HTTPException(409, "Error: Only a finished scenario can be marked as the best.")
        if body.best:
            db.execute(
                "UPDATE ml_runs SET is_best=false WHERE eda_id=%s AND is_best AND ml_id<>%s",
                (str(row["eda_id"]), str(ml_id)),
                conn=conn,
            )
        db.execute("UPDATE ml_runs SET is_best=%s WHERE ml_id=%s", (body.best, str(ml_id)), conn=conn)
        return public_ml(dict(row, is_best=body.best))


def shown_columns(columns, config):
    """Return the indexes of the columns to show, without ``predicted_class`` when it copies the prediction column.

    Predict always writes ``predicted_class`` as a copy of the configured prediction column
    (``<target>_prediction`` by default), so showing both would repeat the same value.
    """
    prediction = config.get("prediction_column") or f"{config.get('target_column')}_prediction"
    duplicate = "predicted_class" if prediction != "predicted_class" and prediction in columns else None
    return [index for index, name in enumerate(columns) if name != duplicate]


def public_review(review):
    """Expose only the editable review fields."""
    if review is None:
        return {"check_status": "not_checked", "approved": False, "comment": "", "updated_at": None}
    return {key: review[key] for key in ("check_status", "approved", "comment", "updated_at")}


@router.get("/ml/{ml_id}/inspection")
def inspection(
    ml_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=PAGE_LIMIT),
    user: dict = Depends(current_user),
):
    """Return one page of parcels, most in need of inspection first, with their saved reviews."""
    row = owned_ml(ml_id, user["app_user_id"])
    columns, rows, identifier = best_predictions(row)
    page = rows[offset : offset + limit]
    ids = [values[identifier] for values in page]
    reviews = {
        review["parcel_id"]: review
        for review in db.all_rows(
            "SELECT * FROM ml_inspection_reviews WHERE ml_id=%s AND parcel_id=ANY(%s)", (str(ml_id), ids)
        )
    }
    counts = db.one(
        "SELECT count(*) FILTER (WHERE check_status<>'not_checked') AS checked, count(*) FILTER (WHERE approved) AS approved "
        "FROM ml_inspection_reviews WHERE ml_id=%s",
        (str(ml_id),),
    )
    need = columns.index("inspection_need")
    shown = shown_columns(columns, row["config"])
    return {
        "columns": [columns[index] for index in shown],
        "id_column": columns[identifier],
        "total": len(rows),
        "offset": offset,
        "checked": counts["checked"],
        "approved": counts["approved"],
        "rows": [
            {
                "values": [values[index] for index in shown],
                "level": need_level(values[need]),
                "review": public_review(reviews.get(values[identifier])),
            }
            for values in page
        ],
    }


class Review(BaseModel):
    """One parcel's review: how it was checked, whether the declaration was confirmed, and notes."""

    model_config = ConfigDict(extra="forbid")
    parcel_id: str = Field(min_length=1, max_length=500)
    check_status: Literal["not_checked", "satellite", "field", "satellite_and_field"]
    approved: bool = False
    comment: str = Field(default="", max_length=2000)


@router.put("/ml/{ml_id}/inspection/review")
def save_review(ml_id: UUID, body: Review, user: dict = Depends(current_user)):
    """Save one parcel's review of the best scenario; only parcels of its predictions are accepted."""
    if body.approved and body.check_status == "not_checked":
        raise HTTPException(422, "Error: Check a parcel before approving it.")
    row = owned_ml(ml_id, user["app_user_id"])
    _, rows, identifier = best_predictions(row)
    if body.parcel_id not in {values[identifier] for values in rows}:
        raise HTTPException(404, "Error: Parcel not found in this scenario's predictions.")
    review = db.one(
        "INSERT INTO ml_inspection_reviews(ml_id,parcel_id,check_status,approved,comment) VALUES(%s,%s,%s,%s,%s) "
        "ON CONFLICT (ml_id,parcel_id) DO UPDATE SET check_status=EXCLUDED.check_status,approved=EXCLUDED.approved,"
        "comment=EXCLUDED.comment,updated_at=now() RETURNING *",
        (str(ml_id), body.parcel_id, body.check_status, body.approved, body.comment),
    )
    return public_review(review)



def parcel_geometry(path, identifier, parcel_id):
    """Read one parcel's geometry from the parcel result as WGS84 GeoJSON; identifiers are compared as text."""
    import geopandas as gpd
    import pyarrow as pa
    import pyarrow.dataset as ds
    from shapely.geometry import mapping

    columns = source_columns(path)
    geometry = next((column["name"] for column in columns if column["geometry"]), None)
    if geometry is None or identifier not in {column["name"] for column in columns}:
        raise HTTPException(409, "Error: The parcel result has no parcel geometries.")
    frame = gpd.read_parquet(
        path, columns=[identifier, geometry], filters=ds.field(identifier).cast(pa.string()) == parcel_id
    )
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    if frame.empty or frame.crs is None:
        raise HTTPException(404, "Error: This parcel has no geometry.")
    shape = frame.to_crs(4326).geometry.union_all()
    return {"type": "Feature", "properties": {}, "geometry": mapping(shape)}


def parcel_statistics(path, identifier, parcel_id, variable):
    """Read one parcel's saved spatial statistics of one variable, one entry per statistics period, oldest first.

    Dated columns are named ``<variable>_<statistic>__YYYYMMDD`` after the period start. Only the
    statistics calculated in the analysis are returned; a missing value (no valid pixels) is ``None``.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    names = {column["name"] for column in source_columns(path)}
    if identifier not in names:
        raise HTTPException(409, "Error: The parcel result has no parcel identifiers.")
    pattern = re.compile(rf"{re.escape(variable)}_({'|'.join(PARCEL_STATISTICS)})__(\d{{8}})")
    dated = {name: match.groups() for name in names if (match := pattern.fullmatch(name))}
    if not dated:
        return []
    table = pq.read_table(path, columns=sorted(dated), filters=ds.field(identifier).cast(pa.string()) == parcel_id)
    if not table.num_rows:
        raise HTTPException(404, "Error: This parcel has no saved statistics.")
    values = table.slice(0, 1).to_pylist()[0]
    periods = {}
    for name, (statistic, label) in dated.items():
        value = values[name]
        # NaN means no valid pixel in the period; JSON has no NaN, so it becomes null.
        periods.setdefault(label, {})[statistic] = float(value) if value is not None and math.isfinite(value) else None
    return [
        {"start": f"{label[:4]}-{label[4:6]}-{label[6:]}", "values": periods[label]} for label in sorted(periods)
    ]


def require_parcel(row, parcel_id):
    """Refuse a parcel that is not among the best scenario's predictions, before any parcel file is read."""
    _, rows, identifier = best_predictions(row)
    if parcel_id not in {values[identifier] for values in rows}:
        raise HTTPException(404, "Error: Parcel not found in this scenario's predictions.")


def parcel_result(row):
    """Return the parcel statistics file behind a scenario."""
    _, source = source_artifact(completed_run(row["run_id"], row["app_user_id"]))
    return str(source)


def inspected_parcel(row, parcel_id):
    """Return the footprint of one parcel of the best scenario's predictions."""
    require_parcel(row, parcel_id)
    return parcel_geometry(parcel_result(row), row["config"]["id_column"], parcel_id)


@router.get("/ml/{ml_id}/inspection/geometry")
def inspection_geometry(ml_id: UUID, parcel_id: str = Query(min_length=1, max_length=500), user: dict = Depends(current_user)):
    """Return one inspected parcel's footprint for the map, without any of its attributes."""
    return inspected_parcel(owned_ml(ml_id, user["app_user_id"]), parcel_id)


def sentinel2_criteria(row):
    """Return the image dates and scene cloud limit of the parcel statistics run behind a scenario, if it has dates."""
    run = owned_run(row["run_id"], row["app_user_id"])
    processing = {**(run["processing"] or {}), **((run["config"] or {}).get("processing") or {})}
    if not processing.get("start_date") or not processing.get("end_date"):
        return None
    cloud = processing.get("sentinel2_max_scene_cloud_cover")
    if cloud is None:
        from data_preparation.parcel_stats.core.base import ParcelStatsBase

        cloud = ParcelStatsBase.SENTINEL2_MAX_SCENE_CLOUD_COVER
    return {"start_date": processing["start_date"], "end_date": processing["end_date"], "max_cloud_cover": cloud}


@router.get("/ml/{ml_id}/inspection/sentinel2")
def inspection_sentinel2(ml_id: UUID, user: dict = Depends(current_user)):
    """Say whether Sentinel-2 tiles are available and which run criteria select their images."""
    row = owned_ml(ml_id, user["app_user_id"])
    run = owned_run(row["run_id"], user["app_user_id"])
    if run.get("source", "openeo") == "hub":
        return {"source": "hub", "available": False}
    criteria = sentinel2_criteria(row)
    available = bool(sentinelhub_instance(user["app_user_id"])) and criteria is not None
    return {"source": "openeo", "available": available, "min_zoom": sentinel2_tiles.MIN_ZOOM, "max_zoom": sentinel2_tiles.MAX_ZOOM,
            **(criteria or {})}


def sentinelhub_instance(app_user_id):
    """Return a random one of the owner's account instances, spreading processing units; else the server's."""
    row = db.one(
        """SELECT sen_instance_id FROM eo_users WHERE app_user_id=%s AND COALESCE(sen_instance_id,'')<>''
        ORDER BY random() LIMIT 1""",
        (app_user_id,),
    )
    return row["sen_instance_id"].strip() if row else get_settings().sentinelhub_instance_id


def configured_criteria(row):
    """Return a Sentinel Hub instance and the run criteria, or explain why Sentinel-2 is unavailable."""
    instance = sentinelhub_instance(row["app_user_id"])
    if not instance:
        raise HTTPException(404, "Error: Sentinel-2 images are not configured on this server.")
    criteria = sentinel2_criteria(row)
    if criteria is None:
        raise HTTPException(404, "Error: The parcel statistics run has no image dates.")
    return instance, criteria


@router.get("/ml/{ml_id}/inspection/sentinel2/dates")
def inspection_sentinel2_dates(
    ml_id: UUID, parcel_id: str = Query(min_length=1, max_length=500), user: dict = Depends(current_user)
):
    """List the parcel's Sentinel-2 dates within the run's dates and cloud limit, oldest first."""
    from shapely.geometry import shape

    row = owned_ml(ml_id, user["app_user_id"])
    instance, criteria = configured_criteria(row)
    bounds = shape(inspected_parcel(row, parcel_id)["geometry"]).bounds
    return sentinel2_tiles.fetch_dates(instance, bounds, **criteria)


@router.get("/ml/{ml_id}/inspection/sentinel2/{z}/{x}/{y}.png")
def inspection_sentinel2_tile(
    ml_id: UUID,
    z: int,
    x: int,
    y: int,
    date: date_type | None = None,
    user: dict = Depends(current_user),
):
    """Return one Sentinel-2 true-colour tile: of ``date`` when given, else the least cloudy within the run's dates.

    Either way the run's dates and cloud limit apply, so a date outside them is refused.
    """
    if not sentinel2_tiles.MIN_ZOOM <= z <= sentinel2_tiles.MAX_ZOOM or not (0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(404, "Error: Tile out of range.")
    instance, criteria = configured_criteria(owned_ml(ml_id, user["app_user_id"]))
    if date is not None:
        if not criteria["start_date"] <= date.isoformat() <= criteria["end_date"]:
            raise HTTPException(404, "Error: The date is outside the parcel statistics run.")
        criteria = {**criteria, "start_date": date.isoformat(), "end_date": date.isoformat()}
    image = sentinel2_tiles.fetch_tile(instance, z, x, y, **criteria)
    return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})
