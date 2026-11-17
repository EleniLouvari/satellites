"""Run classification pipeline steps in the run's own directory and publish their outputs."""

import contextlib
import json
import os
import random
import shutil
import traceback
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from psycopg2.extras import Json

from parcel_app.core import db
from parcel_app.core.storage import checksum, contained, sanitize_text
from parcel_app.jobs.queue import owned_claim
from parcel_app.pipelines.eda.routes import completed_run, source_artifact, source_columns
from parcel_app.settings import get_settings

from .config import CENTROID_COLUMNS, OUTLIER_COLUMN, pipeline_arguments, validate_ml
from .routes import STEPS, annotated_source, stale_artifact

BUNDLE = "ml-results.zip"


def load_frame(path, eda_path, features, config):
    """Build the model table from the parcel result plus the EDA columns the user selected.

    EDA keeps the rows of the parcel result in order, so its calculated columns are attached
    by position. ``is_outlier`` is always attached as inspection metadata; it is a model
    feature only when the user selected it.
    """
    import pandas as pd

    metadata = source_columns(path)
    names = {column["name"] for column in metadata}
    geometry = [column["name"] for column in metadata if column["geometry"]]
    identifier, target = config["id_column"], config["target_column"]
    reliability = [config["inspection_data_reliability_column"], config["inspection_geometry_complexity_column"]]
    wanted = [identifier, target, *reliability, *features]
    columns = list(dict.fromkeys([name for name in wanted if name in names] + geometry))
    if geometry:
        import geopandas as gpd

        frame = gpd.read_parquet(path, columns=columns)
        if frame.crs is not None and frame.crs.is_geographic:
            # Interpolation distances and centroids are in metres, so use a projected CRS.
            frame = frame.to_crs(frame.estimate_utm_crs())
        if config["add_centroid_features"]:
            centroids = frame.geometry.centroid
            frame[CENTROID_COLUMNS[0]], frame[CENTROID_COLUMNS[1]] = centroids.x, centroids.y
    else:
        frame = pd.read_parquet(path, columns=columns)
    eda_names = {column["name"] for column in source_columns(eda_path)}
    calculated = list(dict.fromkeys([OUTLIER_COLUMN, *[name for name in features if name not in names]]))
    check = [identifier] if identifier in eda_names else []
    eda = pd.read_parquet(eda_path, columns=[*check, *calculated])
    same_rows = len(eda) == len(frame) and all(
        (eda[name].astype(str).to_numpy() == frame[name].astype(str).to_numpy()).all() for name in check
    )
    if not same_rows:
        raise ValueError("Error: The EDA result does not match the parcel result row by row.")
    for name in calculated:
        frame[name] = eda[name].to_numpy()
    # Older EDA results stored unscored rows as null; a missing flag means "not an outlier".
    frame[OUTLIER_COLUMN] = pd.Series(frame[OUTLIER_COLUMN].to_numpy(), index=frame.index).eq(True).astype(bool)
    excluded = config["exclude_target_values"]
    if excluded:
        numbers = pd.to_numeric(pd.Series(excluded, dtype="object"), errors="coerce").dropna()
        drop = frame[target].astype(str).isin([str(value) for value in excluded])
        drop |= pd.to_numeric(frame[target], errors="coerce").isin(numbers)
        frame = frame.loc[~drop].copy()
    return frame


def seed_process(config):
    """Apply the notebook's reproducibility setup before the pipeline imports its models."""
    import numpy as np

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["SATELLITES_DISABLE_HTML_REPORT_OPEN"] = "1"
    random.seed(config["random_state"])
    np.random.seed(config["random_state"])
    if {"tensorflow_neural_network", "keras_lstm"} & set(config["selected_models"]):
        os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_CUDNN_USE_AUTOTUNE="0", OMP_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
        with contextlib.suppress(ImportError):
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")
            tf.random.set_seed(config["random_state"])


def run_ml_steps(path, eda_path, output, features, config, start=0, through=None, on_step=None):
    """Run steps ``start + 1`` to ``through`` (default: the last one), reusing the saved earlier steps.

    ``on_step(index, finished)`` is called before a step starts and again when it has
    finished, so progress survives a failure in a later step.
    """
    output = Path(output)
    # The first step starts the run's folder; a retried first step overwrites its partial files.
    output.mkdir(parents=True, exist_ok=True)
    metadata = source_columns(path)
    available = [column["name"] for column in metadata if not column["geometry"]]
    available += [column["name"] for column in source_columns(eda_path) if column["name"] not in available]
    config = validate_ml(config, features, available, geometry=any(column["geometry"] for column in metadata))
    (output / "configuration.json").write_text(json.dumps({"features": features, "config": config}, indent=2), encoding="utf-8")
    report = on_step or (lambda index, finished: None)
    with (
        # Line buffering lets the live view show each message as soon as it is printed.
        (output / "ml.log").open("a", encoding="utf-8", buffering=1) as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            seed_process(config)
            from ml_classification import ClassificationPipelineConfig, GeospatialClassificationPipeline

            # Saved models are reused, so a step interrupted while training continues where it stopped.
            pipeline = GeospatialClassificationPipeline(
                ClassificationPipelineConfig(project_dir=output, **pipeline_arguments(config, features))
            )
            for index in range(start, through or len(STEPS)):
                report(index, False)
                method = getattr(pipeline, STEPS[index][1])
                if index == 0:
                    method(load_frame(path, eda_path, features, config))
                else:
                    method()
                report(index, True)
        except Exception:
            log.write(sanitize_text(traceback.format_exc()))
            raise


def publish(output, complete):
    """List the run's files with checksums; a finished run also gets one ZIP of everything."""
    output = Path(output)
    artifacts = []
    for candidate in sorted(output.rglob("*")):
        if candidate.is_file() and candidate.name not in {BUNDLE, "ml-results.partial.zip"}:
            artifact = contained(output, candidate)
            artifacts.append(
                {
                    "artifact_id": str(uuid4()),
                    "name": artifact.relative_to(output.resolve()).as_posix(),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": checksum(artifact),
                }
            )
    if complete:
        temporary = output / "ml-results.partial.zip"
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for item in artifacts:
                archive.write(contained(output, output / item["name"]), item["name"])
        bundle = output / BUNDLE
        temporary.replace(bundle)
        artifacts.append(
            {"artifact_id": str(uuid4()), "name": BUNDLE, "size_bytes": bundle.stat().st_size, "sha256": checksum(bundle)}
        )
    return artifacts


def clear_for_rerun(output, start, retrain):
    """Remove the outputs a re-run from step index ``start`` replaces.

    Train keeps its folder so that every model not named in ``retrain`` is reused; the
    models named there lose their saved artifacts and are trained afresh.
    """
    output = Path(output)
    train = STEPS[2][3].split("/", 1)[0]
    for candidate in sorted(output.iterdir()) if output.is_dir() else ():
        if candidate.name != train and stale_artifact(candidate.name, start):
            path = contained(output, candidate)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    for model in retrain:
        folder = contained(output, output / train / "models" / model)
        if folder.is_dir():
            shutil.rmtree(folder)


def execute_ml(job):
    """Run the requested steps of one owned ML run, saving progress after every step."""
    target = job["target_id"]
    row = db.one("SELECT * FROM ml_runs WHERE ml_id=%s AND app_user_id=%s", (target, job["app_user_id"]))
    if not row:
        raise ValueError("Error: ML run is unavailable.")
    run = completed_run(row["run_id"], row["app_user_id"])
    artifact, source = source_artifact(run)
    eda = db.one("SELECT * FROM eda_runs WHERE eda_id=%s AND app_user_id=%s", (str(row["eda_id"]), row["app_user_id"]))
    if not eda:
        raise ValueError("Error: The EDA analysis is unavailable.")
    eda_artifact, eda_source = annotated_source(eda)
    start = row["steps_done"]
    through = min(int(job["payload"].get("through", len(STEPS))), len(STEPS))
    if start == 0:
        # Only the first step reads the sources; later steps use the checked data saved by it.
        if artifact["sha256"] != row["input_sha256"] or checksum(source) != row["input_sha256"]:
            raise ValueError("Error: The parcel result changed after ML submission.")
        if eda_artifact["sha256"] != row["eda_sha256"] or checksum(eda_source) != row["eda_sha256"]:
            raise ValueError("Error: The EDA result changed after ML submission.")
    output = contained(get_settings().storage_root, row["output_dir"])

    def save(sql, parameters):
        """Write run state only while this worker still owns the job."""
        with db.connection() as conn:
            owned_claim(job, conn)
            db.execute(sql, parameters, conn=conn)

    save("UPDATE ml_runs SET status='running',started_at=COALESCE(started_at,now()),error=NULL WHERE ml_id=%s", (target,))

    def on_step(index, finished):
        """Record a finished step durably; progress text alone must never fail the run."""
        done = index + finished
        progress = {"completed": done, "total": len(STEPS), "percent": round(100 * done / len(STEPS)), "stage": STEPS[index][2]}
        if finished:
            save("UPDATE ml_runs SET steps_done=%s,progress=%s WHERE ml_id=%s", (done, Json(progress), target))
        else:
            with contextlib.suppress(Exception):
                save("UPDATE ml_runs SET progress=%s WHERE ml_id=%s", (Json(progress), target))

    if job["payload"].get("rerun") is not None:
        clear_for_rerun(output, int(job["payload"]["rerun"]), job["payload"].get("retrain", []))
    try:
        run_ml_steps(source, eda_source, output, row["feature_columns"], row["config"], start, through, on_step)
    except Exception:
        # Keep the reports of the steps that did finish available next to the failure.
        with contextlib.suppress(Exception):
            save("UPDATE ml_runs SET artifacts=%s WHERE ml_id=%s", (Json(publish(output, False)), target))
        raise
    complete = through == len(STEPS)
    save(
        "UPDATE ml_runs SET status=%s,artifacts=%s,finished_at=now() WHERE ml_id=%s",
        ("complete" if complete else "paused", Json(publish(output, complete)), target),
    )


def mark_ml_failed(job, message, conn):
    """Record a failed ML step; the finished steps stay usable for a retry."""
    db.execute(
        "UPDATE ml_runs SET status='failed',error=%s,finished_at=now() WHERE ml_id=%s",
        (message, job["target_id"]),
        conn=conn,
    )
