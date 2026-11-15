"""Execute EDA in a fresh owned directory and publish its immutable outputs."""

import contextlib
import json
import traceback
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from psycopg2.extras import Json

from parcel_app.core import db
from parcel_app.core.storage import checksum, contained, sanitize_text
from parcel_app.jobs.queue import owned_claim
from parcel_app.settings import get_settings

from .config import validate_eda
from .routes import completed_run, source_artifact, source_columns


def load_selected(path, columns):
    """Read only the selected columns, preserving GeoParquet geometry when included."""
    metadata = source_columns(path)
    if any(column["geometry"] and column["name"] in columns for column in metadata):
        import geopandas as gpd

        return gpd.read_parquet(path, columns=columns)
    import pandas as pd

    return pd.read_parquet(path, columns=columns)


def run_eda_files(path, output, columns, config, progress=None):
    """Run the real pipeline without resets; preserve directory structure in the ZIP."""
    from eda import EDAConfig, EDAPipeline

    output = Path(output)
    # Never reset an existing folder; retries are new EDA submissions with new IDs.
    output.mkdir(parents=True, exist_ok=False)
    frame = load_selected(path, columns)
    config = validate_eda(config, columns, list(frame.columns))
    configuration = {"columns": columns, "config": config}
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    with (
        (output / "eda.log").open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            pipeline = EDAPipeline(EDAConfig(output_dir=output, reset_output_dir=False, **config))
            if progress:
                pipeline.run(frame, progress=progress)
            else:
                pipeline.run(frame)
        except Exception:
            log.write(sanitize_text(traceback.format_exc()))
            raise
    artifacts = []
    for candidate in sorted(output.rglob("*")):
        if candidate.is_file():
            artifact = contained(output, candidate)
            artifacts.append(
                {
                    "artifact_id": str(uuid4()),
                    "name": artifact.relative_to(output.resolve()).as_posix(),
                    "size_bytes": artifact.stat().st_size,
                    "sha256": checksum(artifact),
                }
            )
    temporary = output / "eda-results.partial.zip"
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for item in artifacts:
            archive.write(contained(output, output / item["name"]), item["name"])
    bundle = output / "eda-results.zip"
    temporary.replace(bundle)
    artifacts.append(
        {"artifact_id": str(uuid4()), "name": bundle.name, "size_bytes": bundle.stat().st_size, "sha256": checksum(bundle)}
    )
    return artifacts


def execute_eda(job):
    """Verify claim, ownership and source checksum before committing EDA publication."""
    row = db.one("SELECT * FROM eda_runs WHERE eda_id=%s AND app_user_id=%s", (job["target_id"], job["app_user_id"]))
    if not row:
        raise ValueError("Error: EDA analysis is unavailable.")
    run = completed_run(row["run_id"], row["app_user_id"])
    artifact, source = source_artifact(run)
    if artifact["sha256"] != row["input_sha256"] or checksum(source) != row["input_sha256"]:
        raise ValueError("Error: The source result changed after EDA submission.")
    output = contained(get_settings().storage_root, row["output_dir"])
    with db.connection() as conn:
        owned_claim(job, conn)
        db.execute(
            "UPDATE eda_runs SET status='running',started_at=now(),error=NULL WHERE eda_id=%s", (job["target_id"],), conn=conn
        )
    def progress(completed, stage):
        """Persist stage completion without allowing telemetry errors to fail the analysis."""
        with contextlib.suppress(Exception), db.connection() as connection:
            owned_claim(job, connection)
            value = {"completed": completed, "total": 5, "percent": completed * 20,
                     "stage": "Preparing ZIP" if completed == 4 else stage}
            db.execute("UPDATE eda_runs SET progress=%s WHERE eda_id=%s", (Json(value), job["target_id"]), conn=connection)

    artifacts = run_eda_files(source, output, row["selected_columns"], row["config"], progress=progress)
    with db.connection() as conn:
        owned_claim(job, conn)
        db.execute(
            "UPDATE eda_runs SET status='complete',artifacts=%s,finished_at=now(),"
            "progress='{\"stage\":\"Results ready\",\"percent\":100,\"completed\":5,\"total\":5}' WHERE eda_id=%s",
            (Json(artifacts), job["target_id"]),
            conn=conn,
        )


def mark_eda_failed(job, message, conn):
    """Record a failed EDA analysis."""
    db.execute(
        "UPDATE eda_runs SET status='failed',error=%s,finished_at=now() WHERE eda_id=%s",
        (message, job["target_id"]),
        conn=conn,
    )
