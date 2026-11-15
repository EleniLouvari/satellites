"""Publish only verified, contained and sanitized worker deliverables."""

import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from parcel_app.core.storage import checksum, contained, sanitize_text


def safe_config(config):
    """Return a portable config without server paths or secret-valued fields."""
    if isinstance(config, dict):
        return {
            k: safe_config(v)
            for k, v in config.items()
            if k not in {"path", "output_dir"} and not any(s in k.lower() for s in ("password", "token", "secret"))
        }
    if isinstance(config, list):
        return [safe_config(v) for v in config]
    return config


def register_file(run, kind, path, *, conn=None):
    """Upsert a worker-approved artifact; browser downloads use only these rows."""
    from parcel_app.core import db

    path = contained(run["output_dir"], path)
    if not path.is_file():
        raise ValueError("Error: Missing artifact.")
    db.execute(
        """INSERT INTO run_artifacts(artifact_id,run_id,kind,name,storage_path,size_bytes,sha256)
        VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(run_id,kind,name) DO UPDATE SET
        storage_path=EXCLUDED.storage_path,size_bytes=EXCLUDED.size_bytes,sha256=EXCLUDED.sha256""",
        (str(uuid4()), str(run["run_id"]), kind, path.name, str(path), path.stat().st_size, checksum(path)),
        conn=conn,
    )


def register_artifacts(run, secrets=(), *, conn=None):
    """Require completed runner state and checksum before exposing final files."""
    root = Path(run["output_dir"]).resolve()
    manifest = json.loads((root / "artifacts.json").read_text(encoding="utf-8"))
    state = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or state.get("status") != "complete":
        raise ValueError("Error: Runner has not verified completion.")
    final = contained(root, manifest["artifacts"]["time_statistics"])
    if checksum(final) != manifest["time_statistics_sha256"]:
        raise ValueError("Error: Result checksum does not match.")
    for kind in ("time_statistics", "cleaning_report"):
        register_file(run, kind, contained(root, manifest["artifacts"][kind]), conn=conn)
    public = root / "downloads"
    public.mkdir(exist_ok=True)
    register_diagnostics(run, secrets, conn=conn)
    exported = dict(manifest, artifacts={k: Path(v).name for k, v in manifest["artifacts"].items()})
    output = public / "manifest.json"
    output.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    register_file(run, "manifest", output, conn=conn)


def register_diagnostics(run, secrets=(), *, conn=None):
    """Export configuration and redacted logs even when execution failed."""
    root = Path(run["output_dir"]).resolve()
    public = root / "downloads"
    public.mkdir(parents=True, exist_ok=True)
    private_paths = [str(root), root.as_posix(), str(run.get("config", {}).get("input", {}).get("path", ""))]
    config = public / "configuration.json"
    config.write_text(json.dumps(safe_config(run.get("config", {})), indent=2), encoding="utf-8")
    register_file(run, "configuration", config, conn=conn)
    for index, path in enumerate(root.rglob("satellite_*.log")):
        contained(root, path)
        output = public / f"stage-{index}-{path.name}"
        # Read one line at a time; secrets are removed before public persistence.
        with path.open(encoding="utf-8", errors="replace") as source, output.open("w", encoding="utf-8") as dest:
            for line in source:
                dest.write(sanitize_text(line, secrets, private_paths=private_paths))
        register_file(run, "log", output, conn=conn)


def build_bundle(run, *, conn=None):
    """Archive only registered deliverables, never arbitrary checkpoint trees."""
    from parcel_app.core import db

    rows = db.all_rows("SELECT * FROM run_artifacts WHERE run_id=%s AND kind <> 'bundle'", (str(run["run_id"]),), conn=conn)
    root = Path(run["output_dir"])
    (root / "downloads").mkdir(exist_ok=True)
    temporary = root / "downloads" / "results.partial.zip"
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for row in rows:
            path = contained(root, row["storage_path"])
            if checksum(path) != row["sha256"]:
                raise ValueError("Error: Artifact changed after publication.")
            archive.write(path, f"{row['kind']}/{path.name}")
    destination = temporary.with_name("results.zip")
    temporary.replace(destination)
    register_file(run, "bundle", destination, conn=conn)
