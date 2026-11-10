"""Publish only verified, contained and sanitized worker deliverables."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile


def contained(root, value):
    """Resolve and reject files outside a trusted storage boundary."""
    root = Path(root).resolve()
    path = Path(value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Error: Artifact is outside its run directory.")
    return path


_CREDENTIAL_NAME = (
    r"(?:[\w]+[_-])*(?:username|password|passwd|pwd|token|authorization|secret|api[_-]?key|credential[_-]?key|signature)"
)


def _redact_credential_url(match):
    """Hide URLs containing user information or authentication query/fragment keys."""
    value = match.group()
    try:
        url = urlsplit(value)
        fields = parse_qsl(url.query) + parse_qsl(url.fragment)
        sensitive = url.username is not None or any(re.fullmatch(_CREDENTIAL_NAME, key, re.I) for key, _ in fields)
    except ValueError:
        # Malformed authority syntax cannot be safely classified as credential-free.
        sensitive = True
    return "[redacted URL]" if sensitive else value


def sanitize_text(value, secrets=(), *, private_paths=()):
    """Mask credential fields and URLs, never global occurrences of credential values.

    Known values help recognize unquoted credentials containing spaces, but only
    after a credential field label. Private filesystem paths are supplied separately.
    """
    value = re.sub(r"\b[a-z][a-z0-9+.-]*://[^\s\"'<>]+", _redact_credential_url, value, flags=re.I)
    known = "|".join(re.escape(secret) for secret in sorted(set(secrets), key=len, reverse=True) if secret)
    # Quoted values may contain spaces or escaped quotes; auth schemes include the token.
    credential_value = r"\[redacted\]|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?:Bearer|Basic)\s+[^\s,;\"'}\]]+"
    if known:
        credential_value += rf"|(?:{known})(?=$|[\s,;}}\]])"
    credential_value += r"|[^\s,;}\]]+"
    value = re.sub(
        rf"(?<![\w-])([\"']?{_CREDENTIAL_NAME}[\"']?\s*[:=]\s*)(?:{credential_value})",
        r"\1[redacted]",
        value,
        flags=re.I,
    )
    for path in sorted(set(private_paths), key=len, reverse=True):
        if path:
            value = value.replace(path, "[redacted]")
    return value


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


def checksum(path):
    """Hash a file with bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def register_file(run, kind, path, *, conn=None):
    """Upsert a worker-approved artifact; browser downloads use only these rows."""
    from . import db

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
    from . import db

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
