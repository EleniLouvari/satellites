"""Download STAC assets, analyze their sizes, and reconcile them with S3.

What this module checks
-----------------------
The module covers asset retrieval and two related integrity questions:

1. Item download: fetch every selected asset for one catalog item from HTTP,
   S3, or a local file into an isolated collection/item output directory.
2. Asset-size audit: for every selected STAC asset, determine the best available
   physical size from an S3 object, HTTP response, local file, or declared STAC
   metadata. The result records how the size was obtained and classifies the
   asset as ``ok``, ``too_small``, ``size_unavailable``, or ``error`` against a
   configurable minimum-size threshold.
3. Catalog-to-S3 reconciliation: compare the catalog asset report with an S3
   inventory using ``collection_id``, ``item_id``, and ``filename`` as the
   compound identity. The comparison identifies matched assets, duplicate S3
   rows, S3 orphans that are not referenced by the catalog, and catalog assets
   whose corresponding S3 object is missing.

S3 keys are parsed relative to an exact path component named ``assets`` by
default. Keys without the required process path, collection, item, or filename
components are written to a separate error inventory instead of being silently
discarded. Reconciliation validates that every compound-key component is
present and non-blank before calculating mismatches.

Workflow
--------
The intended end-to-end workflow is:

1. Fetch catalog items for one or more collections and call
   :func:`run_asset_size_audit`.
2. Persist each attempted asset immediately in a per-collection SQLite
   checkpoint. Reruns skip terminal rows and can retry unresolved rows without
   rescanning the catalog.
3. Call :func:`export_asset_size_checkpoint_to_excel` after the audit to create
   the formatted ``asset_size_report.xlsx`` used by reconciliation.
4. Call :func:`export_s3_asset_keys` to stream object keys from the selected S3
   prefix into a raw CSV inventory.
5. Call :func:`parse_s3_asset_inventory` to process that CSV in bounded chunks,
   producing a normalized inventory and a rejected-key CSV. Completed outputs
   replace their destinations only after parsing succeeds.
6. Call :func:`check_s3_catalog_mismatches` for all desired collections. The S3
   inventory is scanned once, then each collection receives a formatted
   ``s3_catalog_mismatches.xlsx`` workbook with summary, ``s3_orphans``,
   ``catalog_missing_in_s3``, and ``s3_wrong_path_errors`` sheets. Objects
   stored beneath the temporary ``for_prod`` process path are reported as
   errors while still participating in catalog matching.

The checks are analytical and non-remediating: they read catalog/S3 state and
write checkpoints or reports, but they do not delete S3 objects or modify STAC
items. Matching is based on identifiers and filename; it does not prove that
two matched objects have identical content.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

import pandas as pd
import requests
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from satellites.data_preparation.sources.hub.catalog_library import CatalogSearchUtils, get_auth_session
from satellites.shared.formatting import human_bytes
from satellites.shared.s3 import download_s3_file, get_boto3_client, get_s3_object_size

CHECKPOINT_VERSION = 5
PARQUET_METADATA_KEY = b"asset_size_checkpoint"
REPORT_COLUMNS = [
    "collection_id",
    "item_id",
    "asset_key",
    "title",
    "filename",
    "roles",
    "href",
    "declared_size_bytes",
    "file_size_bytes",
    "file_size_human",
    "size_source",
    "status",
    "issue",
    "checked_at_utc",
]
TERMINAL_STATUSES = {"ok", "too_small"}
ERROR_STATUSES = {"size_unavailable", "error"}
S3_KEY_COLUMNS = ["collection_id", "item_id", "filename"]
S3_INVENTORY_COLUMNS = [
    "s3_path",
    "process_id",
    "asset_folder",
    "collection_id",
    "item_id",
    "filename",
    "file",
]
S3_PARSE_ERROR_COLUMNS = ["file", "component_count", "error_reason"]
S3_KEY_DTYPES = {column: "string" for column in S3_KEY_COLUMNS}
CATALOG_S3_CHECK_COLUMNS = [*S3_KEY_COLUMNS, "file_size_bytes"]
S3_TEMPORARY_PROCESS_ROOT = "for_prod"
EXCEL_MAX_ROWS_PER_SHEET = 1_048_576
EXCEL_MAX_DATA_ROWS_PER_SHEET = EXCEL_MAX_ROWS_PER_SHEET - 1  # Reserve one row for column headers.


@dataclass(frozen=True)
class AssetSizeResult:
    """Result returned by :func:`get_asset_size`."""

    size_bytes: int | None
    source: str
    href: str
    issue: str = ""
    is_error: bool = False


def _utc_now() -> str:
    """Return a sortable, timezone-aware timestamp for checkpoint rows."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Build the compound identity shared by all checkpoint formats."""
    return tuple(str(row.get(column) or "") for column in ("collection_id", "item_id", "asset_key"))


def collection_output_paths(
    output_root: str | Path,
    collection_id: str,
    checkpoint_suffix: str = ".sqlite",
) -> tuple[Path, Path]:
    """Return isolated checkpoint and workbook paths for one collection."""
    safe_collection_id = re.sub(r"[^A-Za-z0-9._-]+", "_", collection_id).strip("._")
    if not safe_collection_id:
        raise ValueError(f"Collection ID does not contain a safe path component: {collection_id!r}")
    if checkpoint_suffix not in {".json", ".parquet", ".sqlite"}:
        raise ValueError(f"Unsupported checkpoint suffix: {checkpoint_suffix!r}")
    collection_dir = Path(output_root) / safe_collection_id
    return collection_dir / f"asset_size_checkpoint{checkpoint_suffix}", collection_dir / "asset_size_report.xlsx"


def _coerce_optional_int(value: Any) -> int | None:
    """Convert persisted numeric values while tolerating empty legacy fields."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_for_size(size_bytes: int | None, min_size_bytes: int, is_error: bool = False) -> str:
    """Map a measurement outcome to the stable checkpoint status vocabulary."""
    if is_error:
        return "error"
    if size_bytes is None:
        return "size_unavailable"
    return "ok" if size_bytes >= min_size_bytes else "too_small"


def _normalized_status(row: dict[str, Any], size_bytes: int | None, min_size_bytes: int) -> str:
    """Return a supported status, recalculating missing legacy values."""
    legacy_status = row.get("status")
    if legacy_status in {"ok", "too_small", "size_unavailable", "error"}:
        return str(legacy_status)
    return _status_for_size(size_bytes, min_size_bytes)


def normalize_report_row(row: dict[str, Any], min_size_bytes: int) -> dict[str, Any]:
    """Migrate a report row from an older checkpoint into the current schema."""
    normalized = {column: row.get(column, "") for column in REPORT_COLUMNS}
    normalized["file_size_bytes"] = _coerce_optional_int(row.get("file_size_bytes", row.get("file_size")))
    normalized["declared_size_bytes"] = _coerce_optional_int(row.get("declared_size_bytes"))
    normalized["roles"] = row.get("roles") or ""
    normalized["title"] = row.get("title") or ""
    normalized["href"] = row.get("href") or ""
    normalized["size_source"] = row.get("size_source") or ""
    normalized["issue"] = row.get("issue") or ""
    normalized["checked_at_utc"] = row.get("checked_at_utc") or ""
    normalized["file_size_human"] = (
        human_bytes(normalized["file_size_bytes"]) if normalized["file_size_bytes"] is not None else ""
    )

    normalized["status"] = _normalized_status(row, normalized["file_size_bytes"], min_size_bytes)
    return normalized


def _import_pyarrow():
    """Import the Parquet engine with an actionable error message."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Parquet checkpoints require pyarrow. Install the project dependencies or run "
            "`python -m pip install pyarrow==25.0.0`."
        ) from exc
    return pa, pq


def _read_json_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load the legacy JSON checkpoint representation."""
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_parquet_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Restore rows and embedded run metadata from a Parquet checkpoint."""
    _, parquet = _import_pyarrow()
    table = parquet.read_table(checkpoint_path)
    metadata = table.schema.metadata or {}
    state_metadata = json.loads(metadata.get(PARQUET_METADATA_KEY, b"{}").decode("utf-8"))
    return {
        "version": state_metadata.get("version", CHECKPOINT_VERSION),
        "updated_at_utc": state_metadata.get("updated_at_utc", ""),
        "config": state_metadata.get("config", {}),
        "report_rows": table.to_pylist(),
    }


def _connect_sqlite_checkpoint(checkpoint_path: Path) -> sqlite3.Connection:
    """Open a checkpoint database and ensure its schema is ready."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS asset_report (
            collection_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            asset_key TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            filename TEXT NOT NULL DEFAULT '',
            roles TEXT NOT NULL DEFAULT '',
            href TEXT NOT NULL DEFAULT '',
            declared_size_bytes INTEGER,
            file_size_bytes INTEGER,
            file_size_human TEXT NOT NULL DEFAULT '',
            size_source TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            issue TEXT NOT NULL DEFAULT '',
            checked_at_utc TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (collection_id, item_id, asset_key)
        );
        CREATE INDEX IF NOT EXISTS asset_report_status_idx ON asset_report (status);
        CREATE TABLE IF NOT EXISTS checkpoint_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(asset_report)")}
    if "title" not in existing_columns:
        connection.execute("ALTER TABLE asset_report ADD COLUMN title TEXT NOT NULL DEFAULT ''")
    connection.commit()
    return connection


def _sqlite_row_values(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return normalized values in ``REPORT_COLUMNS`` order."""
    normalized = normalize_report_row(row, min_size_bytes=0)
    return tuple(normalized[column] for column in REPORT_COLUMNS)


_SQLITE_UPSERT = f"""
    INSERT INTO asset_report ({", ".join(REPORT_COLUMNS)})
    VALUES ({", ".join("?" for _ in REPORT_COLUMNS)})
    ON CONFLICT(collection_id, item_id, asset_key) DO UPDATE SET
        {", ".join(f"{column} = excluded.{column}" for column in REPORT_COLUMNS[3:])}
    WHERE excluded.checked_at_utc >= asset_report.checked_at_utc
"""


def _write_sqlite_metadata(
    connection: sqlite3.Connection,
    config: dict[str, Any] | None,
    updated_at_utc: str | None = None,
) -> None:
    """Upsert checkpoint version, timestamp, and optional run configuration."""
    metadata = {
        "version": str(CHECKPOINT_VERSION),
        "updated_at_utc": updated_at_utc or _utc_now(),
    }
    if config is not None:
        metadata["config"] = json.dumps(config, ensure_ascii=False, default=str)
    connection.executemany(
        """
        INSERT INTO checkpoint_metadata (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        metadata.items(),
    )


def _read_sqlite_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load the live SQLite checkpoint into the common in-memory schema."""
    connection = _connect_sqlite_checkpoint(checkpoint_path)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM checkpoint_metadata"))
        report_rows = [dict(row) for row in connection.execute(f"SELECT {', '.join(REPORT_COLUMNS)} FROM asset_report")]
    finally:
        connection.close()
    try:
        config = json.loads(metadata.get("config", "{}"))
    except json.JSONDecodeError:
        config = {}
    return {
        "version": int(metadata.get("version", CHECKPOINT_VERSION)),
        "updated_at_utc": metadata.get("updated_at_utc", ""),
        "config": config,
        "report_rows": report_rows,
    }


def _read_checkpoint_file(checkpoint_path: Path) -> dict[str, Any]:
    """Dispatch checkpoint loading from the filename's format suffix."""
    suffixes = {suffix.lower() for suffix in checkpoint_path.suffixes}
    if ".json" in suffixes:
        return _read_json_checkpoint(checkpoint_path)
    if ".sqlite" in suffixes:
        return _read_sqlite_checkpoint(checkpoint_path)
    return _read_parquet_checkpoint(checkpoint_path)


def _newer_row(existing_row: dict[str, Any] | None, candidate_row: dict[str, Any]) -> dict[str, Any]:
    """Choose the latest check, with later input winning when timestamps tie."""
    if existing_row is None:
        return candidate_row
    existing_checked_at = str(existing_row.get("checked_at_utc") or "")
    candidate_checked_at = str(candidate_row.get("checked_at_utc") or "")
    return candidate_row if candidate_checked_at >= existing_checked_at else existing_row


def _checkpoint_source_path(checkpoint_path: Path) -> Path:
    """Resolve a requested checkpoint or its supported legacy predecessor."""
    if checkpoint_path.exists():
        return checkpoint_path
    fallback_suffixes = {
        ".sqlite": (".parquet", ".json"),
        ".parquet": (".json",),
    }.get(checkpoint_path.suffix.lower(), ())
    return next(
        (checkpoint_path.with_suffix(suffix) for suffix in fallback_suffixes if checkpoint_path.with_suffix(suffix).exists()),
        checkpoint_path,
    )


def _load_checkpoint_with_backup(source_path: Path) -> dict[str, Any]:
    """Read one checkpoint, recovering from its backup when available."""
    if not source_path.exists():
        return {}
    try:
        return _read_checkpoint_file(source_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        backup_path = source_path.with_suffix(source_path.suffix + ".bak")
        if not backup_path.exists():
            raise ValueError(f"Cannot read asset-size checkpoint {source_path}: {exc}") from exc
        warnings.warn(
            f"Cannot read {source_path}; recovering from {backup_path}",
            RuntimeWarning,
            stacklevel=3,
        )
        try:
            return _read_checkpoint_file(backup_path)
        except (OSError, ValueError, json.JSONDecodeError) as backup_exc:
            raise ValueError(
                f"Cannot read asset-size checkpoint {source_path} or backup {backup_path}: {backup_exc}"
            ) from backup_exc


def _latest_normalized_rows(rows: Iterable[Any], min_size_bytes: int) -> list[dict[str, Any]]:
    """Normalize rows and retain the latest check for each compound asset key."""
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = normalize_report_row(row, min_size_bytes)
        key = _row_key(normalized)
        rows_by_key[key] = _newer_row(rows_by_key.get(key), normalized)
    return list(rows_by_key.values())


def load_asset_size_checkpoint(checkpoint_path: str | Path, min_size_bytes: int) -> dict[str, Any]:
    """Load a checkpoint, keeping the latest ``checked_at_utc`` row per asset.

    SQLite is the live checkpoint format. If a requested SQLite database does
    not exist yet, an older Parquet or JSON checkpoint is loaded for migration.
    """
    source_path = _checkpoint_source_path(Path(checkpoint_path))
    data = _load_checkpoint_with_backup(source_path)

    return {
        "version": CHECKPOINT_VERSION,
        "updated_at_utc": data.get("updated_at_utc", ""),
        "config": data.get("config", {}),
        "report_rows": _latest_normalized_rows(data.get("report_rows", []), min_size_bytes),
    }


def _replace_or_copy_completed_file(temporary_path: Path, output_path: Path) -> None:
    """Install a completed file, with a Windows-compatible fallback.

    Some Windows environments allow writing a file but reject replacing an
    existing path with ``MoveFileEx``/``os.replace``. Retry briefly for transient
    locks, then copy the already-complete temporary file into the destination.
    """
    last_error: PermissionError | None = None
    for retry_index in range(2):
        try:
            os.replace(temporary_path, output_path)
            return
        except PermissionError as exc:
            last_error = exc
            if retry_index == 0:
                time.sleep(0.05)

    warnings.warn(
        f"Windows denied atomic replacement of {output_path}; using a flushed direct-write fallback",
        RuntimeWarning,
        stacklevel=2,
    )
    try:
        with temporary_path.open("rb") as source, output_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot update {output_path}. Close the file if it is open and verify write permission for its directory."
        ) from (last_error or exc)


def _remove_temporary_file(temporary_path: Path) -> None:
    """Remove a temporary output without letting a transient Windows lock abort the audit."""
    for retry_index in range(5):
        try:
            temporary_path.unlink(missing_ok=True)
            return
        except PermissionError:
            if retry_index < 4:
                time.sleep(0.05 * (retry_index + 1))
    warnings.warn(
        f"Temporary file remains locked and could not be removed: {temporary_path}",
        RuntimeWarning,
        stacklevel=2,
    )


def save_asset_size_checkpoint(checkpoint_state: dict[str, Any], checkpoint_path: str | Path) -> Path:
    """Persist a complete checkpoint for migration, fixtures, or legacy formats."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_state["version"] = CHECKPOINT_VERSION
    checkpoint_state["updated_at_utc"] = _utc_now()

    if checkpoint_path.suffix.lower() == ".sqlite":
        report_df = build_report_dataframe(checkpoint_state.get("report_rows", []), min_size_bytes=0)
        connection = _connect_sqlite_checkpoint(checkpoint_path)
        try:
            with connection:
                connection.execute("DELETE FROM asset_report")
                connection.executemany(
                    _SQLITE_UPSERT,
                    (_sqlite_row_values(row) for row in report_df.to_dict(orient="records")),
                )
                _write_sqlite_metadata(
                    connection,
                    checkpoint_state.get("config", {}),
                    checkpoint_state["updated_at_utc"],
                )
        finally:
            connection.close()
        return checkpoint_path

    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    backup_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak")
    try:
        if checkpoint_path.suffix.lower() == ".json":
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(checkpoint_state, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            arrow, parquet = _import_pyarrow()
            report_df = build_report_dataframe(checkpoint_state.get("report_rows", []), min_size_bytes=0)
            table = arrow.Table.from_pandas(report_df, preserve_index=False)
            state_metadata = json.dumps(
                {
                    "version": checkpoint_state["version"],
                    "updated_at_utc": checkpoint_state["updated_at_utc"],
                    "config": checkpoint_state.get("config", {}),
                },
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), PARQUET_METADATA_KEY: state_metadata})
            parquet.write_table(table, temporary_path, compression="zstd")
        if checkpoint_path.exists() and not backup_path.exists():
            shutil.copyfile(checkpoint_path, backup_path)
        _replace_or_copy_completed_file(temporary_path, checkpoint_path)
    finally:
        _remove_temporary_file(temporary_path)
    return checkpoint_path


def extract_size_from_headers(headers: requests.structures.CaseInsensitiveDict | dict[str, str]) -> int | None:
    """Return the full object size from HTTP headers.

    ``Content-Range`` is deliberately checked before ``Content-Length`` because a
    successful ``Range: bytes=0-0`` response normally has a content length of one.
    """
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        total_size = content_range.rsplit("/", maxsplit=1)[-1]
        if total_size != "*":
            try:
                return int(total_size)
            except ValueError:
                pass

    content_length = headers.get("Content-Length")
    if content_length is not None:
        try:
            return int(content_length)
        except ValueError:
            return None
    return None


def _filename_from_reference(reference: Any) -> str | None:
    """Extract a filename from a relative path or URL-like reference."""
    if not isinstance(reference, str) or not reference.strip():
        return None
    parsed = urlparse(reference.strip())
    path_value = parsed.path if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment else reference.strip()
    filename = Path(unquote(path_value).replace("\\", "/")).name
    return filename if filename not in {"", "."} else None


def get_asset_filename(asset_key: str, asset_info: dict[str, Any]) -> str:
    """Return a readable filename from relative path, href, or href fallbacks."""
    filename = _filename_from_reference(asset_info.get("hub:relativePath"))
    if filename:
        return filename

    for href in _candidate_hrefs(asset_info):
        filename = _filename_from_reference(href)
        if filename:
            return filename
    return asset_key


def _template_base_hrefs(asset_info: dict[str, Any]) -> Iterable[str]:
    """Yield base URLs derived from STAC tile and point-query templates."""
    for template_key, template_info in asset_info.items():
        if not template_key.endswith(("-tiles-template", "-query-template")) or not isinstance(template_info, dict):
            continue
        template_href = template_info.get("href")
        if not isinstance(template_href, str) or not template_href:
            continue
        for marker in ("/tiles/", "/point/"):
            if marker in template_href:
                yield template_href.split(marker, maxsplit=1)[0]
                break


def _candidate_hrefs(asset_info: dict[str, Any]) -> list[str]:
    """Collect unique direct, alternate, and templated base asset URLs."""
    candidates: list[str] = []

    def add_candidate(value: Any) -> None:
        """Append one non-empty URL while preserving provider ordering."""
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    add_candidate(asset_info.get("href"))

    # STAC's alternate extension can expose authenticated or cloud-native copies.
    alternate_hrefs = asset_info.get("alternate")
    if isinstance(alternate_hrefs, dict):
        for alternate_info in alternate_hrefs.values():
            if isinstance(alternate_info, dict):
                add_candidate(alternate_info.get("href"))

    for template_href in _template_base_hrefs(asset_info):
        add_candidate(template_href)

    return list(dict.fromkeys(candidates))


def get_asset_download_href(asset_info: dict[str, Any]) -> str | None:
    """Return the preferred direct, alternate, or template-derived asset href."""
    return next(iter(_candidate_hrefs(asset_info)), None)


def _streamed_response_size(response: requests.Response) -> int:
    """Count a streamed response without holding the asset in memory."""
    return sum(len(chunk) for chunk in response.iter_content(chunk_size=1024 * 1024) if chunk)


def _http_head_size(href: str, session: requests.Session, timeout: float) -> tuple[int | None, bool]:
    """Try an HTTP HEAD request and always close its response."""
    response = None
    try:
        response = session.head(href, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        return extract_size_from_headers(response.headers), True
    except requests.RequestException:
        return None, False
    finally:
        if response is not None:
            response.close()


def _http_range_size(
    href: str,
    session: requests.Session,
    timeout: float,
    stream_body_fallback: bool,
) -> tuple[int | None, str, bool]:
    """Try a one-byte range request and optionally count a full 200 response."""
    response = None
    try:
        response = session.get(
            href,
            headers={"Range": "bytes=0-0"},
            stream=True,
            allow_redirects=True,
            timeout=timeout,
        )
        response.raise_for_status()
        size_bytes = extract_size_from_headers(response.headers)
        if response.status_code == requests.codes.partial_content and not response.headers.get("Content-Range"):
            size_bytes = None
        if size_bytes is not None:
            return size_bytes, "http-range", True
        if stream_body_fallback and response.status_code == requests.codes.ok:
            return _streamed_response_size(response), "http-stream", True
        return None, "", True
    except requests.RequestException:
        return None, "", False
    finally:
        if response is not None:
            response.close()


def _http_stream_size(href: str, session: requests.Session, timeout: float) -> tuple[int | None, str, bool]:
    """Stream a normal GET response when neither header-only strategy has a size."""
    response = None
    try:
        response = session.get(
            href,
            headers={"Accept-Encoding": "identity"},
            stream=True,
            allow_redirects=True,
            timeout=timeout,
        )
        response.raise_for_status()
        size_bytes = extract_size_from_headers(response.headers)
        if size_bytes is not None:
            return size_bytes, "http-get", True
        return _streamed_response_size(response), "http-stream", True
    except requests.RequestException:
        return None, "", False
    finally:
        if response is not None:
            response.close()


def _http_asset_size(
    href: str,
    session: requests.Session,
    timeout: float,
    stream_body_fallback: bool,
) -> tuple[int | None, str, bool]:
    """Return size, source, and whether the HTTP asset was reachable."""
    size_bytes, reachable = _http_head_size(href, session, timeout)
    if size_bytes is not None:
        return size_bytes, "http-head", True

    range_size, range_source, range_reachable = _http_range_size(href, session, timeout, stream_body_fallback)
    if range_size is not None:
        return range_size, range_source, True
    reachable = reachable or range_reachable
    if not stream_body_fallback:
        return None, "", reachable

    stream_size, stream_source, stream_reachable = _http_stream_size(href, session, timeout)
    return stream_size, stream_source, reachable or stream_reachable


def _http_candidate_result(
    href: str,
    session: requests.Session,
    timeout: float,
    stream_http_body_fallback: bool,
) -> AssetSizeResult | None:
    """Convert an HTTP size probe into the public result model."""
    size_bytes, source, reachable = _http_asset_size(href, session, timeout, stream_http_body_fallback)
    if size_bytes is not None:
        return AssetSizeResult(size_bytes, source, href)
    if reachable:
        return AssetSizeResult(None, "", href, "asset is reachable but its size is not declared by HTTP headers")
    return None


def _s3_candidate_result(href: str, s3_size_getter: Callable[..., int | None]) -> AssetSizeResult | None:
    """Read the object size for one valid S3 URI."""
    parsed = urlparse(href)
    bucket_name = parsed.netloc
    s3_key = parsed.path.lstrip("/")
    if not bucket_name or not s3_key:
        return None
    size_bytes = s3_size_getter(bucket_name=bucket_name, s3_key=s3_key, verbose=False)
    return AssetSizeResult(int(size_bytes), "s3", href) if size_bytes is not None else None


def _local_candidate_result(href: str) -> AssetSizeResult | None:
    """Measure a file URI or local path candidate."""
    parsed = urlparse(href)
    local_path = Path(parsed.netloc + parsed.path) if parsed.scheme.lower() == "file" else Path(href)
    return AssetSizeResult(local_path.stat().st_size, "file", href) if local_path.is_file() else None


def _measure_candidate_href(
    href: str,
    session: requests.Session,
    timeout: float,
    s3_size_getter: Callable[..., int | None],
    stream_http_body_fallback: bool,
) -> AssetSizeResult | None:
    """Measure one candidate using the transport implied by its URI scheme."""
    scheme = urlparse(href).scheme.lower()

    try:
        if scheme in {"http", "https"}:
            return _http_candidate_result(href, session, timeout, stream_http_body_fallback)
        if scheme == "s3":
            return _s3_candidate_result(href, s3_size_getter)
        if scheme in {"file", ""} or len(scheme) == 1:
            return _local_candidate_result(href)
        return None
    except (OSError, ValueError) as exc:
        return AssetSizeResult(None, "", href, str(exc), True)


def get_asset_size(
    asset_info: dict[str, Any],
    session: requests.Session,
    timeout: float = 20,
    s3_size_getter: Callable[..., int | None] = get_s3_object_size,
    stream_http_body_fallback: bool = False,
) -> AssetSizeResult:
    """Measure an asset, optionally counting an HTTP body when headers omit size."""
    candidates = _candidate_hrefs(asset_info)
    if not candidates:
        return AssetSizeResult(None, "", "", "missing href")

    attempted: list[str] = []
    reachable_without_size: AssetSizeResult | None = None
    for href in candidates:
        attempted.append(href)
        result = _measure_candidate_href(
            href,
            session,
            timeout,
            s3_size_getter,
            stream_http_body_fallback,
        )
        if result is None:
            continue
        if result.size_bytes is not None or result.is_error:
            return result
        if reachable_without_size is None:
            reachable_without_size = result

    if reachable_without_size is not None:
        return reachable_without_size

    return AssetSizeResult(None, "", attempted[0], "size could not be determined from any asset href")


def _roles_label(asset_info: dict[str, Any]) -> str:
    """Flatten STAC roles into a workbook-friendly display value."""
    roles = asset_info.get("roles")
    if isinstance(roles, list):
        return ", ".join(str(role) for role in roles)
    return str(roles or "")


def _declared_size(asset_info: dict[str, Any]) -> int | None:
    """Read the optional STAC file-extension size without trusting its type."""
    return _coerce_optional_int(asset_info.get("file:size"))


def build_asset_report_row(
    collection_id: str,
    item_id: str,
    asset_key: str,
    asset_info: dict[str, Any],
    result: AssetSizeResult,
    min_size_bytes: int,
    title: str = "",
) -> dict[str, Any]:
    """Build a stable, serializable row for the checkpoint and workbook."""
    status = _status_for_size(result.size_bytes, min_size_bytes, result.is_error)
    return {
        "collection_id": collection_id,
        "item_id": item_id,
        "asset_key": asset_key,
        "title": title,
        "filename": get_asset_filename(asset_key, asset_info),
        "roles": _roles_label(asset_info),
        "href": result.href or str(asset_info.get("href") or ""),
        "declared_size_bytes": _declared_size(asset_info),
        "file_size_bytes": result.size_bytes,
        "file_size_human": human_bytes(result.size_bytes) if result.size_bytes is not None else "",
        "size_source": result.source,
        "status": status,
        "issue": result.issue,
        "checked_at_utc": _utc_now(),
    }


def build_report_dataframe(report_rows: Iterable[dict[str, Any]], min_size_bytes: int) -> pd.DataFrame:
    """Return the latest normalized row per asset as a report DataFrame."""
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in report_rows:
        normalized = normalize_report_row(row, min_size_bytes)
        key = _row_key(normalized)
        rows_by_key[key] = _newer_row(rows_by_key.get(key), normalized)

    report_df = pd.DataFrame(rows_by_key.values(), columns=REPORT_COLUMNS)
    if report_df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    for column in ("declared_size_bytes", "file_size_bytes"):
        report_df[column] = pd.to_numeric(report_df[column], errors="coerce").astype("Int64")
    return report_df.sort_values(["collection_id", "item_id", "asset_key"], kind="stable").reset_index(drop=True)


def build_summary_dataframe(report_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize audited assets per collection without mutable counters."""
    columns = ["collection_id", "total_assets", "ok", "too_small", "size_unavailable", "errors"]
    if report_df.empty:
        return pd.DataFrame(columns=columns)

    summary = report_df.groupby("collection_id", dropna=False)["status"].value_counts().unstack(fill_value=0)
    for status in ("ok", "too_small", "size_unavailable", "error"):
        if status not in summary:
            summary[status] = 0
    summary["total_assets"] = summary[["ok", "too_small", "size_unavailable", "error"]].sum(axis=1)
    summary = summary.reset_index().rename(columns={"error": "errors"})
    return summary[columns]


def _display_values(series: pd.Series) -> pd.Series:
    """Convert any pandas dtype to safe strings for width calculations."""
    return series.astype("object").where(series.notna(), "").astype(str)


def _format_sheet(worksheet, dataframe: pd.DataFrame, table_name: str) -> None:
    """Apply consistent navigation, table, header, and width formatting."""
    worksheet.freeze_panes = "A2"
    worksheet.sheet_view.showGridLines = False
    worksheet.row_dimensions[1].height = 24

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font

    # Excel tables require at least one data row in addition to the header.
    if len(dataframe.index) > 0:
        table = Table(displayName=table_name, ref=worksheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)

    for column_index, column in enumerate(dataframe.columns, start=1):
        values = _display_values(dataframe[column])
        max_data_length = int(values.str.len().max()) if not values.empty else 0
        width = min(max(max_data_length, len(column)) + 2, 45)
        worksheet.column_dimensions[get_column_letter(column_index)].width = width


def _asset_report_sheets(report_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a report across Excel worksheets without exceeding its row limit."""
    if report_df.empty:
        return [("asset_report", report_df)]

    sheets = []
    for start_row in range(0, len(report_df.index), EXCEL_MAX_DATA_ROWS_PER_SHEET):
        part_number = len(sheets) + 1
        sheet_name = "asset_report" if part_number == 1 else f"asset_report_{part_number}"
        sheets.append(
            (
                sheet_name,
                report_df.iloc[start_row : start_row + EXCEL_MAX_DATA_ROWS_PER_SHEET],
            )
        )
    return sheets


def _format_asset_report_sheet(worksheet, dataframe: pd.DataFrame, table_name: str) -> None:
    """Apply the common table, status, and numeric formatting to one report part."""
    _format_sheet(worksheet, dataframe, table_name)
    if dataframe.empty:
        return

    status_column = REPORT_COLUMNS.index("status") + 1
    status_letter = get_column_letter(status_column)
    status_range = f"{status_letter}2:{status_letter}{len(dataframe.index) + 1}"
    worksheet.conditional_formatting.add(
        status_range,
        FormulaRule(formula=[f'${status_letter}2="ok"'], fill=PatternFill("solid", fgColor="C6EFCE")),
    )
    worksheet.conditional_formatting.add(
        status_range,
        FormulaRule(formula=[f'${status_letter}2="too_small"'], fill=PatternFill("solid", fgColor="FFC7CE")),
    )
    worksheet.conditional_formatting.add(
        status_range,
        FormulaRule(
            formula=[f'OR(${status_letter}2="size_unavailable",${status_letter}2="error")'],
            fill=PatternFill("solid", fgColor="FFEB9C"),
        ),
    )

    for column_name in ("declared_size_bytes", "file_size_bytes"):
        column_index = REPORT_COLUMNS.index(column_name) + 1
        for cell in worksheet.iter_cols(
            min_col=column_index,
            max_col=column_index,
            min_row=2,
            max_row=len(dataframe.index) + 1,
        ):
            for value_cell in cell:
                value_cell.number_format = "#,##0"


def write_asset_report_excel(
    report_rows: Iterable[dict[str, Any]],
    output_path: str | Path,
    min_size_bytes: int,
    run_info: dict[str, Any] | None = None,
) -> Path:
    """Write the full report, splitting rows across worksheets when necessary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df = build_report_dataframe(report_rows, min_size_bytes)
    summary_df = build_summary_dataframe(report_df)
    run_info_df = pd.DataFrame(
        [(str(key), ", ".join(map(str, value)) if isinstance(value, list) else value) for key, value in (run_info or {}).items()],
        columns=["setting", "value"],
    )

    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    try:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            report_sheets = _asset_report_sheets(report_df)
            for sheet_name, dataframe in report_sheets:
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            run_info_df.to_excel(writer, sheet_name="run_info", index=False)

            _format_sheet(writer.book["summary"], summary_df, "AssetSummary")
            for part_number, (sheet_name, dataframe) in enumerate(report_sheets, start=1):
                table_name = "AssetReport" if part_number == 1 else f"AssetReport{part_number}"
                _format_asset_report_sheet(writer.book[sheet_name], dataframe, table_name)
            _format_sheet(writer.book["run_info"], run_info_df, "RunInfo")

        _replace_or_copy_completed_file(temporary_path, output_path)
    finally:
        _remove_temporary_file(temporary_path)
    return output_path


def export_asset_size_checkpoint_to_excel(
    checkpoint_path: str | Path,
    output_path: str | Path,
    min_size_bytes: int = 1024,
) -> Path:
    """Export the complete SQLite checkpoint to a workbook with bounded sheets."""
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() != ".sqlite":
        raise ValueError("Final Excel export requires a .sqlite checkpoint path")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Asset-size checkpoint does not exist: {checkpoint_path}")
    checkpoint_state = load_asset_size_checkpoint(checkpoint_path, min_size_bytes)
    return write_asset_report_excel(
        checkpoint_state["report_rows"],
        output_path,
        min_size_bytes,
        run_info=checkpoint_state.get("config", {}),
    )


def _rows_grouped_by_collection(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group checkpoint rows that have a non-empty collection identifier."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        collection_id = str(row.get("collection_id") or "")
        if collection_id:
            grouped.setdefault(collection_id, []).append(row)
    return grouped


def _merge_migration_rows(
    legacy_rows: Iterable[dict[str, Any]],
    current_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Merge migration rows while preserving current progress by default."""
    rows_by_key = {_row_key(row): row for row in legacy_rows}
    current_rows_by_key = {_row_key(row): row for row in current_rows}
    for row_key, current_row in current_rows_by_key.items():
        legacy_row = rows_by_key.get(row_key)
        legacy_checked_at = str(legacy_row.get("checked_at_utc") or "") if legacy_row else ""
        current_checked_at = str(current_row.get("checked_at_utc") or "")
        legacy_is_newer = bool(legacy_checked_at and current_checked_at and legacy_checked_at > current_checked_at)
        if not legacy_is_newer:
            rows_by_key[row_key] = current_row
    return list(rows_by_key.values()), current_rows_by_key != rows_by_key


def _migrate_collection_checkpoint(
    collection_id: str,
    legacy_rows: list[dict[str, Any]],
    combined_checkpoint_path: Path,
    output_root: str | Path,
    min_size_bytes: int,
    checkpoint_suffix: str,
) -> dict[str, Any]:
    """Migrate one collection and return its output summary."""
    checkpoint_path, report_path = collection_output_paths(output_root, collection_id, checkpoint_suffix=checkpoint_suffix)
    collection_state = load_asset_size_checkpoint(checkpoint_path, min_size_bytes)
    merged_rows, rows_changed = _merge_migration_rows(legacy_rows, collection_state["report_rows"])
    desired_config = {
        **collection_state.get("config", {}),
        "collection": collection_id,
        "migrated_from": str(combined_checkpoint_path.resolve()),
    }
    config_changed = collection_state.get("config", {}) != desired_config
    collection_state.update(report_rows=merged_rows, config=desired_config)
    if not checkpoint_path.exists() or rows_changed or config_changed:
        save_asset_size_checkpoint(collection_state, checkpoint_path)
    return {
        "rows": len(merged_rows),
        "checkpoint_path": checkpoint_path,
        "report_path": report_path,
    }


def split_checkpoint_by_collection(
    combined_checkpoint_path: str | Path,
    output_root: str | Path,
    min_size_bytes: int,
    checkpoint_suffix: str = ".sqlite",
) -> dict[str, dict[str, Any]]:
    """Migrate a combined checkpoint into independent per-collection outputs.

    Existing per-collection rows take precedence, so rerunning this migration
    never rolls newer collection progress back to an older combined checkpoint.
    The combined files are left untouched as a legacy recovery source.
    """
    combined_checkpoint_path = Path(combined_checkpoint_path)
    if not combined_checkpoint_path.exists():
        return {}

    combined_state = load_asset_size_checkpoint(combined_checkpoint_path, min_size_bytes)
    combined_by_collection = _rows_grouped_by_collection(combined_state["report_rows"])
    return {
        collection_id: _migrate_collection_checkpoint(
            collection_id,
            legacy_rows,
            combined_checkpoint_path,
            output_root,
            min_size_bytes,
            checkpoint_suffix,
        )
        for collection_id, legacy_rows in sorted(combined_by_collection.items())
    }


def _asset_is_selected(asset_info: dict[str, Any], only_data_assets: bool) -> bool:
    """Apply the optional STAC ``data`` role filter to one asset."""
    if not only_data_assets:
        return True
    roles = asset_info.get("roles")
    return isinstance(roles, list) and "data" in roles


def get_item_by_id(
    collection_id: str,
    item_id: str,
    client_id: str = "internal",
    catalog_endpoint: str | None = None,
    fetch_features: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any] | None:
    """Fetch one exact catalog item, returning ``None`` when it is absent.

    ``fetch_features`` is injectable for callers that already own a
    :class:`~common.catalog_library.CatalogSearchUtils` instance. Otherwise the
    catalog endpoint is read from ``catalog_endpoint`` or ``CATALOG_URL``.
    """
    if fetch_features is None:
        endpoint = catalog_endpoint or os.getenv("CATALOG_URL")
        if not endpoint:
            raise ValueError("CATALOG_URL is not configured")
        fetch_features = CatalogSearchUtils(
            catalog_endpoint=endpoint,
            client_id=client_id,
            verbose=False,
        ).fetch_features

    features = fetch_features(
        search_body={"collections": [collection_id], "ids": [item_id]},
        max_items=2,
    )
    return next((feature for feature in features if feature.get("id") == item_id), None)


def _download_http_asset(
    download_href: str,
    destination_path: Path,
    session: requests.Session,
    timeout: float,
) -> None:
    """Stream one HTTP asset to disk and close its response."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    response = None
    try:
        response = session.get(
            download_href,
            stream=True,
            allow_redirects=True,
            timeout=timeout,
        )
        response.raise_for_status()
        with destination_path.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)
    finally:
        if response is not None:
            response.close()


def _download_file_asset(download_href: str, destination_path: Path) -> None:
    """Copy one local-file asset to the requested output path."""
    parsed = urlparse(download_href)
    source_path = Path(parsed.netloc + parsed.path) if parsed.scheme.lower() == "file" else Path(download_href)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)


def _download_s3_asset(
    download_href: str,
    destination_path: Path,
    s3_client: Any,
    s3_downloader: Callable[..., None],
) -> None:
    """Download one valid S3 URI to the requested output path."""
    parsed = urlparse(download_href)
    bucket_name = parsed.netloc
    s3_key = parsed.path.lstrip("/")
    if not bucket_name or not s3_key:
        raise ValueError(f"Invalid S3 href: {download_href}")

    s3_downloader(
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_key=s3_key,
        local_file_path=str(destination_path),
    )


def _close_if_supported(resource: Any) -> None:
    """Close an owned session or client when it exposes ``close``."""
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _print_download_summary(result: dict[str, Any]) -> None:
    """Print the compact item download summary used by catalog notebooks."""
    print("-" * 100)
    print(f"Download summary for {result['collection_id']} / {result['item_id']}")
    print(f"  downloaded={len(result['downloaded'])}")
    print(f"  failed={len(result['failed'])}")
    print(f"  skipped={len(result['skipped'])}")
    print("-" * 100)


def download_all_assets_for_item(
    collection_id: str,
    item_id: str,
    output_dir: str | Path,
    client_id: str = "internal",
    only_data_assets: bool = False,
    timeout: float = 60,
    catalog_endpoint: str | None = None,
    fetch_features: Callable[..., list[dict[str, Any]]] | None = None,
    auth_session_factory: Callable[[str], requests.Session] = get_auth_session,
    s3_client_factory: Callable[[], Any] = get_boto3_client,
    s3_downloader: Callable[..., None] = download_s3_file,
) -> dict[str, Any]:
    """Download all selected assets for one catalog item.

    HTTP and S3 resources are created lazily, so a local-file-only item does not
    require catalog-download credentials beyond those used to fetch the item.
    Individual asset failures are recorded in the returned summary and do not
    stop the remaining downloads.
    """
    item = get_item_by_id(
        collection_id=collection_id,
        item_id=item_id,
        client_id=client_id,
        catalog_endpoint=catalog_endpoint,
        fetch_features=fetch_features,
    )
    if item is None:
        raise ValueError(f"Item not found for collection_id={collection_id}, item_id={item_id}")

    output_root = Path(output_dir) / collection_id / item_id
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "collection_id": collection_id,
        "item_id": item_id,
        "output_dir": str(output_root),
        "downloaded": [],
        "failed": [],
        "skipped": [],
    }
    auth_session = None
    s3_client = None

    try:
        assets = item.get("assets")
        for asset_key, asset_info in assets.items() if isinstance(assets, dict) else ():
            if not isinstance(asset_info, dict):
                result["failed"].append({"asset": asset_key, "reason": "invalid-asset-metadata"})
                continue
            if not _asset_is_selected(asset_info, only_data_assets):
                result["skipped"].append({"asset": asset_key, "reason": "not-data-role"})
                continue

            download_href = get_asset_download_href(asset_info)
            if download_href is None:
                result["failed"].append({"asset": asset_key, "reason": "missing-href"})
                continue

            destination_path = output_root / get_asset_filename(asset_key, asset_info)
            scheme = urlparse(download_href).scheme.lower()
            try:
                if scheme in {"http", "https"}:
                    auth_session = auth_session or auth_session_factory(client_id)
                    _download_http_asset(download_href, destination_path, auth_session, timeout)
                elif scheme == "s3":
                    s3_client = s3_client or s3_client_factory()
                    _download_s3_asset(download_href, destination_path, s3_client, s3_downloader)
                elif scheme in {"file", ""} or len(scheme) == 1:
                    _download_file_asset(download_href, destination_path)
                else:
                    result["failed"].append({"asset": asset_key, "href": download_href, "reason": f"unsupported-scheme:{scheme}"})
                    continue
            except Exception as exc:  # Continue with the item's remaining assets.
                result["failed"].append({"asset": asset_key, "href": download_href, "reason": str(exc)})
                print(f"Failed {asset_key}: {exc}")
                continue

            result["downloaded"].append({"asset": asset_key, "path": str(destination_path), "href": download_href})
            print(f"Downloaded {asset_key} -> {destination_path}")
    finally:
        _close_if_supported(auth_session)
        _close_if_supported(s3_client)

    _print_download_summary(result)
    return result


def _completed_asset_keys(
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    bypass_failed_assets: bool,
) -> set[tuple[str, str, str]]:
    """Return rows that resume logic may skip under the chosen retry policy."""
    completed_statuses = TERMINAL_STATUSES | (ERROR_STATUSES if bypass_failed_assets else set())
    return {key for key, row in rows_by_key.items() if row.get("status") in completed_statuses}


class _SQLiteCheckpointStore:
    """Persist individual asset results without rewriting prior rows."""

    def __init__(
        self,
        checkpoint_path: Path,
        config: dict[str, Any],
        initial_rows: Iterable[dict[str, Any]] = (),
    ) -> None:
        """Open the store and migrate any initial rows in one transaction."""
        if checkpoint_path.suffix.lower() != ".sqlite":
            raise ValueError("Live asset-size audits require a .sqlite checkpoint path")
        self.connection = _connect_sqlite_checkpoint(checkpoint_path)
        with self.connection:
            self.connection.executemany(_SQLITE_UPSERT, (_sqlite_row_values(row) for row in initial_rows))
            _write_sqlite_metadata(self.connection, config)

    def upsert(self, row: dict[str, Any]) -> None:
        """Commit one result as a restart-safe SQLite transaction."""
        with self.connection:
            self.connection.execute(_SQLITE_UPSERT, _sqlite_row_values(row))
            _write_sqlite_metadata(self.connection, config=None)

    def close(self, config: dict[str, Any]) -> None:
        """Save final run metadata and close the database connection."""
        try:
            with self.connection:
                _write_sqlite_metadata(self.connection, config)
        finally:
            self.connection.close()


@dataclass
class _AuditProgress:
    """Own in-memory resume state and row-level SQLite persistence."""

    state: dict[str, Any]
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]]
    completed_keys: set[tuple[str, str, str]]
    checkpoint_store: _SQLiteCheckpointStore
    run_info: dict[str, Any]

    def record(self, key: tuple[str, str, str], row: dict[str, Any]) -> None:
        """Record and commit one result without rewriting previous results."""
        self.rows_by_key[key] = row
        if row["status"] in TERMINAL_STATUSES:
            self.completed_keys.add(key)
        else:
            self.completed_keys.discard(key)
        self.checkpoint_store.upsert(row)

    def close(self) -> None:
        """Close the SQLite store after updating final run metadata."""
        self.state["report_rows"] = list(self.rows_by_key.values())
        self.checkpoint_store.close(self.run_info)


class _RefreshingAuthSession:
    """Create and rotate the authenticated session used by an audit."""

    def __init__(
        self,
        auth_session_factory: Callable[[str], requests.Session],
        client_id: str,
        max_age_seconds: float | None,
        refresh_message: str = "    Refreshed auth session due to token/session age",
    ) -> None:
        """Create the first session and start its monotonic age clock."""
        self._auth_session_factory = auth_session_factory
        self._client_id = client_id
        self._max_age_seconds = max_age_seconds
        self._refresh_message = refresh_message
        self._session = auth_session_factory(client_id)
        self._started_at = time.monotonic()

    def current(self) -> requests.Session:
        """Return a usable session, rotating it after the configured age."""
        if self._is_expired():
            self._session.close()
            self._session = self._auth_session_factory(self._client_id)
            self._started_at = time.monotonic()
            print(self._refresh_message)
        return self._session

    def close(self) -> None:
        """Close the current session."""
        self._session.close()

    def _is_expired(self) -> bool:
        """Return whether session age reached the configured refresh limit."""
        if self._max_age_seconds is None:
            return False
        age_seconds = time.monotonic() - self._started_at
        return self._max_age_seconds >= 0 and age_seconds >= self._max_age_seconds


def _item_asset_context(
    item: dict[str, Any],
    default_collection_id: str,
) -> tuple[str, str, str, dict[str, Any] | None]:
    """Extract the normalized item fields used by the asset iterator."""
    item_collection_id = str(item.get("collection") or default_collection_id)
    item_id = str(item.get("id") or "")
    properties = item.get("properties") or {}
    title = (
        str(properties.get("title") or item.get("title") or "") if isinstance(properties, dict) else str(item.get("title") or "")
    )
    assets = item.get("assets") or {}
    return item_collection_id, item_id, title, assets if isinstance(assets, dict) else None


def _iter_selected_item_assets(
    assets: dict[str, Any],
    only_data_assets: bool,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield valid asset mappings that satisfy the configured role filter."""
    for asset_key, asset_info in assets.items():
        if isinstance(asset_info, dict) and _asset_is_selected(asset_info, only_data_assets):
            yield str(asset_key), asset_info


def _iter_selected_assets(
    collection_id: str,
    items: list[dict[str, Any]],
    only_data_assets: bool,
) -> Iterable[tuple[str, str, str, str, dict[str, Any]]]:
    """Yield selected assets while reporting item-level audit progress."""
    total_items = len(items)
    for processed_items, item in enumerate(items, start=1):
        print("-" * 200)
        print(f"Processing Item {processed_items} of {total_items}")
        item_collection_id, item_id, title, assets = _item_asset_context(item, collection_id)
        if assets is None:
            print(f"Skipping item with invalid assets mapping: {item_collection_id} / {item_id}")
            continue

        print(f"Item: {item_collection_id} / {item_id}")
        for asset_key, asset_info in _iter_selected_item_assets(assets, only_data_assets):
            yield item_collection_id, item_id, asset_key, title, asset_info


def _get_asset_size_safely(
    asset_info: dict[str, Any],
    session: requests.Session,
    request_timeout: float,
    s3_size_getter: Callable[..., int | None],
    stream_http_body_fallback: bool,
) -> AssetSizeResult:
    """Get one asset size without letting ordinary errors abort the workflow."""
    try:
        return get_asset_size(
            asset_info,
            session,
            request_timeout,
            s3_size_getter,
            stream_http_body_fallback,
        )
    except Exception as exc:  # Keep one problematic asset from aborting the remaining work.
        return AssetSizeResult(
            None,
            "",
            str(asset_info.get("href") or ""),
            f"{type(exc).__name__}: {exc}",
            True,
        )


def _audit_collection(
    collection_id: str,
    items: list[dict[str, Any]],
    progress: _AuditProgress,
    auth_session: _RefreshingAuthSession,
    min_size_bytes: int,
    only_data_assets: bool,
    request_timeout: float,
    s3_size_getter: Callable[..., int | None],
    stream_http_body_fallback: bool,
) -> None:
    """Audit all selected, incomplete assets in one collection."""
    for item_collection_id, item_id, asset_key, title, asset_info in _iter_selected_assets(
        collection_id,
        items,
        only_data_assets,
    ):
        session = auth_session.current()
        key = (item_collection_id, item_id, asset_key)
        if key in progress.completed_keys:
            existing_row = progress.rows_by_key[key]
            if title and existing_row.get("title") != title:
                updated_row = {**existing_row, "title": title}
                progress.record(key, updated_row)
                print(f"    SKIP {asset_key}: already checked; title refreshed")
            else:
                print(f"    SKIP {asset_key}: already checked")
            continue

        result = _get_asset_size_safely(
            asset_info,
            session,
            request_timeout,
            s3_size_getter,
            stream_http_body_fallback,
        )
        row = build_asset_report_row(
            item_collection_id,
            item_id,
            asset_key,
            asset_info,
            result,
            min_size_bytes,
            title=title,
        )
        progress.record(key, row)
        size_label = row["file_size_human"] or "N/A"
        print(f"    {row['status'].upper():16} {asset_key}: {size_label}")


def run_asset_size_audit(
    collection_ids: list[str],
    fetch_items: Callable[[str, int | None], list[dict[str, Any]]],
    auth_session_factory: Callable[[str], requests.Session],
    client_id: str,
    checkpoint_path: str | Path,
    min_size_bytes: int = 1024,
    max_items_per_collection: int | None = None,
    only_data_assets: bool = False,
    request_timeout: float = 20,
    s3_size_getter: Callable[..., int | None] = get_s3_object_size,
    stream_http_body_fallback: bool = False,
    auth_session_max_age_seconds: float | None = 55 * 60,
    bypass_failed_assets: bool = True,
) -> pd.DataFrame:
    """Audit selected collections and resume from terminal rows in the checkpoint.

    Rows with ``ok`` or ``too_small`` are terminal and skipped on restart. Rows
    whose size was unavailable or whose request failed are retried unless
    ``bypass_failed_assets`` is enabled. Each attempted asset is committed as one
    SQLite transaction. Excel export is deliberately a separate final operation.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_existed = checkpoint_path.exists()
    state = load_asset_size_checkpoint(checkpoint_path, min_size_bytes)
    rows_by_key = {_row_key(row): row for row in state["report_rows"]}
    run_info = {
        "collections": collection_ids,
        "min_size_bytes": min_size_bytes,
        "max_items_per_collection": "all" if max_items_per_collection is None else max_items_per_collection,
        "only_data_assets": only_data_assets,
        "stream_http_body_fallback": stream_http_body_fallback,
        "auth_session_max_age_seconds": auth_session_max_age_seconds,
        "bypass_failed_assets": bypass_failed_assets,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "last_run_started_at_utc": _utc_now(),
    }
    state["config"] = run_info
    checkpoint_store = _SQLiteCheckpointStore(
        checkpoint_path,
        run_info,
        initial_rows=() if checkpoint_existed else state["report_rows"],
    )
    progress = _AuditProgress(
        state=state,
        rows_by_key=rows_by_key,
        completed_keys=_completed_asset_keys(rows_by_key, bypass_failed_assets),
        checkpoint_store=checkpoint_store,
        run_info=run_info,
    )
    auth_session = _RefreshingAuthSession(
        auth_session_factory,
        client_id,
        auth_session_max_age_seconds,
    )
    try:
        for collection_id in collection_ids:
            print("=" * 200)
            print(f"Processing collection: {collection_id}")
            items = fetch_items(collection_id, max_items_per_collection)
            _audit_collection(
                collection_id,
                items,
                progress,
                auth_session,
                min_size_bytes,
                only_data_assets,
                request_timeout,
                s3_size_getter,
                stream_http_body_fallback,
            )
    finally:
        try:
            progress.close()
        finally:
            auth_session.close()

    report_df = build_report_dataframe(rows_by_key.values(), min_size_bytes)
    print("-" * 100)
    print(build_summary_dataframe(report_df).to_string(index=False))
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    return report_df


def _asset_info_from_checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the asset metadata needed for a direct checkpoint retry."""
    roles = [role.strip() for role in str(row.get("roles") or "").split(",") if role.strip()]
    return {
        "href": str(row.get("href") or ""),
        "roles": roles,
        "file:size": row.get("declared_size_bytes"),
        "hub:relativePath": str(row.get("filename") or ""),
    }


def _retry_checkpoint_assets(
    unresolved_keys: list[tuple[str, str, str]],
    progress: _AuditProgress,
    auth_session: _RefreshingAuthSession,
    min_size_bytes: int,
    request_timeout: float,
    s3_size_getter: Callable[..., int | None],
    stream_http_body_fallback: bool,
) -> None:
    """Retry a fixed snapshot of unresolved checkpoint assets."""
    total_assets = len(unresolved_keys)
    for retry_index, key in enumerate(unresolved_keys, start=1):
        checkpoint_row = progress.rows_by_key[key]
        asset_info = _asset_info_from_checkpoint_row(checkpoint_row)
        result = _get_asset_size_safely(
            asset_info,
            auth_session.current(),
            request_timeout,
            s3_size_getter,
            stream_http_body_fallback,
        )
        row = build_asset_report_row(
            *key,
            asset_info,
            result,
            min_size_bytes,
            title=str(checkpoint_row.get("title") or ""),
        )
        progress.record(key, row)

        size_label = row["file_size_human"] or "N/A"
        print(f"RETRY {retry_index:,}/{total_assets:,} {row['status'].upper():16} {key[1]} / {key[2]}: {size_label}")


def retry_unresolved_asset_sizes(
    auth_session_factory: Callable[[str], requests.Session],
    client_id: str,
    checkpoint_path: str | Path,
    min_size_bytes: int = 1024,
    request_timeout: float = 20,
    s3_size_getter: Callable[..., int | None] = get_s3_object_size,
    stream_http_body_fallback: bool = False,
    auth_session_max_age_seconds: float | None = 55 * 60,
) -> pd.DataFrame:
    """Retry every latest unresolved checkpoint row directly from its saved href.

    This pass deliberately does not refetch catalog items, so checkpoint rows
    outside the notebook's current search dates are still attempted. Each result
    replaces the prior row and is persisted immediately as one SQLite transaction.
    Excel export is deliberately a separate final operation.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_existed = checkpoint_path.exists()
    state = load_asset_size_checkpoint(checkpoint_path, min_size_bytes)
    rows_by_key = {_row_key(row): row for row in state["report_rows"]}
    unresolved_keys = sorted(key for key, row in rows_by_key.items() if row.get("status") not in TERMINAL_STATUSES)
    run_info = {
        **state.get("config", {}),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "retry_started_at_utc": _utc_now(),
        "retry_unresolved_assets": len(unresolved_keys),
        "retry_stream_http_body_fallback": stream_http_body_fallback,
        "retry_auth_session_max_age_seconds": auth_session_max_age_seconds,
    }
    state["config"] = run_info
    checkpoint_store = _SQLiteCheckpointStore(
        checkpoint_path,
        run_info,
        initial_rows=() if checkpoint_existed else state["report_rows"],
    )
    progress = _AuditProgress(
        state=state,
        rows_by_key=rows_by_key,
        completed_keys=set(),
        checkpoint_store=checkpoint_store,
        run_info=run_info,
    )

    if not unresolved_keys:
        progress.close()
        print("No unresolved assets to retry.")
        return build_report_dataframe(rows_by_key.values(), min_size_bytes)

    print(f"Retrying {len(unresolved_keys):,} unresolved assets directly from checkpoint hrefs.")
    auth_session = _RefreshingAuthSession(
        auth_session_factory,
        client_id,
        auth_session_max_age_seconds,
        refresh_message="Refreshed auth session due to token/session age",
    )
    try:
        _retry_checkpoint_assets(
            unresolved_keys,
            progress,
            auth_session,
            min_size_bytes,
            request_timeout,
            s3_size_getter,
            stream_http_body_fallback,
        )
    finally:
        try:
            progress.close()
        finally:
            auth_session.close()

    report_df = build_report_dataframe(rows_by_key.values(), min_size_bytes)
    print("-" * 100)
    print(build_summary_dataframe(report_df).to_string(index=False))
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    return report_df


def _normalized_assets_folder_name(assets_folder_name: str) -> str:
    """Validate the single S3 path component used to identify asset objects."""
    normalized = str(assets_folder_name).strip().strip("/")
    if not normalized or "/" in normalized or "\\" in normalized:
        raise ValueError("assets_folder_name must be one non-empty path component")
    return normalized


def _is_s3_asset_object(s3_key: str, assets_folder_name: str) -> bool:
    """Return whether an S3 key is a file beneath the exact assets component."""
    return not s3_key.endswith("/") and assets_folder_name in s3_key.strip("/").split("/")


def export_s3_asset_keys(
    s3_client: Any,
    bucket_name: str,
    s3_prefix: str,
    output_file: str | Path,
    assets_folder_name: str = "assets",
) -> int:
    """Stream matching S3 object keys into a one-column CSV inventory."""
    assets_folder_name = _normalized_assets_folder_name(assets_folder_name)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0

    # Pagination keeps memory use independent of the number of objects in the prefix.
    paginator = s3_client.get_paginator("list_objects_v2")
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(["file"])
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for s3_object in page.get("Contents", []):
                s3_key = str(s3_object.get("Key") or "")
                # Match the exact path component and exclude S3 folder placeholders.
                if _is_s3_asset_object(s3_key, assets_folder_name):
                    writer.writerow([s3_key])
                    file_count += 1
    return file_count


def _s3_parse_error(s3_key: str, component_count: int, reason: str) -> dict[str, Any]:
    """Build one rejected-key record with a stable schema."""
    return {"file": s3_key, "component_count": component_count, "error_reason": reason}


def parse_s3_asset_key(
    s3_key: Any,
    assets_folder_name: str = "assets",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Parse a processing-result key relative to its exact assets component."""
    assets_folder_name = _normalized_assets_folder_name(assets_folder_name)
    if not isinstance(s3_key, str) or not s3_key.strip():
        return None, _s3_parse_error("", 0, "Empty S3 key")

    normalized_key = s3_key.strip().strip("/")
    parts = normalized_key.split("/")
    component_count = len(parts)
    try:
        assets_index = parts.index(assets_folder_name)
    except ValueError:
        return None, _s3_parse_error(normalized_key, component_count, "Missing 'assets' path component")
    if assets_index < 2:
        return None, _s3_parse_error(normalized_key, component_count, "Expected a process path before 'assets'")
    if component_count < assets_index + 4:
        return None, _s3_parse_error(normalized_key, component_count, "Expected collection/item/filename after 'assets'")

    return {
        "s3_path": parts[0],
        "process_id": "/".join(parts[1:assets_index]),
        "asset_folder": parts[assets_index],
        "collection_id": parts[assets_index + 1],
        "item_id": parts[assets_index + 2],
        "filename": parts[-1],
        "file": normalized_key,
    }, None


def _parse_s3_key_chunk(
    s3_keys: Iterable[Any],
    assets_folder_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one bounded key chunk into valid and rejected dataframes."""
    # Keep successful and rejected records separate so malformed keys remain auditable.
    parsed_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    for s3_key in s3_keys:
        parsed_record, error_record = parse_s3_asset_key(s3_key, assets_folder_name)
        if parsed_record is not None:
            parsed_records.append(parsed_record)
        elif error_record is not None:
            error_records.append(error_record)

    # Explicit columns preserve stable CSV schemas even when either result is empty.
    return (
        pd.DataFrame.from_records(parsed_records, columns=S3_INVENTORY_COLUMNS),
        pd.DataFrame.from_records(error_records, columns=S3_PARSE_ERROR_COLUMNS),
    )


def _append_csv_chunk(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Append a non-empty dataframe while writing its header exactly once."""
    if not dataframe.empty:
        dataframe.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)


def _ensure_csv_schema(output_path: Path, columns: list[str]) -> None:
    """Create a header-only CSV when parsing produced no rows of that type."""
    if not output_path.exists():
        pd.DataFrame(columns=columns).to_csv(output_path, index=False)


def parse_s3_asset_inventory(
    inventory_file: str | Path,
    parsed_output_file: str | Path,
    errors_output_file: str | Path,
    chunk_size: int = 250_000,
    assets_folder_name: str = "assets",
) -> dict[str, int]:
    """Parse a large S3 key inventory into valid and rejected CSV outputs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    inventory_path = Path(inventory_file)
    if not inventory_path.is_file():
        raise FileNotFoundError(f"S3 inventory file not found: {inventory_path}")

    parsed_path = Path(parsed_output_file)
    errors_path = Path(errors_output_file)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_temporary_path = parsed_path.with_name(f"{parsed_path.stem}.tmp{parsed_path.suffix}")
    errors_temporary_path = errors_path.with_name(f"{errors_path.stem}.tmp{errors_path.suffix}")

    # Build both outputs off to the side so an interrupted parse cannot replace valid reports.
    _remove_temporary_file(parsed_temporary_path)
    _remove_temporary_file(errors_temporary_path)

    valid_count = 0
    error_count = 0
    try:
        # Chunking bounds dataframe and record-list memory for inventories with millions of keys.
        for chunk_number, source_chunk in enumerate(pd.read_csv(inventory_path, chunksize=chunk_size), start=1):
            if "file" not in source_chunk.columns:
                raise ValueError(f"S3 inventory is missing the 'file' column: {inventory_path}")
            valid_chunk, error_chunk = _parse_s3_key_chunk(source_chunk["file"], assets_folder_name)
            _append_csv_chunk(valid_chunk, parsed_temporary_path)
            _append_csv_chunk(error_chunk, errors_temporary_path)
            valid_count += len(valid_chunk)
            error_count += len(error_chunk)
            print(f"Chunk {chunk_number}: {len(valid_chunk):,} valid, {len(error_chunk):,} errors")

        # Empty result sets still need headers so downstream reads have predictable schemas.
        _ensure_csv_schema(parsed_temporary_path, S3_INVENTORY_COLUMNS)
        _ensure_csv_schema(errors_temporary_path, S3_PARSE_ERROR_COLUMNS)
        # Install only complete outputs; the shared helper also handles Windows file locks.
        _replace_or_copy_completed_file(parsed_temporary_path, parsed_path)
        _replace_or_copy_completed_file(errors_temporary_path, errors_path)
    finally:
        _remove_temporary_file(parsed_temporary_path)
        _remove_temporary_file(errors_temporary_path)
    return {"valid_rows": valid_count, "error_rows": error_count}


def normalize_and_validate_s3_keys(dataframe: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize and validate the compound keys used for catalog comparison."""
    # Fail early with a source-specific message instead of producing misleading mismatches.
    missing_columns = [column for column in S3_KEY_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{source_name} is missing key columns: {missing_columns}")

    # Compare normalized copies so callers retain their original values and dtypes.
    normalized = dataframe.copy()
    for column in S3_KEY_COLUMNS:
        normalized[column] = normalized[column].astype("string").str.strip()

    # Null or whitespace-only components cannot identify an asset unambiguously.
    invalid_key_mask = normalized[S3_KEY_COLUMNS].isna().any(axis=1) | normalized[S3_KEY_COLUMNS].eq("").any(axis=1)
    if invalid_key_mask.any():
        raise ValueError(f"{source_name} contains {invalid_key_mask.sum():,} rows with incomplete keys")
    return normalized


def _temporary_s3_path_mask(data_in_s3: pd.DataFrame) -> pd.Series:
    """Return rows whose S3 object key contains a ``for_prod`` path segment."""
    if "file" in data_in_s3.columns:
        normalized_files = data_in_s3["file"].astype("string").str.strip().str.replace("\\", "/", regex=False)
        return normalized_files.str.contains(
            rf"(?:^|/){S3_TEMPORARY_PROCESS_ROOT}(?:/|$)",
            regex=True,
            na=False,
        )

    if "process_id" in data_in_s3.columns:
        process_ids = data_in_s3["process_id"].astype("string").str.strip().str.replace("\\", "/", regex=False)
        return process_ids.str.contains(
            rf"(?:^|/){S3_TEMPORARY_PROCESS_ROOT}(?:/|$)",
            regex=True,
            na=False,
        )

    return pd.Series(False, index=data_in_s3.index, dtype=bool)


def build_s3_catalog_mismatch_frames(
    collection_id: str,
    data_in_s3: pd.DataFrame,
    data_in_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return summary and categorized mismatch frames for one collection."""
    # Normalize both sources identically before constructing compound comparison keys.
    data_in_s3 = normalize_and_validate_s3_keys(data_in_s3, f"Parsed S3 inventory ({collection_id})")
    data_in_catalog = normalize_and_validate_s3_keys(data_in_catalog, f"Catalog asset report ({collection_id})")

    # Temporary for_prod objects are path errors, but they still prove that the
    # collection/item/filename compound key exists somewhere in S3.
    temporary_path_mask = _temporary_s3_path_mask(data_in_s3).fillna(False)
    s3_wrong_path_errors = data_in_s3.loc[temporary_path_mask].copy()
    s3_wrong_path_errors["status"] = "error"
    s3_wrong_path_errors["mismatch_reason"] = "S3 object is stored under the temporary 'for_prod' process path"

    # Row-level indexes preserve duplicate S3 objects; unique indexes drive set comparison metrics.
    s3_keys = pd.MultiIndex.from_frame(data_in_s3[S3_KEY_COLUMNS])
    catalog_keys = pd.MultiIndex.from_frame(data_in_catalog[S3_KEY_COLUMNS])
    s3_unique_keys = pd.MultiIndex.from_frame(data_in_s3[S3_KEY_COLUMNS].drop_duplicates())
    catalog_unique_keys = pd.MultiIndex.from_frame(data_in_catalog[S3_KEY_COLUMNS].drop_duplicates())

    # Calculate both mismatch directions because each represents a different cleanup action.
    s3_orphans = data_in_s3.loc[~s3_keys.isin(catalog_unique_keys)].copy()
    s3_orphans["mismatch_reason"] = "S3 object is not referenced in the catalog"
    catalog_missing_in_s3 = data_in_catalog.loc[~catalog_keys.isin(s3_unique_keys)].copy()
    catalog_missing_in_s3["mismatch_reason"] = "Catalog asset is missing in S3"

    # Keep summary construction separate from matching so the comparison remains easy to audit.
    summary = _build_s3_mismatch_summary(
        collection_id,
        data_in_s3,
        data_in_catalog,
        s3_unique_keys,
        catalog_unique_keys,
        s3_orphans,
        catalog_missing_in_s3,
        s3_wrong_path_errors,
    )
    return summary, s3_orphans, catalog_missing_in_s3, s3_wrong_path_errors


def _build_s3_mismatch_summary(
    collection_id: str,
    data_in_s3: pd.DataFrame,
    data_in_catalog: pd.DataFrame,
    s3_unique_keys: pd.MultiIndex,
    catalog_unique_keys: pd.MultiIndex,
    s3_orphans: pd.DataFrame,
    catalog_missing_in_s3: pd.DataFrame,
    s3_wrong_path_errors: pd.DataFrame,
) -> pd.DataFrame:
    """Build the stable long-form mismatch summary used by reports."""
    metrics = [
        ("collection_id", collection_id),
        ("s3_object_rows", len(data_in_s3)),
        ("s3_matchable_object_rows", len(data_in_s3)),
        ("s3_wrong_path_object_rows", len(s3_wrong_path_errors)),
        (
            "s3_wrong_path_unique_asset_keys",
            len(s3_wrong_path_errors[S3_KEY_COLUMNS].drop_duplicates()),
        ),
        ("s3_unique_asset_keys", len(s3_unique_keys)),
        ("catalog_asset_rows", len(data_in_catalog)),
        ("catalog_unique_asset_keys", len(catalog_unique_keys)),
        ("matched_unique_asset_keys", len(s3_unique_keys.intersection(catalog_unique_keys))),
        ("s3_orphan_object_rows", len(s3_orphans)),
        ("s3_orphan_unique_asset_keys", len(s3_orphans[S3_KEY_COLUMNS].drop_duplicates())),
        ("catalog_missing_asset_rows", len(catalog_missing_in_s3)),
        ("catalog_missing_unique_asset_keys", len(catalog_missing_in_s3[S3_KEY_COLUMNS].drop_duplicates())),
        ("duplicate_s3_object_rows", len(data_in_s3) - len(s3_unique_keys)),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def write_s3_catalog_mismatch_report(
    output_file: str | Path,
    summary: pd.DataFrame,
    s3_orphans: pd.DataFrame,
    catalog_missing_in_s3: pd.DataFrame,
    s3_wrong_path_errors: pd.DataFrame,
) -> Path:
    """Write one collection's mismatch dataframes to a formatted workbook."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    output_sheets = {
        "summary": (summary, "MismatchSummary"),
        "s3_orphans": (s3_orphans, "S3Orphans"),
        "catalog_missing_in_s3": (catalog_missing_in_s3, "CatalogMissingInS3"),
        "s3_wrong_path_errors": (s3_wrong_path_errors, "S3WrongPathErrors"),
    }
    try:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            for sheet_name, (dataframe, table_name) in output_sheets.items():
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                _format_sheet(writer.book[sheet_name], dataframe, table_name)
        _replace_or_copy_completed_file(temporary_path, output_path)
    finally:
        _remove_temporary_file(temporary_path)
    return output_path


def _normalized_collection_ids(collection_ids: Iterable[Any]) -> list[str]:
    """Return unique, non-empty collection IDs while preserving input order."""
    normalized = (str(collection_id).strip() for collection_id in collection_ids)
    return list(dict.fromkeys(collection_id for collection_id in normalized if collection_id))


def _available_catalog_report_files(
    collection_ids: list[str],
    reports_folder: Path,
) -> dict[str, Path]:
    """Return existing report paths and warn for collections that are skipped."""
    report_files = {collection_id: reports_folder / collection_id / "asset_size_report.xlsx" for collection_id in collection_ids}
    for collection_id, report_file in list(report_files.items()):
        if report_file.exists():
            continue
        print(f"WARNING: Skipping collection {collection_id}: catalog asset report not found at {report_file}")
        del report_files[collection_id]
    return report_files


def _load_selected_s3_inventory(
    parsed_s3_file: Path,
    collection_ids: Iterable[str],
    csv_chunk_size: int,
) -> dict[str, list[pd.DataFrame]]:
    """Scan one large parsed inventory and retain only selected collections."""
    selected_ids = list(collection_ids)
    selected_set = set(selected_ids)
    chunks_by_collection: dict[str, list[pd.DataFrame]] = {collection_id: [] for collection_id in selected_ids}
    for s3_chunk in pd.read_csv(
        parsed_s3_file,
        usecols=S3_INVENTORY_COLUMNS,
        dtype=S3_KEY_DTYPES,
        chunksize=csv_chunk_size,
    ):
        s3_chunk["collection_id"] = s3_chunk["collection_id"].astype("string").str.strip()
        selected_chunk = s3_chunk.loc[s3_chunk["collection_id"].isin(selected_set)].copy()
        for collection_id, collection_chunk in selected_chunk.groupby("collection_id", sort=False):
            chunks_by_collection[str(collection_id)].append(collection_chunk)
    return chunks_by_collection


def _catalog_rows_for_collection(report_file: Path, collection_id: str) -> pd.DataFrame:
    """Read and filter one asset-size workbook to its expected collection."""
    with pd.ExcelFile(report_file) as workbook:
        report_sheet_names = [
            sheet_name
            for sheet_name in workbook.sheet_names
            if sheet_name == "asset_report"
            or (sheet_name.startswith("asset_report_") and sheet_name.removeprefix("asset_report_").isdigit())
        ]
        if not report_sheet_names:
            raise ValueError(f"Asset-size workbook has no asset_report sheets: {report_file}")
        report_parts = [
            pd.read_excel(
                workbook,
                sheet_name=sheet_name,
                usecols=CATALOG_S3_CHECK_COLUMNS,
                dtype=S3_KEY_DTYPES,
            )
            for sheet_name in report_sheet_names
        ]
    data_in_catalog = pd.concat(report_parts, ignore_index=True)
    collection_mask = data_in_catalog["collection_id"].astype("string").str.strip().eq(collection_id).fillna(False)
    return data_in_catalog.loc[collection_mask].copy()


def _check_s3_collection(
    collection_id: str,
    report_file: Path,
    s3_chunks: list[pd.DataFrame],
    reports_folder: Path,
) -> pd.DataFrame:
    """Compare and write outputs for one collection, returning its summary rows."""
    data_in_s3 = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=S3_INVENTORY_COLUMNS)
    data_in_catalog = _catalog_rows_for_collection(report_file, collection_id)
    summary, s3_orphans, catalog_missing_in_s3, s3_wrong_path_errors = build_s3_catalog_mismatch_frames(
        collection_id,
        data_in_s3,
        data_in_catalog,
    )
    output_file = reports_folder / collection_id / "s3_catalog_mismatches.xlsx"
    write_s3_catalog_mismatch_report(
        output_file,
        summary,
        s3_orphans,
        catalog_missing_in_s3,
        s3_wrong_path_errors,
    )
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"Saved mismatch report to {output_file}")
    collection_summary = summary.iloc[1:].copy()
    collection_summary.insert(0, "collection_id", collection_id)
    return collection_summary


def check_s3_catalog_mismatches(
    collection_ids: Iterable[Any],
    parsed_s3_file: str | Path,
    reports_folder: str | Path,
    csv_chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Compare S3 inventory and catalog reports, writing one workbook per collection."""
    if csv_chunk_size <= 0:
        raise ValueError("csv_chunk_size must be greater than zero")
    selected_collection_ids = _normalized_collection_ids(collection_ids)
    if not selected_collection_ids:
        raise ValueError("Provide at least one collection ID")
    parsed_s3_path = Path(parsed_s3_file)
    if not parsed_s3_path.is_file():
        raise FileNotFoundError(f"Parsed S3 file not found: {parsed_s3_path}")

    reports_path = Path(reports_folder)
    report_files = _available_catalog_report_files(selected_collection_ids, reports_path)
    if not report_files:
        return pd.DataFrame(columns=["collection_id", "metric", "value"])
    chunks_by_collection = _load_selected_s3_inventory(parsed_s3_path, report_files, csv_chunk_size)
    summary_frames = [
        _check_s3_collection(collection_id, report_file, chunks_by_collection[collection_id], reports_path)
        for collection_id, report_file in report_files.items()
    ]
    return pd.concat(summary_frames, ignore_index=True)


def checkpoint_as_dict(checkpoint_state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy; useful for notebook display and debugging."""
    return {
        **{key: value for key, value in checkpoint_state.items() if key != "result"},
        **({"result": asdict(checkpoint_state["result"])} if isinstance(checkpoint_state.get("result"), AssetSizeResult) else {}),
    }
