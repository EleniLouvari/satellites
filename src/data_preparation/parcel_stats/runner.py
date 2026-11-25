"""Run and resume parcel statistics independently of notebooks and web frameworks."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import multiuser
from .core.base import ParcelStatsBase
from .run_config import PreparedRun, engine_workspace, file_sha256, partition_extractor, prepare_run


def _write_json(path: Path, value: dict) -> None:
    """Commit a JSON record only after its complete contents have been written."""
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _package_version() -> str:
    """Identify installed releases while permitting source-tree development."""
    try:
        return version("satellites")
    except PackageNotFoundError:
        return "source-tree"


@contextmanager
def run_lock(directory: Path):
    """Hold an OS lock released on exit or process death; never delete the lock file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".parcel_stats.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Error: This run is already locked by another process or is not writable.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _accounts(accounts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Validate separate in-memory credentials while preserving account ordering."""
    if not isinstance(accounts, list) or not accounts:
        raise ValueError("Error: At least one Copernicus account must be supplied separately.")
    credentials = []
    for item in accounts:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Error: Each account must be a username/password pair.")
        username, password = item
        if any(not isinstance(value, str) or not value.strip() for value in item):
            raise ValueError("Error: Account usernames and passwords must be non-empty text.")
        credentials.append((username.strip(), password))
    if len({user.casefold() for user, _ in credentials}) != len(credentials):
        raise ValueError("Error: Duplicate Copernicus accounts are not allowed.")
    return credentials


def _extraction_config(prepared: PreparedRun) -> dict:
    """Translate the request into the existing multi-account engine contract."""
    config = prepared.config
    return dict(
        config["processing"],
        output_dir=config["output_dir"],
        run_identifier=config["run_identifier"],
        parcel_id_field=config["input"]["parcel_id_field"],
        resume_completed_partitions=True,
    )


def _source_accounts(source, accounts):
    """Require Copernicus credentials only for runs that use openEO."""
    if source == "hub":
        if accounts is not None and accounts != []:
            raise ValueError("Error: HUB uses server credentials; Copernicus accounts must be empty.")
        return []
    return _accounts(accounts)


def _completed_partitions(prepared: PreparedRun, options: dict, hub_snapshots=None) -> dict[int, gpd.GeoDataFrame]:
    """Require signed, complete partitions and exact expected parcel coverage."""
    output = Path(options["output_dir"])
    completed = {}
    expected_names = {f"partition_{index}" for index in range(1, len(prepared.parts) + 1)}
    if {path.name for path in output.glob("partition_*") if path.is_dir()} != expected_names:
        raise ValueError("Error: Run partition directories do not match the expected partitions.")
    for index, parcels in enumerate(prepared.parts, 1):
        directory = output / f"partition_{index}"
        marker = directory / "satellite_partition_completion.json"
        if not marker.is_file():
            raise ValueError(f"Error: Missing completion manifest for partition {index}.")
        extractor = partition_extractor(prepared.config.get("source", "openeo"), parcels, options, index)
        try:
            if prepared.config.get("source") == "hub":
                digest = (hub_snapshots or {}).get(index)
                if not digest:
                    raise ValueError("Error: HUB publication requires the acquired inventory snapshot.")
                # run() already refreshed this inventory; publication verifies that same snapshot.
                extractor._hub_manifest_digest = digest
            result = extractor._load_completed_partition()
        except (ValueError, OSError, KeyError):
            raise ValueError(f"Error: Unreadable completed partition {index}.") from None
        if result is None or result.crs is None or result.crs.to_epsg() != options["working_epsg"]:
            raise ValueError(f"Error: Incomplete or incompatible partition {index}.")
        completed[index] = result
    return completed


def _publish(prepared: PreparedRun, options: dict, hub_snapshots=None) -> dict:
    """Merge only verified partitions and atomically publish the final GeoParquet."""
    completed = _completed_partitions(prepared, options, hub_snapshots)
    output = Path(options["output_dir"])
    # The existing merger retains all attributes; validation above prevents its
    # historical duplicate-dropping behavior from hiding missing parcel results.
    merged, staged = multiuser.merge_and_save_results(
        all_results=completed, output_dir=output / "publication", working_epsg=options["working_epsg"]
    )
    field = options["parcel_id_field"]
    if merged[field].duplicated().any() or set(merged[field].astype(str)) != set(prepared.parcels[field]):
        raise ValueError("Error: Merged partition results do not match the input parcel IDs.")
    destination = output / staged.name
    staged.replace(destination)
    reports = []
    for index in completed:
        report = pd.read_csv(output / f"partition_{index}" / "satellite_pixel_cleaning_report.csv")
        report.insert(0, "partition_number", index)
        reports.append(report)
    report_path = output / "satellite_pixel_cleaning_report.csv"
    temporary = report_path.with_suffix(".partial.csv")
    pd.concat(reports, ignore_index=True).to_csv(temporary, index=False)
    temporary.replace(report_path)
    manifest = {
        "status": "complete",
        "source": prepared.config.get("source", "openeo"),
        "parcel_count": len(merged),
        "partition_count": len(completed),
        "artifacts": {"time_statistics": str(destination), "cleaning_report": str(report_path)},
        "time_statistics_sha256": file_sha256(destination),
    }
    _write_json(output / "artifacts.json", manifest)
    return manifest


def _run_partitions(prepared, accounts, options):
    """Dispatch acquisition while keeping locking, state and publication shared."""
    if prepared.config.get("source") == "hub":
        snapshots = {}
        for index, parcels in enumerate(prepared.parts, 1):
            extractor = partition_extractor("hub", parcels, options, index)
            extractor.run()
            snapshots[index] = extractor._hub_manifest_digest
        return snapshots
    else:
        multiuser.run_parallel_extractions(prepared.parts, accounts, options, scheduling=prepared.config["scheduling"])


def _failure_reason(exc: BaseException) -> str:
    """Describe an exception with its causes, e.g. the endpoint error behind a retry wrapper.

    Causes hidden with ``raise ... from None`` stay hidden.
    """
    parts, current = [], exc
    while current is not None and len(parts) < 4:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return " <- caused by ".join(parts)


def _log_failure(output: Path, exc: BaseException) -> None:
    """Record why a run failed in the run-level orchestration log and caller logging hooks.

    The record uses the pipeline namespace, so Parcel Studio redacts it and
    publishes it as a run event; the file is exported with its diagnostics.
    """
    logger = logging.getLogger(f"{ParcelStatsBase.LOGGER_NAMESPACE}.run")
    try:
        handler = logging.FileHandler(output / ParcelStatsBase.LOG_FILE_NAME, encoding="utf-8")
    except OSError:
        # An unwritable log must not replace the original error; logging hooks still receive it.
        logger.error("Error: Run failed: %s", _failure_reason(exc))
        return
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    try:
        logger.error("Error: Run failed: %s", _failure_reason(exc))
    finally:
        logger.removeHandler(handler)
        handler.close()


def _execute(prepared: PreparedRun, accounts: list[tuple[str, str]], state: dict, *, retry_exhausted_memory=False) -> dict:
    """Persist status around execution; the failure reason is logged, never stored in ``run.json``."""
    output = Path(prepared.config["output_dir"])
    state.update(status="running", updated_at=datetime.now(UTC).isoformat())
    _write_json(output / "run.json", state)
    if (output / "artifacts.json").exists():
        _write_json(output / "artifacts.json", {"status": "running"})
    try:
        options = _extraction_config(prepared)
        if retry_exhausted_memory and prepared.config.get("source", "openeo") == "openeo":
            options["retry_exhausted_memory"] = True
        with engine_workspace(output):
            snapshots = _run_partitions(prepared, accounts, options)
            manifest = _publish(prepared, options, snapshots)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        if isinstance(exc, Exception):
            _log_failure(output, exc)
        raise
    else:
        state.update(status="complete")
        state.pop("error_type", None)
        return manifest
    finally:
        state["updated_at"] = datetime.now(UTC).isoformat()
        _write_json(output / "run.json", state)


def run_parcel_stats(
    config: dict, *, accounts: list[tuple[str, str]] | None = None, base_dir: str | Path | None = None,
    app_user_id: int | None = None,
) -> dict:
    """Execute a new dictionary-configured run and return its completed artifact manifest.

    ``accounts`` are supplied separately by a trusted caller; this function is
    not an authentication boundary. The optional app user ID records ownership
    for later resume checks. Existing nonempty output directories are rejected.
    """
    accounts = _source_accounts(config.get("source", "openeo"), accounts)
    prepared = prepare_run(config, base_dir=base_dir)
    output = Path(prepared.config["output_dir"])
    with run_lock(output):
        if any(path.name != ".parcel_stats.lock" for path in output.iterdir()):
            raise ValueError("Error: A new run requires an empty output directory; use resume for an existing run.")
        _write_json(output / "config.json", prepared.config)
        state = {
            "schema_version": 1,
            "source": prepared.config.get("source", "openeo"),
            "input_sha256": prepared.input_sha256,
            "config_sha256": file_sha256(output / "config.json"),
            "accounts": [name for name, _ in accounts],
            "app_user_id": app_user_id,
            "package_version": _package_version(),
        }
        return _execute(prepared, accounts, state)


def resume_parcel_stats(
    run_dir: str | Path, *, accounts: list[tuple[str, str]] | None = None, app_user_id: int | None = None,
    retry_exhausted_memory: bool = False,
) -> dict:
    """Resume a saved run with unchanged input/configuration and the same account identities."""
    if type(retry_exhausted_memory) is not bool:
        raise TypeError("Error: retry_exhausted_memory must be a boolean.")
    output = Path(run_dir).resolve()
    if not (output / "run.json").is_file() or not (output / "config.json").is_file():
        raise ValueError("Error: run_dir must contain run.json and config.json from a CLI run.")
    with run_lock(output):
        state = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if state.get("app_user_id") != app_user_id:
            raise ValueError("Error: The application user does not own this run.")
        if state.get("config_sha256") != file_sha256(output / "config.json"):
            raise ValueError("Error: Saved configuration changed; create a new run.")
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        accounts = _source_accounts(config.get("source", "openeo"), accounts)
        if state.get("accounts") != [name for name, _ in accounts]:
            raise ValueError("Error: Resume requires the same account usernames and order; passwords may change.")
        if Path(config["output_dir"]).resolve() != output:
            raise ValueError("Error: Run directories cannot be relocated; persisted remote jobs contain absolute paths.")
        prepared = prepare_run(config)
        if state.get("input_sha256") != prepared.input_sha256:
            raise ValueError("Error: The input file changed; create a new run.")
        return _execute(prepared, accounts, state, retry_exhausted_memory=retry_exhausted_memory)
