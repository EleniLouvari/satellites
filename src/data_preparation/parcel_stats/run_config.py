"""Offline input preparation for dictionary and JSON parcel-statistics requests."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import geopandas as gpd

from .core.base import ParcelStatsBase
from .core.crs import estimate_utm_epsg_from_parcels
from .core.parcel_batches import ParcelBatchPlanner
from .job_manager import OpenEOJobManagerZonalStats

_RESERVED = {
    "parcels",
    "output_dir",
    "parcel_id_field",
    "openeo_username",
    "openeo_password",
    "run_identifier",
    "resume_completed_partitions",
    "retry_exhausted_memory",
    "hub_catalog_url",
    "hub_client_id",
}


@dataclass
class PreparedRun:
    """Hold validated configuration and partitions without acquiring imagery."""

    config: dict
    parcels: gpd.GeoDataFrame
    parts: list[gpd.GeoDataFrame]
    input_sha256: str
    batch_count: int

    def summary(self) -> dict:
        """Return a JSON-safe preflight summary for CLI or application callers."""
        return {
            "parcel_count": len(self.parcels),
            "partition_count": len(self.parts),
            "batch_count": self.batch_count,
            "config": self.config,
        }


def file_sha256(path: Path) -> str:
    """Fingerprint input bytes without loading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(value, allowed: set[str], name: str) -> dict:
    """Reject unrecognized keys rather than silently ignoring settings."""
    if not isinstance(value, dict):
        raise ValueError(f"Error: {name} must be an object.")  # noqa: TRY004 - all invalid input is reported as ValueError (CLI exit code 2, HTTP 422)
    if set(value).difference(allowed):
        raise ValueError(f"Error: {name} contains unsupported fields; see the CLI configuration guide.")
    return dict(value)


def _path(value, base: Path, name: str) -> Path:
    """Resolve a local path relative to the configuration file or caller directory."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Error: {name} must be a non-empty local path.")
    path = Path(value).expanduser()
    return (base / path).resolve()


def extractor_class(source: str = "openeo"):
    """Resolve the selected acquisition adapter only when its contract is needed."""
    if source == "openeo":
        return OpenEOJobManagerZonalStats
    if source == "hub":
        from .hub import HubZonalStats

        return HubZonalStats
    raise ValueError("Error: source must be openeo or hub.")


def partition_extractor(source, parcels, options, index):
    """Build one provider's partition without passing incompatible orchestration options."""
    settings = dict(options, output_dir=Path(options["output_dir"]) / f"partition_{index}")
    identifier = settings.pop("run_identifier", "run")
    if source == "openeo":
        settings["run_identifier"] = f"{identifier}_part{index}"
    return extractor_class(source)(parcels=parcels, **settings)


def _processing_parameters(source: str = "openeo") -> dict:
    """Read the engine's constructor contract without maintaining duplicate defaults."""
    parameters = {}
    for cls in (ParcelStatsBase, extractor_class(source)):
        parameters.update(
            {
                name: param
                for name, param in inspect.signature(cls.__init__).parameters.items()
                if name != "self" and name not in _RESERVED and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            }
        )
    return parameters


def _processing(value, source: str = "openeo") -> dict:
    """Check JSON types before calling the engine's scientific validators."""
    parameters = _processing_parameters(source)
    options = _object(value, set(parameters), "processing")
    for name in ("start_date", "end_date"):
        if name not in options or not isinstance(options[name], str):
            raise ValueError(f"Error: processing.{name} is required as an ISO date string.")
        # Dates in the request are calendar dates, never timestamps or local timezones.
        from datetime import date

        date.fromisoformat(options[name])
    for name, option in options.items():
        default = parameters[name].default
        if isinstance(default, bool) and not isinstance(option, bool):
            raise ValueError(f"Error: {name} must be a boolean.")  # noqa: TRY004 - all invalid input is reported as ValueError (CLI exit code 2, HTTP 422)
        # This optional integer accepts explicit null for automatic sizing even
        # when the configured constructor default is a concrete row count.
        if type(default) is int and name != "temporal_fill_block_rows" and type(option) is not int:
            raise ValueError(f"Error: {name} must be an integer.")
    if source == "openeo":
        _validate_remote_options(options)
    ParcelStatsBase._validate_temporal_fill_block_rows(options.get("temporal_fill_block_rows"))
    # Reject NaN/Infinity and non-JSON objects before persisting any configuration.
    json.dumps(options, allow_nan=False)
    return options


def _validate_remote_options(options):
    """Validate only openEO's job and executor-memory controls."""
    if options.get("openeo_parallel_jobs", 2) not in (1, 2):
        raise ValueError("Error: openeo_parallel_jobs must be 1 or 2 per account.")
    from data_preparation.sources.openeo.job_options import memory_bytes, validate_job_options

    remote_options = validate_job_options(options.get("openeo_job_options"))
    if memory_bytes(options.get("openeo_max_executor_memory", "8G")) < memory_bytes(remote_options["executor-memory"]):
        raise ValueError("Error: openeo_max_executor_memory must be at least the initial executor memory.")


def _read_parcel_file(source: dict, path: Path) -> gpd.GeoDataFrame:
    """Read one supported local vector file and preserve its explicit geometry metadata."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".geoparquet"}:
        if source.get("layer") is not None:
            raise ValueError("Error: layer is only supported for GeoPackage input.")
        parcels = gpd.read_parquet(path)
    elif suffix in {".gpkg", ".geojson"}:
        if suffix == ".geojson" and source.get("layer") is not None:
            raise ValueError("Error: layer is only supported for GeoPackage input.")
        parcels = gpd.read_file(path, **({"layer": source["layer"]} if source.get("layer") else {}))
    else:
        raise ValueError("Error: Input must be GeoParquet, GeoPackage or GeoJSON.")
    if parcels.geometry.name != "geometry":
        if "geometry" in parcels.columns:
            raise ValueError("Error: Rename the ordinary geometry attribute before normalizing the active geometry column.")
        parcels = parcels.rename_geometry("geometry")
    return parcels


def _load_parcels(source: dict, path: Path) -> gpd.GeoDataFrame:
    """Require explicit IDs and retain every polygon, repairing where possible."""
    parcels = _read_parcel_file(source, path)
    field = source.get("parcel_id_field")
    if not isinstance(field, str) or field != field.strip() or field == parcels.geometry.name or field not in parcels:
        raise ValueError("Error: input.parcel_id_field must name an existing non-geometry column.")
    ids = parcels[field]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError("Error: Parcel IDs must be non-null, non-blank and unique after string conversion.")
    planner = ParcelBatchPlanner()
    planner.PARCEL_ID_FIELD = field
    planner.logger = logging.getLogger(__name__)
    prepared = planner._prepare_parcels(parcels)
    if len(prepared) != len(parcels):
        raise ValueError("Error: Input contains empty or non-polygon geometries; repair them before running.")
    return prepared


def _partition(parcels: gpd.GeoDataFrame, field: str | None) -> list[gpd.GeoDataFrame]:
    """Group parcels deterministically without silently dropping missing labels."""
    if field is None:
        return [parcels]
    if not isinstance(field, str) or field not in parcels or field == parcels.geometry.name:
        raise ValueError("Error: partition_field must name an existing non-geometry column.")
    if parcels[field].isna().any():
        raise ValueError("Error: Every parcel must have a partition label.")
    return [part.copy() for _, part in parcels.groupby(field, sort=True, observed=True)]


@contextmanager
def engine_workspace(directory: Path):
    """Release pipeline file handlers, including those left by a failed constructor."""
    try:
        yield
    finally:
        # Pipeline loggers are process-global; close only handlers belonging to this workspace.
        for logger in list(logging.Logger.manager.loggerDict.values()):
            if not isinstance(logger, logging.Logger) or not logger.name.startswith("parcel_stats_pipeline."):
                continue
            owned = any(
                isinstance(h, logging.FileHandler) and Path(h.baseFilename).is_relative_to(directory) for h in logger.handlers
            )
            if owned:
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)


def _request(config: dict, base_dir: str | Path | None) -> tuple[dict, dict, Path, Path]:
    """Resolve the request envelope and enforce input/output path separation."""
    config = _object(config, {"schema_version", "source", "input", "processing", "output_dir", "run_identifier", "scheduling"}, "config")
    provider = config.get("source", "openeo")
    if provider not in ("openeo", "hub"):
        raise ValueError("Error: source must be openeo or hub.")
    config["source"] = provider
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Error: schema_version must be 1.")
    source = _object(config.get("input"), {"path", "layer", "parcel_id_field", "partition_field"}, "input")
    base = Path(base_dir or Path.cwd()).resolve()
    path = _path(source.get("path"), base, "input.path")
    output = _path(config.get("output_dir"), base, "output_dir")
    if not path.is_file():
        raise ValueError("Error: input.path must identify an existing local file.")
    if path.is_relative_to(output):
        raise ValueError("Error: The parcel input must be outside output_dir.")
    if output.exists() and not output.is_dir():
        raise ValueError("Error: output_dir must be a directory.")
    scheduling = config.get("scheduling", "batches")
    if scheduling not in {"batches", "partitions"}:
        raise ValueError("Error: scheduling must be batches or partitions.")
    if provider == "hub":
        scheduling = "batches"
    identifier = config.get("run_identifier", f"run-{uuid4().hex[:12]}")
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 60:
        raise ValueError("Error: run_identifier must contain 1 to 60 characters.")
    config.update(run_identifier=identifier, scheduling=scheduling)
    return config, source, path, output


def _preflight(parts: list[gpd.GeoDataFrame], field: str, identifier: str, options: dict, source="openeo") -> tuple[dict, int]:
    """Use the existing engine to normalize all settings and validate every tile."""
    count = 0
    normalized = {}
    with TemporaryDirectory(prefix="parcel-preflight-") as temporary:
        directory = Path(temporary).resolve()
        with engine_workspace(directory):
            for index, part in enumerate(parts, 1):
                extractor = partition_extractor(
                    source, part, dict(options, output_dir=directory, parcel_id_field=field, run_identifier=identifier), index,
                )
                if source == "hub":
                    count += sum(1 for _ in extractor._hub_iter_batches())
                else:
                    plans = extractor._build_tile_plan(extractor.output_dir)
                    if not extractor.fit_tiles_to_parcels and len(plans) >= 10 and all(len(p["parcels"]) == 1 for p in plans):
                        raise ValueError("Error: Pathological tile plan; select fitted tiles or increase tile size.")
                    count += len(plans)
                normalized = {name: getattr(extractor, name) for name in _processing_parameters(source)}
    normalized["start_date"] = normalized["start_date"].date().isoformat()
    normalized["end_date"] = normalized["end_date"].date().isoformat()
    normalized = json.loads(json.dumps(normalized, allow_nan=False))
    return normalized, count


def prepare_run(config: dict, *, base_dir: str | Path | None = None) -> PreparedRun:
    """Validate input and plan tiles locally; never authenticate or submit remote jobs.

    Paths are relative to ``base_dir`` (the current directory for Python callers).
    Engine validation uses an automatically cleaned temporary directory, leaving
    the requested output directory untouched. The supplied dictionary is not mutated.
    """
    config, source, path, output = _request(config, base_dir)
    options = _processing(config.get("processing"), config["source"])
    fingerprint = file_sha256(path)
    parcels = _load_parcels(source, path)
    parts = _partition(parcels, source.get("partition_field"))
    if "working_epsg" not in options:
        options["working_epsg"] = 2100 if config["source"] == "hub" else estimate_utm_epsg_from_parcels(parcels)
    normalized, count = _preflight(parts, source["parcel_id_field"], config["run_identifier"], options, config["source"])
    if fingerprint != file_sha256(path):
        raise ValueError("Error: The input file changed during validation.")
    resolved = dict(config, input=dict(source, path=str(path)), output_dir=str(output), processing=normalized)
    return PreparedRun(resolved, parcels, parts, fingerprint, count)
