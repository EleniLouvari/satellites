"""Read Planet delivery-bucket files without running HUB ingestion or registration.

The credential names, monthly order layout and manifest grouping follow
esa_govermental_hub/pipeline/download/download_planet.py, called by the running
upload_batch_process_planet notebook. This adapter only lists and downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
import pandas as pd
import rasterio

MONTH_NAMES = (
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
)


def create_planet_delivery_client(env_file=None):
    """Use PLANET_* settings from this process or a read-only .env file.

    Only the four required keys are read; the process environment and the other
    repository's files are never modified. File values must be literal dotenv
    assignments (quoted values and comments are supported).
    """
    names = ("PLANET_AWS_ACCESS_KEY_ID", "PLANET_AWS_SECRET_ACCESS_KEY", "PLANET_S3_REGION", "PLANET_S3_URL")
    settings = {}
    if env_file is not None:
        for line in Path(env_file).read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.removeprefix("export ").partition("=")
            if separator and name.strip() in names:
                settings[name.strip()] = " ".join(shlex.split(value, comments=True))
    settings.update({name: os.environ[name] for name in names if os.getenv(name)})
    missing = [name for name in names[:3] if not settings.get(name)]
    if missing:
        raise ValueError(f"Error: Missing Planet delivery configuration: {', '.join(missing)}")
    return boto3.client(
        "s3",
        aws_access_key_id=settings[names[0]],
        aws_secret_access_key=settings[names[1]],
        region_name=settings[names[2]],
        endpoint_url=settings.get(names[3]) or None,
        config=Config(connect_timeout=30, read_timeout=180, retries={"mode": "standard", "max_attempts": 5}),
    )


def _read_s3_json(client, bucket, key):
    response = client.get_object(Bucket=bucket, Key=key)
    with response["Body"] as body:
        return json.load(body)


def _requested_pairs(quad_ids, start_date, end_date):
    quads = sorted(set(str(quad) for quad in quad_ids))
    if not quads or any(re.fullmatch(r"\d+-\d+", quad) is None for quad in quads):
        raise ValueError("Error: quad_ids must contain Planet grid IDs such as 1154-1279")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Error: Dates must be valid and ordered")
    return quads, pd.period_range(start, end, freq="M")


def discover_planet_delivery_tiles(client, bucket, quad_ids, start_date, end_date, *, require_complete=True):
    """Resolve complete order manifests and return selected file records before transferring imagery.

    Each quad/month must have one analysis TIFF and its metadata. Duplicate
    deliveries are rejected explicitly rather than silently choosing an order when
    ``require_complete`` is True. Auxiliary quality/provenance files listed for
    the same quad are retained.
    """
    quads, months = _requested_pairs(quad_ids, start_date, end_date)
    selected = []
    missing_pairs = []
    for month in months:
        prefix = f"PL_BSM/PL_BSM_{MONTH_NAMES[month.month - 1]}_{month.year}/"
        objects = {}
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            objects.update({obj["Key"]: obj for obj in page.get("Contents", [])})
        # Inspect only order manifests that contain one of the requested quads.
        roots = sorted(
            {"/".join(key.split("/")[:3]) + "/" for key in objects if PurePosixPath(key).name.split("_", 1)[0] in quads}
        )
        records = []
        for root in roots:
            manifest_key = root + "manifest.json"
            if manifest_key not in objects:
                continue  # A manifest is the delivery-completion marker.
            manifest = _read_s3_json(client, bucket, manifest_key)
            for entry in manifest.get("files", []):
                relative = entry.get("path", "")
                filename = PurePosixPath(relative).name
                quad = str(entry.get("annotations", {}).get("planet/quad_id") or filename.split("_", 1)[0])
                if quad not in quads:
                    continue
                key = root + relative
                if key not in objects or objects[key]["Size"] != entry.get("size"):
                    raise ValueError(f"Error: Missing or incomplete delivered file: {key}")
                if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts or "\\" in relative:
                    raise ValueError("Error: Unsafe path in Planet delivery manifest")
                records.append(
                    {
                        "period_start": str(month.start_time.date()),
                        "quad_id": quad,
                        "key": key,
                        "filename": filename,
                        "size": objects[key]["Size"],
                        "etag": objects[key]["ETag"],
                        "sha256": entry.get("digests", {}).get("sha256", ""),
                        "order_id": root.rstrip("/").split("/")[-1],
                        "manifest_key": manifest_key,
                    }
                )
        for quad in quads:
            group = [row for row in records if row["quad_id"] == quad]
            analyses = [row for row in group if row["filename"] in {f"{quad}_quad_bandmath.tif", f"{quad}_quad.tif"}]
            metadata = [row for row in group if row["filename"] == f"{quad}_metadata.json"]
            if len(analyses) != 1 or len(metadata) != 1:
                message = f"Expected one analysis TIFF and metadata for {month}/{quad}; found {len(analyses)}/{len(metadata)}"
                if require_complete:
                    raise ValueError(message)
                missing_pairs.append({"period": str(month), "quad_id": quad, "reason": message})
                continue
            selected.extend(group)
        print(f"Resolved {month}: {len(quads)} quads, {len(records)} files", flush=True)
    if not selected:
        raise ValueError("Error: No completed Planet deliveries found for the requested quad/month range")
    if missing_pairs and not require_complete:
        print(
            f"Continuing with partial delivery coverage: missing {len(missing_pairs)} quad/month pairs", flush=True
        )
    return pd.DataFrame(selected).sort_values(["period_start", "quad_id", "filename"]).reset_index(drop=True)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_planet_delivery_tiles(client, bucket, inventory, output_dir, *, include_auxiliary=True):
    """Download/resume selected delivery files with size and SHA256 verification.

    Files are published by atomic rename after verification. Existing completed
    files are reused; changed or partial files are downloaded again. Transfers
    are sequential with at most two multipart workers to limit shared bandwidth.
    """
    if inventory.empty:
        raise ValueError("Error: A nonempty delivery inventory is required")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(root / "planet_delivery_inventory.csv", index=False)
    transfer = TransferConfig(max_concurrency=2, multipart_threshold=64 * 1024**2, multipart_chunksize=16 * 1024**2)
    for number, row in enumerate(inventory.itertuples(), 1):
        analysis = row.filename in {f"{row.quad_id}_quad_bandmath.tif", f"{row.quad_id}_quad.tif"}
        if not include_auxiliary and not analysis and not row.filename.endswith("_metadata.json"):
            continue
        target = (root / row.period_start / row.quad_id / row.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Error: Delivery path escapes download directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        marker = target.with_suffix(target.suffix + ".complete.json")
        expected = {"bucket": bucket, "key": row.key, "size": row.size, "etag": row.etag, "sha256": row.sha256}
        try:
            saved = json.loads(marker.read_text())
        except (FileNotFoundError, ValueError):
            saved = {}
        if target.is_file() and target.stat().st_size == row.size:
            stat = target.stat()
            if saved == {**expected, "mtime_ns": stat.st_mtime_ns}:
                print(f"Reuse {number}/{len(inventory)}: {row.period_start}/{row.filename}", flush=True)
                continue
            # Adopt an already-downloaded file only after checking its manifest digest.
            if row.sha256 and _sha256(target) == row.sha256:
                marker.write_text(json.dumps({**expected, "mtime_ns": stat.st_mtime_ns}))
                continue
        partial = target.with_suffix(target.suffix + ".partial")
        print(f"Download {number}/{len(inventory)}: {row.period_start}/{row.filename} ({row.size / 1e6:.1f} MB)", flush=True)
        client.download_file(bucket, row.key, str(partial), Config=transfer)
        if partial.stat().st_size != row.size or (row.sha256 and _sha256(partial) != row.sha256):
            raise ValueError(f"Error: Downloaded content does not match the delivery manifest: {row.filename}")
        partial.replace(target)
        marker.write_text(json.dumps({**expected, "mtime_ns": target.stat().st_mtime_ns}))
    # A retained directory may also contain an unfinished, unrelated date range.
    # Only the requested tile/month set belongs to this completed download.
    manifest = build_local_planet_manifest(
        root, quad_ids=inventory.quad_id.unique(), start_date=inventory.period_start.min(), end_date=inventory.period_start.max()
    )
    pairs = inventory[["period_start", "quad_id"]].drop_duplicates()
    manifest = manifest.merge(pairs, on=["period_start", "quad_id"], validate="one_to_one")
    if len(manifest) != len(pairs):
        raise ValueError("Error: Downloaded manifest does not contain every requested tile/month")
    manifest.to_csv(root / "planet_local_manifest.csv", index=False)
    return manifest


def build_local_planet_manifest(source, *, quad_ids=None, start_date=None, end_date=None):
    """Load a manifest CSV or discover raw Planet TIFFs from neighboring metadata.

    Discovery uses mosaic acquisition dates and quad IDs, not catalog names or
    directory names. Delivery last_acquired is an exclusive end timestamp.
    Raster scaling is deliberately left to PlanetBasemapZonalStats.
    """
    source = Path(source).resolve()
    labels = None
    if start_date is not None or end_date is not None:
        if start_date is None or end_date is None:
            raise ValueError("Error: Supply both start_date and end_date")
        labels = set(pd.period_range(start_date, end_date, freq="M").start_time.strftime("%Y-%m-%d"))
    requested_quads = None if quad_ids is None else {str(quad) for quad in quad_ids}
    if source.is_file():
        manifest = pd.read_csv(source, dtype={"quad_id": str})
        required = {"period_start", "quad_id", "path"}
        if not required.issubset(manifest):
            raise ValueError(f"Error: Local manifest requires {sorted(required)}")
        manifest["path"] = manifest.path.map(lambda p: str((source.parent / p).resolve()))
    else:
        records = []
        for metadata_path in sorted(source.rglob("*_metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "quad" not in metadata or "mosaic" not in metadata:
                continue
            quad = str(metadata["quad"]["id"])
            start = pd.Timestamp(metadata["mosaic"]["first_acquired"])
            # Ignore unrelated or still-downloading months before inspecting TIFFs.
            if (requested_quads is not None and quad not in requested_quads) or (
                labels is not None and str(start.date()) not in labels
            ):
                continue
            end = pd.Timestamp(metadata["mosaic"]["last_acquired"])
            if start.day != 1 or start.normalize() != start or end != start + pd.DateOffset(months=1):
                raise ValueError(f"Error: Expected one complete monthly mosaic: {metadata_path}")
            candidates = [metadata_path.parent / f"{quad}_{suffix}.tif" for suffix in ("quad_bandmath", "quad")]
            candidates = [path for path in candidates if path.is_file()]
            if len(candidates) != 1:
                raise ValueError(f"Error: Expected one local analysis TIFF beside {metadata_path}; found {len(candidates)}")
            with rasterio.open(candidates[0]) as raster:
                if raster.crs is None or raster.count < 8:
                    raise ValueError(f"Error: Expected a georeferenced eight-band Planet raster: {candidates[0]}")
            records.append(
                {
                    "period_start": str(start.date()),
                    "quad_id": quad,
                    "path": str(candidates[0]),
                    "metadata_path": str(metadata_path),
                    "mosaic_name": metadata["mosaic"].get("name", ""),
                }
            )
        manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError(f"Error: No local Planet tiles found: {source}")
    if requested_quads is not None:
        manifest = manifest[manifest.quad_id.isin(requested_quads)]
    if labels is not None:
        manifest = manifest[manifest.period_start.isin(labels)]
    if manifest.empty or manifest.duplicated(["period_start", "quad_id"]).any():
        raise ValueError("Error: Local manifest is empty or contains duplicate quad/month entries")
    return manifest.sort_values(["period_start", "quad_id"]).reset_index(drop=True)
