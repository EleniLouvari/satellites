"""Read saved progress without executing pipelines or estimating remaining time."""

import json
import re
from pathlib import Path

from parcel_app.core.storage import contained


def counted(stage, completed, total):
    """Describe a measured work count with an explicit denominator."""
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "percent": round(min(completed, total) * 100 / total) if total else None,
    }


def committed_batches(root, folders, planned, signature=None):
    """Return planned cube names with both committed tables, even after their NetCDFs are removed."""
    pattern = re.compile(rf"(batch_[0-9]+_monthly)_stats_({signature or '[0-9a-f]{16}'})\.parquet")
    completed = set()
    for folder in folders:
        for path in folder.glob("*_stats_*.parquet"):
            match = pattern.fullmatch(path.name)
            if match and match[1] in planned:
                report = path.with_name(f"{path.stem}_report.parquet")
                if contained(root, path).is_file() and contained(root, report).is_file():
                    # Multiple processing signatures must never count the same planned cube twice.
                    completed.add(match[1])
    return completed


def completed_batches(root, job_path, targets):
    """Count planned openEO cubes listed in one ``tile_jobs.parquet``."""
    planned = {Path(value.replace("\\", "/")).stem for value in targets if isinstance(value, str)}
    return len(committed_batches(root, [job_path.parent], planned))


def partition_marker(root, index):
    """Return one partition's completion marker, or an empty record before it exists."""
    path = root / f"partition_{index}" / "satellite_partition_completion.json"
    if not path.is_file():
        return {}
    return json.loads(contained(root, path).read_text(encoding="utf-8"))


def completed_partition_count(root, total):
    """Count committed partition markers without importing acquisition libraries."""
    return sum(partition_marker(root, index).get("status") == "complete" for index in range(1, total + 1))


def hub_batch_counts(root, partitions):
    """Count HUB batches planned in each partition manifest and those with committed statistics.

    The HUB batch grid is planned when the catalog is resolved, so the preflight
    ``batch_count`` does not describe it. Partitions whose manifest is not yet
    written contribute nothing to either count.
    """
    import pyarrow.parquet as pq

    done, total = 0, 0
    for index in range(1, partitions + 1):
        folder = root / f"partition_{index}"
        manifest = folder / "hub_item_manifest.parquet"
        if not manifest.is_file():
            continue
        numbers = pq.read_table(contained(root, manifest), columns=["batch_number"]).column("batch_number").to_pylist()
        planned = {f"batch_{int(number):05d}_monthly" for number in numbers}
        total += len(planned)
        marker = partition_marker(root, index)
        if marker.get("status") == "complete":
            done += len(planned)
            continue
        signature = marker.get("signature")
        signature = signature if isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{16}", signature) else None
        cube_dirs = [path for path in (folder / "monthly_cubes").glob("*") if path.is_dir()]
        done += len(committed_batches(root, cube_dirs, planned, signature))
    return done, total


def remote_counts(root):
    """Read only openEO remote-job and local-checkpoint count columns."""
    import pyarrow.parquet as pq

    remote_done, local_done = 0, 0
    for path in root.rglob("tile_jobs.parquet"):
        path = contained(root, path)
        columns = pq.read_schema(path).names
        table = pq.read_table(path, columns=[name for name in ("status", "target_path") if name in columns])
        remote_done += sum(value == "finished" for value in table.column("status").to_pylist())
        if "target_path" in columns:
            local_done += completed_batches(root, path, table.column("target_path").to_pylist())
    return remote_done, local_done


def inactive_progress(run, scan):
    """Describe lifecycle states that need no checkpoint scan."""
    status = run.get("status")
    if status == "complete":
        return {"stage": "Results ready", "percent": 100}
    if status == "deleting":
        manifest = run.get("deletion_manifest") or {}
        if manifest.get("remote_complete"):
            return {"stage": "Removing local files and history", "percent": None}
        return counted("Remote cleanup", manifest.get("remote_deleted", 0), len(manifest.get("jobs", [])))
    if status not in {"running", "failed", "interrupted"}:
        return {"stage": "Waiting for processing", "percent": None}
    if not scan and status != "running":
        return {"stage": "Stopped; open for saved progress", "percent": None}
    return None


def active_progress(summary, counts, hub):
    """Choose a measured provider-specific stage from committed checkpoint counts.

    For HUB, ``summary["batch_count"]`` must be the batch total read from the
    partition manifests rather than the preflight estimate.
    """
    remote_done, local_done, partitions_done = counts
    batches, total = summary.get("batch_count", 0), summary.get("partition_count", 0)
    partitions = counted("Partition assembly", partitions_done, total)
    if hub:
        local = counted("HUB batches", local_done, batches)
        if not batches:
            local = {"stage": "Resolving HUB catalog", "percent": None}
        current = local if not batches or local_done < batches else partitions
        details = {"local": local, "partitions": partitions}
    else:
        remote = counted("Remote jobs", remote_done, batches)
        local = counted("Local batches", local_done, batches)
        current = remote if batches and remote_done < batches else local
        if batches and local_done >= batches:
            current = partitions
        details = {"remote": remote, "local": local, "partitions": partitions}
    if total and partitions_done >= total:
        current = {"stage": "Verifying and publishing results", "percent": None}
    return {**current, **details}


def run_progress(run, *, scan=True):
    """Read saved progress without running acquisition or inventing job percentages."""
    inactive = inactive_progress(run, scan)
    if inactive is not None:
        return inactive
    summary = run.get("summary") or {}
    root = Path(run["output_dir"])
    hub = run.get("source", "openeo") == "hub"
    try:
        if hub:
            remote_done, (local_done, batch_total) = 0, hub_batch_counts(root, summary.get("partition_count", 0))
            summary = {**summary, "batch_count": batch_total}
        else:
            remote_done, local_done = remote_counts(root)
        partitions_done = completed_partition_count(root, summary.get("partition_count", 0))
    except (OSError, ValueError, KeyError):
        return {"stage": "Updating saved progress", "percent": None}
    return active_progress(summary, (remote_done, local_done, partitions_done), hub)
