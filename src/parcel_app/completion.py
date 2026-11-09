"""Read saved progress without executing pipelines or estimating remaining time."""

import json
import re
from pathlib import Path

from .artifacts import contained


def counted(stage, completed, total):
    """Describe a measured work count with an explicit denominator."""
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "percent": round(min(completed, total) * 100 / total) if total else None,
    }


def completed_batches(root, job_path, targets):
    """Count planned cubes with both committed tables, even after their NetCDFs are removed."""
    planned = {Path(value.replace("\\", "/")).stem for value in targets if isinstance(value, str)}
    completed = set()
    for path in job_path.parent.glob("*_stats_*.parquet"):
        match = re.fullmatch(r"(batch_[0-9]+_monthly)_stats_[0-9a-f]{16}\.parquet", path.name)
        if match and match[1] in planned:
            report = path.with_name(f"{path.stem}_report.parquet")
            if contained(root, path).is_file() and contained(root, report).is_file():
                # Multiple processing signatures must never count the same planned cube twice.
                completed.add(match[1])
    return len(completed)


def run_progress(run):
    """Separate remote jobs, committed local batches, partition assembly and publication."""
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
    summary = run.get("summary") or {}
    root = Path(run["output_dir"])
    remote_done, local_done, partitions_done = 0, 0, 0
    batch_count, partition_count = summary.get("batch_count", 0), summary.get("partition_count", 0)
    readable = True
    try:
        import pyarrow.parquet as pq

        for path in root.rglob("tile_jobs.parquet"):
            path = contained(root, path)
            columns = pq.read_schema(path).names
            table = pq.read_table(path, columns=[name for name in ("status", "target_path") if name in columns])
            remote_done += sum(value == "finished" for value in table.column("status").to_pylist())
            if "target_path" in columns:
                local_done += completed_batches(root, path, table.column("target_path").to_pylist())
        for index in range(1, partition_count + 1):
            path = root / f"partition_{index}" / "satellite_partition_completion.json"
            if path.is_file() and json.loads(contained(root, path).read_text(encoding="utf-8")).get("status") == "complete":
                partitions_done += 1
    except (OSError, ValueError, KeyError):
        # A checkpoint may be being replaced. Do not turn a transient read failure into fake progress.
        readable = False
    remote = counted("Remote jobs", remote_done, batch_count)
    local = counted("Local batches", local_done, batch_count)
    partitions = counted("Partition assembly", partitions_done, partition_count)
    if not readable:
        return {"stage": "Updating saved progress", "percent": None}
    current = remote if batch_count and remote_done < batch_count else local
    if batch_count and local_done >= batch_count:
        current = partitions
    if partition_count and partitions_done >= partition_count:
        current = {"stage": "Verifying and publishing results", "percent": None}
    return {**current, "remote": remote, "local": local, "partitions": partitions}
