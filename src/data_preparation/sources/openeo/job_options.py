"""Validate CDSE resource options and classify saved remote failure evidence."""

import copy
import json
import re
from pathlib import Path

MEMORY_KEYS = {"executor-memory", "executor-memoryOverhead", "python-memory", "driver-memory", "driver-memoryOverhead"}
INTEGER_KEYS = {"executor-cores", "task-cpus", "max-executors", "driver-cores"}
OTHER_KEYS = {"udf-dependency-archives", "logging-threshold"}
INVALID_CODES = {"ProcessGraphInvalid", "ProcessGraphMissing", "ProcessParameterInvalid", "ProcessParameterMissing",
                 "ProcessUnsupported", "CollectionNotFound", "BandNotAvailable"}


def memory_bytes(value):
    """Parse positive integer Spark memory quantities with K/M/G/T units."""
    match = re.fullmatch(r"([1-9][0-9]*)([kmgt])", value, re.IGNORECASE) if isinstance(value, str) else None
    if match is None:
        raise ValueError("Error: Memory must be a positive quantity such as 2048m or 4G.")
    return int(match[1]) * 1024 ** ("kmgt".index(match[2].lower()) + 1)


def validate_job_options(value):
    """Copy a supported, JSON-safe resource configuration with an explicit heap baseline."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - MEMORY_KEYS - INTEGER_KEYS - OTHER_KEYS:
        raise ValueError("Error: openeo_job_options contains unsupported options or is not an object.")
    options = {"executor-memory": "2G", **copy.deepcopy(value)}
    for name, item in options.items():
        if name in MEMORY_KEYS and not (name == "python-memory" and item is None):
            memory_bytes(item)
        elif name in INTEGER_KEYS and (type(item) is not int or item <= 0):
            raise ValueError(f"Error: {name} must be a positive integer.")
    _validate_auxiliary_options(options)
    json.dumps(options, allow_nan=False)
    return options


def _validate_auxiliary_options(options):
    """Validate archive lists and logging levels separately from resource quantities."""
    archives = options.get("udf-dependency-archives", [])
    if not isinstance(archives, list) or any(not isinstance(item, str) or not item.strip() for item in archives):
        raise ValueError("Error: udf-dependency-archives must be a list of nonempty URLs.")
    if options.get("logging-threshold", "info") not in {"debug", "info", "warning", "error"}:
        raise ValueError("Error: logging-threshold must be debug, info, warning or error.")


def next_executor_memory(options, maximum):
    """Double only the JVM heap, stopping at the configured resource ceiling."""
    current, limit = memory_bytes(options["executor-memory"]), memory_bytes(maximum)
    if current >= limit:
        return None
    amount = min(current * 2, limit)
    return f"{amount // 1024**3}G" if amount % 1024**3 == 0 else f"{amount // 1024}K"


def failure_kind(row):
    """Classify only the exact attempt's saved logs; shuffle errors alone prove no OOM."""
    job_id, target = row.get("id"), row.get("target_path")
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", job_id) or not isinstance(target, str):
        return "unknown"
    path = Path(target).parent / "job_manager" / f"job_{job_id}" / f"job_{job_id}_errors.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    entries = data.get("logs", []) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return "unknown"
    entries = [item for item in entries if isinstance(item, dict)]
    messages = "\n".join(str(item.get("message", "")) for item in entries)
    if any(str(item.get("code")) in INVALID_CODES for item in entries) or any(
        re.search(rf"\b{code}\s*:", messages) for code in INVALID_CODES
    ):
        return "invalid_request"
    if re.search(r"JVM OOM|exit code 52\b|OutOfMemoryError|Java heap space|GC overhead limit exceeded", messages, re.IGNORECASE):
        return "jvm_oom"
    return "unknown"
