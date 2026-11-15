"""Durable per-tile retry decisions, independent of scientific cache identity."""

import json

import pandas as pd

from data_preparation.sources.openeo.job_options import (
    executor_memory_for_area,
    failure_kind,
    next_executor_memory,
    validate_job_options,
)


class TileJobRetryMixin:
    """Extend bounded retries with explicit remote memory failure recovery."""

    openeo_job_options = None
    openeo_auto_memory_retry = True
    openeo_max_executor_memory = "8G"
    openeo_memory_reference_area_km2 = None
    retry_exhausted_memory = False

    def _initial_job_options(self, row):
        """Scale the configured heap with the tile area; other options stay as configured."""
        options = validate_job_options(self.openeo_job_options)
        bounds = [row.get(name) for name in ("west", "south", "east", "north")]
        if self.openeo_memory_reference_area_km2 is None or any(value is None or pd.isna(value) for value in bounds):
            return options
        west, south, east, north = (float(value) for value in bounds)
        # Tile extents are stored in the projected working CRS, so the units are metres.
        area_km2 = (east - west) * (north - south) / 1e6
        memory = executor_memory_for_area(
            options["executor-memory"], area_km2, self.openeo_memory_reference_area_km2, self.openeo_max_executor_memory,
        )
        return {**options, "executor-memory": memory}

    def _job_options_for_row(self, row):
        """Prefer the durable options of this batch over the run's initial options."""
        saved = row.get("job_options")
        return validate_job_options(json.loads(saved)) if isinstance(saved, str) and saved else self._initial_job_options(row)

    def _ensure_job_options(self, job_db):
        """Initialize legacy metadata once without changing IDs, states or retry counts."""
        jobs = job_db.df.copy()
        changed = False
        if "job_options" not in jobs:
            jobs["job_options"] = None
        missing = jobs["job_options"].isna()
        for index in jobs.index[missing]:
            options = self._initial_job_options(jobs.loc[index])
            jobs.loc[index, "job_options"] = json.dumps(options, sort_keys=True)
            if options["executor-memory"] != validate_job_options(self.openeo_job_options)["executor-memory"]:
                self.source_logger.info(
                    "Batch %s uses executor-memory=%s for its tile area (reference %s km2, cap %s).",
                    int(jobs.loc[index, "batch_number"]), options["executor-memory"],
                    self.openeo_memory_reference_area_km2, self.openeo_max_executor_memory,
                )
        changed = bool(missing.any())
        for column, default in {"job_options_history": "{}", "failure_kind": ""}.items():
            if column not in jobs:
                jobs[column] = default
                changed = True
            elif jobs[column].isna().any():
                jobs[column] = jobs[column].fillna(default)
                changed = True
        if changed:
            job_db.persist(jobs)

    def _retry_options(self, row, kind):
        """Return replacement options or None when retrying cannot improve the failure."""
        options = self._job_options_for_row(row)
        within_budget = int(row["retry_count"]) < self.max_job_retries
        if kind == "invalid_request":
            return None
        if kind == "jvm_oom" and self.openeo_auto_memory_retry:
            memory = next_executor_memory(options, self.openeo_max_executor_memory)
            if memory is not None and (within_budget or self.retry_exhausted_memory):
                return {**options, "executor-memory": memory}
            return None
        return options if within_budget else None

    def _prepare_failed_retry(self, row):
        """Record evidence and prepare a single attempt without resetting past history."""
        failed_id = self._clean_job_id(row.get("id"))
        restart = row["status"] in {"start_failed", "queued_for_start_failed"} and bool(failed_id)
        kind = "start_failure" if restart else failure_kind(row)
        row["failure_kind"] = kind
        options = self._retry_options(row, kind)
        if options is None:
            self.source_logger.error(
                "Batch %s will not retry: %s; retry count %s, executor memory %s, cap %s. "
                "For JVM OOM at the cap, use smaller spatial batches in a new run.",
                int(row["batch_number"]), kind, row["retry_count"],
                self._job_options_for_row(row)["executor-memory"], self.openeo_max_executor_memory,
            )
            return row, False
        history = [item for item in str(row.get("failed_job_ids") or "").split(";") if item]
        if not restart and failed_id:
            if failed_id not in history:
                history.append(failed_id)
            previous = json.loads(row["job_options_history"])
            previous[failed_id] = self._job_options_for_row(row)
            row["job_options_history"] = json.dumps(previous, sort_keys=True)
        row["failed_job_ids"] = ";".join(history)
        row["job_options"] = json.dumps(options, sort_keys=True)
        row["retry_count"] = int(row["retry_count"]) + 1
        row["retry_mode"] = "restart" if restart else "replace"
        if not restart:
            # A failed create request must not later restart the old job with
            # its old resources. Its identity remains in failed_job_ids/history.
            row["id"] = ""
        row["status"] = "not_started"
        self.source_logger.warning(
            "Retrying batch %s for openEO job %s (retry %s; mode=%s; failure=%s; executor-memory=%s).",
            int(row["batch_number"]), failed_id or "<unknown>", row["retry_count"], row["retry_mode"], kind,
            options["executor-memory"],
        )
        return row, True
