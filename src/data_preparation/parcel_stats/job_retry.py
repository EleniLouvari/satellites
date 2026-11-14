"""Durable per-tile retry decisions, independent of scientific cache identity."""

import json

from data_preparation.sources.openeo.job_options import failure_kind, next_executor_memory, validate_job_options


class TileJobRetryMixin:
    """Extend bounded retries with explicit remote memory failure recovery."""

    openeo_job_options = None
    openeo_auto_memory_retry = True
    openeo_max_executor_memory = "8G"
    retry_exhausted_memory = False

    def _job_options_for_row(self, row):
        """Prefer the durable options of this batch over the run's initial options."""
        saved = row.get("job_options")
        return validate_job_options(json.loads(saved) if isinstance(saved, str) and saved else self.openeo_job_options)

    def _ensure_job_options(self, job_db):
        """Initialize legacy metadata once without changing IDs, states or retry counts."""
        jobs = job_db.df.copy()
        defaults = {"job_options": json.dumps(validate_job_options(self.openeo_job_options)),
                    "job_options_history": "{}", "failure_kind": ""}
        changed = False
        for column, default in defaults.items():
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
