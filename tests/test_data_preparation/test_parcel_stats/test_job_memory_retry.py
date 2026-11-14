"""Remote memory recovery is evidence-based, bounded and durable per batch."""

import json
import logging

import pandas as pd
import pytest

from data_preparation.parcel_stats.job_manager import OpenEOJobManagerZonalStats
from tests.test_data_preparation.test_parcel_stats.test_zonal_stats_job_manager import _MemoryJobDatabase


def planner():
    """Build only the remote retry component without raster or network work."""
    value = object.__new__(OpenEOJobManagerZonalStats)
    value.max_job_retries = 3
    value.source_logger = logging.getLogger("memory-retry-test")
    value.openeo_job_options = {"executor-memory": "2G", "max-executors": 20}
    value.openeo_auto_memory_retry = True
    value.openeo_max_executor_memory = "8G"
    value.retry_exhausted_memory = False
    return value


def failed_batch(tmp_path, message="Lost executor: exit code 52(JVM OOM).", retries=0):
    """Write the same saved log layout as MultiBackendJobManager."""
    folder = tmp_path / "job_manager" / "job_failed-1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job_failed-1_errors.json").write_text(json.dumps([{"level": "error", "message": message}]))
    return _MemoryJobDatabase(pd.DataFrame([{
        "batch_number": 2, "id": "failed-1", "status": "error", "retry_count": retries,
        "target_path": str(tmp_path / "batch_2.nc"), "openeo_user": "original-owner",
    }]))


def test_oom_increases_only_failed_batch_options_and_preserves_history(tmp_path):
    """A JVM memory failure changes heap only and records its previous settings."""
    engine = planner()
    database = failed_batch(tmp_path)
    assert engine._reset_retryable_failed_jobs(database) == 1
    row = database.df.iloc[0]
    assert json.loads(row.job_options) == {"executor-memory": "4G", "max-executors": 20}
    assert json.loads(row.job_options_history)["failed-1"]["executor-memory"] == "2G"
    assert row.failure_kind == "jvm_oom"
    assert row.openeo_user == "original-owner" and row.retry_count == 1
    assert engine.openeo_job_options["executor-memory"] == "2G"
    assert engine._reset_retryable_failed_jobs(database) == 0  # prepared retry survives restart


def test_exhausted_memory_recovery_requires_opt_in_and_stops_at_cap(tmp_path):
    """Old exhausted jobs may advance memory, but never reset their retry count."""
    engine = planner()
    database = failed_batch(tmp_path, retries=3)
    assert engine._reset_retryable_failed_jobs(database) == 0
    engine.retry_exhausted_memory = True
    for expected in ("4G", "8G"):
        assert engine._reset_retryable_failed_jobs(database) == 1
        assert json.loads(database.df.iloc[0].job_options)["executor-memory"] == expected
        database.df.loc[0, "status"] = "error"
        database.df.loc[0, "id"] = "failed-1"  # simulate the next submitted job's saved failure
    assert engine._reset_retryable_failed_jobs(database) == 0
    assert database.df.iloc[0].retry_count == 5
    assert database.df.iloc[0].status == "error"


@pytest.mark.parametrize("message,expected", [
    ("Missing an output location for shuffle 4 partition 82", 1),
    ("Service temporarily unavailable", 1),
    ("ProcessGraphInvalid: missing node", 0),
    ("ProcessParameterInvalid: invalid bands", 0),
])
def test_nonmemory_failures_do_not_increase_memory(tmp_path, message, expected):
    """Shuffle errors alone are not proof of OOM; invalid inputs are terminal."""
    engine = planner()
    database = failed_batch(tmp_path, message)
    assert engine._reset_retryable_failed_jobs(database) == expected
    assert json.loads(database.df.iloc[0].job_options)["executor-memory"] == "2G"


def test_corrupt_logs_keep_retry_budget_and_options(tmp_path):
    """Missing diagnostics cannot authorize extra retries or resource escalation."""
    engine = planner()
    engine.retry_exhausted_memory = True
    database = failed_batch(tmp_path, retries=3)
    next(tmp_path.rglob("*errors.json")).write_text("{")
    assert engine._reset_retryable_failed_jobs(database) == 0


@pytest.mark.parametrize("options", [[], {"password": "secret"}, {"executor-memory": "0G"},
                                    {"executor-cores": True}, {"max-executors": 0}, {"python-memory": -1}])
def test_invalid_remote_options_rejected(options):
    """Reject malformed or unsupported job settings before any submission."""
    from data_preparation.sources.openeo.job_options import validate_job_options

    with pytest.raises((ValueError, TypeError)):
        validate_job_options(options)


def test_supported_options_and_memory_units():
    """Accept the documented backend fields while copying mutable inputs."""
    from data_preparation.sources.openeo.job_options import validate_job_options, next_executor_memory

    original = {"executor-memory": "2048m", "executor-memoryOverhead": "1800m", "python-memory": None,
                "executor-cores": 1, "task-cpus": 1, "max-executors": 20, "driver-memory": "2G",
                "driver-memoryOverhead": "1G", "driver-cores": 1, "udf-dependency-archives": [],
                "logging-threshold": "info"}
    assert validate_job_options(original) == original
    assert next_executor_memory(original, "8G") == "4G"
    assert next_executor_memory({"executor-memory": "6G"}, "8G") == "8G"
    assert next_executor_memory({"executor-memory": "8G"}, "8G") is None


def test_replacement_submission_uses_persisted_batch_options(tmp_path):
    """The next remote job actually receives the options selected by retry policy."""
    from types import SimpleNamespace

    engine = planner()
    database = failed_batch(tmp_path)
    engine._reset_retryable_failed_jobs(database)
    calls = []
    engine._build_tile_cube = lambda *args: SimpleNamespace(create_job=lambda **kw: calls.append(kw))
    engine.run_identifier = "test-run"
    engine._build_or_reuse_tile_job(database.df.iloc[0], None, {"spatial_extent": {}, "tile_id": "tile", "batch_number": 2})
    assert calls[0]["job_options"]["executor-memory"] == "4G"


def test_real_parquet_and_shared_batch_view_preserve_options(tmp_path):
    """A fresh process sees selected memory, while sibling rows retain their own options."""
    from threading import RLock
    from openeo.extra.job_management import create_job_db, get_job_db
    from data_preparation.parcel_stats.batch_scheduler import _BatchJobDatabase

    engine = planner()
    rows = failed_batch(tmp_path).df
    sibling = rows.iloc[0].copy()
    sibling["batch_number"], sibling["id"], sibling["status"] = 3, "successful", "finished"
    rows = pd.concat([rows, sibling.to_frame().T], ignore_index=True)
    database = create_job_db(tmp_path / "tile_jobs.parquet", df=rows)
    selected = _BatchJobDatabase(database, 0, RLock())
    assert engine._reset_retryable_failed_jobs(selected) == 1
    reopened = get_job_db(tmp_path / "tile_jobs.parquet").df
    assert json.loads(reopened.iloc[0].job_options)["executor-memory"] == "4G"
    assert reopened.iloc[1].id == "successful" and reopened.iloc[1].status == "finished"


def test_web_remote_options_validation_and_metadata():
    """New run settings expose a JSON object and exclude the runtime-only recovery flag."""
    from parcel_app.config import check_processing, configuration

    fields = {f["name"]: f for f in configuration(user={})["fields"]}
    assert fields["openeo_job_options"]["type"] == "object"
    assert "retry_exhausted_memory" not in fields
    base = {"start_date": "2024-01-01", "end_date": "2024-01-31"}
    with pytest.raises(ValueError):
        check_processing({**base, "openeo_job_options": {"executor-memory": "bad"}})
    with pytest.raises(ValueError):
        check_processing({**base, "openeo_job_options": {"executor-memory": "16G"}, "openeo_max_executor_memory": "8G"})


def test_resume_recovery_preserves_saved_configuration(tmp_path, monkeypatch):
    """Recovery is runtime-only and still passes the original input/configuration checks."""
    from types import SimpleNamespace
    from data_preparation.parcel_stats import runner

    config = {"output_dir": str(tmp_path)}
    runner._write_json(tmp_path / "config.json", config)
    runner._write_json(tmp_path / "run.json", {"accounts": ["owner"], "app_user_id": 4, "input_sha256": "input",
                                              "config_sha256": runner.file_sha256(tmp_path / "config.json")})
    before = (tmp_path / "config.json").read_bytes()
    monkeypatch.setattr(runner, "prepare_run", lambda config: SimpleNamespace(input_sha256="input"))
    calls = []
    monkeypatch.setattr(runner, "_execute", lambda *args, **kw: calls.append(kw))
    runner.resume_parcel_stats(tmp_path, accounts=[("owner", "password")], app_user_id=4, retry_exhausted_memory=True)
    assert calls == [{"retry_exhausted_memory": True}]
    assert (tmp_path / "config.json").read_bytes() == before
    with pytest.raises(ValueError, match="own"):
        runner.resume_parcel_stats(tmp_path, accounts=[("owner", "password")], app_user_id=5, retry_exhausted_memory=True)


def test_resume_cli_recovery_flag():
    """CLI users can explicitly authorize bounded exhausted-memory recovery."""
    from data_preparation.cli import _parser

    args = _parser().parse_args(["parcel-stats", "resume", "--run-dir", "example", "--app-user-id", "4",
                                 "--retry-exhausted-memory"])
    assert args.retry_exhausted_memory is True


def test_replacement_preparation_clears_old_id_before_network_submission(tmp_path):
    """A create-request connection failure must not restart the old low-memory job."""
    engine = planner()
    database = failed_batch(tmp_path)
    assert engine._reset_retryable_failed_jobs(database) == 1
    assert database.df.iloc[0].id == ""
    database.df.loc[0, "status"] = "start_failed"
    assert engine._reset_retryable_failed_jobs(database) == 1
    assert database.df.iloc[0].retry_mode == "replace"
    assert json.loads(database.df.iloc[0].job_options)["executor-memory"] == "4G"


def test_malformed_log_entries_do_not_crash_retry_classification(tmp_path):
    """Unusable diagnostic fields are treated as missing evidence."""
    engine = planner()
    database = failed_batch(tmp_path)
    next(tmp_path.rglob("*errors.json")).write_text(json.dumps([{"code": {"unexpected": True}}, None]))
    assert engine._reset_retryable_failed_jobs(database) == 1
    assert json.loads(database.df.iloc[0].job_options)["executor-memory"] == "2G"


def test_prepared_replacement_stays_pinned_to_original_account(tmp_path, monkeypatch):
    """Resume after preparing a retry must not move it to another account's shared slot."""
    from queue import Queue
    from data_preparation.parcel_stats import batch_scheduler as scheduler

    engine = planner()
    database = failed_batch(tmp_path)
    engine._reset_retryable_failed_jobs(database)
    plan = {"batch_number": 2, "target_path": str(tmp_path / "batch_2.nc")}
    monkeypatch.setattr(scheduler, "_prepare_database", lambda *args: database)
    shared = Queue()
    pinned = {"original-owner": Queue(), "other-owner": Queue()}
    scheduler._enqueue_partition_tasks(1, engine, [plan], [plan],
                                       [("other-owner", "pw"), ("original-owner", "pw")], shared, pinned, tmp_path)
    assert shared.empty() and pinned["other-owner"].empty()
    assert pinned["original-owner"].qsize() == 1
