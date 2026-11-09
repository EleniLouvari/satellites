"""Progress must count persisted work, never equate remote success with final results."""

import json
import pandas as pd


def test_run_progress_separates_remote_jobs_from_completed_results(tmp_path):
    from parcel_app.completion import run_progress

    run = {"status": "running", "output_dir": str(tmp_path), "summary": {"batch_count": 2, "partition_count": 1}}
    folder = tmp_path / "partition_1" / "monthly_cubes"
    folder.mkdir(parents=True)
    pd.DataFrame({"status": ["finished", "running"]}).to_parquet(folder / "tile_jobs.parquet")
    result = run_progress(run)
    assert result["remote"]["percent"] == 50
    assert result["local"]["percent"] == 0
    pd.DataFrame({"status": ["finished", "finished"]}).to_parquet(folder / "tile_jobs.parquet")
    assert run_progress(run)["remote"]["percent"] == 100
    assert run_progress(run)["local"]["percent"] == 0
    (folder.parent / "satellite_partition_completion.json").write_text(json.dumps({"status": "processing"}))
    assert run_progress(run)["local"]["percent"] == 0
    run["status"] = "complete"
    assert run_progress(run)["percent"] == 100


def test_unknown_progress_is_not_invented(tmp_path):
    from parcel_app.completion import run_progress

    assert run_progress({"status": "running", "output_dir": str(tmp_path), "summary": {}})["percent"] is None


def test_local_batches_advance_before_partition_assembly(tmp_path):
    from parcel_app.completion import run_progress

    run = {"status": "running", "output_dir": str(tmp_path), "summary": {"batch_count": 4, "partition_count": 1}}
    folder = tmp_path / "partition_1" / "monthly_cubes" / "signature"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "status": ["finished"] * 4,
        "target_path": [f"/container/batch_{i:05d}_monthly.nc" for i in range(1, 5)],
    }).to_parquet(folder / "tile_jobs.parquet")
    # One committed pair counts; duplicate signatures, partial writes and orphan reports do not.
    for name in (
        "batch_00001_monthly_stats_aaaaaaaaaaaaaaaa.parquet",
        "batch_00001_monthly_stats_aaaaaaaaaaaaaaaa_report.parquet",
        "batch_00001_monthly_stats_bbbbbbbbbbbbbbbb.parquet",
        "batch_00001_monthly_stats_bbbbbbbbbbbbbbbb_report.parquet",
        "batch_00002_monthly_stats_aaaaaaaaaaaaaaaa.parquet",
        "batch_00003_monthly_stats_aaaaaaaaaaaaaaaa.partial.parquet",
        "batch_00003_monthly_stats_aaaaaaaaaaaaaaaa_report.parquet",
        "batch_00004_monthly_cleaning_report.parquet",
        "batch_99999_monthly_stats_aaaaaaaaaaaaaaaa.parquet",
        "batch_99999_monthly_stats_aaaaaaaaaaaaaaaa_report.parquet",
    ):
        pd.DataFrame({"value": [1]}).to_parquet(folder / name)
    result = run_progress(run)
    assert result["local"] == {"stage": "Local batches", "completed": 1, "total": 4, "percent": 25}
    assert result["partitions"]["completed"] == 0
    assert result["stage"] == "Local batches" and result["percent"] == 25
    for i in range(2, 5):
        for suffix in ("", "_report"):
            pd.DataFrame({"value": [1]}).to_parquet(folder / f"batch_{i:05d}_monthly_stats_aaaaaaaaaaaaaaaa{suffix}.parquet")
    result = run_progress(run)
    assert result["local"]["percent"] == 100
    assert result["stage"] == "Partition assembly" and result["percent"] == 0
    (folder.parent.parent / "satellite_partition_completion.json").write_text(json.dumps({"status": "complete"}))
    assert run_progress(run)["stage"] == "Verifying and publishing results"


def test_deletion_progress_uses_checkpointed_remote_count():
    from parcel_app.completion import run_progress

    result = run_progress(
        {"status": "deleting", "deletion_manifest": {"jobs": [{}, {}, {}, {}], "remote_deleted": 1, "remote_complete": False}}
    )
    assert result["percent"] == 25 and result["stage"] == "Remote cleanup"
