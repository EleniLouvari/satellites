"""Exercise offline CLI validation, publication and restart boundaries."""

import copy
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


@pytest.fixture
def config(tmp_path):
    """Create a real parcel file and a small two-partition request."""
    parcels = gpd.GeoDataFrame(
        {"id": ["a", "b"], "group": ["west", "east"]}, geometry=[box(0, 0, 100, 100), box(200, 0, 300, 100)], crs=3857
    )
    path = tmp_path / "parcels.geoparquet"
    parcels.to_parquet(path)
    return {
        "schema_version": 1,
        "input": {"path": str(path), "parcel_id_field": "id", "partition_field": "group"},
        "output_dir": str(tmp_path / "run"),
        "run_identifier": "test",
        "processing": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "working_epsg": 3857,
            "sentinel2_bands": ["B02"],
            "spatial_statistics": ["median"],
            "fit_tiles_to_parcels": True,
            "tile_size_metres": 1000,
            "tile_buffer_metres": 0,
        },
    }


@pytest.fixture
def offline_extraction(monkeypatch):
    """Replace remote execution with complete, correctly signed partition artifacts."""
    from data_preparation.parcel_stats import multiuser
    from data_preparation.parcel_stats.job_manager import OpenEOJobManagerZonalStats

    calls = []

    def extract(spatial_parts, users_list, config, *, scheduling):
        """Persist the actual engine's completion contract without remote jobs."""
        calls.append((users_list, scheduling))
        for index, parcels in enumerate(spatial_parts, 1):
            options = dict(
                config,
                output_dir=Path(config["output_dir"]) / f"partition_{index}",
                run_identifier=f"{config['run_identifier']}_part{index}",
            )
            extractor = OpenEOJobManagerZonalStats(parcels=parcels, **options)
            result = extractor.parcels.to_crs(config["working_epsg"])
            result["B02_mean__20240101"] = 2.0
            result.to_parquet(extractor.output_dir / extractor.PARCEL_OUTPUT_FILE_NAME)
            result.to_parquet(extractor.output_dir / extractor.REDUCED_PARCEL_OUTPUT_FILE_NAME)
            pd.DataFrame({"batch_number": [1], "final_null_count": [0]}).to_csv(
                extractor.output_dir / "satellite_pixel_cleaning_report.csv", index=False
            )
            extractor._write_partition_completion("complete")

    monkeypatch.setattr(multiuser, "run_parallel_extractions", extract)
    return calls


def test_validation_resolves_defaults_without_remote_or_output(config, monkeypatch):
    """Preflight must validate real inputs without making jobs or run files."""
    import openeo
    from data_preparation.parcel_stats.run_config import prepare_run

    monkeypatch.setattr(openeo, "connect", lambda *a, **k: pytest.fail("Remote access during validation"))
    original = copy.deepcopy(config)
    prepared = prepare_run(config)
    assert len(prepared.parcels) == 2
    assert len(prepared.parts) == 2
    assert prepared.batch_count == 2
    assert prepared.config["processing"]["spatial_statistics"] == ["mean", "median"]
    assert not Path(config["output_dir"]).exists()
    assert config == original


@pytest.mark.parametrize(
    "option,value",
    [
        ("openeo_password", "never-persist"),
        ("unknown_option", 3),
        ("output_dir", "elsewhere"),
        ("resume_completed_partitions", True),
        ("batch_workers", 1.5),
        ("fill_nulls", "false"),
        ("openeo_parallel_jobs", 3),
        ("start_date", "not-a-date"),
        ("working_epsg", 4326),
        ("interpolation_method", "idw"),
    ],
)
def test_invalid_processing_fails_before_output(config, option, value):
    """Unsafe, misspelled and incompatible settings cannot reach remote execution."""
    from data_preparation.parcel_stats.run_config import prepare_run

    config["processing"][option] = value
    with pytest.raises((ValueError, TypeError)):
        prepare_run(config)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("problem", ["duplicate", "null_id", "missing_id", "missing_group", "null_group", "nonpolygon"])
def test_ambiguous_parcels_are_rejected(config, problem):
    """The CLI must never silently lose parcels or invent IDs."""
    from data_preparation.parcel_stats.run_config import prepare_run

    path = config["input"]["path"]
    parcels = gpd.read_parquet(path)
    if problem == "duplicate":
        parcels["id"] = [1, "1"]
        parcels["id"] = parcels["id"].astype(str)
    elif problem == "null_id":
        parcels.loc[0, "id"] = None
    elif problem == "missing_id":
        parcels = parcels.drop(columns="id")
    elif problem == "missing_group":
        parcels = parcels.drop(columns="group")
    elif problem == "null_group":
        parcels.loc[0, "group"] = None
    else:
        parcels.geometry = parcels.geometry.centroid
    parcels.to_parquet(path)
    with pytest.raises(ValueError):
        prepare_run(config)


def test_run_publishes_complete_results_without_secrets(config, offline_extraction):
    """The runner must merge every partition and expose only completed artifacts."""
    from data_preparation.parcel_stats.runner import run_parcel_stats

    manifest = run_parcel_stats(config, accounts=[("first", " secret1 "), ("second", "secret2")])
    result = gpd.read_parquet(manifest["artifacts"]["time_statistics"])
    assert set(result["id"]) == {"a", "b"}
    assert result.crs.to_epsg() == 3857
    assert list(result["B02_mean__20240101"]) == [2.0, 2.0]
    assert manifest["status"] == "complete"
    assert offline_extraction == [([("first", " secret1 "), ("second", "secret2")], "batches")]
    for path in Path(config["output_dir"]).rglob("*.json"):
        assert "secret1" not in path.read_text()
        assert "secret2" not in path.read_text()


def test_resume_checks_input_and_account_identity(config, offline_extraction):
    """Resume must preserve input and remote job ownership, but allow password rotation."""
    from data_preparation.parcel_stats.runner import resume_parcel_stats, run_parcel_stats

    run_parcel_stats(config, accounts=[("first", "old")])
    resumed = resume_parcel_stats(config["output_dir"], accounts=[("first", "new")])
    assert resumed["parcel_count"] == 2
    with pytest.raises(ValueError, match="account"):
        resume_parcel_stats(config["output_dir"], accounts=[("different", "new")])
    path = Path(config["input"]["path"])
    parcels = gpd.read_parquet(path)
    parcels["changed"] = True
    parcels.to_parquet(path)
    with pytest.raises(ValueError, match="input"):
        resume_parcel_stats(config["output_dir"], accounts=[("first", "new")])


def test_run_refuses_existing_output(config, offline_extraction):
    """A fresh request must not overwrite an existing run or unrelated files."""
    from data_preparation.parcel_stats.runner import run_parcel_stats

    output = Path(config["output_dir"])
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        run_parcel_stats(config, accounts=[("first", "secret")])
    assert (output / "keep.txt").read_text() == "keep"
    assert not offline_extraction


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_id", "missing_report", "wrong_signature"])
def test_incomplete_partition_never_publishes_success(config, offline_extraction, monkeypatch, problem):
    """A present result file alone must not be mistaken for a complete run."""
    from data_preparation.parcel_stats import multiuser
    from data_preparation.parcel_stats.runner import run_parcel_stats

    original = multiuser.run_parallel_extractions

    def incomplete(*args, **kwargs):
        """Simulate partial or inconsistent artifacts at the execution boundary."""
        original(*args, **kwargs)
        output = Path(config["output_dir"]) / "partition_2"
        path = output / "satellite_parcel_time_stats.geoparquet"
        if problem == "missing":
            path.unlink()
        elif problem == "missing_report":
            (output / "satellite_pixel_cleaning_report.csv").unlink()
        elif problem == "wrong_signature":
            (output / "satellite_partition_completion.json").write_text('{"status": "complete", "signature": "bad"}')
        else:
            frame = gpd.read_parquet(path)
            if problem == "duplicate":
                frame = pd.concat([frame, frame], ignore_index=True)
            else:
                frame["id"] = "wrong"
            frame.to_parquet(path)

    monkeypatch.setattr(multiuser, "run_parallel_extractions", incomplete)
    with pytest.raises(ValueError, match="partition"):
        run_parcel_stats(config, accounts=[("first", "secret")])
    output = Path(config["output_dir"])
    assert not (output / "artifacts.json").exists()
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_run_lock_excludes_another_writer(tmp_path):
    """Concurrent operators must not write to the same checkpoints."""
    from data_preparation.parcel_stats.runner import run_lock

    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already"):
            with run_lock(tmp_path):
                pytest.fail("Second writer acquired the lock")
    with run_lock(tmp_path):
        pass


def test_cli_validate_relative_paths_and_errors(config, tmp_path, capsys):
    """JSON paths are relative to the config file and stdout stays machine readable."""
    from data_preparation.cli import main

    config["input"]["path"] = "parcels.geoparquet"
    config["output_dir"] = "run"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert main(["parcel-stats", "validate", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["parcel_count"] == 2
    path.write_text("{not json")
    assert main(["parcel-stats", "validate", "--config", str(path)]) == 2
    assert "Error:" in capsys.readouterr().err


def test_cli_run_requires_explicit_database_scope(config, tmp_path, capsys):
    """An omitted user must not silently load every database account."""
    from data_preparation.cli import main

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit) as exc:
        main(["parcel-stats", "run", "--config", str(path)])
    assert exc.value.code == 2
    assert not Path(config["output_dir"]).exists()


def test_preflight_normalizes_active_geometry_name(config):
    """A GeoParquet's named active geometry must work with the engine's geometry column."""
    from data_preparation.parcel_stats.run_config import prepare_run

    path = config["input"]["path"]
    gpd.read_parquet(path).rename_geometry("geom").to_parquet(path)
    assert prepare_run(config).parcels.geometry.name == "geometry"


def test_preflight_rejects_geometry_name_collision(config):
    """Normalizing the geometry column must not overwrite an ordinary input attribute."""
    from data_preparation.parcel_stats.run_config import prepare_run

    path = config["input"]["path"]
    parcels = gpd.read_parquet(path).rename_geometry("geom")
    parcels["geometry"] = "keep this attribute"
    parcels.to_parquet(path)
    with pytest.raises(ValueError, match="geometry"):
        prepare_run(config)


def test_preflight_rejects_id_field_whitespace(config):
    """The engine must not strip the requested field and invent replacement IDs."""
    from data_preparation.parcel_stats.run_config import prepare_run

    path = config["input"]["path"]
    gpd.read_parquet(path).rename(columns={"id": " id "}).to_parquet(path)
    config["input"]["parcel_id_field"] = " id "
    with pytest.raises(ValueError, match="parcel_id_field"):
        prepare_run(config)


def test_resume_rejects_changed_config_and_wrong_owner(config, offline_extraction):
    """Owner and saved-configuration checks run before acquiring any new imagery."""
    from data_preparation.parcel_stats.runner import resume_parcel_stats, run_parcel_stats

    run_parcel_stats(config, accounts=[("first", "secret")], app_user_id=42)
    with pytest.raises(ValueError, match="user"):
        resume_parcel_stats(config["output_dir"], accounts=[("first", "secret")], app_user_id=43)
    path = Path(config["output_dir"]) / "config.json"
    saved = json.loads(path.read_text())
    saved["processing"]["fill_nulls"] = True
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="configuration"):
        resume_parcel_stats(config["output_dir"], accounts=[("first", "secret")], app_user_id=42)
    assert len(offline_extraction) == 1


def test_cli_failed_run_redacts_password_and_can_resume(config, tmp_path, offline_extraction, monkeypatch, capsys):
    """A failed CLI run records safe state and resumes using the same database owner."""
    from data_preparation.cli import main
    from data_preparation.parcel_stats import accounts, multiuser

    monkeypatch.setattr(accounts, "load_app_openeo_accounts", lambda owner: [("first", "private-secret")])
    original = multiuser.run_parallel_extractions

    def failure(*args, **kwargs):
        """Simulate an upstream error containing a credential."""
        raise RuntimeError("upstream rejected private-secret")

    monkeypatch.setattr(multiuser, "run_parallel_extractions", failure)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert main(["parcel-stats", "run", "--config", str(path), "--app-user-id", "42"]) == 1
    captured = capsys.readouterr()
    assert "private-secret" not in captured.err + captured.out
    state = json.loads((Path(config["output_dir"]) / "run.json").read_text())
    assert state["status"] == "failed"
    assert "private-secret" not in json.dumps(state)
    monkeypatch.setattr(multiuser, "run_parallel_extractions", original)
    assert main(["parcel-stats", "resume", "--run-dir", config["output_dir"], "--app-user-id", "42"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_runner_executes_real_local_pipeline_from_synthetic_download(config, monkeypatch):
    """Exercise scheduling, cleaning, statistics and merging; replace only remote downloads."""
    import numpy as np
    import openeo
    import xarray as xr
    from data_preparation.parcel_stats import batch_scheduler
    from data_preparation.parcel_stats.runner import run_parcel_stats

    monkeypatch.setattr(openeo, "connect", lambda *a, **k: pytest.fail("Unexpected live connection"))

    def download(shared, pinned, users, jobs_per_user, batch_workers=1):
        """Supply a deterministic physical-band cube for each scheduled batch."""
        assert users == [("first", "secret")]
        assert jobs_per_user == 2
        assert all(queue.empty() for queue in pinned.values())
        while not shared.empty():
            task = shared.get_nowait()
            west, south, east, north = task.plan["parcels"].to_crs(3857).total_bounds
            cube = xr.Dataset(
                {"B02": (("t", "y", "x"), np.full((1, 10, 10), 0.25, dtype="float32"))},
                coords={
                    "t": pd.to_datetime(["2024-01-01"]),
                    "x": west + np.arange(10) * 10 + 5,
                    "y": north - np.arange(10) * 10 - 5,
                },
            ).rio.write_crs(3857)
            cube.to_netcdf(task.plan["target_path"])
            cube.close()

    monkeypatch.setattr(batch_scheduler, "_drain_batches", download)
    result = run_parcel_stats(config, accounts=[("first", "secret")])
    frame = gpd.read_parquet(result["artifacts"]["time_statistics"])
    assert set(frame["id"]) == {"a", "b"}
    assert list(frame["B02_mean__20240101"]) == pytest.approx([0.25, 0.25])
    assert list(frame["B02_median__20240101"]) == pytest.approx([0.25, 0.25])
