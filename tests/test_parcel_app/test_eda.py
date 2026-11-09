"""Web EDA configuration, input isolation and ownership contract."""

from dataclasses import fields

import pytest
from fastapi import HTTPException


def test_all_eda_options_are_exposed_or_explicitly_managed():
    from eda import EDAConfig
    from parcel_app.eda_config import eda_schema

    schema = eda_schema([{"name": "id"}, {"name": "NDVI"}])
    assert {f["name"] for f in schema["fields"]} | set(schema["managed"]) == {f.name for f in fields(EDAConfig)}
    assert schema["managed"]["reset_output_dir"] is False
    assert schema["defaults"]["include_html_report"] is True


def test_eda_validates_selected_columns_types_and_managed_paths():
    from parcel_app.eda_config import validate_eda

    columns = ["id", "NDVI", "crop"]
    assert validate_eda({"target_column": "crop", "id_column": "id"}, columns, columns)["target_column"] == "crop"
    for settings, selected in [
        ({"output_dir": "/etc"}, columns),
        ({"artifact_names": {}}, columns),
        ({"reset_output_dir": True}, columns),
        ({"max_features_per_plot": 1.5}, columns),
        ({"target_column": "crop"}, ["NDVI"]),
        ({}, []),
        ({}, ["missing"]),
        ({"feature_selection_exclude_columns": ["missing"]}, columns),
        ({"include_plots": "false"}, columns),
        ({"iqr_multiplier": float("nan")}, columns),
    ]:
        with pytest.raises(ValueError):
            validate_eda(settings, selected, columns)


def test_eda_requires_completed_owned_parent(monkeypatch):
    from parcel_app import eda_api

    def foreign(*args, **kwargs):
        raise HTTPException(404, "Error: Analysis not found.")

    monkeypatch.setattr(eda_api, "owned_run", foreign)
    with pytest.raises(HTTPException) as error:
        eda_api.completed_run("unused", 2)
    assert error.value.status_code == 404
    monkeypatch.setattr(eda_api, "owned_run", lambda *a, **kw: {"status": "running"})
    with pytest.raises(HTTPException) as error:
        eda_api.completed_run("unused", 2)
    assert error.value.status_code == 409


def test_real_eda_uses_only_selected_columns_and_keeps_source(tmp_path):
    """Exercise real EDA, report output and ZIP layout without touching parcel results."""
    import json
    from zipfile import ZipFile
    import pandas as pd
    from parcel_app.artifacts import checksum
    from parcel_app.eda_task import run_eda_files

    source = tmp_path / "parcel-result.parquet"
    pd.DataFrame({"id": list(range(20)), "value": [n * 0.1 for n in range(20)], "excluded": ["private"] * 20}).to_parquet(source)
    before = checksum(source)
    output = tmp_path / "eda"
    artifacts = run_eda_files(source, output, ["id", "value"], {"id_column": "id", "include_plots": False})
    assert checksum(source) == before
    annotated = pd.read_parquet(output / "eda_annotated_data.parquet")
    assert "excluded" not in annotated.columns
    assert (output / "report.html").is_file()
    assert json.loads((output / "configuration.json").read_text())["columns"] == ["id", "value"]
    with ZipFile(output / "eda-results.zip") as archive:
        assert set(archive.namelist()) == {item["name"] for item in artifacts if item["name"] != "eda-results.zip"}
        assert "tables/overview/column_profile.csv" in archive.namelist()
    with pytest.raises(FileExistsError):
        run_eda_files(source, output, ["value"], {})
    stages = []
    run_eda_files(source, tmp_path / "progress-eda", ["value"], {"include_plots": False},
                  progress=lambda completed, stage: stages.append((completed, stage)))
    assert [completed for completed, _ in stages] == [0, 1, 2, 3, 4]


def test_geometry_is_preserved_only_when_selected(tmp_path):
    """GeoParquet inputs work both with and without geospatial diagnostics."""
    import geopandas as gpd
    from shapely.geometry import Point
    from parcel_app.eda_api import source_columns
    from parcel_app.eda_task import load_selected

    path = tmp_path / "source.parquet"
    gpd.GeoDataFrame({"value": [1, 2]}, geometry=[Point(1, 2), Point(3, 4)], crs=4326).to_parquet(path)
    assert next(column for column in source_columns(path) if column["name"] == "geometry")["geometry"]
    assert list(load_selected(path, ["value"]).columns) == ["value"]
    selected = load_selected(path, ["value", "geometry"])
    assert isinstance(selected, gpd.GeoDataFrame)
    assert selected.crs.to_epsg() == 4326


def test_browser_report_is_private_sandboxed_and_self_contained(tmp_path, monkeypatch):
    """Only registered plots are embedded; inline report scripts cannot access the app."""
    from parcel_app import eda_api

    (tmp_path / "report.html").write_text("<img src='plots/a.png'><img src='../secret.png'>", encoding="utf-8")
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots/a.png").write_bytes(b"png-image")
    row = {"status": "complete", "output_dir": str(tmp_path), "artifacts": [{"name": "report.html"}, {"name": "plots/a.png"}]}
    monkeypatch.setattr(eda_api, "owned_eda", lambda *a: row)
    response = eda_api.open_report("id", {"app_user_id": 1})
    assert b"data:image/png;base64," in response.body
    assert "sandbox allow-scripts;" in response.headers["content-security-policy"]
    assert "allow-same-origin" not in response.headers["content-security-policy"]
    assert "no-store" in response.headers["cache-control"]
    row["status"] = "running"
    with pytest.raises(HTTPException) as error:
        eda_api.open_report("id", {"app_user_id": 1})
    assert error.value.status_code == 409
    row.update(status="complete", artifacts=[])
    with pytest.raises(HTTPException) as error:
        eda_api.open_report("id", {"app_user_id": 1})
    assert error.value.status_code == 404


def test_eda_failure_preserves_diagnostic_traceback(tmp_path, monkeypatch):
    """The worker must leave diagnostic evidence instead of losing the exception."""
    import pandas as pd
    from eda import EDAPipeline
    from parcel_app.eda_task import run_eda_files

    source = tmp_path / "source.parquet"
    pd.DataFrame({"value": [1, 2]}).to_parquet(source)

    def fail(*args):
        raise ValueError("Error: example diagnostic")

    monkeypatch.setattr(EDAPipeline, "run", fail)
    with pytest.raises(ValueError):
        run_eda_files(source, tmp_path / "eda", ["value"], {})
    log = (tmp_path / "eda/eda.log").read_text()
    assert "Traceback" in log and "ValueError: Error: example diagnostic" in log


def test_failed_eda_log_is_downloadable_and_redacted(tmp_path, monkeypatch):
    """Failed-run diagnostics must not expose passwords or require successful publication."""
    from parcel_app import eda_api

    (tmp_path / "eda.log").write_text("ValueError: password=secret https://example.invalid/token", encoding="utf-8")
    monkeypatch.setattr(eda_api, "owned_eda", lambda *a: {"status": "failed", "output_dir": str(tmp_path)})
    response = eda_api.download_log("id", {"app_user_id": 1})
    assert b"ValueError" in response.body
    assert b"secret" not in response.body and b"https://" not in response.body
    assert "attachment" in response.headers["content-disposition"]
