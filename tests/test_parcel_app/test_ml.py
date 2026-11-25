"""Web ML classification: grouped configuration, input assembly, step-by-step runs and ownership."""

from dataclasses import fields
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from tests.test_parcel_app.test_routes import clients as _clients_fixture

clients = _clients_fixture
COLUMNS = ["parcel_id", "crop", "NDVI_median__20240101", "NDVI_median__20240201", "data_reliability_score",
           "geom_shape_complexity_score"]
FEATURES = ["NDVI_median__20240101", "NDVI_median__20240201"]
# Three quick models keep the real pipeline to a few seconds.
FAST = {
    "target_column": "crop",
    "id_column": "parcel_id",
    "selected_models": ["decision_tree", "bayesian", "knn"],
    "max_search_candidates": 2,
    "cv_folds": 3,
    "n_jobs": 1,
    "spatial_split": True,
    # Match the two test features so the config does not warn about capping the default of 15.
    "feature_importance_top_n": len(FEATURES),
}


def write_sources(folder, rows=240):
    """Write a parcel result and the row-aligned EDA table with its calculated columns."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    rng = np.random.default_rng(0)
    label = rng.integers(0, 3, rows)
    x, y = 22 + rng.random(rows) * 0.2, 39 + rng.random(rows) * 0.2
    parcels = gpd.GeoDataFrame(
        {
            "parcel_id": [f"p{i}" for i in range(rows)],
            "crop": np.where(np.arange(rows) < 10, 9999, label).astype(float),
            "NDVI_median__20240101": label + rng.normal(0, 0.4, rows),
            "NDVI_median__20240201": label * 2 + rng.normal(0, 0.4, rows),
            "data_reliability_score": rng.random(rows),
            "geom_shape_complexity_score": rng.random(rows),
        },
        geometry=[box(a, b, a + 0.001, b + 0.001) for a, b in zip(x, y)],
        crs=4326,
    )
    parcels.loc[10:19, "crop"] = np.nan
    source = Path(folder) / "result.parquet"
    parcels.to_parquet(source)
    eda = Path(folder) / "eda_annotated_data.parquet"
    pd.DataFrame(
        {
            "parcel_id": parcels["parcel_id"],
            "NDVI_median__20240101": parcels["NDVI_median__20240101"],
            "is_outlier": np.arange(rows) % 20 == 0,
            "mahalanobis_distance": rng.random(rows),
        }
    ).to_parquet(eda)
    return source, eda


def test_all_ml_options_are_grouped_described_or_explicitly_managed():
    from ml_classification import ClassificationPipelineConfig
    from parcel_app.pipelines.ml.config import MANAGED, SEPARATE, TUNE_PREFIX, WEB_DEFAULTS, ml_schema

    schema = ml_schema([{"name": "parcel_id"}, {"name": "crop"}, {"name": "geometry", "geometry": True}])
    settings = [field for field in schema["fields"] if not field["name"].startswith(TUNE_PREFIX)]
    exposed = {field["name"] for field in settings}
    assert (exposed - set(WEB_DEFAULTS)) | set(MANAGED) | SEPARATE == {f.name for f in fields(ClassificationPipelineConfig)}
    assert {field["name"] for field in schema["fields"]} == set(schema["defaults"])
    groups = {group["id"] for group in schema["groups"]}
    assert all(field["group"] in groups and field["description"] for field in settings)
    assert all(group["description"] for group in schema["groups"])
    by_name = {field["name"]: field for field in schema["fields"]}
    assert by_name["target_column"]["choices"] == [None, "parcel_id", "crop"]
    assert by_name["smote_k_neighbors"]["enabled_when"] == {"label_balancing_method": ["smote"]}
    assert by_name["exclude_outliers_from_training"]["group"] == "data"
    assert by_name["exclude_outliers_from_training"]["type"] == "boolean"
    assert schema["defaults"]["exclude_outliers_from_training"] is False
    assert by_name[f"{TUNE_PREFIX}random_forest"]["type"] == "object"
    assert schema["defaults"]["feature_importance_top_n"] == 15
    assert {name: schema["defaults"][name] for name in (
        "spatial_split", "spatial_split_method", "top_voting_models",
        "optimize_class_probabilities", "probability_optimization_iterations",
    )} == {
        "spatial_split": True, "spatial_split_method": "by_row", "top_voting_models": 5,
        "optimize_class_probabilities": True, "probability_optimization_iterations": 5,
    }
    assert ml_schema([{"name": "parcel_id"}, {"name": "crop"}])["defaults"]["spatial_split"] is False
    assert schema["managed"]["open_html_report"] is False and schema["managed"]["reset_project_dir_on_run_check"] is False


def test_ml_subgroups_explain_fields_of_their_own_group():
    from parcel_app.pipelines.ml.config import GROUPS, SUBGROUPS, ml_schema

    members = {group: set(names) for group, _, _, names in GROUPS}
    placed = [name for sections in SUBGROUPS.values() for section in sections for name in section[4]]
    assert len(placed) == len(set(placed))
    assert all(set(section[4]) <= members[group] for group, sections in SUBGROUPS.items() for section in sections)
    schema = ml_schema([{"name": "parcel_id"}, {"name": "crop"}])
    groups = {group["id"]: group for group in schema["groups"]}
    assert [s["id"] for s in groups["inspection"]["subgroups"]] == ["inputs", "weights", "thresholds", "data_reliability", "geometry"]
    assert [s["id"] for s in groups["confidence"]["subgroups"]] == ["probability", "borda", "class_reliability"]
    assert all(s["description"] for group in groups.values() for s in group.get("subgroups", []))
    by_name = {field["name"]: field for field in schema["fields"]}
    assert by_name["rank_confidence_high_min_borda"]["subgroup"] == "borda"
    assert by_name["inspection_data_weight"]["subgroup"] == "weights"
    # The percentile calibration is a choice whose two percentiles only apply in percentile mode.
    mode = by_name["inspection_data_reliability_threshold_mode"]
    assert (mode["default"], mode["choices"], mode["subgroup"]) == ("absolute", ["absolute", "percentile"], "data_reliability")
    assert by_name["inspection_medium_data_reliability_percentile"]["enabled_when"] == {
        "inspection_scoring_enabled": [True],
        "inspection_data_reliability_threshold_mode": ["percentile"],
    }
    sections = {section["id"]: section for section in groups["inspection"]["subgroups"]}
    assert sections["data_reliability"]["parent"] == "thresholds"
    assert sections["geometry"]["parent"] == "thresholds"
    assert by_name["inspection_geometry_threshold_mode"]["subgroup"] == "geometry"
    assert by_name["inspection_geometry_threshold_mode"]["default"] == "absolute"
    assert by_name["inspection_medium_score_threshold"]["subgroup"] == "thresholds"
    assert all("percentile" in " ".join(sections[name]["description"]).lower()
               and "absolute" in " ".join(sections[name]["description"]).lower()
               for name in ("data_reliability", "geometry"))
    assert by_name["inspection_check_matching_declarations"]["default"] is False
    assert by_name["inspection_check_matching_declarations"]["subgroup"] == "inputs"
    assert by_name["target_column"]["subgroup"] is None


def test_ml_validates_columns_types_limits_and_managed_settings():
    from parcel_app.pipelines.ml.config import pipeline_arguments, validate_ml

    options = validate_ml(FAST, FEATURES, COLUMNS)
    arguments = pipeline_arguments(options, FEATURES)
    assert arguments["feature_columns"] == FEATURES and arguments["open_html_report"] is False
    assert arguments["tune_params"]["neural_network"]["model__hidden_layer_sizes"] == [(32,), (64,), (64, 32)]
    assert "exclude_target_values" not in arguments
    # FAST selects three models, so the default five-model vote uses all three.
    assert arguments["top_voting_models"] == 3
    centroids = pipeline_arguments(validate_ml({**FAST, "add_centroid_features": True}, FEATURES, COLUMNS), FEATURES)
    assert centroids["feature_columns"][-2:] == ["centroid_x", "centroid_y"]
    for settings, features in [
        ({"project_dir": "/etc"}, FEATURES),
        ({"open_html_report": True}, FEATURES),
        ({"target_column": None}, FEATURES),
        ({"target_column": "missing"}, FEATURES),
        ({"id_column": "crop"}, FEATURES),
        ({}, []),
        ({}, ["missing"]),
        ({}, ["crop"]),
        ({}, FEATURES * 2),
        ({"cv_folds": 1}, FEATURES),
        ({"cv_folds": 2.5}, FEATURES),
        ({"n_jobs": 99}, FEATURES),
        ({"n_jobs": -1}, FEATURES),
        ({"selected_models": []}, FEATURES),
        ({"selected_models": ["unknown"]}, FEATURES),
        ({"spatial_split": "true"}, FEATURES),
        ({"test_size": float("nan")}, FEATURES),
        ({"fill_nulls": True}, FEATURES),
        ({"inspection_model_weight": 0.9}, FEATURES),
        ({"tune_params.knn": {"n_neighbors": [3]}}, FEATURES),
        ({"tune_params.knn": {"model__n_neighbors": []}}, FEATURES),
    ]:
        with pytest.raises(ValueError, match="^Error:"):
            validate_ml(FAST | settings, features, COLUMNS)
    with pytest.raises(ValueError, match="no geometry"):
        validate_ml(FAST, FEATURES, COLUMNS, geometry=False)
    # An EDA-calculated column is offered like any other; only the target and ID are reserved.
    assert validate_ml(FAST, [*FEATURES, "is_outlier"], [*COLUMNS, "is_outlier"])["target_column"] == "crop"


@pytest.mark.parametrize("add_centroids", [False, True])
def test_existing_centroid_columns_can_be_selected_once(add_centroids):
    """Accept coordinates offered by the picker without duplicating generated features."""
    from parcel_app.pipelines.ml.config import pipeline_arguments, validate_ml

    features = [*FEATURES, "centroid_x", "centroid_y"]
    settings = validate_ml(FAST | {"add_centroid_features": add_centroids}, features, [*COLUMNS, "centroid_x", "centroid_y"])
    assert pipeline_arguments(settings, features)["feature_columns"] == features
    # Coordinates still cannot leak into the model when assigned as the target or ID.
    for role in ("target_column", "id_column"):
        with pytest.raises(ValueError, match="excluding the target and ID"):
            validate_ml(FAST | {role: "centroid_x"}, features, [*COLUMNS, "centroid_x", "centroid_y"])


def test_input_is_the_parcel_result_plus_selected_eda_columns(tmp_path):
    import pandas as pd

    from parcel_app.pipelines.ml.config import validate_ml
    from parcel_app.pipelines.ml.tasks import load_frame

    source, eda = write_sources(tmp_path)
    available = [*COLUMNS, "is_outlier", "mahalanobis_distance"]
    frame = load_frame(source, eda, FEATURES, validate_ml(FAST, FEATURES, available))
    # The outlier flag is always attached for inspection; other EDA columns only when selected.
    assert set(frame.columns) == {*COLUMNS, "geometry", "is_outlier"}
    assert len(frame) == 240 and int(frame["is_outlier"].sum()) == 12
    assert frame.crs.is_projected
    features = [*FEATURES, "mahalanobis_distance"]
    settings = FAST | {"exclude_target_values": [9999], "add_centroid_features": True}
    frame = load_frame(source, eda, features, validate_ml(settings, features, available))
    assert len(frame) == 230 and not frame["crop"].eq(9999).any()
    assert frame["crop"].isna().sum() == 10 and frame["mahalanobis_distance"].notna().all()
    assert frame["centroid_x"].between(100_000, 900_000).all()
    # Older EDA results stored unscored rows as null; they load as "not an outlier".
    older = pd.read_parquet(eda)
    older["is_outlier"] = pd.array([True, None, False, None] * 60, dtype="boolean")
    older.to_parquet(eda)
    frame = load_frame(source, eda, FEATURES, validate_ml(FAST, FEATURES, available))
    assert frame["is_outlier"].dtype == bool and int(frame["is_outlier"].sum()) == 60
    shuffled = pd.read_parquet(eda).iloc[::-1]
    shuffled.to_parquet(eda)
    with pytest.raises(ValueError, match="row by row"):
        load_frame(source, eda, FEATURES, validate_ml(FAST, FEATURES, available))


def test_real_ml_steps_run_separately_and_continue_from_saved_outputs(tmp_path):
    """Exercise the real pipeline one stage at a time; later calls reuse what earlier ones saved."""
    import json
    from zipfile import ZipFile

    import pandas as pd

    from parcel_app.core.storage import checksum
    from parcel_app.pipelines.ml.routes import REPORTS
    from parcel_app.pipelines.ml.tasks import publish, run_ml_steps

    source, eda = write_sources(tmp_path)
    before = checksum(source), checksum(eda)
    output = tmp_path / "ml"
    settings = FAST | {"exclude_target_values": [9999]}
    calls = []
    run_ml_steps(source, eda, output, FEATURES, settings, 0, 1, on_step=lambda index, finished: calls.append((index, finished)))
    assert calls == [(0, False), (0, True)]
    names = {item["name"] for item in publish(output, False)}
    assert REPORTS["check"] in names and REPORTS["prepare"] not in names and "ml-results.zip" not in names
    # The sources are read by the first step only: later steps must work without them.
    hidden = [path.rename(path.with_suffix(".hidden")) for path in (source, eda)]
    with pytest.raises(OSError):
        run_ml_steps(source, eda, output, FEATURES, settings, 0, 1)
    for path, original in zip(hidden, (source, eda)):
        path.rename(original)
    run_ml_steps(source, eda, output, FEATURES, settings, 1, 3, on_step=lambda index, finished: calls.append((index, finished)))
    assert calls[2:] == [(1, False), (1, True), (2, False), (2, True)]
    assert (output / REPORTS["train"]).is_file() and not (output / REPORTS["predict"]).exists()
    run_ml_steps(source, eda, output, FEATURES, settings, 3, 5)
    assert (checksum(source), checksum(eda)) == before
    artifacts = publish(output, True)
    names = {item["name"] for item in artifacts}
    assert set(REPORTS.values()) <= names
    predictions = pd.read_csv(output / "05_predict/final_predictions_preview.csv")
    assert len(predictions) == 230 and {"crop_prediction", "is_outlier", "inspection_need"} <= set(predictions.columns)
    assert predictions["crop_prediction"].notna().all()
    saved = json.loads((output / "configuration.json").read_text())
    assert saved["features"] == FEATURES and saved["config"]["cv_folds"] == 3
    with ZipFile(output / "ml-results.zip") as archive:
        assert set(archive.namelist()) == names - {"ml-results.zip"}
    log = (output / "ml.log").read_text()
    assert log.count("run_check") >= 1 and "run_predict" in log  # one log across all calls


def test_ml_failure_preserves_diagnostic_traceback(tmp_path, monkeypatch):
    from ml_classification import GeospatialClassificationPipeline
    from parcel_app.pipelines.ml.tasks import run_ml_steps

    source, eda = write_sources(tmp_path)

    def fail(*args):
        raise ValueError("Error: example diagnostic")

    monkeypatch.setattr(GeospatialClassificationPipeline, "run_check", fail)
    with pytest.raises(ValueError):
        run_ml_steps(source, eda, tmp_path / "ml", FEATURES, FAST)
    log = (tmp_path / "ml/ml.log").read_text()
    assert "Traceback" in log and "ValueError: Error: example diagnostic" in log


def test_browser_report_is_private_sandboxed_and_names_no_paths(tmp_path, monkeypatch):
    """Only registered plots of a fixed report page are embedded; the page cannot reach the app."""
    from parcel_app.pipelines.ml import routes as ml_api

    (tmp_path / "03_train/plots").mkdir(parents=True)
    (tmp_path / "03_train/report.html").write_text(
        f"<img src='plots/a.png' alt='a'><img src='../secret.png' alt='b'><img src='plots/unregistered.png' alt='c'>{tmp_path}/x",
        encoding="utf-8",
    )
    (tmp_path / "03_train/plots/a.png").write_bytes(b"png-image")
    (tmp_path / "03_train/plots/unregistered.png").write_bytes(b"other")
    (tmp_path / "secret.png").write_bytes(b"secret")
    row = {
        "status": "paused",
        "output_dir": str(tmp_path),
        "artifacts": [{"name": "03_train/report.html"}, {"name": "03_train/plots/a.png"}],
    }
    monkeypatch.setattr(ml_api, "owned_ml", lambda *a: row)
    response = ml_api.open_report("id", "train", {"app_user_id": 1})
    assert response.body.count(b"data:image/png;base64,") == 1
    assert b"src='../secret.png'" in response.body and b"src='plots/unregistered.png'" in response.body
    assert str(tmp_path).encode() not in response.body
    assert "sandbox allow-scripts;" in response.headers["content-security-policy"]
    assert "allow-same-origin" not in response.headers["content-security-policy"]
    # A step that has not run yet, or anything that is not a page name, has no report.
    for page in ("evaluate", "../03_train/report.html", "03_train/report.html"):
        with pytest.raises(HTTPException) as error:
            ml_api.open_report("id", page, {"app_user_id": 1})
        assert error.value.status_code == 404


def test_ml_log_is_readable_while_running_and_redacted(tmp_path, monkeypatch):
    from parcel_app.pipelines.ml import routes as ml_api

    separator = "_" * 100  # the pipeline's own separator lines must not stall redaction
    (tmp_path / "ml.log").write_text(
        f"{separator}\n\x1b[1;32mTraining\x1b[0m password=secret {tmp_path}/03_train", encoding="utf-8"
    )
    row = {"status": "running", "output_dir": str(tmp_path)}
    monkeypatch.setattr(ml_api, "owned_ml", lambda *a: row)
    body = ml_api.download_log("id", {"app_user_id": 1}).body
    assert b"Training password=[redacted] [redacted]/03_train" in body and b"\x1b" not in body
    row["status"] = "queued"
    with pytest.raises(HTTPException) as error:
        ml_api.download_log("id", {"app_user_id": 1})
    assert error.value.status_code == 404


def test_rerun_rules_keep_check_and_prepare_settings_and_force_stale_models():
    from parcel_app.pipelines.ml.config import ml_defaults, required_retrain, rerun_rules

    rules = rerun_rules()
    train, evaluate, predict = set(rules["train"]), set(rules["evaluate"]), set(rules["predict"])
    # Check and Prepare settings, and the columns Check loads, never change in a re-run.
    for name in ("target_column", "apply_iqr", "cv_folds", "random_state", "inspection_data_reliability_column",
                 "max_map_geometries"):
        assert name not in train and name not in evaluate and name not in predict
    assert {"selected_models", "tune_params.knn", "rank_confidence_enabled", "inspection_model_weight"} <= train
    assert predict < evaluate < train
    assert {"top_voting_models", "rank_confidence_enabled", "interpretability_top_models"} <= evaluate
    # Settings read by Train stay locked when the trained models are reused.
    assert not {"selected_models", "tune_params.knn", "scoring_primary", "label_balancing_method", "n_jobs"} & evaluate
    assert "inspection_model_weight" in predict and "prediction_column" in predict
    assert not {"selected_models", "tune_params.knn", "rank_confidence_enabled"} & predict

    previous = ml_defaults() | FAST
    assert required_retrain(previous, dict(previous)) == set()
    grid = previous | {"tune_params.knn": {"model__n_neighbors": [3]}}
    assert required_retrain(previous, grid) == {"knn"}
    assert required_retrain(previous, previous | {"scoring_primary": "accuracy"}) == set(FAST["selected_models"])
    lstm = previous | {"selected_models": ["knn", "keras_lstm"], "lstm_min_timesteps": 3}
    assert required_retrain(previous | {"selected_models": ["knn", "keras_lstm"]}, lstm) == {"keras_lstm"}


def test_rerun_cleanup_removes_replaced_steps_and_only_the_retrained_models(tmp_path):
    from parcel_app.pipelines.ml.routes import stale_artifact
    from parcel_app.pipelines.ml.tasks import clear_for_rerun

    names = ["01_check/report.html", "03_train/models/knn/best_model.joblib", "04_evaluate/report.html",
             "05_predict/inspection_queue.csv", "final_dashboard/report.html", "ml-results.zip", "ml.log"]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    (tmp_path / "03_train/models/bayesian").mkdir()
    assert [name for name in names if stale_artifact(name, 4)] == names[3:6]
    assert [name for name in names if stale_artifact(name, 2)] == names[1:6]

    clear_for_rerun(tmp_path, 2, ["bayesian"])
    remaining = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert remaining == ["01_check/report.html", "03_train/models/knn/best_model.joblib", "ml.log"]
    assert not (tmp_path / "03_train/models/bayesian").exists()
    assert [name for name in names if stale_artifact(name, 3)] == names[2:6]


def claim(kind, target):
    """Claim one queued job the way a worker does."""
    from parcel_app.core import db

    return db.one(
        "UPDATE web_jobs SET status='running',claim_token=%s WHERE target_id=%s AND kind=%s AND status='pending' RETURNING *",
        (str(uuid4()), target, kind),
    )


def test_ml_scenarios_of_an_eda_run_step_by_step_through_the_queue(clients, tmp_path):
    """Run EDA, then two ML scenarios step by step on an owned result; verify access and cleanup."""
    import io
    from zipfile import ZipFile

    from parcel_app.core import db
    from parcel_app.core.storage import checksum
    from parcel_app.jobs.dispatch import execute_task
    from parcel_app.pipelines.parcel_stats.artifacts import register_file

    owner, stranger = clients
    user_id = owner.get("/api/auth/me").json()["app_user_id"]
    upload = owner.post("/api/uploads?filename=original.geojson", content=b"{}").json()
    # Finish the upload's inspection so that only ML work decides whether deletion is allowed.
    db.execute("UPDATE parcel_uploads SET status='ready' WHERE upload_id=%s", (upload["upload_id"],))
    db.execute("UPDATE web_jobs SET status='done' WHERE target_id=%s", (upload["upload_id"],))
    run_id = str(uuid4())
    output = tmp_path / "users" / str(user_id) / "runs" / run_id / "output"
    output.mkdir(parents=True)
    source, unused = write_sources(output)
    unused.unlink()
    source_hash = checksum(source)
    run = db.one(
        "INSERT INTO analysis_runs(run_id,app_user_id,name,upload_id,parcel_id_field,processing,account_ids,output_dir,status) "
        "VALUES(%s,%s,'ML source',%s,'parcel_id',%s,'[]',%s,'complete') RETURNING *",
        (run_id, user_id, upload["upload_id"], '{"start_date": "2024-03-01", "end_date": "2024-09-30"}', str(output)),
    )
    register_file(run, "time_statistics", source)
    eda_body = {
        "request_id": str(uuid4()),
        "columns": ["parcel_id", "crop", *FEATURES],
        "config": {"id_column": "parcel_id", "target_column": "crop", "target_task": "classification", "include_plots": False},
    }
    eda_id = owner.post(f"/api/runs/{run_id}/eda", json=eda_body).json()["eda_id"]
    base = f"/api/eda/{eda_id}/ml"
    assert owner.get(f"{base}/options").status_code == 409  # ML is offered only after EDA has finished
    execute_task(claim("eda", eda_id))

    assert stranger.get(f"{base}/options").status_code == 404
    options = owner.get(f"{base}/options").json()
    sources = {column["name"]: column["source"] for column in options["columns"]}
    assert [name for name in sources if sources[name] == "eda"] == ["parcel_id", "crop", *FEATURES]
    assert sources["is_outlier"] == "calculated" and sources["data_reliability_score"] == "result"
    # Columns EDA analysed come first, then the ones it calculated, then the rest of the parcel result.
    order = [column["source"] for column in options["columns"]]
    assert order == sorted(order, key=["eda", "calculated", "result"].index)
    proposed = [column["name"] for column in options["columns"] if column["proposed"]]
    assert proposed and set(proposed) <= set(FEATURES) and options["default_features"] == proposed
    assert "is_outlier" not in options["default_features"]
    defaults = options["schema"]["defaults"]
    assert (defaults["target_column"], defaults["id_column"]) == ("crop", "parcel_id")
    targets = next(field for field in options["schema"]["fields"] if field["name"] == "target_column")["choices"]
    assert "crop" in targets and "is_outlier" not in targets

    features = [*FEATURES, "mahalanobis_distance"]
    body = {"request_id": str(uuid4()), "name": "Trees", "features": features, "through": 1,
            "config": FAST | {"exclude_target_values": [9999]}}
    assert stranger.post(base, json=body).status_code == 404
    assert owner.post(base, json={**body, "config": FAST | {"project_dir": "/etc"}}).status_code == 422
    assert owner.post(base, json={**body, "features": ["crop"]}).status_code == 422
    assert owner.post(base, json={**body, "config": FAST | {"target_column": "is_outlier"}}).status_code == 422
    assert owner.post(base, json={**body, "through": 6}).status_code == 422
    response = owner.post(base, json=body)
    assert response.status_code == 202, response.text
    ml_id = response.json()["ml_id"]
    assert owner.post(base, json=body).json()["ml_id"] == ml_id
    assert owner.post(base, json={**body, "name": "Renamed"}).status_code == 409
    assert db.one("SELECT count(*) AS n FROM web_jobs WHERE target_id=%s AND kind='ml'", (ml_id,))["n"] == 1
    # The run list and EDA list count what lies beneath each entry for the run tree.
    listed = next(item for item in owner.get("/api/runs").json() if item["run_id"] == run_id)
    assert (listed["eda_count"], listed["ml_count"]) == (1, 1)
    assert owner.get(f"/api/runs/{run_id}/eda").json()[0]["ml_count"] == 1
    link = f"/api/ml/{ml_id}"
    assert owner.post(f"{link}/continue", json={"through": 2}).status_code == 409  # a step is already queued
    assert owner.delete(link).status_code == 409  # queued work blocks deletion

    def state():
        return next(item for item in owner.get(base).json() if item["ml_id"] == ml_id)

    execute_task(claim("ml", ml_id))
    row = state()
    assert (row["status"], row["steps_done"]) == ("paused", 1), row
    assert not {"output_dir", "app_user_id", "input_sha256", "eda_sha256"} & set(row)
    assert owner.get(f"{link}/report/check").status_code == 200
    assert owner.get(f"{link}/report/prepare").status_code == 404
    assert stranger.get(f"{link}/report/check").status_code == 404
    assert stranger.post(f"{link}/continue", json={"through": 2}).status_code == 404
    assert owner.post(f"{link}/continue", json={"through": 1}).status_code == 409  # already done
    assert owner.post(f"{link}/continue", json={"through": 2}).json()["status"] == "queued"
    execute_task(claim("ml", ml_id))
    assert (state()["status"], state()["steps_done"]) == ("paused", 2)

    # A step that fails keeps the finished steps; retrying continues from the failed one.
    ml_root = output.parent / "eda" / eda_id / "ml"  # runs/<run>/eda/<eda>/ml/<ml>
    prepared = ml_root / ml_id / "02_prepare"
    prepared.rename(prepared.with_name("moved"))
    assert owner.post(f"{link}/continue", json={"through": 5}).status_code == 202
    execute_task(claim("ml", ml_id))
    row = state()
    assert (row["status"], row["steps_done"]) == ("failed", 2) and row["error"].startswith("Error: ML classification failed")
    assert owner.get(f"{link}/report/check").status_code == 200
    assert b"Traceback" in owner.get(f"{link}/log").content
    # The failed step leaves an empty 02_prepare behind; Windows cannot rename a folder onto it.
    if prepared.exists():
        prepared.rmdir()
    prepared.with_name("moved").rename(prepared)
    assert owner.post(f"{link}/continue", json={"through": 5}).status_code == 202
    execute_task(claim("ml", ml_id))
    row = state()
    assert (row["status"], row["steps_done"], row["error"]) == ("complete", 5, None), row
    assert row["progress"]["percent"] == 100
    assert owner.post(f"{link}/continue", json={"through": 5}).status_code == 409
    assert checksum(source) == source_hash
    report = owner.get(f"{link}/report/train")
    assert report.status_code == 200 and "data:image/png;base64," in report.text
    assert owner.get(f"{link}/report/dashboard").status_code == 200
    files = {item["name"]: item["artifact_id"] for item in row["artifacts"]}
    archive_link = f"{link}/files/{files['ml-results.zip']}"
    assert stranger.get(archive_link).status_code == 404
    with ZipFile(io.BytesIO(owner.get(archive_link).content)) as archive:
        assert "05_predict/final_predictions_preview.csv" in archive.namelist()
    predictions = owner.get(f"{link}/files/{files['05_predict/final_predictions_preview.csv']}")
    assert predictions.status_code == 200 and b"crop_prediction" in predictions.content
    assert "05_predict/inspection_queue.csv" in files

    # Predict runs again with new inspection weights; settings of earlier steps stay locked.
    saved = row["config"]
    weights = {"inspection_model_weight": 0.5, "inspection_data_weight": 0.3, "inspection_geometry_weight": 0.2}
    rerun = {"step": "predict", "config": saved | weights}
    assert stranger.post(f"{link}/rerun", json=rerun).status_code == 404
    assert owner.post(f"{link}/rerun", json={**rerun, "config": saved | {"cv_folds": 4}}).status_code == 422
    assert owner.post(f"{link}/rerun", json={**rerun, "config": saved | {"selected_models": ["knn"]}}).status_code == 422
    assert owner.post(f"{link}/rerun", json={**rerun, "retrain": ["knn"]}).status_code == 422
    response = owner.post(f"{link}/rerun", json=rerun)
    assert response.status_code == 202, response.text
    queued = response.json()
    assert (queued["status"], queued["steps_done"]) == ("queued", 4)
    assert not any(item["name"].startswith("05_predict/") for item in queued["artifacts"])
    assert owner.get(f"{link}/report/evaluate").status_code == 200
    assert owner.post(f"{link}/rerun", json=rerun).status_code == 409  # already queued
    execute_task(claim("ml", ml_id))
    row = state()
    assert (row["status"], row["steps_done"], row["config"]["inspection_model_weight"]) == ("complete", 5, 0.5), row
    assert "05_predict/inspection_queue.csv" in {item["name"] for item in row["artifacts"]}

    # Evaluate runs again with a smaller vote; the trained models are reused and Train's settings stay locked.
    models = ml_root / ml_id / "03_train" / "models"
    trained = {name: (models / name / "best_model.joblib").stat().st_mtime_ns for name in ("bayesian", "decision_tree")}
    # Lower the Borda minimum with the vote so two voting models are still graded by rank agreement.
    smaller_vote = {"top_voting_models": 2, "rank_confidence_minimum_models": 2}
    rerun = {"step": "evaluate", "config": row["config"] | smaller_vote}
    locked = row["config"] | {"scoring_primary": "accuracy"}
    assert owner.post(f"{link}/rerun", json={**rerun, "config": locked}).status_code == 422
    assert owner.post(f"{link}/rerun", json={**rerun, "retrain": ["bayesian"]}).status_code == 422
    response = owner.post(f"{link}/rerun", json=rerun)
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["steps_done"] == 3
    assert not any(item["name"].startswith(("04_evaluate/", "05_predict/")) for item in queued["artifacts"])
    assert owner.get(f"{link}/report/train").status_code == 200
    execute_task(claim("ml", ml_id))
    row = state()
    assert (row["status"], row["steps_done"], row["config"]["top_voting_models"]) == ("complete", 5, 2), row
    assert trained == {name: (models / name / "best_model.joblib").stat().st_mtime_ns for name in trained}

    # Train runs again: the ticked model and the one whose grid changed are retrained, the third is reused.
    # The default knn grid needs more parcels than these folds have, so knn trains only with the edited grid.
    assert not (models / "knn" / "best_model.joblib").exists()
    trained = {name: (models / name / "best_model.joblib").stat().st_mtime_ns for name in ("bayesian", "decision_tree")}
    grid = {"tune_params.knn": {"model__n_neighbors": [3, 5]}}
    response = owner.post(f"{link}/rerun", json={"step": "train", "config": row["config"] | grid, "retrain": ["bayesian"]})
    assert response.status_code == 202, response.text
    assert response.json()["steps_done"] == 2
    job = db.one("SELECT payload FROM web_jobs WHERE target_id=%s AND status='pending'", (ml_id,))
    assert job["payload"] == {"through": 5, "rerun": 2, "retrain": ["bayesian", "knn"]}
    execute_task(claim("ml", ml_id))
    row = state()
    assert (row["status"], row["steps_done"]) == ("complete", 5), row
    after = {name: (models / name / "best_model.joblib").stat().st_mtime_ns for name in ("bayesian", "decision_tree")}
    assert after["decision_tree"] == trained["decision_tree"] and after["bayesian"] != trained["bayesian"]
    assert (models / "knn" / "best_model.joblib").is_file()
    files = {item["name"]: item["artifact_id"] for item in row["artifacts"]}
    archive_link = f"{link}/files/{files['ml-results.zip']}"

    # A second scenario of the same EDA runs all steps at once with other models.
    other = {**body, "request_id": str(uuid4()), "name": "Bayes only", "features": FEATURES, "through": 5,
             # Two models: lower the defaults of three so the config does not warn about capping them.
             "config": FAST | {"selected_models": ["bayesian", "decision_tree"], "interpretability_top_models": 2,
                               "rank_confidence_minimum_models": 2}}
    second = owner.post(base, json=other).json()["ml_id"]
    execute_task(claim("ml", second))
    assert [(item["name"], item["status"]) for item in owner.get(base).json()] == [("Bayes only", "complete"), ("Trees", "complete")]

    # Inspection: only the single best, finished scenario lists its parcels, most urgent first, with reviews.
    inspect = f"{link}/inspection"
    assert owner.get(inspect).status_code == 409  # not the best scenario yet
    assert stranger.put(f"{link}/best", json={"best": True}).status_code == 404
    assert owner.put(f"{link}/best", json={"best": True}).json()["is_best"] is True
    assert owner.put(f"/api/ml/{second}/best", json={"best": True}).status_code == 200
    assert {item["ml_id"]: item["is_best"] for item in owner.get(base).json()} == {ml_id: False, second: True}
    assert owner.put(f"{link}/best", json={"best": True}).status_code == 200
    page = owner.get(inspect, params={"limit": 500}).json()
    assert page["total"] == 230 and page["id_column"] == "parcel_id" and (page["checked"], page["approved"]) == (0, 0)
    levels = [item["level"] for item in page["rows"] if item["level"] is not None]
    assert levels and levels == sorted(levels, reverse=True)
    # predicted_class repeats crop_prediction, so the table shows only the configured prediction column.
    assert "crop_prediction" in page["columns"] and "predicted_class" not in page["columns"]
    assert all(len(item["values"]) == len(page["columns"]) for item in page["rows"])
    parcel = page["rows"][0]["values"][page["columns"].index("parcel_id")]
    assert page["rows"][0]["review"] == {"check_status": "not_checked", "approved": False, "comment": "", "updated_at": None}
    review = f"{inspect}/review"
    assert stranger.put(review, json={"parcel_id": parcel, "check_status": "field"}).status_code == 404
    assert owner.put(review, json={"parcel_id": parcel, "check_status": "not_checked", "approved": True}).status_code == 422
    assert owner.put(review, json={"parcel_id": parcel, "check_status": "drone"}).status_code == 422
    assert owner.put(review, json={"parcel_id": "no-such-parcel", "check_status": "field"}).status_code == 404
    saved = owner.put(review, json={"parcel_id": parcel, "check_status": "field", "approved": True, "comment": "Wheat seen"})
    assert saved.status_code == 200 and saved.json()["approved"] is True
    shape = owner.get(f"{inspect}/geometry", params={"parcel_id": parcel})
    assert shape.status_code == 200 and shape.json()["geometry"]["type"] == "Polygon"
    assert stranger.get(f"{inspect}/geometry", params={"parcel_id": parcel}).status_code == 404
    assert owner.get(f"{inspect}/geometry", params={"parcel_id": "no-such-parcel"}).status_code == 404
    # Sentinel-2 tiles use the parcel statistics run's dates and cloud limit; without an instance ID there are none.
    sentinel2 = owner.get(f"{inspect}/sentinel2").json()
    assert sentinel2["available"] is False and (sentinel2["start_date"], sentinel2["end_date"]) == ("2024-03-01", "2024-09-30")
    assert sentinel2["max_cloud_cover"] == 70  # the parcel statistics default when the run did not set it
    assert owner.get(f"{inspect}/sentinel2/14/9200/6250.png").status_code == 404
    assert owner.get(f"{inspect}/sentinel2/3/0/0.png").status_code == 404  # below the minimum zoom
    assert owner.get(f"{inspect}/sentinel2/dates", params={"parcel_id": parcel}).status_code == 404
    assert owner.get(f"{inspect}/sentinel2/14/9200/6250.png", params={"date": "not-a-date"}).status_code == 422
    assert stranger.get(f"{inspect}/sentinel2").status_code == 404
    first = owner.get(inspect, params={"limit": 1}).json()
    assert first["rows"][0]["review"]["comment"] == "Wheat seen" and (first["checked"], first["approved"]) == (1, 1)
    assert owner.put(f"{link}/best", json={"best": False}).json()["is_best"] is False
    assert owner.put(review, json={"parcel_id": parcel, "check_status": "field"}).status_code == 409

    # A changed parcel result stops new scenarios and leaves finished ones intact.
    third = owner.post(base, json={**other, "request_id": str(uuid4())}).json()["ml_id"]
    source.write_bytes(b"changed-after-submission")
    execute_task(claim("ml", third))
    assert db.one("SELECT status FROM ml_runs WHERE ml_id=%s", (third,))["status"] == "failed"
    assert owner.get(archive_link).status_code == 200

    ml_output = ml_root / ml_id
    assert owner.delete(f"/api/eda/{eda_id}").status_code == 409  # still used by ML runs
    assert stranger.delete(link).status_code == 404
    assert owner.delete(link, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert owner.delete(link).status_code == 200
    assert not ml_output.exists() and output.exists()
    assert {item["ml_id"] for item in owner.get(base).json()} == {second, third}
    assert owner.delete(f"/api/runs/{run_id}").status_code == 200  # the run removes its EDA and ML records
    assert not db.one("SELECT 1 FROM ml_runs WHERE run_id=%s", (run_id,))
    assert not db.one("SELECT 1 FROM web_jobs WHERE target_id=ANY(%s)", ([ml_id, second, third],))
    assert not output.parent.exists()


def test_interrupted_ml_job_can_be_continued(clients, tmp_path):
    from parcel_app.core import db
    from parcel_app.jobs.queue import reconcile
    from tests.test_parcel_app.test_deletion import seed_run

    owner, _ = clients
    run, _, eda_id, _ = seed_run(owner, tmp_path)
    ml_id = str(uuid4())
    db.execute(
        "INSERT INTO ml_runs(ml_id,run_id,eda_id,app_user_id,name,feature_columns,config,input_sha256,eda_sha256,output_dir,"
        "status,steps_done) VALUES(%s,%s,%s,%s,'Interrupted','[]','{}','a','b',%s,'running',2)",
        (ml_id, str(run["run_id"]), eda_id, run["app_user_id"], str(tmp_path / "unused")),
    )
    db.execute(
        "INSERT INTO web_jobs(job_id,app_user_id,kind,target_id,status) VALUES(%s,%s,'ml',%s,'running')",
        (str(uuid4()), run["app_user_id"], ml_id),
    )
    reconcile("run")
    row = db.one("SELECT status,error,steps_done FROM ml_runs WHERE ml_id=%s", (ml_id,))
    assert row["status"] == "interrupted" and row["steps_done"] == 2 and row["error"].startswith("Error: ML worker interrupted")
    assert owner.post(f"/api/ml/{ml_id}/continue", json={"through": 3}).json()["status"] == "queued"
    job = db.one("SELECT payload FROM web_jobs WHERE target_id=%s AND status='pending'", (ml_id,))
    assert job["payload"] == {"through": 3}


def test_live_log_tail_keeps_the_last_lines_without_colours_or_private_paths(tmp_path, monkeypatch):
    from parcel_app.pipelines.ml import routes

    monkeypatch.setattr(routes, "LIVE_LOG_BYTES", 2000)
    monkeypatch.setattr(routes, "LIVE_LOG_LINES", 3)
    log = tmp_path / "ml.log"
    log.write_text("".join(f"line {n} \x1b[0;90m{tmp_path}/x.csv\x1b[0m\n" for n in range(50)), encoding="utf-8")
    tail = routes.log_tail(log, tmp_path).splitlines()
    assert len(tail) == 3 and tail[-1].startswith("line 49")
    assert "\x1b" not in "".join(tail) and str(tmp_path) not in "".join(tail)


def test_inspection_order_puts_the_most_urgent_parcels_first_and_colours_them(tmp_path):
    from parcel_app.pipelines.ml.inspection import need_level, ordered_predictions

    path = tmp_path / "predictions.csv"
    lines = ["parcel_id,inspection_need,inspection_score", "a,VERY_LOW,0.9", "b,HIGH,0.2", "c,VERY_HIGH,",
             "d,HIGH,0.7", "e,UNKNOWN,0.99", "f,VERY_HIGH,0.1"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    columns, rows = ordered_predictions(str(path), "sha")
    assert columns == ["parcel_id", "inspection_need", "inspection_score"]
    # Need band first, then score (highest first); a missing score follows its band, an unknown band goes last.
    assert [row[0] for row in rows] == ["f", "c", "d", "b", "a", "e"]
    assert (need_level("VERY_HIGH"), need_level("VERY_LOW"), need_level("UNKNOWN")) == (1, 0, None)
    assert 0 < need_level("MEDIUM") < need_level("HIGH") < 1



def test_inspection_geometry_is_one_attribute_free_wgs84_parcel(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    from parcel_app.pipelines.ml.inspection import parcel_geometry

    # Integer identifiers are matched by their text, as the predictions table shows them; metres become degrees.
    path = tmp_path / "parcels.parquet"
    gpd.GeoDataFrame({"id": [1, 2]}, geometry=[box(0, 0, 10, 10), box(500000, 4300000, 500100, 4300100)],
                     crs=32634).to_parquet(path)
    feature = parcel_geometry(str(path), "id", "2")
    assert feature["type"] == "Feature" and feature["properties"] == {}
    x, y = feature["geometry"]["coordinates"][0][0]
    assert 20 < x < 22 and 38 < y < 40
    with pytest.raises(HTTPException) as missing:
        parcel_geometry(str(path), "id", "3")
    assert missing.value.status_code == 404


def test_inspection_hides_predicted_class_only_when_it_copies_the_prediction_column():
    from parcel_app.pipelines.ml.inspection import shown_columns

    columns = ["parcel_id", "crop", "crop_prediction", "predicted_class", "inspection_need"]
    assert shown_columns(columns, {"target_column": "crop", "prediction_column": None}) == [0, 1, 2, 4]
    assert shown_columns(columns, {"target_column": "crop", "prediction_column": "crop_prediction"}) == [0, 1, 2, 4]
    # The configured prediction column is predicted_class itself, or is missing: nothing is hidden.
    named = ["parcel_id", "crop", "predicted_class", "inspection_need"]
    assert shown_columns(named, {"target_column": "crop", "prediction_column": "predicted_class"}) == [0, 1, 2, 3]
    assert shown_columns(named, {"target_column": "crop", "prediction_column": None}) == [0, 1, 2, 3]
