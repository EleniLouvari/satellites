"""Unit tests for ML persistence, label-map sampling and sensitivity result helpers."""

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from ml_classification.sensitivity.libraries import results
from ml_classification.shared import mapping, persistence
from tests.utils import expect_equal, expect_false, expect_in, expect_true

matplotlib.use("Agg")


# --------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------


def test_ensure_dir_creates_nested_directories(tmp_path):
    """Nested folders are created and returned as a Path."""
    folder = persistence.ensure_dir(tmp_path / "a" / "b")
    expect_true(folder.is_dir())
    expect_true(isinstance(folder, Path))


def test_delete_dir_removes_read_only_trees_and_ignores_missing(tmp_path):
    """Read-only files are made writable before removal; missing folders are ignored."""
    folder = tmp_path / "outputs"
    folder.mkdir()
    locked = folder / "locked.txt"
    locked.write_text("x", encoding="utf-8")
    os.chmod(locked, stat.S_IREAD)

    persistence.delete_dir(folder)
    persistence.delete_dir(tmp_path / "never_created")

    expect_false(folder.exists())


def test_delete_dir_tolerates_permission_errors(tmp_path, monkeypatch):
    """A PermissionError from the fallback deletion is swallowed."""
    folder = tmp_path / "busy"
    folder.mkdir()

    def deny(*_args, **_kwargs):
        raise PermissionError("in use")

    monkeypatch.setattr(persistence.shutil, "rmtree", deny)
    persistence.delete_dir(folder)

    expect_true(folder.exists())


def test_clear_directory_removes_children_but_keeps_root(tmp_path):
    """Files and sub-folders are removed while the root folder remains."""
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("y", encoding="utf-8")

    persistence.clear_directory(tmp_path)

    expect_true(tmp_path.is_dir())
    expect_equal(list(tmp_path.iterdir()), [])


def test_json_round_trip_serializes_numpy_and_paths_and_logs(tmp_path):
    """NumPy arrays and paths are serialized, and an explicit log records the write."""
    target = tmp_path / "nested" / "payload.json"
    persistence.save_json({"values": np.array([1, 2]), "path": Path("a/b")}, target)

    loaded = persistence.load_json(target)

    expect_equal(loaded, {"values": [1, 2], "path": str(Path("a/b"))})
    expect_equal(json.loads(target.read_text(encoding="utf-8"))["values"], [1, 2])


def test_json_default_rejects_unknown_types():
    """Unsupported objects raise a TypeError with the project prefix."""
    with pytest.raises(TypeError, match="Error: Object of type"):
        persistence.save_json({"bad": object()}, Path("unused.json"))


def test_joblib_and_csv_round_trip(tmp_path):
    """Joblib and plain CSV artifacts round-trip through their helpers."""
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    persistence.save_joblib({"frame": frame}, tmp_path / "model" / "artifact.joblib")
    persistence.save_frame_csv(frame, tmp_path / "tables" / "frame.csv")

    restored = persistence.load_joblib(tmp_path / "model" / "artifact.joblib")

    pd.testing.assert_frame_equal(restored["frame"], frame)
    expect_equal((tmp_path / "tables" / "frame.csv").read_text(encoding="utf-8").splitlines()[0], "a,b")


def _project_config(tmp_path, fail_on_cleanup_error=False):
    """Return the subset of pipeline configuration used by ``reset_project_outputs``."""
    project = tmp_path / "project"
    return SimpleNamespace(
        project_dir=project,
        log_path=tmp_path / "logs" / "pipeline.log",
        check_dir=project / "01_check",
        prepare_dir=project / "02_prepare",
        train_dir=project / "03_train",
        evaluate_dir=project / "04_evaluate",
        predict_dir=project / "05_predict",
        fail_on_cleanup_error=fail_on_cleanup_error,
    )


def test_reset_project_outputs_clears_previous_run_and_logs(tmp_path):
    """Earlier outputs are removed and both reset messages reach the log file."""
    config = _project_config(tmp_path, fail_on_cleanup_error=True)
    config.train_dir.mkdir(parents=True)
    (config.train_dir / "model.joblib").write_text("old", encoding="utf-8")
    (config.project_dir / "report_index.html").write_text("old", encoding="utf-8")

    persistence.reset_project_outputs(config)

    expect_equal(list(config.project_dir.iterdir()), [])
    log_text = config.log_path.read_text(encoding="utf-8")
    expect_in("Resetting project outputs", log_text)
    expect_in("Project outputs ready", log_text)


def test_reset_project_outputs_raises_when_strict_cleanup_leaves_files(tmp_path, monkeypatch):
    """Strict cleanup fails if anything survives in an existing project folder."""
    config = _project_config(tmp_path, fail_on_cleanup_error=True)
    config.project_dir.mkdir()
    (config.project_dir / "stuck.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(persistence, "clear_directory", lambda _path: None)

    with pytest.raises(PermissionError, match="Error: Could not fully clear"):
        persistence.reset_project_outputs(config)


# --------------------------------------------------------------------------------
# mapping
# --------------------------------------------------------------------------------


def _polygons(count, crs="EPSG:2100"):
    """Return squares of increasing size so area ranking is predictable."""
    return gpd.GeoDataFrame(
        {"label": [f"c{i % 3}" for i in range(count)]},
        geometry=[box(i * 100, 0, i * 100 + i + 1, i + 1) for i in range(count)],
        crs=crs,
    )


def test_sample_geometries_returns_small_frames_unchanged():
    """Frames at or below the limit are returned as-is."""
    df = _polygons(3)
    expect_true(mapping._sample_geometries_for_plot(df, max_geometries=3) is df)


def test_sample_geometries_samples_points_and_keeps_largest_polygons():
    """Points are randomly sampled; polygons keep the largest areas."""
    points = gpd.GeoDataFrame({"label": list("abcdef")}, geometry=[Point(i, i) for i in range(6)], crs="EPSG:2100")
    sampled_points = mapping._sample_geometries_for_plot(points, max_geometries=4)
    sampled_polygons = mapping._sample_geometries_for_plot(_polygons(6), max_geometries=2)

    expect_equal(len(sampled_points), 4)
    expect_equal(sorted(sampled_polygons.index), [4, 5])


def test_sample_geometries_mixes_polygons_and_other_geometries():
    """Mixed inputs keep up to 80% largest polygons and fill the rest with other geometries."""
    polygons = _polygons(8)
    lines = gpd.GeoDataFrame(
        {"label": ["line"] * 4}, geometry=[LineString([(0, i), (1, i)]) for i in range(4)], crs="EPSG:2100"
    )
    mixed = pd.concat([polygons, lines], ignore_index=True)

    sampled = mapping._sample_geometries_for_plot(mixed, max_geometries=5)

    expect_equal(len(sampled), 5)
    expect_equal(int((sampled.geom_type == "Polygon").sum()), 4)
    expect_equal(int((sampled.geom_type == "LineString").sum()), 1)


def test_sample_geometries_samples_non_polygon_mixtures():
    """Point and line mixtures without polygons fall back to random sampling."""
    mixed = gpd.GeoDataFrame(
        {"label": ["p", "p", "l", "l"]},
        geometry=[Point(0, 0), Point(1, 1), LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
        crs="EPSG:2100",
    )
    expect_equal(len(mapping._sample_geometries_for_plot(mixed, max_geometries=3)), 3)


def test_select_largest_polygons_handles_missing_crs_and_reprojection_failure(monkeypatch):
    """Area ranking works without CRS and falls back to native areas if reprojection fails."""
    no_crs = _polygons(4, crs=None)
    expect_equal(sorted(mapping._select_largest_polygons(no_crs, 2).index), [2, 3])

    def fail_to_crs(*_args, **_kwargs):
        raise RuntimeError("projection unavailable")

    projected = _polygons(4)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", fail_to_crs)
    expect_equal(sorted(mapping._select_largest_polygons(projected, 1).index), [3])


def test_geometry_is_points_and_marker_sizes():
    """Point detection and marker sizing follow their documented thresholds."""
    points = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:2100")
    expect_true(mapping._geometry_is_points(points))
    expect_false(mapping._geometry_is_points(_polygons(1)))
    expect_false(mapping._geometry_is_points(points.iloc[:0]))
    sizes = [mapping._marker_sizes_for_row_count(rows) for rows in (10, 500, 1000, 2000)]
    expect_equal(sizes, [(22, 28), (16, 20), (12, 15), (8, 10)])


def test_save_label_map_skips_frames_without_usable_geometry(tmp_path):
    """No file is written when geometry is missing, empty or entirely null."""
    output = tmp_path / "map.png"
    mapping.save_label_map(pd.DataFrame({"label": ["a"]}), "label", output)
    mapping.save_label_map(_polygons(0), "label", output)
    mapping.save_label_map(gpd.GeoDataFrame({"label": ["a"], "geometry": [None]}), "label", output)

    expect_false(output.exists())


def test_save_predicted_labels_map_writes_png(tmp_path):
    """The historical helper renders a map through the shared renderer."""
    output = tmp_path / "maps" / "predicted.png"
    mapping.save_predicted_labels_map(_polygons(5), "label", output, max_geometries=3)

    expect_true(output.exists() and output.stat().st_size > 0, "Map image must be written")


# --------------------------------------------------------------------------------
# sensitivity results
# --------------------------------------------------------------------------------


def test_resolve_selected_model_handles_soft_voting_and_single_models():
    """Soft voting maps to its table name; otherwise the first selected model is used."""
    expect_equal(results.resolve_selected_model({"selection_type": "soft_voting"}), "soft_voting")
    expect_equal(results.resolve_selected_model({"selection_type": "best", "selected_models": ["rf", "et"]}), "rf")
    with pytest.raises(ValueError, match="Error: No selected model"):
        results.resolve_selected_model({"selection_type": "best"})


def test_read_test_metrics_prefers_voting_table_and_falls_back(tmp_path):
    """The voting-aware table wins when present; otherwise the plain table is read."""
    config = SimpleNamespace(evaluate_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Error: Could not find test metrics CSV"):
        results.read_test_metrics(config)

    pd.DataFrame({"model": ["rf"], "accuracy": [0.8]}).to_csv(tmp_path / "model_metrics_test.csv", index=False)
    expect_equal(results.read_test_metrics(config)["model"].tolist(), ["rf"])

    pd.DataFrame({"model": ["soft_voting"], "accuracy": [0.9]}).to_csv(
        tmp_path / "model_metrics_test_with_voting.csv", index=False
    )
    expect_equal(results.read_test_metrics(config)["model"].tolist(), ["soft_voting"])


def test_extract_model_accuracy_returns_float_or_lists_available_models():
    """Accuracy is returned for a known model; unknown models list the alternatives."""
    metrics = pd.DataFrame({"model": ["rf", "et"], "accuracy": [0.75, 0.5]})
    expect_equal(results.extract_model_accuracy(metrics, "rf"), 0.75)
    with pytest.raises(ValueError, match=r"Available: \['et', 'rf'\]"):
        results.extract_model_accuracy(metrics, "svm")


def test_results_to_frame_sorts_by_seed_and_handles_empty_input():
    """Rows are sorted by seed with a stable column order, even when empty."""
    columns = ["seed", "success", "selected_model", "test_accuracy", "project_dir", "error"]
    records = [
        results.SensitivityResult(seed=7, success=False, selected_model=None, test_accuracy=None, project_dir="b", error="x"),
        results.SensitivityResult(seed=1, success=True, selected_model="rf", test_accuracy=0.9, project_dir="a"),
    ]

    frame = results.results_to_frame(records)
    empty = results.results_to_frame([])

    expect_equal(frame["seed"].tolist(), [1, 7])
    expect_equal(list(frame.columns), columns)
    expect_true(empty.empty)
    expect_equal(list(empty.columns), columns)


def test_write_results_csv_writes_plain_table(tmp_path):
    """The summary CSV is written without typed headers."""
    frame = results.results_to_frame(
        [results.SensitivityResult(seed=1, success=True, selected_model="rf", test_accuracy=0.9, project_dir="a")]
    )

    path = results.write_results_csv(tmp_path, frame)

    expect_equal(path, tmp_path / "sensitivity_results.csv")
    expect_equal(path.read_text(encoding="utf-8").splitlines()[0], "seed,success,selected_model,test_accuracy,project_dir,error")
