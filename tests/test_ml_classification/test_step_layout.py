"""Behavioral and dependency checks for the numbered step packages."""

from __future__ import annotations

import ast
import importlib
import io
import pickle
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from tests.utils import expect_equal, expect_false, expect_in, expect_not_in, expect_true

matplotlib.use("Agg")

from ml_classification import ClassificationPipelineConfig, GeospatialClassificationPipeline
from ml_classification.sensitivity import ClassificationSensitivityRunner, SensitivityResult
from ml_classification.shared.logging import TeeStream
from ml_classification.shared.models.models import ContiguousLabelClassifier

PACKAGE = Path(__file__).resolve().parents[2] / "src/ml_classification"
PREFIX = "ml_classification."
LEGACY = {"core", "steps", "reporting", "visuals", "to_delete"}


def test_log_handler_can_close_tee_after_its_log_context_exits():
    console, log = io.StringIO(), io.StringIO()
    stream = TeeStream(console, log)
    stream.write("completed\n")
    log.close()
    stream.close()
    expect_false(console.closed)
    expect_equal(console.getvalue(), "completed\n")


def test_implementation_dependencies_follow_step_ownership():
    """Keep the new structure from drifting back into cross-step coupling or cycles."""
    edges = {}
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        owner = relative.parts[0]
        if owner in LEGACY:
            continue
        name = PREFIX + relative.with_suffix("").as_posix().replace("/", ".").removesuffix(".__init__")
        edges[name] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            expect_false(any(alias.name == "*" for alias in node.names), msg = path)
            target = node.module or ""
            if not target.startswith(PREFIX):
                continue
            target_owner = target[len(PREFIX) :].split(".")[0]
            expect_not_in(target_owner, LEGACY, msg = (path, target))
            if owner == "shared":
                expect_equal(target_owner, "shared", msg = (path, target))
            if owner.startswith("step_") and target_owner.startswith("step_"):
                expect_equal(owner, target_owner, msg = (path, target))
            edges[name].add(target)

    visited = set()

    def visit(name, active):
        expect_not_in(name, active, msg = f"Import cycle: {' -> '.join([*active, name])}")
        if name in visited:
            return
        for dependency in edges.get(name, ()):
            visit(dependency, (*active, name))
        visited.add(name)

    for name in edges:
        visit(name, ())


def test_fitted_model_round_trip_preserves_predictions():
    data = pd.DataFrame({"feature": [0.0, 0.2, 0.8, 1.0]})
    estimator = ContiguousLabelClassifier(LogisticRegression(), num_classes=2).fit(data, np.array([0, 0, 1, 1]))
    expected = estimator.predict_proba(data)
    restored = pickle.loads(pickle.dumps(estimator))
    expect_true(type(restored) is ContiguousLabelClassifier)
    np.testing.assert_allclose(restored.predict_proba(data), expected)


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("ml_classification.shared.config.config", "ClassificationPipelineConfig"),
        ("ml_classification.shared.models.tensorflow_models", "TensorFlowDenseClassifier"),
        ("ml_classification.shared.models.tensorflow_lstm_models", "KerasLSTMClassifier"),
    ],
)
def test_objects_round_trip_with_current_module_paths(module_name, class_name, tmp_path):
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = (
        {"project_dir": tmp_path, "target_column": "label", "feature_columns": ["value"], "feature_importance_top_n": 1}
        if class_name == "ClassificationPipelineConfig"
        else {"random_state": 42}
    )
    original = cls(**kwargs)
    restored = pickle.loads(pickle.dumps(original))
    expect_true(type(restored) is cls)
    if class_name == "ClassificationPipelineConfig":
        expect_equal(restored.train_dir, original.train_dir)
        expect_equal(restored.tune_params, original.tune_params)
    else:
        expect_equal(restored.get_params(), original.get_params())


@pytest.mark.parametrize("confidence_enabled", [False, True])
def test_numbered_steps_complete_resume_and_regenerate_reports(tmp_path, confidence_enabled):
    """Exercise real helper imports, persisted artifacts, confidence, and restart behavior."""
    rng = np.random.default_rng(42)
    classes = np.tile(np.arange(3), 32)
    data = pd.DataFrame(
        {
            "row_id": [f"parcel-{i}" for i in range(len(classes))],
            "label": np.array([f"crop-{i}" for i in classes], dtype=object),
            "feature": classes + rng.normal(0, 0.15, len(classes)),
            "other_feature": rng.normal(size=len(classes)),
        }
    )
    data.loc[data.index[-6:], "label"] = None
    config = ClassificationPipelineConfig(
        project_dir=tmp_path / "run",
        target_column="label",
        feature_columns=["feature", "other_feature"],
        cv_folds=2,
        n_jobs=1,
        selected_models=("logistic_regression", "decision_tree", "bayesian"),
        max_search_candidates=2,
        tune_params={
            "logistic_regression": {"model__C": [1.0], "model__solver": ["lbfgs"]},
            "decision_tree": {"model__max_depth": [3], "model__min_samples_leaf": [2]},
            "bayesian": {"model__var_smoothing": [1e-9]},
        },
        interpretability_top_models=0,
        feature_importance_top_n=2,
        inspection_scoring_enabled=False,
        class_reliability_minimum_oof_support=2,
        rank_confidence_enabled=confidence_enabled,
        open_html_report=False,
        reset_project_dir_on_run_check=False,
    )
    pipeline = GeospatialClassificationPipeline(config)
    summary = pipeline.run_all(data)
    expect_equal(list(summary), ["check", "prepare", "train", "evaluate", "predict"])
    expect_equal(summary["predict"]["rows_filled"], 6)
    expect_equal((summary["evaluate"]["confidence"]["class_reliability"] is not None), confidence_enabled)
    original_predictions = joblib.load(config.predict_dir / "final_predictions.joblib")
    expect_true(original_predictions[config.prediction_filled_column].notna().all())
    paths = [config.train_model_dir(name) / "best_model.joblib" for name in config.selected_models]
    modification_times = [path.stat().st_mtime_ns for path in paths]

    resumed = GeospatialClassificationPipeline(config)
    resumed.run_train()
    expect_equal([path.stat().st_mtime_ns for path in paths], modification_times)
    resumed.run_predict()
    replayed = joblib.load(config.predict_dir / "final_predictions.joblib")
    pd.testing.assert_frame_equal(replayed, original_predictions)
    resumed.recalculate_prediction_quality()
    recalculated = joblib.load(config.predict_dir / "final_predictions.joblib")
    pd.testing.assert_frame_equal(recalculated, original_predictions)
    reports = resumed.create_reports()
    expect_true(Path(reports["index"]).is_file())
    for name in ("check", "prepare", "train", "evaluate", "predict"):
        report = reports[name]
        expect_true(Path(report["report"] if isinstance(report, dict) else report).is_file())


def test_sensitivity_uses_owned_result_plot_and_report_helpers(tmp_path):
    runner = ClassificationSensitivityRunner(tmp_path / "sensitivity")
    frame = runner._results_to_frame(
        [SensitivityResult(3, True, "soft_voting", 0.8, "run-3"), SensitivityResult(7, False, None, None, "run-7", "example")]
    )
    outputs = runner.create_report(frame, write_csv=True)
    expect_true(outputs)
    expect_true((runner.output_dir / "sensitivity_results.csv").is_file())
    expect_true((runner.output_dir / "accuracy_by_seed.png").is_file())
    expect_in("Sensitivity", (runner.output_dir / "sensitivity_report.html").read_text(encoding="utf-8"))
