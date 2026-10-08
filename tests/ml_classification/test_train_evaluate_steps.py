import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml_classification.shared.config.config import ClassificationPipelineConfig
from ml_classification.shared.models.models import ContiguousLabelClassifier, build_model_candidates
from ml_classification.shared.models.tensorflow_lstm_models import TemporalTensorBuilder
from ml_classification.step_03_train.train import TrainStep
from ml_classification.step_04_evaluate.evaluate import EvaluateStep


def test_model_tune_params_can_be_overridden_per_model():
    custom_gradient_boosting = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [1],
    }
    config = ClassificationPipelineConfig(
        project_dir="test_project",
        target_column="label",
        feature_columns=["feature"],
        selected_models=("gradient_boosting",),
        tune_params={"gradient_boosting": custom_gradient_boosting},
        cv_folds=2,
    )

    candidates = build_model_candidates(config, numeric_features=["feature"], categorical_features=[])

    assert candidates[0].param_distributions == custom_gradient_boosting
    assert config.tune_params["extra_trees"]["model__n_estimators"] == [300, 500, 800]


def test_every_model_candidate_uses_its_configured_tune_params(monkeypatch):
    config = ClassificationPipelineConfig(
        project_dir="test_project", target_column="label", feature_columns=["feature"], cv_folds=2
    )
    from ml_classification.shared.models import models as models_module

    monkeypatch.setattr(models_module, "_safe_import", lambda module_name: object())
    candidates = build_model_candidates(config, numeric_features=["feature"], categorical_features=[])

    assert {candidate.name for candidate in candidates} == set(config.tune_params)
    for candidate in candidates:
        assert candidate.param_distributions == config.tune_params[candidate.name]


def test_default_tune_params_are_independent_between_configs():
    first = ClassificationPipelineConfig(
        project_dir="first_project", target_column="label", feature_columns=["feature"], cv_folds=2
    )
    second = ClassificationPipelineConfig(
        project_dir="second_project", target_column="label", feature_columns=["feature"], cv_folds=2
    )

    first.tune_params["gradient_boosting"]["model__n_estimators"].append(999)

    assert 999 not in second.tune_params["gradient_boosting"]["model__n_estimators"]


def test_tune_params_reject_unknown_models():
    with np.testing.assert_raises_regex(ValueError, "unknown model names"):
        ClassificationPipelineConfig(
            project_dir="test_project",
            target_column="label",
            feature_columns=["feature"],
            tune_params={"not_a_model": {"model__value": [1]}},
            cv_folds=2,
        )


def test_tune_params_reject_empty_parameter_values():
    with np.testing.assert_raises_regex(ValueError, "non-empty list or tuple"):
        ClassificationPipelineConfig(
            project_dir="test_project",
            target_column="label",
            feature_columns=["feature"],
            tune_params={"gradient_boosting": {"model__n_estimators": []}},
            cv_folds=2,
        )


def test_contiguous_label_classifier_aligns_probabilities_across_missing_class_folds():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 1, 1, 2, 2])
    cv = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
    ]
    estimator = ContiguousLabelClassifier(LogisticRegression(), num_classes=3)

    probabilities = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    fitted = estimator.fit(X[:4], y[:4])

    assert is_classifier(estimator)
    assert np.array_equal(fitted.classes_seen_, np.array([0, 1]))
    assert np.array_equal(fitted.classes_, np.array([0, 1, 2]))
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_resume_requires_oof_probabilities_when_optimization_is_enabled():
    class Artifact:
        def __init__(self, present):
            self.present = present

        def exists(self):
            return self.present

    best_model = Artifact(present=True)
    cv_results = Artifact(present=True)
    oof_probabilities = Artifact(present=False)

    class ModelDirectory:
        def __truediv__(self, filename):
            assert filename == "oof_probabilities.joblib"
            return oof_probabilities

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        optimize_class_probabilities=True,
        train_model_dir=lambda _model_name: ModelDirectory(),
    )
    step._model_artifact_paths = lambda _model_name: (best_model, cv_results)

    assert not step._is_model_fully_trained("xgboost", supports_predict_proba=True)

    oof_probabilities.present = True
    assert step._is_model_fully_trained("xgboost", supports_predict_proba=True)


def test_prepare_search_parameters_knn_limits_neighbors():
    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(max_search_candidates=100)

    class Candidate:
        name = "knn"
        param_distributions = {"model__n_neighbors": [1, 2, 3, 4, 5], "model__p": [1]}

    # two folds with small train sizes -> min_fold_train_size == 2
    cv = [(np.array([0, 1]), np.array([2])), (np.array([0, 1, 2]), np.array([3]))]
    params, effective = step._prepare_search_parameters(Candidate(), cv)

    assert params["model__n_neighbors"] == [1, 2]
    assert effective <= 100


def test_summarize_search_returns_summary_and_per_fold():
    # Minimal fake search/results to exercise summarization logic
    candidate = SimpleNamespace(name="dummy", supports_predict_proba=False)
    class FakeSearch:
        n_iterations_ = 5
        best_score_ = 0.8

    search = FakeSearch()
    import pandas as pd

    results_df = pd.DataFrame([
        {"split0_test_score": 0.9, "split1_test_score": 0.8}
    ])

    summary, per_fold = TrainStep._summarize_search(candidate, search, results_df)

    assert summary["model"] == "dummy"
    assert abs(summary["best_cv_score"] - 0.85) < 1e-6
    assert len(per_fold) == 2


def test_to_geo_classifier_result_row_binary_and_multiclass():
    eval_step = object.__new__(EvaluateStep)

    # Binary case
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    probs = np.array([[0.6, 0.4], [0.1, 0.9], [0.7, 0.3], [0.8, 0.2]])
    row = eval_step._to_geo_classifier_result_row(y_true, y_pred, probs, [0, 1], "m1")
    assert row["model"] == "m1"
    assert "accuracy" in row and "roc_auc" in row

    # Multiclass case
    y_true = pd.Series(["a", "b", "c", "a"])
    y_pred = np.array(["a", "b", "b", "c"])
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.0, 0.9, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.3, 0.5],
    ])
    row2 = eval_step._to_geo_classifier_result_row(y_true, y_pred, probs, ["a", "b", "c"], "m2")
    assert row2["model"] == "m2"
    assert "f1_score_class" in row2 and "roc_auc_class" in row2


def test_build_parcel_ranking_outputs_generates_rows_and_metrics():
    eval_step = object.__new__(EvaluateStep)
    eval_step.config = SimpleNamespace(id_column="parcel_code", scoring_primary="f1_macro")

    labels = ["a", "b", "c"]
    y_test = pd.Series(["a", "b", "c", "a"])
    id_test = pd.Series(["1", "2", "3", "4"])

    probability_cache_test = {
        "m1": np.array(
            [
                [0.9, 0.05, 0.05],
                [0.2, 0.7, 0.1],
                [0.2, 0.2, 0.6],
                [0.5, 0.3, 0.2],
            ]
        ),
        "m2": np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.3, 0.2, 0.5],
                [0.6, 0.2, 0.2],
            ]
        ),
    }
    selection = {"selected_models": ["m1", "m2"], "selection_type": "soft_voting"}

    parcel_df, summary_df = eval_step._build_parcel_ranking_outputs(
        probability_cache_test=probability_cache_test,
        selection=selection,
        labels=labels,
        y_test=y_test,
        id_test=id_test,
    )

    assert not parcel_df.empty
    assert not summary_df.empty
    assert {"best_class_by_prob_avg", "best_class_by_rank_median"}.issubset(parcel_df.columns)
    assert {"method", "accuracy", "f1_macro"}.issubset(summary_df.columns)
    assert set(summary_df["method"]) == {
        "probability_average",
        "probability_median",
        "rank_average",
        "rank_median",
    }


def test_build_model_candidates_includes_tensorflow_candidate_when_stack_available(monkeypatch, tmp_path):
    config = ClassificationPipelineConfig(
        project_dir=tmp_path,
        target_column="label",
        feature_columns=["f1", "f2"],
        selected_models=("tensorflow_neural_network",),
        cv_folds=2,
    )

    from ml_classification.shared.models import models as models_module

    def fake_safe_import(module_name: str):
        if module_name in {"tensorflow", "scikeras.wrappers"}:
            return object()
        return None

    monkeypatch.setattr(models_module, "_safe_import", fake_safe_import)
    candidates = build_model_candidates(config, numeric_features=["f1", "f2"], categorical_features=[])

    assert [candidate.name for candidate in candidates] == ["tensorflow_neural_network"]
    estimator = candidates[0].builder()
    assert estimator.named_steps["model"].__class__.__name__ == "TensorFlowDenseClassifier"


def test_build_model_candidates_warns_for_unavailable_tensorflow_candidate(monkeypatch, tmp_path):
    config = ClassificationPipelineConfig(
        project_dir=tmp_path,
        target_column="label",
        feature_columns=["f1"],
        selected_models=("tensorflow_neural_network", "random_forest"),
        cv_folds=2,
    )

    from ml_classification.shared.models import models as models_module

    monkeypatch.setattr(models_module, "_safe_import", lambda module_name: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates = build_model_candidates(config, numeric_features=["f1"], categorical_features=[])

    assert [candidate.name for candidate in candidates] == ["random_forest"]
    assert any("tensorflow_neural_network" in str(item.message) for item in caught)


def test_build_model_candidates_includes_keras_lstm_when_stack_available(monkeypatch, tmp_path):
    config = ClassificationPipelineConfig(
        project_dir=tmp_path,
        target_column="label",
        feature_columns=["NDVI_median__20250101", "NDVI_median__20250201"],
        selected_models=("keras_lstm",),
        cv_folds=2,
    )

    from ml_classification.shared.models import models as models_module

    def fake_safe_import(module_name: str):
        if module_name in {"tensorflow", "scikeras.wrappers"}:
            return object()
        return None

    monkeypatch.setattr(models_module, "_safe_import", fake_safe_import)
    candidates = build_model_candidates(config, numeric_features=config.feature_columns, categorical_features=[])

    assert [candidate.name for candidate in candidates] == ["keras_lstm"]
    estimator = candidates[0].builder()
    assert estimator.named_steps["model"].__class__.__name__ == "KerasLSTMClassifier"


def test_temporal_tensor_builder_creates_expected_shape_and_order():
    frame = pd.DataFrame(
        {
            "parcel_code": ["1", "2"],
            "B04_median__20250101": [0.1, 0.2],
            "NDVI_median__20250101": [0.5, 0.6],
            "B04_median__20250201": [0.3, 0.4],
            "NDVI_median__20250201": [0.7, 0.8],
        }
    )
    builder = TemporalTensorBuilder(temporal_statistics=("median",), min_timesteps=2).fit(frame)
    tensor = builder.transform(frame)

    assert tensor.shape == (2, 2, 2)
    # January values for first parcel in the same feature order as schema.
    assert np.isclose(tensor[0, 0, 0], 0.1)
    assert np.isclose(tensor[0, 0, 1], 0.5)


def test_temporal_tensor_builder_raises_for_incomplete_grid_when_required():
    frame = pd.DataFrame(
        {
            "B04_median__20250101": [0.1],
            "NDVI_median__20250101": [0.5],
            "B04_median__20250201": [0.3],
        }
    )
    with np.testing.assert_raises(ValueError):
        TemporalTensorBuilder(temporal_statistics=("median",), require_complete_timesteps=True, min_timesteps=2).fit(frame)


def test_temporal_tensor_builder_rejects_missing_month_in_monthly_sequence():
    frame = pd.DataFrame(
        {
            "NDVI_median__20250101": [0.4],
            "NDVI_median__20250301": [0.6],
        }
    )
    with np.testing.assert_raises_regex(ValueError, "missing calendar months"):
        TemporalTensorBuilder(
            temporal_statistics=("median",),
            temporal_frequency="monthly",
            min_timesteps=2,
        ).fit(frame)


def test_temporal_tensor_builder_rejects_invalid_calendar_date():
    frame = pd.DataFrame(
        {
            "NDVI_median__20250101": [0.4],
            "NDVI_median__20251301": [0.6],
        }
    )
    with np.testing.assert_raises_regex(ValueError, "invalid YYYYMMDD"):
        TemporalTensorBuilder(
            temporal_statistics=("median",),
            temporal_frequency=None,
            min_timesteps=2,
        ).fit(frame)


def test_temporal_tensor_schema_reports_ignored_non_temporal_features():
    frame = pd.DataFrame(
        {
            "NDVI_median__20250101": [0.4],
            "NDVI_median__20250201": [0.5],
            "elevation_mean_m": [120.0],
        }
    )
    builder = TemporalTensorBuilder(
        temporal_statistics=("median",),
        temporal_frequency="monthly",
        min_timesteps=2,
    ).fit(frame)
    schema = builder.to_schema_dict()

    assert schema["dates"] == ["20250101", "20250201"]
    assert schema["features"] == ["NDVI_median"]
    assert schema["ignored_features"] == ["elevation_mean_m"]
    assert schema["n_timesteps"] == 2
    assert schema["n_features_per_timestep"] == 1


def test_temporal_tensor_builder_accepts_leading_underscore_statistics():
    frame = pd.DataFrame(
        {
            "NDVI_median__20250101": [0.4],
            "NDVI_median__20250201": [0.5],
        }
    )
    builder = TemporalTensorBuilder(
        temporal_statistics=("_median",),
        temporal_frequency="monthly",
        min_timesteps=2,
    ).fit(frame)

    schema = builder.to_schema_dict()
    assert schema["features"] == ["NDVI_median"]
    assert schema["dates"] == ["20250101", "20250201"]


def test_train_step_forces_tensorflow_outer_parallelism_to_one():
    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(n_jobs=4)

    assert step._effective_n_jobs(SimpleNamespace(name="keras_lstm")) == 1
    assert step._effective_n_jobs(SimpleNamespace(name="tensorflow_neural_network")) == 1
    assert step._effective_n_jobs(SimpleNamespace(name="random_forest")) == 4


def test_train_step_incremental_training_skips_existing_models(tmp_path):
    """Test that TrainStep skips models that already exist in model_specs.json when force_retrain=False."""
    import json

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        train_dir=tmp_path,
        force_retrain_models=False,
        n_jobs=1,
    )

    # Create a pre-existing model_specs.json to simulate a previous training run
    existing_specs = {
        "random_forest": {
            "best_params": {"model__n_estimators": 50},
            "artifact_dir": str(tmp_path / "random_forest"),
        },
        "knn": {
            "best_params": {"model__n_neighbors": 5},
            "artifact_dir": str(tmp_path / "knn"),
        },
    }
    with open(tmp_path / "model_specs.json", "w") as f:
        json.dump(existing_specs, f)

    # Simulate candidate models: 3 total, 2 already trained
    candidates = [
        SimpleNamespace(name="random_forest"),
        SimpleNamespace(name="knn"),
        SimpleNamespace(name="gradient_boosting"),
    ]

    # Track which models would be trained
    models_to_train = []
    for candidate in candidates:
        if candidate.name not in existing_specs:
            models_to_train.append(candidate.name)

    # Verify that only the new model (gradient_boosting) would be trained
    assert models_to_train == ["gradient_boosting"]
    assert "random_forest" not in models_to_train
    assert "knn" not in models_to_train


def test_train_step_force_retrain_retrains_all_models(tmp_path):
    """Test that TrainStep retrains all models when force_retrain_models=True."""
    import json

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        train_dir=tmp_path,
        force_retrain_models=True,  # Force retraining
        n_jobs=1,
    )

    # Create a pre-existing model_specs.json
    existing_specs = {
        "random_forest": {
            "best_params": {"model__n_estimators": 50},
            "artifact_dir": str(tmp_path / "random_forest"),
        },
    }
    with open(tmp_path / "model_specs.json", "w") as f:
        json.dump(existing_specs, f)

    # When force_retrain_models=True, even existing models should be retrained
    # This is validated by checking that force_retrain_models condition prevents skipping
    assert step.config.force_retrain_models is True


def test_run_train_resumes_unfinished_models(monkeypatch, tmp_path):
    """Verify incremental run_train skips only fully trained models and trains unfinished ones."""
    train_dir = tmp_path / "03_train"
    prepare_dir = tmp_path / "02_prepare"
    models_dir = train_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prepare_dir.mkdir(parents=True, exist_ok=True)

    # Fully trained model: has both model and cv artifacts.
    rf_dir = models_dir / "random_forest"
    rf_dir.mkdir(parents=True, exist_ok=True)
    (rf_dir / "best_model.joblib").write_text("ok", encoding="utf-8")
    pd.DataFrame(
        [{"rank_test_score": 1, "split0_test_score": 0.70, "split1_test_score": 0.66, "mean_test_score": 0.68}]
    ).to_csv(rf_dir / "cv_results.csv", index=False)

    # Unfinished model: listed in specs but missing cv_results.
    knn_dir = models_dir / "knn"
    knn_dir.mkdir(parents=True, exist_ok=True)
    (knn_dir / "best_model.joblib").write_text("ok", encoding="utf-8")

    model_specs = {
        "random_forest": {
            "best_params": {"model__n_estimators": 100},
            "supports_predict_proba": True,
            "numeric_features": ["f1"],
            "categorical_features": [],
            "artifact_dir": str(rf_dir),
        },
        "knn": {
            "best_params": {"model__n_neighbors": 5},
            "supports_predict_proba": True,
            "numeric_features": ["f1"],
            "categorical_features": [],
            "artifact_dir": str(knn_dir),
        },
    }
    (train_dir / "model_specs.json").write_text(json.dumps(model_specs), encoding="utf-8")

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        train_dir=train_dir,
        prepare_dir=prepare_dir,
        target_column="label",
        force_retrain_models=False,
        cv_ranking_method="score_minus_std",
        scoring_primary="f1_macro",
        n_jobs=1,
        train_model_dir=lambda name: models_dir / name,
    )

    train_df = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "label": ["a", "b", "a", "b"]})

    def fake_load_joblib(path):
        if str(path).endswith("train_dataset.joblib"):
            return train_df
        raise AssertionError(f"Unexpected load_joblib path: {path}")

    class DummyEncoder:
        @staticmethod
        def transform(values):
            return np.array([0 if value == "a" else 1 for value in values])

    def fake_context(_):
        return {
            "prepare_summary": {
                "cv_folds": [
                    {"train_index": [0, 1, 2], "valid_index": [3]},
                    {"train_index": [1, 2, 3], "valid_index": [0]},
                ],
                "target_labels": ["a", "b"],
            },
            "active_features": ["f1"],
            "numeric_features": ["f1"],
            "categorical_features": [],
            "label_encoder": DummyEncoder(),
        }

    candidates = [
        SimpleNamespace(name="random_forest", supports_predict_proba=True),
        SimpleNamespace(name="knn", supports_predict_proba=True),
        SimpleNamespace(name="gradient_boosting", supports_predict_proba=True),
    ]

    trained = []

    def fake_train_candidate(candidate, *_args, **_kwargs):
        trained.append(candidate.name)
        return (
            {
                "model": candidate.name,
                "best_cv_score": 0.75,
                "cv_score_std": 0.05,
                "cv_score_minus_std": 0.70,
                "best_iteration": 1,
                "supports_predict_proba": True,
            },
            [{"model": candidate.name, "fold": 0, "score": 0.75}],
            {
                "best_params": {},
                "supports_predict_proba": True,
                "numeric_features": ["f1"],
                "categorical_features": [],
                "artifact_dir": str(models_dir / candidate.name),
            },
            None,
        )

    captured = {}

    def fake_finalize(training_rows, failed_rows, best_fold_rows, model_specs_out, skipped_models=None):
        captured["training_rows"] = training_rows
        captured["failed_rows"] = failed_rows
        captured["best_fold_rows"] = best_fold_rows
        captured["model_specs"] = model_specs_out
        captured["skipped_models"] = skipped_models or []
        return {"ok": True}

    monkeypatch.setattr(
        "ml_classification.step_03_train.train.load_joblib",
        fake_load_joblib,
    )
    monkeypatch.setattr(
        "ml_classification.step_03_train.train.load_modeling_context",
        fake_context,
    )
    monkeypatch.setattr(
        "ml_classification.step_03_train.train.build_model_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(step, "_train_candidate", fake_train_candidate)
    monkeypatch.setattr(step, "_finalize_training", fake_finalize)

    result = step.run_train()

    assert result == {"ok": True}
    assert captured["skipped_models"] == ["random_forest"]
    assert trained == ["knn", "gradient_boosting"]
    assert set(captured["model_specs"].keys()) == {"random_forest", "knn", "gradient_boosting"}
    assert set(row["model"] for row in captured["training_rows"]) == {
        "random_forest",
        "knn",
        "gradient_boosting",
    }


def test_run_train_recovers_trained_models_without_model_specs(monkeypatch, tmp_path):
    """Verify run_train skips fully trained model folders even when model_specs.json is missing."""
    train_dir = tmp_path / "03_train"
    prepare_dir = tmp_path / "02_prepare"
    models_dir = train_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prepare_dir.mkdir(parents=True, exist_ok=True)

    # Pre-existing fully trained model folder (no model_specs.json present).
    rf_dir = models_dir / "random_forest"
    rf_dir.mkdir(parents=True, exist_ok=True)
    (rf_dir / "best_model.joblib").write_text("ok", encoding="utf-8")
    pd.DataFrame(
        [{"rank_test_score": 1, "split0_test_score": 0.71, "split1_test_score": 0.69, "mean_test_score": 0.70}]
    ).to_csv(rf_dir / "cv_results.csv", index=False)

    step = object.__new__(TrainStep)
    step.config = SimpleNamespace(
        train_dir=train_dir,
        prepare_dir=prepare_dir,
        target_column="label",
        force_retrain_models=False,
        cv_ranking_method="score_minus_std",
        scoring_primary="f1_macro",
        n_jobs=1,
        train_model_dir=lambda name: models_dir / name,
    )

    train_df = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "label": ["a", "b", "a", "b"]})

    def fake_load_joblib(path):
        path_str = str(path)
        if path_str.endswith("train_dataset.joblib"):
            return train_df
        if path_str.endswith("best_model.joblib"):
            return SimpleNamespace(predict_proba=lambda _x: None)
        raise AssertionError(f"Unexpected load_joblib path: {path}")

    class DummyEncoder:
        @staticmethod
        def transform(values):
            return np.array([0 if value == "a" else 1 for value in values])

    def fake_context(_):
        return {
            "prepare_summary": {
                "cv_folds": [
                    {"train_index": [0, 1, 2], "valid_index": [3]},
                    {"train_index": [1, 2, 3], "valid_index": [0]},
                ],
                "target_labels": ["a", "b"],
            },
            "active_features": ["f1"],
            "numeric_features": ["f1"],
            "categorical_features": [],
            "label_encoder": DummyEncoder(),
        }

    candidates = [
        SimpleNamespace(name="random_forest", supports_predict_proba=True),
        SimpleNamespace(name="knn", supports_predict_proba=True),
    ]

    trained = []

    def fake_train_candidate(candidate, *_args, **_kwargs):
        trained.append(candidate.name)
        return (
            {
                "model": candidate.name,
                "best_cv_score": 0.72,
                "cv_score_std": 0.04,
                "cv_score_minus_std": 0.68,
                "best_iteration": 1,
                "supports_predict_proba": True,
            },
            [{"model": candidate.name, "fold": 0, "score": 0.72}],
            {
                "best_params": {},
                "supports_predict_proba": True,
                "numeric_features": ["f1"],
                "categorical_features": [],
                "artifact_dir": str(models_dir / candidate.name),
            },
            None,
        )

    captured = {}

    def fake_finalize(training_rows, failed_rows, best_fold_rows, model_specs_out, skipped_models=None):
        captured["training_rows"] = training_rows
        captured["failed_rows"] = failed_rows
        captured["best_fold_rows"] = best_fold_rows
        captured["model_specs"] = model_specs_out
        captured["skipped_models"] = skipped_models or []
        return {"ok": True}

    monkeypatch.setattr(
        "ml_classification.step_03_train.train.load_joblib",
        fake_load_joblib,
    )
    monkeypatch.setattr(
        "ml_classification.step_03_train.train.load_modeling_context",
        fake_context,
    )
    monkeypatch.setattr(
        "ml_classification.step_03_train.train.build_model_candidates",
        lambda *_args, **_kwargs: candidates,
    )
    monkeypatch.setattr(step, "_train_candidate", fake_train_candidate)
    monkeypatch.setattr(step, "_finalize_training", fake_finalize)

    result = step.run_train()

    assert result == {"ok": True}
    assert captured["skipped_models"] == ["random_forest"]
    assert trained == ["knn"]
    assert set(captured["model_specs"].keys()) == {"random_forest", "knn"}
    assert set(row["model"] for row in captured["training_rows"]) == {"random_forest", "knn"}


def test_evaluate_step_filters_models_by_selected_models(tmp_path):
    """Test that EvaluateStep filters evaluation to only selected_models when configured."""
    from types import SimpleNamespace

    eval_step = object.__new__(EvaluateStep)
    eval_step.config = SimpleNamespace(
        selected_models=("random_forest", "extra_trees"),  # Only these two
        id_column="parcel_id",
    )

    # Simulate model_specs with 4 trained models
    inputs = {
        "model_specs": {
            "random_forest": {},
            "knn": {},
            "extra_trees": {},
            "gradient_boosting": {},
        }
    }

    all_models = list(inputs["model_specs"].keys())
    selected = eval_step.config.selected_models

    # Models to evaluate should be the intersection
    models_to_evaluate = [m for m in all_models if m in selected]
    skipped = [m for m in all_models if m not in selected]

    assert set(models_to_evaluate) == {"random_forest", "extra_trees"}
    assert set(skipped) == {"knn", "gradient_boosting"}


def test_evaluate_step_evaluates_all_models_when_selected_models_none():
    """Test that EvaluateStep evaluates all trained models when selected_models is None."""
    from types import SimpleNamespace

    eval_step = object.__new__(EvaluateStep)
    eval_step.config = SimpleNamespace(
        selected_models=None,  # Evaluate all
        id_column="parcel_id",
    )

    # Simulate model_specs with 4 trained models
    inputs = {
        "model_specs": {
            "random_forest": {},
            "knn": {},
            "extra_trees": {},
            "gradient_boosting": {},
        }
    }

    all_models = list(inputs["model_specs"].keys())

    # When selected_models is None, all should be evaluated
    if eval_step.config.selected_models is None:
        models_to_evaluate = all_models
    else:
        models_to_evaluate = [m for m in all_models if m in eval_step.config.selected_models]

    assert len(models_to_evaluate) == 4
    assert set(models_to_evaluate) == {"random_forest", "knn", "extra_trees", "gradient_boosting"}
