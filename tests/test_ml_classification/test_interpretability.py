import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml_classification.step_04_evaluate.libraries import interpretability
from tests.utils import expect_equal, expect_false, expect_true


class _FakeSparseArray:
    def __init__(self, values):
        self._values = np.asarray(values)

    def toarray(self):
        return self._values


class _FakePreprocessor:
    def __init__(self, transformed, feature_names_out=None, raise_names=False):
        self._transformed = transformed
        self._feature_names_out = feature_names_out
        self._raise_names = raise_names

    def transform(self, frame):
        return self._transformed

    def get_feature_names_out(self):
        if self._raise_names:
            raise ValueError("boom")
        return self._feature_names_out


def test_resolve_shap_estimator_parts_without_named_steps():
    model = object()
    preprocessor, resolved_model = interpretability._resolve_shap_estimator_parts(model)

    expect_equal(preprocessor, None)
    expect_true(resolved_model is model)


def test_resolve_shap_estimator_parts_unwraps_contiguous_label_wrapper():
    inner_model = object()
    wrapper_type = type("ContiguousLabelClassifier", (), {})
    wrapped_model = wrapper_type()
    wrapped_model.estimator_ = inner_model
    estimator = SimpleNamespace(named_steps={"preprocessor": "prep", "model": wrapped_model})

    preprocessor, resolved_model = interpretability._resolve_shap_estimator_parts(estimator)

    expect_equal(preprocessor, "prep")
    expect_true(resolved_model is inner_model)


def test_extract_feature_names_prefers_preprocessor_names():
    frame = pd.DataFrame({"f1": [1], "f2": [2]})
    preprocessor = _FakePreprocessor(
        transformed=np.array([[1.0, 2.0]]),
        feature_names_out=np.array(["scale__NDVI__20250101", "cat__B04__20250101"]),
    )

    names = interpretability._extract_feature_names_for_shap(preprocessor, frame)

    expect_equal(names, ["scale__NDVI__20250101", "cat__B04__20250101"])


def test_extract_feature_names_falls_back_to_frame_columns_on_preprocessor_error():
    frame = pd.DataFrame({"f1": [1], "f2": [2]})
    preprocessor = _FakePreprocessor(
        transformed=np.array([[1.0, 2.0]]),
        feature_names_out=np.array(["unused"]),
        raise_names=True,
    )

    names = interpretability._extract_feature_names_for_shap(preprocessor, frame)

    expect_equal(names, ["f1", "f2"])


def test_prepare_shap_inputs_uses_toarray_and_normalizes_feature_names():
    frame = pd.DataFrame({"source_a": [1.0], "source_b": [2.0]})
    preprocessor = _FakePreprocessor(
        transformed=_FakeSparseArray([[1.0, 2.0]]),
        feature_names_out=np.array(["num__NDVI__20250101", "num__B04__20250101"]),
    )

    transformed, names = interpretability._prepare_shap_inputs(preprocessor, frame)

    expect_equal(transformed.shape, (1, 2))
    expect_equal(names, ["NDVI__20250101", "B04__20250101"])


def test_prepare_shap_inputs_returns_none_for_non_2d_data():
    frame = pd.DataFrame({"f1": [1], "f2": [2]})
    preprocessor = _FakePreprocessor(
        transformed=np.array([1.0, 2.0]),
        feature_names_out=np.array(["num__f1", "num__f2"]),
    )

    transformed, names = interpretability._prepare_shap_inputs(preprocessor, frame)

    expect_equal(transformed, None)
    expect_equal(names, ["num__f1", "num__f2"])


def test_run_shap_attempt_returns_none_on_exception():
    result = interpretability._run_shap_attempt(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    expect_equal(result, None)


def test_compute_predictor_shap_values_prefers_predict_proba():
    class Model:
        def predict_proba(self, values):
            return np.array([[0.7, 0.3]])

        def predict(self, values):
            return np.array([0])

    class FakeExplainer:
        def __init__(self, label):
            self.label = label

        def __call__(self, eval_data):
            return self.label

    model = Model()

    def explainer_factory(target, background, feature_names):
        if target.__name__ == "predict_proba":
            return FakeExplainer("from_predict_proba")
        return FakeExplainer("from_predict")

    shap = SimpleNamespace(Explainer=explainer_factory)
    result = interpretability._compute_predictor_shap_values(
        shap=shap,
        fitted_model=model,
        eval_data=np.array([[1.0, 2.0]]),
        background=np.array([[1.0, 2.0]]),
        feature_names=["f1", "f2"],
    )

    expect_equal(result, "from_predict_proba")


def test_compute_shap_values_uses_tree_fallback_when_direct_explainer_fails():
    model_type = type("RandomForestClassifier", (), {})
    model = model_type()

    class FailingDirectExplainer:
        def shap_values(self, eval_data):
            raise RuntimeError("direct failure")

    class TreeExplainerSuccess:
        def __init__(self, *args, **kwargs):
            pass

        def shap_values(self, eval_data):
            return "tree-values"

    shap = SimpleNamespace(
        Explainer=lambda target, *args, **kwargs: FailingDirectExplainer(),
        TreeExplainer=TreeExplainerSuccess,
    )

    result = interpretability._compute_shap_values(
        shap=shap,
        fitted_model=model,
        eval_data=np.array([[1.0, 2.0]]),
        background=np.array([[1.0, 2.0]]),
        feature_names=["f1", "f2"],
    )

    expect_equal(result, "tree-values")


def test_compute_shap_values_raises_for_unsupported_non_callable_model():
    model_type = type("UnsupportedModel", (), {})
    model = model_type()

    class FailingDirectExplainer:
        def shap_values(self, eval_data):
            raise RuntimeError("direct failure")

    shap = SimpleNamespace(
        Explainer=lambda target, *args, **kwargs: FailingDirectExplainer(),
        TreeExplainer=lambda *args, **kwargs: None,
    )

    with np.testing.assert_raises_regex(RuntimeError, "SHAP is not supported for model type UnsupportedModel"):
        interpretability._compute_shap_values(
            shap=shap,
            fitted_model=model,
            eval_data=np.array([[1.0, 2.0]]),
            background=np.array([[1.0, 2.0]]),
            feature_names=["f1", "f2"],
        )


def test_save_shap_summary_plot_returns_early_for_invalid_transformed_shape(monkeypatch, tmp_path):
    fake_shap_module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "shap", fake_shap_module)
    render_called = {"value": False}

    monkeypatch.setattr(
        interpretability,
        "_prepare_shap_inputs",
        lambda preprocessor, frame: (None, ["f1"]),
    )
    monkeypatch.setattr(
        interpretability,
        "_render_shap_summary_figure",
        lambda *args, **kwargs: render_called.update(value=True),
    )

    frame = pd.DataFrame({"f1": [1.0]})
    interpretability.save_shap_summary_plot(
        estimator=SimpleNamespace(),
        X_sample=frame,
        output_path=tmp_path / "plot.png",
        model_name="model_a",
    )

    expect_false(render_called["value"])


def test_save_shap_summary_plot_raises_when_shap_is_unavailable(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("missing shap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    frame = pd.DataFrame({"f1": [1.0]})

    with np.testing.assert_raises_regex(RuntimeError, "SHAP is not installed in the current environment"):
        interpretability.save_shap_summary_plot(
            estimator=SimpleNamespace(),
            X_sample=frame,
            output_path=tmp_path / "plot.png",
            model_name="model_b",
        )
