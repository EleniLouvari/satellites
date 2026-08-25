"""Model factories, wrappers, and preprocessing builders for classification training."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# Optional estimators are imported lazily below so the base pipeline remains lightweight.
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from .config import ClassificationPipelineConfig


@dataclass(slots=True)
class ModelCandidate:
    """Describe a trainable model option with builder and hyperparameter search space."""

    name: str
    builder: Callable[[], Pipeline]
    param_distributions: dict[str, list]
    supports_predict_proba: bool = True


class ContiguousLabelClassifier(ClassifierMixin, BaseEstimator):
    """Wrap estimators so probabilistic outputs align with contiguous encoded labels."""

    def __init__(self, base_estimator, num_classes: int | None = None):
        """Store the wrapped estimator and optional global class count."""
        # Keep constructor state minimal so sklearn cloning remains compatible.
        self.base_estimator = base_estimator
        self.num_classes = num_classes

    def get_params(self, deep=True):
        """Return estimator parameters following the scikit-learn protocol."""
        # Expose nested estimator parameters when deep inspection is requested.
        params = {"base_estimator": self.base_estimator, "num_classes": self.num_classes}
        if deep and hasattr(self.base_estimator, "get_params"):
            for key, value in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{key}"] = value
        return params

    def set_params(self, **params):
        """Set estimator parameters following the scikit-learn protocol."""
        # Route prefixed parameters to the wrapped base estimator.
        base_estimator_params = {}
        for key, value in params.items():
            if key.startswith("base_estimator__"):
                base_estimator_params[key.replace("base_estimator__", "", 1)] = value
            else:
                setattr(self, key, value)
        if base_estimator_params and hasattr(self.base_estimator, "set_params"):
            self.base_estimator.set_params(**base_estimator_params)
        return self

    def fit(self, X, y):
        """Fit the wrapped estimator using contiguous local class indices."""
        # Preserve feature names to support downstream dataframe reconstruction.
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        else:
            n_features = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 0
            self.feature_names_in_ = [f"f{idx}" for idx in range(n_features)] if n_features else None
        y_array = np.asarray(y)
        self.classes_seen_ = np.unique(y_array)
        self.estimator_ = clone(self.base_estimator)
        local_mapping = {label: idx for idx, label in enumerate(self.classes_seen_)}
        y_local = np.array([local_mapping[label] for label in y_array], dtype=int)
        self.estimator_.fit(self._ensure_feature_frame(X), y_local)
        # ``cross_val_predict`` and other sklearn classifier utilities use
        # ``classes_`` to align probability columns across folds.  The wrapper
        # already expands probabilities to the global encoded class space, so
        # advertise that same space instead of only the labels seen in this fit.
        self.classes_ = (
            np.arange(self.num_classes, dtype=int)
            if self.num_classes is not None
            else self.classes_seen_.copy()
        )
        return self

    def predict(self, X):
        """Predict labels and map local class indices back to original labels."""
        # Decode wrapped-estimator outputs to the original class domain.
        y_local = self.estimator_.predict(self._ensure_feature_frame(X))
        y_local = np.asarray(y_local, dtype=int)
        return self.classes_seen_[y_local]

    def predict_proba(self, X):
        """Return class probabilities aligned to the configured global classes."""
        # Expand probabilities to the full class space when needed.
        probabilities = self.estimator_.predict_proba(self._ensure_feature_frame(X))
        if self.num_classes is None:
            return probabilities
        full_probabilities = np.zeros((probabilities.shape[0], self.num_classes), dtype=float)
        for idx, original_label in enumerate(self.classes_seen_):
            full_probabilities[:, int(original_label)] = probabilities[:, idx]
        return full_probabilities

    def _ensure_feature_frame(self, X):
        """Ensure feature input is a dataframe when stored names are available."""
        # Convert array-like data back to a labeled frame for pipeline compatibility.
        if hasattr(X, "columns"):
            return X
        if self.feature_names_in_ is None:
            return X
        return pd.DataFrame(X, columns=self.feature_names_in_)


def _to_dense_matrix(X):
    """Convert sparse matrices to dense arrays while leaving dense input unchanged."""
    # Keep this helper lightweight for use in preprocessing pipelines.
    return X.toarray() if hasattr(X, "toarray") else X


def _build_pipeline_with_optional_balancer(config: ClassificationPipelineConfig, preprocessor, model):
    """Build a preprocessing/model pipeline with optional label balancing."""
    # Start from the baseline pipeline used when balancing is disabled.
    method = config.label_balancing_method
    base_steps = [("preprocessor", preprocessor), ("model", model)]
    if method == "none":
        return Pipeline(steps=base_steps)

    imblearn_module = _safe_import("imblearn")
    if imblearn_module is None:
        warnings.warn(
            "label_balancing_method was requested but imbalanced-learn is not installed. Falling back to no balancing.",
            stacklevel=2,
        )
        return Pipeline(steps=base_steps)

    over_sampling_module = _safe_import("imblearn.over_sampling")
    pipeline_module = _safe_import("imblearn.pipeline")
    if over_sampling_module is None or pipeline_module is None:
        warnings.warn("Could not import imbalanced-learn over-sampling modules. Falling back to no balancing.", stacklevel=2)
        return Pipeline(steps=base_steps)

    if method == "random_oversample":
        sampler = over_sampling_module.RandomOverSampler(random_state=config.random_state)
        return pipeline_module.Pipeline(steps=[("preprocessor", preprocessor), ("sampler", sampler), ("model", model)])

    if method == "smote":
        sampler = over_sampling_module.SMOTE(random_state=config.random_state, k_neighbors=config.smote_k_neighbors)
        return pipeline_module.Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("to_dense", FunctionTransformer(_to_dense_matrix, accept_sparse=True)),
                ("sampler", sampler),
                ("model", model),
            ]
        )

    return Pipeline(steps=base_steps)


def _build_tensorflow_estimator(config: ClassificationPipelineConfig):
    """Build the optional TensorFlow dense classifier wrapper lazily."""
    # Import lazily so the base pipeline does not require TensorFlow at import time.
    from .tensorflow_models import TensorFlowDenseClassifier

    return TensorFlowDenseClassifier(random_state=config.random_state)


def _build_tensorflow_lstm_estimator(config: ClassificationPipelineConfig):
    """Build the optional TensorFlow LSTM classifier wrapper lazily."""
    # Import lazily so base pipeline imports do not require TensorFlow.
    from .tensorflow_lstm_models import KerasLSTMClassifier

    return KerasLSTMClassifier(
        temporal_statistics=config.lstm_temporal_statistics,
        require_complete_timesteps=config.lstm_require_complete_timesteps,
        temporal_frequency=config.lstm_temporal_frequency,
        min_timesteps=config.lstm_min_timesteps,
        class_balancing_method=config.lstm_class_balancing_method,
        random_state=config.random_state,
    )


def build_model_candidates(
    config: ClassificationPipelineConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    num_classes: int | None = None,
) -> list[ModelCandidate]:
    """Build the catalog of configured model candidates and search spaces."""
    # Define base candidates first, then add optional external-library estimators.
    candidates: list[ModelCandidate] = [
        ModelCandidate(
            name="logistic_regression",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_linear_preprocessor(config, numeric_features, categorical_features),
                LogisticRegression(max_iter=2500, solver="saga", class_weight="balanced", random_state=config.random_state),
            ),
            param_distributions=config.tune_params["logistic_regression"],
        ),
        ModelCandidate(
            name="linear_sgd_classifier",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_linear_preprocessor(config, numeric_features, categorical_features),
                SGDClassifier(loss="log_loss", penalty="elasticnet", class_weight="balanced", random_state=config.random_state),
            ),
            param_distributions=config.tune_params["linear_sgd_classifier"],
        ),
        ModelCandidate(
            name="random_forest",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_tree_preprocessor(config, numeric_features, categorical_features),
                RandomForestClassifier(random_state=config.random_state, n_jobs=1, class_weight="balanced_subsample"),
            ),
            param_distributions=config.tune_params["random_forest"],
        ),
        ModelCandidate(
            name="extra_trees",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_tree_preprocessor(config, numeric_features, categorical_features),
                ExtraTreesClassifier(
                    random_state=config.random_state, n_jobs=1, class_weight="balanced", bootstrap=True
                ),
            ),
            param_distributions=config.tune_params["extra_trees"],
        ),
        ModelCandidate(
            name="gradient_boosting",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_dense_tree_preprocessor(config, numeric_features, categorical_features),
                GradientBoostingClassifier(random_state=config.random_state),
            ),
            param_distributions=config.tune_params["gradient_boosting"],
        ),
        ModelCandidate(
            name="decision_tree",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_tree_preprocessor(config, numeric_features, categorical_features),
                DecisionTreeClassifier(random_state=config.random_state, class_weight="balanced"),
            ),
            param_distributions=config.tune_params["decision_tree"],
        ),
        ModelCandidate(
            name="knn",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_dense_linear_preprocessor(config, numeric_features, categorical_features),
                KNeighborsClassifier(n_jobs=1),
            ),
            param_distributions=config.tune_params["knn"],
        ),
        ModelCandidate(
            name="neural_network",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_dense_linear_preprocessor(config, numeric_features, categorical_features),
                MLPClassifier(
                    random_state=config.random_state,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    batch_size="auto",
                ),
            ),
            param_distributions=config.tune_params["neural_network"],
        ),
        ModelCandidate(
            name="bagging",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config,
                _build_dense_tree_preprocessor(config, numeric_features, categorical_features),
                BaggingClassifier(
                    estimator=DecisionTreeClassifier(
                        random_state=config.random_state,
                        class_weight="balanced",
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                    ),
                    random_state=config.random_state,
                    n_jobs=1,
                ),
            ),
            param_distributions=config.tune_params["bagging"],
        ),
        ModelCandidate(
            name="bayesian",
            builder=lambda: _build_pipeline_with_optional_balancer(
                config, _build_dense_linear_preprocessor(config, numeric_features, categorical_features), GaussianNB()
            ),
            param_distributions=config.tune_params["bayesian"],
        ),
    ]

    if _safe_import("tensorflow") is not None and _safe_import("scikeras.wrappers") is not None:
        candidates.append(
            ModelCandidate(
                name="tensorflow_neural_network",
                builder=lambda: _build_pipeline_with_optional_balancer(
                    config,
                    _build_dense_linear_preprocessor(config, numeric_features, categorical_features),
                    _build_tensorflow_estimator(config),
                ),
                param_distributions=config.tune_params["tensorflow_neural_network"],
            )
        )
        candidates.append(
            ModelCandidate(
                name="keras_lstm",
                builder=lambda: Pipeline(steps=[("model", _build_tensorflow_lstm_estimator(config))]),
                param_distributions=config.tune_params["keras_lstm"],
            )
        )

    if not categorical_features:
        candidates.append(
            ModelCandidate(
                name="hist_gradient_boosting",
                builder=lambda: _build_pipeline_with_optional_balancer(
                    config,
                    _build_hist_preprocessor(numeric_features),
                    HistGradientBoostingClassifier(random_state=config.random_state, class_weight="balanced"),
                ),
                param_distributions=config.tune_params["hist_gradient_boosting"],
            )
        )
    xgboost_module = _safe_import("xgboost")
    if xgboost_module is not None:
        candidates.append(
            ModelCandidate(
                name="xgboost",
                builder=lambda: _build_pipeline_with_optional_balancer(
                    config,
                    _build_dense_tree_preprocessor(config, numeric_features, categorical_features),
                    ContiguousLabelClassifier(
                        xgboost_module.XGBClassifier(
                            random_state=config.random_state, n_jobs=1, eval_metric="mlogloss", verbosity=0
                        ),
                        num_classes=num_classes,
                    ),
                ),
                param_distributions=config.tune_params["xgboost"],
            )
        )
    lightgbm_module = _safe_import("lightgbm")
    if lightgbm_module is not None:
        candidates.append(
            ModelCandidate(
                name="lightgbm",
                builder=lambda: _build_pipeline_with_optional_balancer(
                    config,
                    _build_dense_tree_preprocessor(config, numeric_features, categorical_features),
                    ContiguousLabelClassifier(
                        lightgbm_module.LGBMClassifier(
                            random_state=config.random_state, n_jobs=1, verbose=-1, class_weight="balanced"
                        ),
                        num_classes=num_classes,
                    ),
                ),
                param_distributions=config.tune_params["lightgbm"],
            )
        )
    if config.selected_models:
        selected = set(config.selected_models)
        available = {candidate.name for candidate in candidates}
        missing_selected = sorted(selected.difference(available))
        if missing_selected:
            warnings.warn(
                (
                    "Some selected_models are unavailable in this environment and will be skipped: "
                    f"{missing_selected}"
                ),
                stacklevel=2,
            )
        candidates = [candidate for candidate in candidates if candidate.name in selected]
    if not candidates:
        raise ValueError("No model candidates are active. Check config.selected_models and optional dependencies.")
    return candidates


def build_estimator_by_name(
    name: str,
    config: ClassificationPipelineConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    num_classes: int | None = None,
) -> Pipeline:
    """Instantiate a configured estimator pipeline by candidate name."""
    # Rebuild the candidate map on demand to keep configuration-driven behavior.
    candidate_map = {
        candidate.name: candidate
        for candidate in build_model_candidates(config, numeric_features, categorical_features, num_classes=num_classes)
    }
    if name not in candidate_map:
        raise KeyError(f"Unknown model candidate: {name}")
    return candidate_map[name].builder()


def _build_linear_preprocessor(
    config: ClassificationPipelineConfig, numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a preprocessor suitable for linear models."""
    # Apply numeric scaling while keeping categorical output sparse.
    return _build_preprocessor(config, numeric_features, categorical_features, scale_numeric=True, dense_categorical=False)


def _build_tree_preprocessor(
    config: ClassificationPipelineConfig, numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a preprocessor suitable for tree-based models."""
    # Skip scaling because tree models are scale-invariant.
    return _build_preprocessor(config, numeric_features, categorical_features, scale_numeric=False, dense_categorical=False)


def _build_hist_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Build a numeric-only preprocessor for histogram-based boosting."""
    # Keep preprocessing minimal because this estimator handles nonlinearity internally.
    return ColumnTransformer(transformers=[("numeric", SimpleImputer(strategy="median"), numeric_features)], remainder="drop")


def _build_dense_linear_preprocessor(
    config: ClassificationPipelineConfig, numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a linear-model preprocessor that outputs dense features."""
    # Emit dense categorical matrices for estimators that require dense input.
    return _build_preprocessor(config, numeric_features, categorical_features, scale_numeric=True, dense_categorical=True)


def _build_dense_tree_preprocessor(
    config: ClassificationPipelineConfig, numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a tree-model preprocessor that outputs dense features."""
    # Preserve tree-friendly numeric treatment while densifying categorical features.
    return _build_preprocessor(config, numeric_features, categorical_features, scale_numeric=False, dense_categorical=True)


def _build_preprocessor(
    config: ClassificationPipelineConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool,
    dense_categorical: bool,
) -> ColumnTransformer:
    """Create the shared numeric/categorical preprocessing transformer."""
    # Build numeric and categorical branches from the requested mode flags.
    numeric_transformer = SimpleImputer(strategy="median")
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    encoder_kwargs = {"handle_unknown": "infrequent_if_exist", "min_frequency": config.rare_category_min_frequency}
    if dense_categorical:
        encoder_kwargs["sparse_output"] = False

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(**encoder_kwargs))]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def _safe_import(module_name: str):
    """Import a module by name and return None when import fails."""
    # Allow optional dependencies without making them hard requirements.
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None
