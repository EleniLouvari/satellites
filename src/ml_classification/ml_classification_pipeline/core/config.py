"""Configuration object defining runtime options and validation rules for the pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import warnings


def _default_tune_params() -> dict[str, dict[str, list]]:
    """Return an independent copy of the default search space for every model."""
    return deepcopy(
        {
            "logistic_regression": {
                "model__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            },
            "linear_sgd_classifier": {
                "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "model__l1_ratio": [0.0, 0.15, 0.5, 0.8, 1.0],
            },
            "random_forest": {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [5, 8, 12, 16],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [3, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__max_samples": [0.6, 0.8, None],
                "model__ccp_alpha": [0.0, 0.0001, 0.001, 0.005],
            },
            "extra_trees": {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [5, 8, 12, 16],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [3, 5, 10, 20],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__max_samples": [0.6, 0.8, None],
                "model__ccp_alpha": [0.0, 0.0001, 0.001],
            },
            "decision_tree": {
                "model__max_depth": [3, 5, 8, 12, 16],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [5, 10, 20, 40],
                "model__criterion": ["gini", "entropy", "log_loss"],
                "model__max_features": [None, "sqrt", 0.5],
                "model__ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01],
            },
            "knn": {
                "model__n_neighbors": [5, 9, 15, 25, 40],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 30, 50],
            },
            "neural_network": {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "model__alpha": [0.0001, 0.001, 0.01, 0.1],
                "model__learning_rate_init": [0.0001, 0.0005, 0.001],
            },
            "tensorflow_neural_network": {
                "model__hidden_layer_sizes": [(32,), (64,), (128,), (128, 64)],
                "model__activation": ["relu", "selu", "gelu"],
                "model__dropout_rate": [0.0, 0.2, 0.4],
                "model__use_batch_normalization": [False, True],
                "model__learning_rate": [0.0001, 0.0005, 0.001],
                "model__batch_size": [32, 64],
                "model__l2_regularization": [0.0, 0.0001, 0.001],
            },
            "bayesian": {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
            "keras_lstm": {
                "model__lstm_units_1": [32, 64],
                "model__lstm_units_2": [16, 32],
                "model__dense_units": [16, 32],
                "model__dropout_rate": [0.10, 0.25],
                "model__bidirectional": [False, True],
                "model__learning_rate": [0.0005, 0.001],
                "model__batch_size": [32, 64],
            },
            "bagging": {
                "model__n_estimators": [100, 200, 400],
                "model__max_samples": [0.5, 0.7, 0.9],
                "model__max_features": [0.5, 0.7, 0.9],
                "model__estimator__max_depth": [5, 8, 12, 16],
                "model__estimator__min_samples_split": [10, 20, 40],
                "model__estimator__min_samples_leaf": [3, 5, 10, 20],
            },
            "gradient_boosting": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_split": [10, 20, 40],
                "model__min_samples_leaf": [5, 10, 20],
                "model__subsample": [0.6, 0.8],
                "model__max_features": ["sqrt", 0.5, None],
            },
            "hist_gradient_boosting": {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_iter": [100, 200, 400],
                "model__max_depth": [3, 5, 8],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__min_samples_leaf": [20, 40, 60],
                "model__l2_regularization": [0.01, 0.1, 1.0, 10.0],
            },
            "xgboost": {
                "model__base_estimator__n_estimators": [200, 400, 600],
                "model__base_estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__base_estimator__max_depth": [2, 3, 4, 5],
                "model__base_estimator__min_child_weight": [3, 5, 10],
                "model__base_estimator__subsample": [0.6, 0.8, 1.0],
                "model__base_estimator__colsample_bytree": [0.5, 0.7, 0.9],
                "model__base_estimator__reg_alpha": [0.0, 0.01, 0.1, 1.0],
                "model__base_estimator__reg_lambda": [1.0, 5.0, 10.0],
                "model__base_estimator__gamma": [0.0, 0.1, 0.5],
            },
            "lightgbm": {
                "model__base_estimator__n_estimators": [200, 400, 600],
                "model__base_estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__base_estimator__num_leaves": [7, 15, 31],
                "model__base_estimator__max_depth": [3, 5, 8],
                "model__base_estimator__min_child_samples": [20, 40, 80],
                "model__base_estimator__subsample": [0.6, 0.8, 1.0],
                "model__base_estimator__colsample_bytree": [0.5, 0.7, 0.9],
                "model__base_estimator__reg_alpha": [0.0, 0.1, 1.0],
                "model__base_estimator__reg_lambda": [0.1, 1.0, 10.0],
            },
        }
    )


# Slots prevent accidental runtime configuration attributes caused by misspellings.
@dataclass(slots=True)
class ClassificationPipelineConfig:
    """Store validated configuration values used by all pipeline steps."""

    # ================================================================================
    # Core Pipeline & Dataset Configuration
    # Required inputs: project directory, target variable, and input features
    # ================================================================================
    project_dir: str | Path
    target_column: str
    feature_columns: list[str]
    id_column: str = "row_id"

    # ================================================================================
    # Pipeline Control & Restart Behavior
    # Controls how the pipeline handles previous runs and error recovery
    # ================================================================================
    reset_project_dir_on_run_check: bool = True
    fail_on_cleanup_error: bool = False
    force_retrain_models: bool = False

    # ================================================================================
    # Data Validation & Quality Control
    # IQR-based outlier detection and spatial/temporal imputation
    # ================================================================================
    apply_iqr: bool = False
    iqr_lower_quantile: float = 0.25
    iqr_upper_quantile: float = 0.75
    iqr_multiplier: float = 1.5
    enforce_unique_ids: bool = True
    spatial_interpolation_method: str | None = None
    spatial_interpolation_max_distance_in_meters: float | None = None
    spatial_interpolation_variogram_lags: int = 15
    spatial_interpolation_variogram_max_distance_in_meters: float | None = None

    # ================================================================================
    # Data Splitting & Cross-Validation
    # Holdout test split and fold configuration for train/validation/test
    # ================================================================================
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42

    # ================================================================================
    # Spatial Splitting (for GeoDataFrame projects)
    # Hold out complete geographic regions/tiles for honest spatial evaluation
    # ================================================================================
    spatial_split: bool = False
    spatial_split_method: str = "by_row"
    spatial_split_grid_size: int = 10

    # ================================================================================
    # Model Selection & Hyperparameter Search
    # Which models to train and how to search their hyperparameter spaces
    # ================================================================================
    selected_models: tuple[str, ...] | None = None
    max_search_candidates: int = 40
    n_jobs: int = -1
    tune_params: dict[str, dict[str, list]] = field(default_factory=_default_tune_params)

    # ================================================================================
    # Model Training & Scoring Configuration
    # Cross-validation scoring and ranking criteria for model comparison
    # ================================================================================
    scoring_primary: str = "f1_macro"
    scoring_secondary: tuple[str, ...] = ("balanced_accuracy", "f1_weighted", "accuracy")
    cv_ranking_method: str = "score_minus_std"

    # ================================================================================
    # Class Imbalance Handling
    # Strategies for balancing underrepresented classes during training
    # ================================================================================
    label_balancing_method: str = "none"
    smote_k_neighbors: int = 5
    rare_category_min_frequency: int | float = 10

    # ================================================================================
    # LSTM-Specific Configuration
    # Temporal sequence parsing and class balancing for Keras LSTM model
    # ================================================================================
    lstm_temporal_statistics: tuple[str, ...] | None = ("median",)
    lstm_require_complete_timesteps: bool = True
    lstm_temporal_frequency: str | None = "monthly"
    lstm_min_timesteps: int = 6
    lstm_class_balancing_method: str = "balanced_class_weight"

    # ================================================================================
    # Ensemble & Voting Configuration
    # Soft voting strategy for combining multiple model predictions
    # ================================================================================
    selection_type: str = "soft_voting"
    top_voting_models: int | None = None

    # ================================================================================
    # Class Probability Optimization
    # Post-hoc tuning of class probability multipliers to maximize OOF macro F1
    # ================================================================================
    optimize_class_probabilities: bool = False
    probability_multiplier_grid: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0)
    probability_optimization_iterations: int = 2
    probability_optimization_max_accuracy_drop: float = 0.02

    # ================================================================================
    # Prediction Output Configuration
    # Column names and thresholds for final predictions and confidence flags
    # ================================================================================
    prediction_column: str | None = None
    prediction_filled_column: str | None = None
    probability_prefix: str = "probability"
    prediction_confidence_threshold: float = 0.60
    # This field stores the maximum final class probability for diagnostics;
    # it is not the rank-based confidence level.
    prediction_confidence_column: str = "prediction_max_probability"
    prediction_confidence_level_column: str = "prediction_confidence_level"
    prediction_review_column: str = "prediction_needs_review"

    # ================================================================================
    # Rank-Based Ensemble Confidence
    # Qualitative confidence derived from within-model class rankings
    # ================================================================================
    rank_confidence_enabled: bool = True
    rank_confidence_minimum_models: int = 3
    rank_confidence_high_min_borda: float = 90.0
    rank_confidence_high_max_range: float = 2.0
    rank_confidence_medium_min_borda: float = 75.0
    rank_confidence_medium_max_range: float = 4.0
    class_reliability_enabled: bool = True
    class_reliability_minimum_oof_support: int = 100
    class_reliability_high_min_precision: float = 0.80
    class_reliability_medium_min_precision: float = 0.60

    # ================================================================================
    # Interpretability & Feature Importance
    # Model explanation via feature importance and SHAP values
    # ================================================================================
    interpretability_top_models: int = 3
    interpretability_include_shap: bool = False
    shap_sample_size: int = 500
    feature_importance_top_n: int = 15

    # ================================================================================
    # Output & Reporting Configuration
    # Data types, report generation, and directory structure management
    # ================================================================================
    float_dtype: str = "float32"
    max_distribution_features: int = 9
    max_map_geometries: int = 20000
    output_schema_version: str = "1.2.0"
    log_filename: str = "pipeline.log"
    step_names: tuple[str, ...] = field(default=("01_check", "02_prepare", "03_train", "04_evaluate", "05_predict"))
    open_html_report: bool = True


    def __post_init__(self) -> None:
        """Normalize and validate configuration right after dataclass initialization."""
        # Convert user path input into a `Path` for consistent filesystem handling.
        self.project_dir = Path(self.project_dir)
        self._normalize_inputs()
        self._validate_strings_and_sequences()
        self._validate_numeric_ranges()
        self._apply_safe_caps()
        self._set_default_output_columns()

    # The following helper groups encapsulate normalization, validation,
    # and safe-capping logic so the constructor remains easy to read and any
    # validation failures raise clear, contextual errors during initialization.

    def _normalize_inputs(self) -> None:
        """Normalize configurable string and sequence inputs to canonical forms."""
        # Standardize model selection type values before validation.
        if self.selected_models is not None:
            self.selected_models = tuple(self.selected_models)
        self.selection_type = self.selection_type.strip().lower()
        self.cv_ranking_method = self.cv_ranking_method.strip().lower()
        self.label_balancing_method = self.label_balancing_method.strip().lower()
        self.lstm_class_balancing_method = self.lstm_class_balancing_method.strip().lower()
        self.spatial_split_method = self.spatial_split_method.strip().lower()
        if self.lstm_temporal_frequency is not None:
            self.lstm_temporal_frequency = self.lstm_temporal_frequency.strip().lower()
        if self.lstm_temporal_statistics is not None:
            self.lstm_temporal_statistics = tuple(str(value).strip().lower() for value in self.lstm_temporal_statistics)
        self.probability_multiplier_grid = tuple(float(value) for value in self.probability_multiplier_grid)
        if self.spatial_interpolation_method is not None:
            # Normalize user-provided method names to lowercase for later comparisons.
            self.spatial_interpolation_method = self.spatial_interpolation_method.strip().lower()
        self._normalize_tune_params()

    def _normalize_tune_params(self) -> None:
        """Merge model-level user search-space overrides with the default catalog."""
        if not isinstance(self.tune_params, dict):
            raise TypeError("tune_params must be a dictionary keyed by model name.")
        defaults = _default_tune_params()
        invalid_model_names = [name for name in self.tune_params if not isinstance(name, str)]
        if invalid_model_names:
            raise TypeError("tune_params model names must be strings.")
        unknown_models = sorted(set(self.tune_params).difference(defaults))
        if unknown_models:
            raise ValueError(f"tune_params contains unknown model names: {unknown_models}")
        for model_name, param_grid in self.tune_params.items():
            defaults[model_name] = deepcopy(param_grid)
        self.tune_params = defaults

    def _validate_strings_and_sequences(self) -> None:
        """Validate categorical options and sequence-based configuration fields."""
        # Enforce supported voting/selection strategies.
        if self.selection_type not in {"soft_voting", "single_model"}:
            raise ValueError("selection_type must be either 'soft_voting' or 'single_model'.")
        if self.cv_ranking_method not in {"mean_score", "score_minus_std"}:
            raise ValueError("cv_ranking_method must be either 'mean_score' or 'score_minus_std'.")

        if self.label_balancing_method not in {"none", "random_oversample", "smote"}:
            raise ValueError("label_balancing_method must be one of: 'none', 'random_oversample', or 'smote'.")
        if self.spatial_split_method not in {"by_group", "by_row"}:
            raise ValueError("spatial_split_method must be either 'by_group' or 'by_row'.")
        if self.lstm_temporal_frequency not in {None, "monthly"}:
            raise ValueError("lstm_temporal_frequency must be None or 'monthly'.")
        if self.lstm_class_balancing_method not in {"none", "balanced_class_weight"}:
            raise ValueError(
                "lstm_class_balancing_method must be either 'none' or 'balanced_class_weight'."
            )
        if self.lstm_temporal_statistics is not None and not self.lstm_temporal_statistics:
            raise ValueError("lstm_temporal_statistics cannot be empty when provided.")
        if not isinstance(self.lstm_require_complete_timesteps, bool):
            raise TypeError("lstm_require_complete_timesteps must be a bool.")
        if not self.feature_columns:
            raise ValueError("feature_columns must contain at least one feature name.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns contains duplicate names. Provide unique feature names only.")
        if not isinstance(self.apply_iqr, bool):
            raise TypeError("apply_iqr must be a bool.")
        if not isinstance(self.optimize_class_probabilities, bool):
            raise TypeError("optimize_class_probabilities must be a bool.")
        if not isinstance(self.rank_confidence_enabled, bool):
            raise TypeError("rank_confidence_enabled must be a bool.")
        if not isinstance(self.class_reliability_enabled, bool):
            raise TypeError("class_reliability_enabled must be a bool.")
        for model_name, param_grid in self.tune_params.items():
            if not isinstance(param_grid, dict):
                raise TypeError(f"tune_params['{model_name}'] must be a dictionary.")
            for parameter_name, values in param_grid.items():
                if not isinstance(parameter_name, str) or not parameter_name.startswith("model__"):
                    raise ValueError(
                        f"tune_params['{model_name}'] parameter names must start with 'model__'."
                    )
                if not isinstance(values, (list, tuple)) or not values:
                    raise ValueError(
                        f"tune_params['{model_name}']['{parameter_name}'] must be a non-empty list or tuple."
                    )
        if not self.probability_multiplier_grid:
            raise ValueError("probability_multiplier_grid must contain at least one value.")
        if self.spatial_interpolation_method not in {None, "nearest", "idw", "kriging"}:
            raise ValueError("spatial_interpolation_method must be None, 'nearest', 'idw', or 'kriging'.")
        if self.spatial_interpolation_method == "idw" and self.spatial_interpolation_max_distance_in_meters is None:
            raise ValueError("spatial_interpolation_max_distance_in_meters is required for idw interpolation.")
        if self.spatial_interpolation_method == "kriging" and self.spatial_interpolation_variogram_max_distance_in_meters is None:
            raise ValueError("spatial_interpolation_variogram_max_distance_in_meters is required for kriging interpolation.")
        forbidden_features = {self.id_column, self.target_column, "geometry"}
        leaked_features = sorted(forbidden_features.intersection(self.feature_columns))
        if leaked_features:
            raise ValueError(
                f"Identifiers, target, geometry and the raw time column cannot be model features. Remove: {leaked_features}"
            )

        # Acceptable values for selected models:
        # ----------------------------------------
        # None -> use all available models
        # logistic_regression
        # linear_sgd_classifier
        # random_forest
        # extra_trees
        # gradient_boosting
        # decision_tree
        # knn
        # neural_network
        # tensorflow_neural_network
        # keras_lstm
        # bagging
        # bayesian

        # Conditionally available:
        # ----------------------------------------
        # hist_gradient_boosting (only when there are no categorical features)
        # xgboost (only if xgboost is installed)
        # lightgbm (only if lightgbm is installed)
        # tensorflow_neural_network (only if tensorflow and scikeras are installed)
        # keras_lstm (only if tensorflow and scikeras are installed)
        if self.selected_models is None:
            return
        if not self.selected_models:
            raise ValueError("selected_models cannot be empty when provided.")
        if len(set(self.selected_models)) != len(self.selected_models):
            raise ValueError("selected_models contains duplicate names. Provide unique model names only.")

    # Numeric validations are centralized so range checks produce consistent
    # error messages and are easy to extend when new numeric options are added.

    def _validate_numeric_ranges(self) -> None:
        """Validate all numeric bounds and range constraints in configuration."""
        # Group validation rules to keep error messages explicit and consistent.
        numeric_validations = (
            (0.0 < float(self.test_size) < 1.0, "test_size must be a float strictly between 0 and 1."),
            (int(self.cv_folds) >= 2, "cv_folds must be >= 2."),
            (int(self.n_jobs) != 0, "n_jobs cannot be 0. Use -1 or a positive integer."),
            (int(self.max_search_candidates) >= 2, "max_search_candidates must be >= 2."),
            (self.top_voting_models is None or int(self.top_voting_models) >= 2, "top_voting_models must be >= 2 when provided."),
            (int(self.interpretability_top_models) >= 0, "interpretability_top_models must be >= 0."),
            (int(self.shap_sample_size) >= 10, "shap_sample_size must be >= 10."),
            (int(self.feature_importance_top_n) >= 1, "feature_importance_top_n must be >= 1."),
            (int(self.max_distribution_features) >= 1, "max_distribution_features must be >= 1."),
            (int(self.max_map_geometries) >= 1, "max_map_geometries must be >= 1."),
            (int(self.spatial_split_grid_size) >= 2, "spatial_split_grid_size must be >= 2."),
            (int(self.lstm_min_timesteps) >= 2, "lstm_min_timesteps must be >= 2."),
            (int(self.smote_k_neighbors) >= 1, "smote_k_neighbors must be >= 1."),
            (
                all(value > 0 for value in self.probability_multiplier_grid),
                "probability_multiplier_grid values must all be > 0.",
            ),
            (
                int(self.probability_optimization_iterations) >= 1,
                "probability_optimization_iterations must be >= 1.",
            ),
            (
                0.0 <= float(self.probability_optimization_max_accuracy_drop) < 1.0,
                "probability_optimization_max_accuracy_drop must be between 0 and 1.",
            ),
            (
                0.0 <= float(self.iqr_lower_quantile) < float(self.iqr_upper_quantile) <= 1.0,
                "IQR quantiles must satisfy 0 <= iqr_lower_quantile < iqr_upper_quantile <= 1.",
            ),
            (float(self.iqr_multiplier) > 0, "iqr_multiplier must be > 0."),
            (
                self.spatial_interpolation_max_distance_in_meters is None
                or float(self.spatial_interpolation_max_distance_in_meters) > 0,
                "spatial_interpolation_max_distance_in_meters must be positive when provided.",
            ),
            (int(self.spatial_interpolation_variogram_lags) >= 1, "spatial_interpolation_variogram_lags must be >= 1."),
            (
                self.spatial_interpolation_variogram_max_distance_in_meters is None
                or float(self.spatial_interpolation_variogram_max_distance_in_meters) > 0,
                "spatial_interpolation_variogram_max_distance_in_meters must be positive when provided.",
            ),
            (
                0.0 <= float(self.prediction_confidence_threshold) <= 1.0,
                "prediction_confidence_threshold must be between 0 and 1.",
            ),
            (int(self.rank_confidence_minimum_models) >= 1, "rank_confidence_minimum_models must be >= 1."),
            (
                int(self.class_reliability_minimum_oof_support) >= 1,
                "class_reliability_minimum_oof_support must be >= 1.",
            ),
        )
        for condition, message in numeric_validations:
            if not condition:
                raise ValueError(message)

        high_borda = float(self.rank_confidence_high_min_borda)
        medium_borda = float(self.rank_confidence_medium_min_borda)
        if not 0.0 <= medium_borda <= high_borda <= 100.0:
            raise ValueError("Rank-confidence Borda thresholds must satisfy 0 <= MEDIUM <= HIGH <= 100.")
        high_range = float(self.rank_confidence_high_max_range)
        medium_range = float(self.rank_confidence_medium_max_range)
        if high_range < 0.0 or medium_range < high_range:
            raise ValueError("Rank-confidence range thresholds must satisfy 0 <= HIGH <= MEDIUM.")
        reliability_high = float(self.class_reliability_high_min_precision)
        reliability_medium = float(self.class_reliability_medium_min_precision)
        if not 0.0 <= reliability_medium <= reliability_high <= 1.0:
            raise ValueError("Class-reliability precision thresholds must satisfy 0 <= MEDIUM <= HIGH <= 1.")

    def _apply_safe_caps(self) -> None:
        """Apply safe caps to dependent limits and emit warnings when clipped."""
        # Infer selected model count for dependent cap calculations.
        selected_model_count = len(self.selected_models) if self.selected_models is not None else None
        if self.selection_type == "soft_voting" and selected_model_count is not None and selected_model_count < 2:
            warnings.warn(
                "selection_type='soft_voting' requires at least 2 selected_models. Falling back to 'single_model'.", stacklevel=2
            )
            self.selection_type = "single_model"
        self.top_voting_models = self._cap_with_warning(
            value=self.top_voting_models,
            maximum=selected_model_count,
            field_name="top_voting_models",
            maximum_label="selected_models",
        )
        self.interpretability_top_models = self._cap_with_warning(
            value=self.interpretability_top_models,
            maximum=selected_model_count,
            field_name="interpretability_top_models",
            maximum_label="selected_models",
        )
        self.feature_importance_top_n = self._cap_with_warning(
            value=self.feature_importance_top_n,
            maximum=len(self.feature_columns),
            field_name="feature_importance_top_n",
            maximum_label="feature_columns",
        )
        confidence_model_limit = self.top_voting_models or selected_model_count
        if (
            self.rank_confidence_enabled
            and self.selection_type == "soft_voting"
            and confidence_model_limit is not None
            and confidence_model_limit < self.rank_confidence_minimum_models
        ):
            warnings.warn(
                "The configured soft-voting strategy can use fewer models than "
                "rank_confidence_minimum_models; affected predictions will receive LOW confidence.",
                stacklevel=2,
            )

    def _set_default_output_columns(self) -> None:
        """Set default output column names when explicit values are not provided."""
        # Derive prediction column names from the configured target field.
        if not self.prediction_column:
            self.prediction_column = f"{self.target_column}_prediction"
        if not self.prediction_filled_column:
            self.prediction_filled_column = f"{self.target_column}_filled"

    def _cap_with_warning(self, value: int | None, maximum: int | None, field_name: str, maximum_label: str) -> int | None:
        """Clip a numeric value to a maximum and warn when clipping occurs."""
        # Return early when clipping is not required.
        if value is None or maximum is None or value <= maximum:
            return value
        warnings.warn(
            (f"{field_name}={value} is larger than the number of {maximum_label} ({maximum}). Using {maximum} instead."),
            stacklevel=2,
        )
        return maximum

    @property
    def check_dir(self) -> Path:
        """Return the filesystem path for step 1 outputs."""
        # Step directory names are controlled centrally via `step_names`.
        return self.project_dir / self.step_names[0]

    @property
    def prepare_dir(self) -> Path:
        """Return the filesystem path for step 2 outputs."""
        # Use deterministic subfolder naming for reproducible output layouts.
        return self.project_dir / self.step_names[1]

    @property
    def train_dir(self) -> Path:
        """Return the filesystem path for step 3 outputs."""
        # Keep training artifacts isolated under their own step folder.
        return self.project_dir / self.step_names[2]

    @property
    def train_models_dir(self) -> Path:
        """Return the directory that stores per-model training artifacts."""
        # Group model-specific files under a shared models subdirectory.
        return self.train_dir / "models"

    def train_model_dir(self, model_name: str) -> Path:
        """Return the artifact directory path for a single trained model."""
        # Keep each model's outputs in its own subfolder.
        return self.train_models_dir / model_name

    @property
    def evaluate_dir(self) -> Path:
        """Return the filesystem path for step 4 outputs."""
        # Evaluation artifacts are written under the fourth pipeline step.
        return self.project_dir / self.step_names[3]

    @property
    def predict_dir(self) -> Path:
        """Return the filesystem path for step 5 outputs."""
        # Prediction artifacts are written under the fifth pipeline step.
        return self.project_dir / self.step_names[4]

    @property
    def final_dashboard_dir(self) -> Path:
        """Return the output directory for the post-evaluation dashboard."""
        return self.project_dir / "final_dashboard"

    @property
    def log_path(self) -> Path:
        """Return the absolute path to the pipeline log file."""
        # Log file path is derived from the configured project directory.
        return self.project_dir / self.log_filename

    def ensure_directories(self) -> None:
        """Create all required pipeline directories if they do not already exist."""
        # Ensure all step directories are present before writing artifacts.
        for folder in (
            self.project_dir,
            self.check_dir,
            self.prepare_dir,
            self.train_dir,
            self.train_models_dir,
            self.evaluate_dir,
            self.predict_dir,
            self.final_dashboard_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)
