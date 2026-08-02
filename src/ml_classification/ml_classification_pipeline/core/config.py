"""Configuration object defining runtime options and validation rules for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings



@dataclass(slots=True)
class ClassificationPipelineConfig:
    """Store validated configuration values used by all pipeline steps."""

    project_dir: str | Path
    target_column: str
    feature_columns: list[str]
    selected_models: tuple[str, ...] | None = None
    reset_project_dir_on_run_check: bool = True
    fail_on_cleanup_error: bool = False
    id_column: str = "row_id"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    scoring_primary: str = "f1_macro"
    scoring_secondary: tuple[str, ...] = ("balanced_accuracy", "f1_weighted", "accuracy")
    selection_type: str = "soft_voting"
    cv_ranking_method: str = "score_minus_std"
    rare_category_min_frequency: int | float = 10
    max_search_candidates: int = 24
    top_voting_models: int | None = None
    interpretability_top_models: int = 3
    interpretability_include_shap: bool = False
    shap_sample_size: int = 500
    feature_importance_top_n: int = 15
    max_distribution_features: int = 9
    max_map_geometries: int = 20000
    float_dtype: str = "float32"
    spatial_split: bool = False
    spatial_split_grid_size: int = 10
    label_balancing_method: str = "none"
    smote_k_neighbors: int = 5
    prediction_column: str | None = None
    prediction_filled_column: str | None = None
    probability_prefix: str = "probability"
    output_schema_version: str = "1.0.0"
    log_filename: str = "pipeline.log"
    step_names: tuple[str, ...] = field(
        default=("01_check", "02_prepare", "03_train", "04_evaluate", "05_predict"),
    )
    open_html_report: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate configuration right after dataclass initialization."""
        # Convert user path input into a `Path` for consistent filesystem handling.
        self.project_dir = Path(self.project_dir)
        self._normalize_inputs()
        self._validate_strings_and_sequences()
        self._validate_numeric_ranges()
        self._apply_safe_caps()
        self._set_default_output_columns()

    def _normalize_inputs(self) -> None:
        """Normalize configurable string and sequence inputs to canonical forms."""
        # Standardize model selection type values before validation.
        if self.selected_models is not None:
            self.selected_models = tuple(self.selected_models)
        self.selection_type = self.selection_type.strip().lower()
        self.cv_ranking_method = self.cv_ranking_method.strip().lower()
        self.label_balancing_method = self.label_balancing_method.strip().lower()

    def _validate_strings_and_sequences(self) -> None:
        """Validate categorical options and sequence-based configuration fields."""
        # Enforce supported voting/selection strategies.
        if self.selection_type not in {"soft_voting", "single_model"}:
            raise ValueError("selection_type must be either 'soft_voting' or 'single_model'.")
        if self.cv_ranking_method not in {"mean_score", "score_minus_std"}:
            raise ValueError("cv_ranking_method must be either 'mean_score' or 'score_minus_std'.")
        if self.label_balancing_method not in {"none", "random_oversample", "smote"}:
            raise ValueError(
                "label_balancing_method must be one of: 'none', 'random_oversample', or 'smote'."
            )
        if not self.feature_columns:
            raise ValueError("feature_columns must contain at least one feature name.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns contains duplicate names. Provide unique feature names only.")

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
        # bagging
        # bayesian

        # Conditionally available:
        # ----------------------------------------
        # hist_gradient_boosting (only when there are no categorical features)
        # xgboost (only if xgboost is installed)
        # lightgbm (only if lightgbm is installed)
        if self.selected_models is None:
            return
        if not self.selected_models:
            raise ValueError("selected_models cannot be empty when provided.")
        if len(set(self.selected_models)) != len(self.selected_models):
            raise ValueError("selected_models contains duplicate names. Provide unique model names only.")

    def _validate_numeric_ranges(self) -> None:
        """Validate all numeric bounds and range constraints in configuration."""
        # Group validation rules to keep error messages explicit and consistent.
        numeric_validations = (
            (0.0 < float(self.test_size) < 1.0, "test_size must be a float strictly between 0 and 1."),
            (int(self.cv_folds) >= 2, "cv_folds must be >= 2."),
            (int(self.n_jobs) != 0, "n_jobs cannot be 0. Use -1 or a positive integer."),
            (int(self.max_search_candidates) >= 2, "max_search_candidates must be >= 2."),
            (
                self.top_voting_models is None or int(self.top_voting_models) >= 2,
                "top_voting_models must be >= 2 when provided.",
            ),
            (int(self.interpretability_top_models) >= 0, "interpretability_top_models must be >= 0."),
            (int(self.shap_sample_size) >= 10, "shap_sample_size must be >= 10."),
            (int(self.feature_importance_top_n) >= 1, "feature_importance_top_n must be >= 1."),
            (int(self.max_distribution_features) >= 1, "max_distribution_features must be >= 1."),
            (int(self.max_map_geometries) >= 1, "max_map_geometries must be >= 1."),
            (int(self.spatial_split_grid_size) >= 2, "spatial_split_grid_size must be >= 2."),
            (int(self.smote_k_neighbors) >= 1, "smote_k_neighbors must be >= 1."),
        )
        for condition, message in numeric_validations:
            if not condition:
                raise ValueError(message)

    def _apply_safe_caps(self) -> None:
        """Apply safe caps to dependent limits and emit warnings when clipped."""
        # Infer selected model count for dependent cap calculations.
        selected_model_count = len(self.selected_models) if self.selected_models is not None else None
        if self.selection_type == "soft_voting" and selected_model_count is not None and selected_model_count < 2:
            warnings.warn(
                "selection_type='soft_voting' requires at least 2 selected_models. Falling back to 'single_model'.",
                stacklevel=2,
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

    def _set_default_output_columns(self) -> None:
        """Set default output column names when explicit values are not provided."""
        # Derive prediction column names from the configured target field.
        if not self.prediction_column:
            self.prediction_column = f"{self.target_column}_prediction"
        if not self.prediction_filled_column:
            self.prediction_filled_column = f"{self.target_column}_filled"

    def _cap_with_warning(
        self,
        value: int | None,
        maximum: int | None,
        field_name: str,
        maximum_label: str,
    ) -> int | None:
        """Clip a numeric value to a maximum and warn when clipping occurs."""
        # Return early when clipping is not required.
        if value is None or maximum is None or value <= maximum:
            return value
        warnings.warn(
            (
                f"{field_name}={value} is larger than the number of {maximum_label} "
                f"({maximum}). Using {maximum} instead."
            ),
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
        ):
            folder.mkdir(parents=True, exist_ok=True)
