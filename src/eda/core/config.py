"""Configuration for the dataframe and geodataframe EDA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class EDAConfig:
    """Runtime options for exploratory data analysis.

    The defaults are intentionally broad enough for first-pass profiling while
    keeping reports responsive on wider datasets.
    """

    output_dir: str | Path
    report_title: str = "Exploratory Data Analysis Report"
    target_column: str | None = None
    target_task: str = "auto"
    id_column: str | None = None
    max_features_per_plot: int = 12
    max_categories: int = 15
    categorical_association_max_unique_ratio: float = 0.95
    high_missing_percent_threshold: float = 40.0
    quasi_constant_threshold: float = 0.98
    max_target_levels_for_plots: int = 8
    max_multivariate_features: int = 8
    correlation_threshold: float = 0.75
    normality_sample_size: int = 5000
    outlier_zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    feature_selection_alpha: float = 0.05
    feature_selection_redundancy_threshold: float = 0.90
    feature_selection_max_features: int | None = 50
    feature_selection_exclude_columns: tuple[str, ...] = ()
    print_feature_selection_summary: bool = True
    include_html_report: bool = True
    include_plots: bool = True
    include_geospatial: bool = True
    reset_output_dir: bool = True
    random_state: int = 42
    schema_version: str = "1.5.0"
    artifact_names: dict[str, str] = field(
        default_factory=lambda: {
            "summary": "eda_summary.json",
            "column_profile": "tables/overview/column_profile.csv",
            "data_quality_flags": "tables/data_quality/data_quality_flags.csv",
            "numeric_summary": "tables/numeric/numeric_summary.csv",
            "robust_numeric_summary": "tables/numeric/robust_numeric_summary.csv",
            "categorical_summary": "tables/categorical/categorical_summary.csv",
            "datetime_summary": "tables/overview/datetime_summary.csv",
            "normality_tests": "tables/numeric/normality_tests.csv",
            "outlier_summary": "tables/numeric/outlier_summary.csv",
            "correlation_pairs": "tables/correlation/strong_correlation_pairs.csv",
            "multicollinearity": "tables/correlation/multicollinearity.csv",
            "missingness_summary": "tables/data_quality/missingness_summary.csv",
            "missingness_target_tests": "tables/target/missingness_target_tests.csv",
            "categorical_associations": "tables/categorical/categorical_associations.csv",
            "categorical_target_tests": "tables/target/categorical_target_tests.csv",
            "categorical_numeric_target_tests": "tables/target/categorical_numeric_target_tests.csv",
            "numeric_target_tests": "tables/target/numeric_target_tests.csv",
            "numeric_target_correlations": "tables/target/numeric_target_correlations.csv",
            "feature_target_associations": "tables/feature_selection/feature_target_associations.csv",
            "feature_selection_criteria": "tables/feature_selection/feature_selection_criteria.csv",
            "feature_selection_proposal": "tables/feature_selection/feature_selection_proposal.csv",
            "feature_selection_message": "tables/feature_selection/feature_selection_proposal.txt",
            "pca_summary": "tables/multivariate/pca_summary.csv",
            "multivariate_outliers": "tables/multivariate/multivariate_outliers.csv",
            "annotated_data": "eda_annotated_data.parquet",
            "target_summary": "tables/target/target_summary.csv",
            "geospatial_summary": "tables/geospatial/geospatial_summary.csv",
            "report": "report.html",
        }
    )

    def __post_init__(self) -> None:
        """Normalize and validate user-provided options."""
        self.output_dir = Path(self.output_dir)
        if self.target_task not in {"auto", "classification", "regression"}:
            raise ValueError("Error: target_task must be 'auto', 'classification', or 'regression'.")
        if self.max_features_per_plot < 1:
            raise ValueError("Error: max_features_per_plot must be >= 1.")
        if self.max_categories < 2:
            raise ValueError("Error: max_categories must be >= 2.")
        if not 0 < self.categorical_association_max_unique_ratio <= 1:
            raise ValueError("Error: categorical_association_max_unique_ratio must be between 0 and 1.")
        if not 0 <= self.high_missing_percent_threshold <= 100:
            raise ValueError("Error: high_missing_percent_threshold must be between 0 and 100.")
        if not 0 < self.quasi_constant_threshold <= 1:
            raise ValueError("Error: quasi_constant_threshold must be between 0 and 1.")
        if self.max_target_levels_for_plots < 2:
            raise ValueError("Error: max_target_levels_for_plots must be >= 2.")
        if self.max_multivariate_features < 2:
            raise ValueError("Error: max_multivariate_features must be >= 2.")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("Error: correlation_threshold must be between 0 and 1.")
        if self.normality_sample_size < 3:
            raise ValueError("Error: normality_sample_size must be >= 3.")
        if self.outlier_zscore_threshold <= 0:
            raise ValueError("Error: outlier_zscore_threshold must be > 0.")
        if self.iqr_multiplier <= 0:
            raise ValueError("Error: iqr_multiplier must be > 0.")
        if not 0 < self.feature_selection_alpha < 1:
            raise ValueError("Error: feature_selection_alpha must be between 0 and 1.")
        if not 0 <= self.feature_selection_redundancy_threshold <= 1:
            raise ValueError("Error: feature_selection_redundancy_threshold must be between 0 and 1.")
        if self.feature_selection_max_features is not None and self.feature_selection_max_features < 1:
            raise ValueError("Error: feature_selection_max_features must be >= 1 or None.")

    @property
    def plots_dir(self) -> Path:
        """Directory used for generated plot images."""
        return self.output_dir / "plots"

    @property
    def report_path(self) -> Path:
        """Path to the generated HTML report."""
        return self.output_dir / self.artifact_names["report"]
