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
    id_column: str | None = None
    max_columns_for_plots: int = 12
    max_categories: int = 15
    categorical_association_max_unique_ratio: float = 0.95
    high_missing_percent_threshold: float = 40.0
    quasi_constant_threshold: float = 0.98
    max_target_scatter_features: int = 6
    max_multivariate_features: int = 8
    max_pairplot_features: int = 10
    correlation_threshold: float = 0.75
    normality_sample_size: int = 5000
    outlier_zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    include_html_report: bool = True
    include_plots: bool = True
    include_geospatial: bool = True
    reset_output_dir: bool = True
    random_state: int = 42
    schema_version: str = "1.0.0"
    artifact_names: dict[str, str] = field(
        default_factory=lambda: {
            "summary": "eda_summary.json",
            "column_profile": "column_profile.csv",
            "data_quality_flags": "data_quality_flags.csv",
            "numeric_summary": "numeric_summary.csv",
            "robust_numeric_summary": "robust_numeric_summary.csv",
            "categorical_summary": "categorical_summary.csv",
            "datetime_summary": "datetime_summary.csv",
            "normality_tests": "normality_tests.csv",
            "outlier_summary": "outlier_summary.csv",
            "correlation_pairs": "strong_correlation_pairs.csv",
            "multicollinearity": "multicollinearity.csv",
            "missingness_summary": "missingness_summary.csv",
            "missingness_target_tests": "missingness_target_tests.csv",
            "categorical_associations": "categorical_associations.csv",
            "categorical_target_tests": "categorical_target_tests.csv",
            "numeric_target_tests": "numeric_target_tests.csv",
            "numeric_target_correlations": "numeric_target_correlations.csv",
            "pca_summary": "pca_summary.csv",
            "multivariate_outliers": "multivariate_outliers.csv",
            "target_summary": "target_summary.csv",
            "geospatial_summary": "geospatial_summary.csv",
            "report": "report.html",
        }
    )

    def __post_init__(self) -> None:
        """Normalize and validate user-provided options."""
        self.output_dir = Path(self.output_dir)
        if self.max_columns_for_plots < 1:
            raise ValueError("max_columns_for_plots must be >= 1.")
        if self.max_categories < 2:
            raise ValueError("max_categories must be >= 2.")
        if not 0 < self.categorical_association_max_unique_ratio <= 1:
            raise ValueError("categorical_association_max_unique_ratio must be between 0 and 1.")
        if not 0 <= self.high_missing_percent_threshold <= 100:
            raise ValueError("high_missing_percent_threshold must be between 0 and 100.")
        if not 0 < self.quasi_constant_threshold <= 1:
            raise ValueError("quasi_constant_threshold must be between 0 and 1.")
        if self.max_target_scatter_features < 1:
            raise ValueError("max_target_scatter_features must be >= 1.")
        if self.max_multivariate_features < 2:
            raise ValueError("max_multivariate_features must be >= 2.")
        if self.max_pairplot_features < 2:
            raise ValueError("max_pairplot_features must be >= 2.")
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1.")
        if self.normality_sample_size < 3:
            raise ValueError("normality_sample_size must be >= 3.")
        if self.outlier_zscore_threshold <= 0:
            raise ValueError("outlier_zscore_threshold must be > 0.")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be > 0.")

    @property
    def plots_dir(self) -> Path:
        """Directory used for generated plot images."""
        return self.output_dir / "plots"

    @property
    def report_path(self) -> Path:
        """Path to the generated HTML report."""
        return self.output_dir / self.artifact_names["report"]
