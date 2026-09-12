"""End-to-end EDA pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .core import EDAConfig, build_eda_artifacts
from .core.io import ensure_dir, reset_dir, save_frame, save_json, save_parquet_frame, save_text
from .reporting import write_eda_html_report
from .visuals import create_eda_plots


class EDAPipeline:
    """Create statistical EDA artifacts for a DataFrame or GeoDataFrame."""

    def __init__(self, config: EDAConfig):
        self.config = config

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run the EDA pipeline and write artifacts to disk."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("EDAPipeline.run expects a pandas DataFrame or GeoPandas GeoDataFrame.")
        if self.config.reset_output_dir:
            reset_dir(self.config.output_dir)
        else:
            ensure_dir(self.config.output_dir)
        ensure_dir(self.config.plots_dir)

        artifacts = build_eda_artifacts(df, self.config)
        if self.config.print_feature_selection_summary and self.config.target_column:
            print(artifacts["feature_selection_message"])
        plot_paths = create_eda_plots(df, artifacts, self.config)
        saved_paths = self._save_artifacts(artifacts)

        report_path: Path | None = None
        if self.config.include_html_report:
            report_path = write_eda_html_report(artifacts, plot_paths, self.config)
            saved_paths["html_report"] = report_path

        return {
            "summary": artifacts["summary"],
            "artifacts": artifacts,
            "plot_paths": plot_paths,
            "saved_paths": saved_paths,
            "report_path": report_path,
            "annotated_data": artifacts["annotated_data"],
            "proposed_features": artifacts["feature_selection_proposal"].loc[
                artifacts["feature_selection_proposal"]["proposed_action"] == "use", "feature"
            ].tolist(),
        }

    def _save_artifacts(self, artifacts: dict[str, Any]) -> dict[str, Path]:
        """Persist the pipeline artifacts using configured filenames."""
        paths: dict[str, Path] = {}
        names = self.config.artifact_names
        paths["summary"] = save_json(artifacts["summary"], self.config.output_dir / names["summary"])
        csv_artifacts = {
            "column_profile": artifacts["column_profile"],
            "data_quality_flags": artifacts["data_quality_flags"],
            "numeric_summary": artifacts["numeric_summary"],
            "robust_numeric_summary": artifacts["robust_numeric_summary"],
            "categorical_summary": artifacts["categorical_summary"],
            "datetime_summary": artifacts["datetime_summary"],
            "normality_tests": artifacts["normality_tests"],
            "outlier_summary": artifacts["outlier_summary"],
            "correlation_pairs": artifacts["strong_correlation_pairs"],
            "multicollinearity": artifacts["multicollinearity"],
            "missingness_summary": artifacts["missingness_summary"],
            "missingness_target_tests": artifacts["missingness_target_tests"],
            "categorical_associations": artifacts["categorical_associations"],
            "categorical_target_tests": artifacts["categorical_target_tests"],
            "categorical_numeric_target_tests": artifacts["categorical_numeric_target_tests"],
            "numeric_target_tests": artifacts["numeric_target_tests"],
            "numeric_target_correlations": artifacts["numeric_target_correlations"],
            "feature_target_associations": artifacts["feature_target_associations"],
            "feature_selection_criteria": artifacts["feature_selection_criteria"],
            "feature_selection_proposal": artifacts["feature_selection_proposal"],
            "pca_summary": artifacts["pca_summary"],
            "multivariate_outliers": artifacts["multivariate_outliers"],
            "target_summary": artifacts["target_summary"],
            "geospatial_summary": artifacts["geospatial_summary"],
        }
        for key, table in csv_artifacts.items():
            paths[key] = save_frame(table, self.config.output_dir / names[key])
        paths["feature_selection_message"] = save_text(
            artifacts["feature_selection_message"], self.config.output_dir / names["feature_selection_message"]
        )
        paths["annotated_data"] = save_parquet_frame(
            artifacts["annotated_data"], self.config.output_dir / names["annotated_data"]
        )
        for method, matrix in artifacts["correlation_matrices"].items():
            if not matrix.empty:
                matrix_path = self.config.output_dir / "tables" / "correlation" / f"{method}_correlation_matrix.csv"
                paths[f"{method}_correlation_matrix"] = save_frame(
                    matrix.reset_index(names="column"), matrix_path
                )
        return paths


def run_eda(
    df: pd.DataFrame,
    output_dir: str | Path,
    target_column: str | None = None,
    report_title: str = "Exploratory Data Analysis Report",
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience function for one-call EDA execution."""
    config = EDAConfig(output_dir=output_dir, target_column=target_column, report_title=report_title, **kwargs)
    return EDAPipeline(config).run(df)
