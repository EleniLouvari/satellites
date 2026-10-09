"""Run multi-seed sensitivity experiments and generate an HTML summary report."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ml_classification.pipeline import GeospatialClassificationPipeline
from ml_classification.sensitivity.libraries.plots import create_accuracy_plot
from ml_classification.sensitivity.libraries.report import write_html_report
from ml_classification.sensitivity.libraries.results import (
    SensitivityResult,
    extract_model_accuracy,
    read_test_metrics,
    resolve_selected_model,
    results_to_frame,
    write_results_csv,
)
from ml_classification.shared.config.config import ClassificationPipelineConfig


class ClassificationSensitivityRunner:
    """Execute `GeospatialClassificationPipeline` across random seeds.

    The runner executes the full pipeline for each seed, captures test accuracy,
    and writes three outputs into ``output_dir``:
    1) ``sensitivity_results.csv``
    2) ``accuracy_by_seed.png``
    3) ``sensitivity_report.html``
    """

    def __init__(self, output_dir: str | Path):
        """Initialize the runner.

        Args:
            output_dir: Folder where summary artifacts are written.
        """
        # Normalize and create the destination folder for all sensitivity artifacts.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, gdf: pd.DataFrame, seeds: list[int], config_builder: Callable[[int], ClassificationPipelineConfig]
    ) -> pd.DataFrame:
        """Run the full sensitivity workflow in one call.

        Args:
            gdf: Input GeoDataFrame/DataFrame used by each pipeline run.
            seeds: Random seed values to evaluate.
            config_builder: Callback that returns a ``ClassificationPipelineConfig``
                for a given seed.

        Returns:
            A dataframe with one row per seed run.
        """
        # Step 1: run all seed experiments and persist CSV.
        results_df = self.run_seeds(gdf=gdf, seeds=seeds, config_builder=config_builder, write_csv=True)

        # Step 2: create plot and HTML report from the aggregated table.
        self.create_report(results_df)
        return results_df

    def run_seed(
        self, gdf: pd.DataFrame, seed: int, config_builder: Callable[[int], ClassificationPipelineConfig]
    ) -> SensitivityResult:
        """Run the pipeline for one seed and return a structured result.

        Args:
            gdf: Input GeoDataFrame/DataFrame used by the pipeline.
            seed: Random seed value for this execution.
            config_builder: Callback that returns a seed-specific config.

        Returns:
            A single ``SensitivityResult`` with success flag and metrics/error.
        """
        try:
            print(f"Running seed: {seed}...")
            config = config_builder(int(seed))
            pipeline = GeospatialClassificationPipeline(config)

            # Execute all pipeline steps for this seed.
            pipeline.run_check(gdf)
            pipeline.run_prepare()
            pipeline.run_train()
            selection_summary = pipeline.run_evaluate()
            pipeline.run_predict()

            # Resolve the selected model and extract its test accuracy.
            selected_model = self._resolve_selected_model(selection_summary)
            test_metrics = self._read_test_metrics(config)
            test_accuracy = self._extract_model_accuracy(test_metrics, selected_model)

            return SensitivityResult(
                seed=int(seed),
                success=True,
                selected_model=selected_model,
                test_accuracy=test_accuracy,
                project_dir=str(config.project_dir),
                error=None,
            )
        except Exception as exc:
            logging.getLogger(__name__).debug("Error: run_seed failed; using its fallback.", exc_info=True)
            # Return a failed row instead of raising, so outer loops can continue.
            return SensitivityResult(
                seed=int(seed),
                success=False,
                selected_model=None,
                test_accuracy=None,
                project_dir="",
                error=f"{exc.__class__.__name__}: {exc}",
            )

    def run_seeds(
        self,
        gdf: pd.DataFrame,
        seeds: list[int],
        config_builder: Callable[[int], ClassificationPipelineConfig],
        write_csv: bool = True,
    ) -> pd.DataFrame:
        """Run multiple seeds and optionally persist only the CSV summary.

        Args:
            gdf: Input GeoDataFrame/DataFrame used by each pipeline run.
            seeds: Random seed values to evaluate.
            config_builder: Callback that returns a seed-specific config.
            write_csv: Whether to write ``sensitivity_results.csv``.

        Returns:
            A dataframe with one row per seed run.
        """
        # Collect per-seed outcomes, continuing even if one run fails.
        results = [self.run_seed(gdf=gdf, seed=int(seed), config_builder=config_builder) for seed in seeds]
        results_df = self._results_to_frame(results)
        if write_csv:
            # Persist intermediate tabular results for step-wise workflows.
            self._write_results_csv(results_df)
        return results_df

    def create_report(self, results_df: pd.DataFrame, write_csv: bool = False) -> dict[str, str]:
        """Generate plot/report artifacts from an existing results dataframe.

        Args:
            results_df: Aggregated sensitivity results dataframe.
            write_csv: Whether to also rewrite ``sensitivity_results.csv``.

        Returns:
            Dictionary with artifact paths for ``csv_path``, ``plot_path``, and
            ``report_path`` (empty string when plot is not generated).
        """
        # Optional CSV write allows calling this method independently.
        csv_path = self._write_results_csv(results_df) if write_csv else (self.output_dir / "sensitivity_results.csv")
        plot_path = self._create_accuracy_plot(results_df)
        report_path = self._write_html_report(results_df, plot_path)
        return {
            "csv_path": str(csv_path),
            "plot_path": "" if plot_path is None else str(plot_path),
            "report_path": str(report_path),
        }

    def _resolve_selected_model(self, selection_summary: dict) -> str:
        """Resolve which model name should be searched in test metrics.

        Args:
            selection_summary: Selection metadata returned by ``run_evaluate()``.

        Returns:
            The model name expected in the evaluation metrics table.
        """
        return resolve_selected_model(selection_summary)

    def _read_test_metrics(self, config: ClassificationPipelineConfig) -> pd.DataFrame:
        """Load the test-metrics dataframe generated by the evaluate step.

        Args:
            config: Seed-specific pipeline configuration.

        Returns:
            A dataframe with test metrics for model candidates.
        """
        return read_test_metrics(config)

    def _extract_model_accuracy(self, test_metrics: pd.DataFrame, model_name: str) -> float:
        """Extract test accuracy for a target model name.

        Args:
            test_metrics: Evaluation metrics table.
            model_name: Name of the model to locate.

        Returns:
            The model accuracy as a float.
        """
        return extract_model_accuracy(test_metrics, model_name)

    def _results_to_frame(self, results: list[SensitivityResult]) -> pd.DataFrame:
        """Convert dataclass results into a stable, sorted dataframe.

        Args:
            results: In-memory seed-run records.

        Returns:
            A dataframe with deterministic column order.
        """
        return results_to_frame(results)

    def _write_results_csv(self, results_df: pd.DataFrame) -> Path:
        """Write the per-seed result table to CSV.

        Args:
            results_df: Aggregated sensitivity results dataframe.

        Returns:
            Path to the written CSV file.
        """
        return write_results_csv(self.output_dir, results_df)

    def _create_accuracy_plot(self, results_df: pd.DataFrame) -> Path | None:
        """Create and save an accuracy-versus-seed plot.

        Args:
            results_df: Aggregated sensitivity results dataframe.

        Returns:
            Path to the saved plot, or ``None`` when no successful rows exist.
        """
        return create_accuracy_plot(self.output_dir, results_df)

    def _write_html_report(self, results_df: pd.DataFrame, plot_path: Path | None) -> Path:
        """Generate an HTML summary report for sensitivity analysis.

        Args:
            results_df: Aggregated sensitivity results dataframe.
            plot_path: Optional plot image path.

        Returns:
            Path to the generated HTML report.
        """
        return write_html_report(self.output_dir, results_df, plot_path)
