"""Run multi-seed sensitivity experiments and generate an HTML summary report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_classification.ml_classification_pipeline.core.config import ClassificationPipelineConfig
from ml_classification.ml_classification_pipeline.pipeline import GeospatialClassificationPipeline


@dataclass(slots=True)
class SensitivityResult:
    """Store one seed run outcome.

    Attributes:
        seed: Random seed used for the pipeline run.
        success: Whether the run completed successfully.
        selected_model: Final selected model name (or ``soft_voting``).
        test_accuracy: Accuracy measured on the test split for the selected model.
        project_dir: Project directory used by the seed-specific run.
        error: Short exception text when a run fails.
    """

    seed: int
    success: bool
    selected_model: str | None
    test_accuracy: float | None
    project_dir: str
    error: str | None = None


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
        self,
        gdf: pd.DataFrame,
        seeds: list[int],
        config_builder: Callable[[int], ClassificationPipelineConfig],
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
        results_df = self.run_seeds(
            gdf=gdf,
            seeds=seeds,
            config_builder=config_builder,
            write_csv=True,
        )

        # Step 2: create plot and HTML report from the aggregated table.
        self.create_report(results_df)
        return results_df

    def run_seed(
        self,
        gdf: pd.DataFrame,
        seed: int,
        config_builder: Callable[[int], ClassificationPipelineConfig],
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
        # Convert soft-voting selection into the table model name convention.
        if selection_summary.get("selection_type") == "soft_voting":
            return "soft_voting"
        selected_models = selection_summary.get("selected_models", [])
        if not selected_models:
            raise ValueError("No selected model was returned by run_evaluate().")
        return str(selected_models[0])

    def _read_test_metrics(self, config: ClassificationPipelineConfig) -> pd.DataFrame:
        """Load the test-metrics dataframe generated by the evaluate step.

        Args:
            config: Seed-specific pipeline configuration.

        Returns:
            A dataframe with test metrics for model candidates.
        """
        # Prefer file that includes soft voting rows when available.
        preferred = config.evaluate_dir / "model_metrics_test_with_voting.csv"
        fallback = config.evaluate_dir / "model_metrics_test.csv"
        metrics_path = preferred if preferred.exists() else fallback
        if not metrics_path.exists():
            raise FileNotFoundError(f"Could not find test metrics CSV under: {config.evaluate_dir}")
        return pd.read_csv(metrics_path)

    def _extract_model_accuracy(self, test_metrics: pd.DataFrame, model_name: str) -> float:
        """Extract test accuracy for a target model name.

        Args:
            test_metrics: Evaluation metrics table.
            model_name: Name of the model to locate.

        Returns:
            The model accuracy as a float.
        """
        # Match the model row and return accuracy as float.
        model_row = test_metrics.loc[test_metrics["model"].astype(str) == str(model_name)]
        if model_row.empty:
            available = sorted(test_metrics["model"].astype(str).unique().tolist())
            raise ValueError(f"Model '{model_name}' not found in test metrics. Available: {available}")
        return float(model_row.iloc[0]["accuracy"])

    def _results_to_frame(self, results: list[SensitivityResult]) -> pd.DataFrame:
        """Convert dataclass results into a stable, sorted dataframe.

        Args:
            results: In-memory seed-run records.

        Returns:
            A dataframe with deterministic column order.
        """
        # Keep column order stable for easier report consumption.
        frame = pd.DataFrame([asdict(result) for result in results])
        if frame.empty:
            return pd.DataFrame(columns=["seed", "success", "selected_model", "test_accuracy", "project_dir", "error"])
        return frame[["seed", "success", "selected_model", "test_accuracy", "project_dir", "error"]].sort_values("seed")

    def _write_results_csv(self, results_df: pd.DataFrame) -> Path:
        """Write the per-seed result table to CSV.

        Args:
            results_df: Aggregated sensitivity results dataframe.

        Returns:
            Path to the written CSV file.
        """
        # Save machine-readable summary for later analysis.
        csv_path = self.output_dir / "sensitivity_results.csv"
        results_df.to_csv(csv_path, index=False)
        return csv_path

    def _create_accuracy_plot(self, results_df: pd.DataFrame) -> Path | None:
        """Create and save an accuracy-versus-seed plot.

        Args:
            results_df: Aggregated sensitivity results dataframe.

        Returns:
            Path to the saved plot, or ``None`` when no successful rows exist.
        """
        # Plot only successful runs with non-null accuracy values.
        plot_df = results_df.loc[results_df["success"] & results_df["test_accuracy"].notna(), ["seed", "test_accuracy"]].copy()
        if plot_df.empty:
            # No successful metrics available, so a plot cannot be generated.
            return None

        plot_df.sort_values("seed", inplace=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(plot_df["seed"], plot_df["test_accuracy"], marker="o", linewidth=2, color="#1f77b4")
        ax.set_title("Sensitivity Analysis: Test Accuracy by Random Seed")
        ax.set_xlabel("Random Seed")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)

        mean_acc = float(plot_df["test_accuracy"].mean())
        std_acc = float(plot_df["test_accuracy"].std(ddof=0))

        # Draw mean, ±1σ, and ±2σ reference lines for quick visual spread checks.
        ax.axhline(mean_acc, linestyle="--", linewidth=1.6, color="#d62728", label=f"mean={mean_acc:.4f}")
        if std_acc > 0:
            ax.axhline(mean_acc + std_acc, linestyle=":", linewidth=1.2, color="#ff7f0e", label="+1σ")
            ax.axhline(mean_acc - std_acc, linestyle=":", linewidth=1.2, color="#ff7f0e", label="-1σ")
            ax.axhline(mean_acc + (2 * std_acc), linestyle="-.", linewidth=1.1, color="#2ca02c", label="+2σ")
            ax.axhline(mean_acc - (2 * std_acc), linestyle="-.", linewidth=1.1, color="#2ca02c", label="-2σ")

        ax.legend(loc="best")
        fig.tight_layout()

        plot_path = self.output_dir / "accuracy_by_seed.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

        # Also persist basic aggregate statistics.
        stats_df = pd.DataFrame(
            [
                {"metric": "accuracy_mean", "value": mean_acc},
                {"metric": "accuracy_std", "value": std_acc},
                {"metric": "accuracy_plus_1std", "value": mean_acc + std_acc},
                {"metric": "accuracy_minus_1std", "value": mean_acc - std_acc},
                {"metric": "accuracy_plus_2std", "value": mean_acc + (2 * std_acc)},
                {"metric": "accuracy_minus_2std", "value": mean_acc - (2 * std_acc)},
            ]
        )
        stats_df.to_csv(self.output_dir / "accuracy_summary_stats.csv", index=False)
        return plot_path

    def _write_html_report(self, results_df: pd.DataFrame, plot_path: Path | None) -> Path:
        """Generate an HTML summary report for sensitivity analysis.

        Args:
            results_df: Aggregated sensitivity results dataframe.
            plot_path: Optional plot image path.

        Returns:
            Path to the generated HTML report.
        """
        # Prepare aggregates for successful runs only.
        success_df = results_df.loc[results_df["success"] & results_df["test_accuracy"].notna()].copy()
        run_count = int(len(results_df))
        success_count = int(len(success_df))
        failure_count = int(run_count - success_count)
        mean_acc = float(success_df["test_accuracy"].mean()) if not success_df.empty else float("nan")
        std_acc = float(success_df["test_accuracy"].std(ddof=0)) if not success_df.empty else float("nan")

        # Build table rows with escaped content.
        table_rows = []
        for _, row in results_df.sort_values("seed").iterrows():
            accuracy_text = "" if pd.isna(row["test_accuracy"]) else f"{float(row['test_accuracy']):.4f}"
            table_rows.append(
                "<tr>"
                f"<td>{int(row['seed'])}</td>"
                f"<td>{'✅' if bool(row['success']) else '❌'}</td>"
                f"<td>{html.escape(str(row['selected_model'])) if pd.notna(row['selected_model']) else ''}</td>"
                f"<td>{accuracy_text}</td>"
                f"<td>{html.escape(str(row['error'])) if pd.notna(row['error']) else ''}</td>"
                "</tr>"
            )

        img_html = ""
        if plot_path is not None and plot_path.exists():
            # Use a relative image reference so HTML remains portable with the folder.
            img_html = (
                "<h2>Accuracy Plot</h2>"
                f"<img src='{html.escape(plot_path.name)}' alt='Accuracy by seed' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;'>"
            )

        html_text = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>ML Classification Sensitivity Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .meta {{ padding: 12px; background: #f3f4f6; border-radius: 8px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f9fafb; }}
    .muted {{ color: #6b7280; }}
  </style>
</head>
<body>
  <h1>ML Classification Sensitivity Report</h1>
  <p class='muted'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

  <div class='meta'>
    <b>Total runs:</b> {run_count} &nbsp; | &nbsp;
    <b>Successful:</b> {success_count} &nbsp; | &nbsp;
    <b>Failed:</b> {failure_count} &nbsp; | &nbsp;
    <b>Accuracy mean:</b> {'' if np.isnan(mean_acc) else f'{mean_acc:.4f}'} &nbsp; | &nbsp;
    <b>Accuracy std:</b> {'' if np.isnan(std_acc) else f'{std_acc:.4f}'}
  </div>

  {img_html}

  <h2>Per-seed Results</h2>
  <table>
    <thead>
      <tr>
        <th>Seed</th><th>Success</th><th>Selected Model</th><th>Test Accuracy</th><th>Error</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>"""

        report_path = self.output_dir / "sensitivity_report.html"
        report_path.write_text(html_text, encoding="utf-8")
        return report_path
