"""Top-level orchestration entrypoint for the classification pipeline steps."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_classification.shared.logging import print_formatted_txt, time_decorator
from ml_classification.shared.reports.report_index import write_index_report
from ml_classification.step_01_check.check import CheckStep
from ml_classification.step_02_prepare.prepare import PrepareStep
from ml_classification.step_03_train.train import TrainStep
from ml_classification.step_04_evaluate.evaluate import EvaluateStep
from ml_classification.step_05_predict.predict import PredictStep

# Compose the individual step mixins here so callers can run the workflow through one object.


class GeospatialClassificationPipeline(CheckStep, PrepareStep, TrainStep, EvaluateStep, PredictStep):
    """Compose all step mixins into a geospatial-ready classification workflow."""

    @staticmethod
    def _normalize_reports_step(step: str | None) -> list[str]:
        """Normalize a step selector to one or more canonical step names."""
        step_order = ["check", "prepare", "train", "evaluate", "predict"]
        if step is None:
            return step_order

        normalized = str(step).strip().lower()
        alias_map = {
            "all": step_order,
            "*": step_order,
            "check": ["check"],
            "01_check": ["check"],
            "prepare": ["prepare"],
            "02_prepare": ["prepare"],
            "train": ["train"],
            "03_train": ["train"],
            "evaluate": ["evaluate"],
            "04_evaluate": ["evaluate"],
            "predict": ["predict"],
            "05_predict": ["predict"],
        }
        if normalized not in alias_map:
            valid = ", ".join(["all"] + step_order)
            raise ValueError(f"Error: Unknown step '{step}'. Expected one of: {valid}.")
        return alias_map[normalized]

    @time_decorator
    def create_reports(self, step: str | None = None) -> dict[str, Any]:
        """Recreate plots and HTML reports from previously persisted artifacts."""
        selected_steps = self._normalize_reports_step(step)
        print_formatted_txt("Recreating reports from persisted artifacts...", "SUBSECTION")
        outputs: dict[str, Any] = {"requested_steps": selected_steps}

        for selected_step in selected_steps:
            if selected_step == "check":
                outputs["check"] = self._create_check_reports()
            elif selected_step == "prepare":
                outputs["prepare"] = self._create_prepare_reports()
            elif selected_step == "train":
                outputs["train"] = self._create_train_reports()
            elif selected_step == "evaluate":
                outputs["evaluate"] = self._create_evaluate_reports()
            elif selected_step == "predict":
                outputs["predict"] = self._create_predict_reports()

        write_index_report(self.config)
        outputs["index"] = str(self.config.project_dir / "report_index.html")
        print_formatted_txt(f"Recreated reports for steps: {selected_steps}", "RESULTS")
        return outputs

    @time_decorator
    def run_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all pipeline steps in order and return step summaries."""
        # Execute the end-to-end pipeline workflow.
        print_formatted_txt("ML Classification Pipeline...", "SECTION")
        return {
            "check": self.run_check(df),
            "prepare": self.run_prepare(),
            "train": self.run_train(),
            "evaluate": self.run_evaluate(),
            "predict": self.run_predict(),
        }
