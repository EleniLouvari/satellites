"""Top-level orchestration entrypoint for the classification pipeline steps."""

from __future__ import annotations

from typing import Any

import pandas as pd
from satellites.shared.io import read_data

from .core.persistence import load_joblib, load_json, print_formatted_txt, time_decorator
from .reporting import (
    write_check_report,
    write_evaluate_report,
    write_index_report,
    write_predict_report,
    write_prepare_report,
    write_train_report,
)
from .reporting.final_dashboard import write_pipeline_final_dashboard
from .steps import CheckStep, EvaluateStep, PredictStep, PrepareStep, TrainStep
from .visuals import save_cv_fold_comparison_plot, save_model_comparison_plot

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
            raise ValueError(f"Unknown step '{step}'. Expected one of: {valid}.")
        return alias_map[normalized]

    @staticmethod
    def _read_optional_csv(path) -> pd.DataFrame | None:
        """Read a CSV artifact when present, otherwise return None."""
        if not path.exists():
            return None
        return read_data(str(path), watch_curly_brackets=False)

    @staticmethod
    def _ensure_artifact(path, message: str) -> None:
        """Raise a clear error when a required artifact is missing."""
        if not path.exists():
            raise FileNotFoundError(f"{message} Missing artifact: {path}")

    def _create_check_reports(self) -> str:
        """Regenerate check-step plots and HTML report from persisted artifacts."""
        dataset_path = self.config.check_dir / "input_dataset.joblib"
        summary_path = self.config.check_dir / "check_summary.json"
        feature_profile_path = self.config.check_dir / "feature_profile.csv"
        self._ensure_artifact(dataset_path, "Cannot recreate check reports.")
        self._ensure_artifact(summary_path, "Cannot recreate check reports.")
        self._ensure_artifact(feature_profile_path, "Cannot recreate check reports.")

        dataset = load_joblib(dataset_path)
        summary = load_json(summary_path)
        feature_profile = read_data(str(feature_profile_path), watch_curly_brackets=False)
        requested_features = [name for name in self.config.feature_columns if name in dataset.columns]
        if not requested_features:
            requested_features = list(summary.get("active_features", []))
        labeled_df = dataset.loc[dataset[self.config.target_column].notna()].copy()
        self._save_check_plots(dataset, labeled_df, requested_features)
        write_check_report(self.config, summary, feature_profile)
        return str(self.config.check_dir / "report.html")

    def _create_prepare_reports(self) -> str:
        """Regenerate prepare-step plots and HTML report from persisted artifacts."""
        prepare_summary_path = self.config.prepare_dir / "prepare_summary.json"
        train_path = self.config.prepare_dir / "train_dataset.joblib"
        test_path = self.config.prepare_dir / "test_dataset.joblib"
        self._ensure_artifact(prepare_summary_path, "Cannot recreate prepare reports.")
        self._ensure_artifact(train_path, "Cannot recreate prepare reports.")
        self._ensure_artifact(test_path, "Cannot recreate prepare reports.")

        prepare_summary = load_json(prepare_summary_path)
        train_df = load_joblib(train_path)
        test_df = load_joblib(test_path)
        active_features = list(prepare_summary.get("active_features", []))
        self._save_prepare_plots(train_df, test_df, active_features)
        write_prepare_report(self.config, prepare_summary, train_df, test_df)
        return str(self.config.prepare_dir / "report.html")

    def _create_train_reports(self) -> str:
        """Regenerate train-step plots and HTML report from persisted artifacts."""
        summary_path = self.config.train_dir / "training_summary.csv"
        specs_path = self.config.train_dir / "model_specs.json"
        failed_path = self.config.train_dir / "training_failed_models.csv"
        self._ensure_artifact(summary_path, "Cannot recreate train reports.")
        self._ensure_artifact(specs_path, "Cannot recreate train reports.")

        training_summary_df = read_data(str(summary_path), watch_curly_brackets=False)
        model_specs = load_json(specs_path)
        failed_models_df = (
            read_data(str(failed_path), watch_curly_brackets=False)
            if failed_path.exists()
            else pd.DataFrame(columns=["model", "error_type", "error_message"])
        )
        fold_scores_path = self.config.train_dir / "best_cv_fold_scores.csv"
        if fold_scores_path.exists():
            fold_scores_df = read_data(str(fold_scores_path), watch_curly_brackets=False)
            save_cv_fold_comparison_plot(
                fold_scores_df,
                self.config.train_dir / "plots" / "best_cv_fold_scores.png",
                self.config.scoring_primary,
            )
        write_train_report(self.config, training_summary_df, model_specs, failed_models_df)
        return str(self.config.train_dir / "report.html")

    def _create_evaluate_reports(self) -> dict[str, str]:
        """Regenerate evaluate-step plots, HTML report, and final dashboard."""
        selection_path = self.config.evaluate_dir / "selection_summary.json"
        train_metrics_path = self.config.evaluate_dir / "model_metrics_train.csv"
        test_metrics_path = self.config.evaluate_dir / "model_metrics_test.csv"
        self._ensure_artifact(selection_path, "Cannot recreate evaluate reports.")
        self._ensure_artifact(train_metrics_path, "Cannot recreate evaluate reports.")
        self._ensure_artifact(test_metrics_path, "Cannot recreate evaluate reports.")

        selection = load_json(selection_path)
        train_metrics_df = read_data(str(train_metrics_path), watch_curly_brackets=False)
        test_metrics_df = read_data(str(test_metrics_path), watch_curly_brackets=False)
        geo_train_df = self._read_optional_csv(self.config.evaluate_dir / "geo_classifier_style_train_metrics.csv")
        geo_test_df = self._read_optional_csv(self.config.evaluate_dir / "geo_classifier_style_test_metrics.csv")
        ranking_metrics_df = self._read_optional_csv(self.config.evaluate_dir / "ranking" / "ranking_method_metrics.csv")
        ranking_train_df = self._read_optional_csv(self.config.evaluate_dir / "ranking" / "ranking_method_metrics_train.csv")
        confidence_metrics_df = self._read_optional_csv(
            self.config.evaluate_dir / "confidence" / "confidence_level_metrics_test.csv"
        )
        oof_confidence_metrics_df = self._read_optional_csv(
            self.config.evaluate_dir / "confidence" / "confidence_level_metrics_oof.csv"
        )
        confidence_by_class_df = self._read_optional_csv(self.config.evaluate_dir / "confidence" / "confidence_by_class_test.csv")

        save_model_comparison_plot(
            test_metrics_df,
            self.config.evaluate_dir / "plots" / "model_comparison.png",
            self.config.scoring_primary,
        )
        train_report_df = self._read_optional_csv(self.config.evaluate_dir / "model_metrics_train_with_voting.csv")
        if train_report_df is None:
            train_report_df = train_metrics_df
        test_report_df = self._read_optional_csv(self.config.evaluate_dir / "model_metrics_test_with_voting.csv")
        if test_report_df is None:
            test_report_df = test_metrics_df
        write_evaluate_report(
            self.config,
            train_report_df,
            test_report_df,
            selection,
            geo_train_metrics_df=geo_train_df,
            geo_test_metrics_df=geo_test_df,
            ranking_method_metrics_df=ranking_metrics_df,
            ranking_train_metrics_df=ranking_train_df,
            confidence_metrics_df=confidence_metrics_df,
            oof_confidence_metrics_df=oof_confidence_metrics_df,
            confidence_by_class_df=confidence_by_class_df,
        )
        dashboard_path = write_pipeline_final_dashboard(self.config, train_report_df, test_report_df, selection)
        return {
            "report": str(self.config.evaluate_dir / "report.html"),
            "final_dashboard": str(dashboard_path),
        }

    def _create_predict_reports(self) -> dict[str, str]:
        """Regenerate predict-step plots, HTML report, and refreshed final dashboard."""
        predictions_path = self.config.predict_dir / "final_predictions.joblib"
        selection_path = self.config.evaluate_dir / "selection_summary.json"
        self._ensure_artifact(predictions_path, "Cannot recreate predict reports.")
        self._ensure_artifact(selection_path, "Cannot recreate predict reports.")

        final_df = load_joblib(predictions_path)
        selection = load_json(selection_path)
        self._save_prediction_plots(final_df)
        write_predict_report(self.config, final_df, selection)

        train_metrics_df = self._read_optional_csv(self.config.evaluate_dir / "model_metrics_train_with_voting.csv")
        if train_metrics_df is None:
            train_metrics_df = read_data(str(self.config.evaluate_dir / "model_metrics_train.csv"), watch_curly_brackets=False)
        test_metrics_df = self._read_optional_csv(self.config.evaluate_dir / "model_metrics_test_with_voting.csv")
        if test_metrics_df is None:
            test_metrics_df = read_data(str(self.config.evaluate_dir / "model_metrics_test.csv"), watch_curly_brackets=False)
        dashboard_path = write_pipeline_final_dashboard(
            self.config,
            train_metrics_df,
            test_metrics_df,
            selection,
            prediction_df=final_df,
        )
        return {
            "report": str(self.config.predict_dir / "report.html"),
            "final_dashboard": str(dashboard_path),
        }

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
    def recalculate_prediction_quality(self) -> dict[str, Any]:
        """Recalculate prediction confidence and inspection outputs from saved artifacts."""
        predictions_path = self.config.predict_dir / "final_predictions.joblib"
        selection_path = self.config.evaluate_dir / "selection_summary.json"
        member_probabilities_path = self.config.predict_dir / "member_probabilities.joblib"
        self._ensure_artifact(predictions_path, "Cannot recalculate prediction quality.")
        self._ensure_artifact(selection_path, "Cannot recalculate prediction quality.")
        self._ensure_artifact(member_probabilities_path, "Cannot recalculate prediction quality.")

        final_df = load_joblib(predictions_path)
        selection = load_json(selection_path)
        member_probabilities = load_joblib(member_probabilities_path)
        predictions = final_df[self.config.prediction_column].to_numpy()
        rank_confidence, rank_contract = self._calculate_rank_confidence(selection, predictions, member_probabilities)
        final_df = self._apply_rank_confidence_columns(final_df, rank_confidence)
        final_df, inspection_summary = self._add_inspection_metrics(final_df)
        summary = {
            "rank_confidence_enabled": bool(rank_contract["enabled"]),
            "rank_confidence_contract_source": rank_contract["source"],
            "confidence_method": rank_contract.get("method"),
            "class_reliability_enabled": bool(rank_contract.get("class_reliability_enabled", False)),
            "class_reliability_applied_to_final_confidence": bool(
                final_df.get("prediction_class_reliability_applied", pd.Series([False])).iloc[0]
            ),
            "class_reliability_method": (
                rank_contract["class_reliability"].get("method") if rank_contract["class_reliability"] is not None else None
            ),
            "final_confidence_rule": (
                "rank_plus_class_reliability"
                if bool(final_df.get("prediction_class_reliability_applied", pd.Series([False])).iloc[0])
                else "rank_only"
            ),
            "rank_confidence_minimum_models": rank_contract["minimum_models"],
            **inspection_summary,
        }
        predict_summary = self._with_schema(summary, "predict_summary")
        self._persist_predictions(final_df, predict_summary)
        self._save_prediction_plots(final_df)
        write_predict_report(self.config, final_df, selection)
        return predict_summary

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
