"""Pipeline step for model evaluation, comparison, and selection reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from satellites.ml_classification.shared.config.base import PipelineStepBase
from satellites.ml_classification.shared.reports.final_dashboard import write_pipeline_final_dashboard
from satellites.ml_classification.shared.logging import print_formatted_txt, time_decorator
from satellites.ml_classification.shared.models.modeling_context import load_modeling_context
from satellites.ml_classification.shared.persistence import load_joblib, load_json, save_frame_csv, save_json
from satellites.ml_classification.shared.probabilities import apply_class_probability_multipliers
from satellites.ml_classification.shared.reports.report_index import write_index_report
from satellites.ml_classification.step_04_evaluate.libraries.confidence import evaluate_rank_confidence

# Evaluation operates only on held-out data to keep reported model comparisons unbiased.
from satellites.ml_classification.step_04_evaluate.libraries.interpretability import save_interpretability
from satellites.ml_classification.step_04_evaluate.libraries.model_evaluation import (
    evaluate_base_models,
    model_artifact_path,
    save_model_evaluation_plot,
    to_geo_classifier_result_row,
    validate_probability_class_order,
)
from satellites.ml_classification.step_04_evaluate.libraries.model_selection import (
    evaluate_voting_candidate,
    sort_metrics_without_voting,
)
from satellites.ml_classification.step_04_evaluate.libraries.plots import save_model_comparison_plot
from satellites.ml_classification.step_04_evaluate.libraries.ranking import (
    apply_class_reliability_guard,
    build_parcel_ranking_outputs,
    rank_positions_desc,
    summarize_rank_confidence,
)
from satellites.ml_classification.step_04_evaluate.libraries.report import write_evaluate_report
from satellites.shared.io import read_data


class EvaluateStep(PipelineStepBase):
    """Evaluate trained models on train/test sets and choose prediction strategy."""

    @staticmethod
    def _validate_probability_class_order(estimator: Any, labels: list[str], model_name: str) -> None:
        """Ensure predict_proba columns match the pipeline's encoded class order."""
        return validate_probability_class_order(estimator, labels, model_name)

    @staticmethod
    def _rank_positions_desc(matrix: np.ndarray) -> np.ndarray:
        """Return descending within-row average ranks, including deterministic ties."""
        return rank_positions_desc(matrix)

    def _build_parcel_ranking_outputs(
        self,
        probability_cache: dict[str, np.ndarray],
        selection: dict[str, Any],
        labels: list[str],
        y_true: pd.Series,
        id_col: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create per-parcel best-class outputs and ranking-method summary metrics."""
        return build_parcel_ranking_outputs(self.config, probability_cache, selection, labels, y_true, id_col)

    @staticmethod
    def _summarize_rank_confidence(parcel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Summarize empirical correctness overall and by predicted class."""
        return summarize_rank_confidence(parcel_df)

    @staticmethod
    def _apply_class_reliability_guard(
        parcel_df: pd.DataFrame, reliability_contract: dict[str, Any], *, combine_with_rank: bool = True
    ) -> pd.DataFrame:
        """Attach frozen OOF class reliability and optionally combine it with rank confidence."""
        return apply_class_reliability_guard(parcel_df, reliability_contract, combine_with_rank=combine_with_rank)

    def _to_geo_classifier_result_row(
        self, y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray | None, labels: list[str], model_name: str
    ) -> dict[str, Any]:
        """Build legacy GeoDataFrameClassifier-style metric rows for reporting."""
        return to_geo_classifier_result_row(y_true, y_pred, probabilities, labels, model_name)

    def _load_evaluation_inputs(self):
        # Load modeling context and prepared train/test datasets used for holdout evaluation.
        context = load_modeling_context(self.config)
        train_df = load_joblib(self.config.prepare_dir / "train_dataset.joblib")
        test_df = load_joblib(self.config.prepare_dir / "test_dataset.joblib")
        features = context["active_features"]
        # Return inputs packaged as a dict for downstream evaluation routines.
        return {
            "context": context,
            "model_specs": load_json(self.config.train_dir / "model_specs.json"),
            "cv_metrics": read_data(str(self.config.train_dir / "training_summary.csv"), watch_curly_brackets=False),
            "X_train": train_df[features].copy(),
            "y_train": train_df[self.config.target_column].astype(str).copy(),
            "X_test": test_df[features].copy(),
            "y_test": test_df[self.config.target_column].astype(str).copy(),
            "id_test": test_df[self.config.id_column].copy(),
            "id_train": train_df[self.config.id_column].copy(),
        }

    def _model_artifact_path(self, model_name):
        # Prefer model artefact inside model-specific folder, fallback to top-level train dir.
        return model_artifact_path(self.config, model_name)

    def _save_model_evaluation_plot(self, model_name, y_test, predictions, probabilities, labels):
        # If no probabilities, only save confusion matrix for the model.
        return save_model_evaluation_plot(self.config, model_name, y_test, predictions, probabilities, labels)

    def _evaluate_base_models(self, inputs):
        # Evaluate each trained model on train/test splits, collecting metrics and artifacts.
        return evaluate_base_models(self.config, inputs)

    def _save_interpretability(self, cv_metrics_df, test_metrics_df, fitted_estimators, X_train):
        # Select top models for interpretability analysis and optionally produce SHAP/importance outputs.
        return save_interpretability(self.config, cv_metrics_df, test_metrics_df, fitted_estimators, X_train)

    @time_decorator
    def run_evaluate(self) -> dict[str, Any]:
        """Evaluate fitted models, generate reports, and select final strategy."""
        # Load prepared data and model artifacts for holdout evaluation.
        print_formatted_txt("Evaluating models...", "SUBSECTION")
        inputs = self._load_evaluation_inputs()
        context = inputs["context"]
        cv_metrics_df = inputs["cv_metrics"]
        labels = context["labels"]
        # Unpack training/test arrays and run base model evaluations to collect metrics and plots.
        X_train, y_train, y_test = inputs["X_train"], inputs["y_train"], inputs["y_test"]
        metric_rows, geo_rows, caches, fitted_estimators = self._evaluate_base_models(inputs)
        train_rows, test_rows = metric_rows["train"], metric_rows["test"]
        geo_train_rows, geo_test_rows = geo_rows["train"], geo_rows["test"]
        probability_cache_train, probability_cache_test = caches["train"], caches["test"]
        oof_probability_cache = caches["oof"]

        train_metrics_df = pd.DataFrame(train_rows)
        test_metrics_df = pd.DataFrame(test_rows)
        train_metrics_df = sort_metrics_without_voting(train_metrics_df, self.config)
        test_metrics_df = sort_metrics_without_voting(test_metrics_df, self.config)
        save_frame_csv(train_metrics_df, self.config.evaluate_dir / "model_metrics_train.csv")
        save_frame_csv(test_metrics_df, self.config.evaluate_dir / "model_metrics_test.csv")
        save_model_comparison_plot(
            test_metrics_df, self.config.evaluate_dir / "plots" / "model_comparison.png", self.config.scoring_primary
        )

        self._save_interpretability(cv_metrics_df, test_metrics_df, fitted_estimators, X_train)

        selection = evaluate_voting_candidate(
            config=self.config,
            cv_metrics_df=cv_metrics_df,
            train_metrics_df=train_metrics_df,
            test_metrics_df=test_metrics_df,
            probability_cache_train=probability_cache_train,
            probability_cache_test=probability_cache_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels,
            oof_probability_cache=oof_probability_cache,
        )

        prob_models = list(selection["selected_models"]) if selection["selection_type"] == "soft_voting" else []
        if len(prob_models) >= 2:
            averaged_probs_train = np.zeros_like(probability_cache_train[prob_models[0]], dtype=np.float64)
            averaged_probs_test = np.zeros_like(probability_cache_test[prob_models[0]], dtype=np.float64)
            for model_name in prob_models:
                averaged_probs_train += probability_cache_train[model_name]
                averaged_probs_test += probability_cache_test[model_name]
            averaged_probs_train /= len(prob_models)
            averaged_probs_test /= len(prob_models)
            averaged_probs_train = apply_class_probability_multipliers(
                averaged_probs_train, labels, selection.get("class_probability_multipliers")
            )
            averaged_probs_test = apply_class_probability_multipliers(
                averaged_probs_test, labels, selection.get("class_probability_multipliers")
            )

            voting_predictions_train = np.asarray(labels)[np.argmax(averaged_probs_train, axis=1)]
            voting_predictions_test = np.asarray(labels)[np.argmax(averaged_probs_test, axis=1)]
            geo_train_rows.append(
                self._to_geo_classifier_result_row(
                    y_true=y_train,
                    y_pred=voting_predictions_train,
                    probabilities=averaged_probs_train,
                    labels=labels,
                    model_name="Voting",
                )
            )
            geo_test_rows.append(
                self._to_geo_classifier_result_row(
                    y_true=y_test,
                    y_pred=voting_predictions_test,
                    probabilities=averaged_probs_test,
                    labels=labels,
                    model_name="Voting",
                )
            )

        geo_train_report_df = pd.DataFrame(geo_train_rows)
        geo_test_report_df = pd.DataFrame(geo_test_rows)
        for group, report_df in {"train": geo_train_report_df, "test": geo_test_report_df}.items():
            voting_df = report_df[report_df["model"] == "Voting"].copy()
            base_df = report_df[report_df["model"] != "Voting"].copy()
            if "f1_score" in base_df.columns:
                base_df = base_df.sort_values(by=["f1_score"], ascending=False)
            ordered_df = pd.concat([base_df, voting_df], ignore_index=True)
            if group == "train":
                geo_train_report_df = ordered_df
            else:
                geo_test_report_df = ordered_df
            save_frame_csv(ordered_df, self.config.evaluate_dir / f"geo_classifier_style_{group}_metrics.csv")

        confidence = evaluate_rank_confidence(self.config, inputs, selection, caches)
        selection_with_schema = self._with_schema(selection, "selection_summary")

        csv_contracts = {
            **confidence.csv_contracts,
            "model_metrics_train.csv": train_metrics_df.columns.tolist(),
            "model_metrics_test.csv": test_metrics_df.columns.tolist(),
            "geo_classifier_style_train_metrics.csv": geo_train_report_df.columns.tolist(),
            "geo_classifier_style_test_metrics.csv": geo_test_report_df.columns.tolist(),
        }
        interpretability_summary_path = self.config.evaluate_dir / "interpretability" / "interpretability_summary.csv"
        if interpretability_summary_path.exists():
            interpretability_summary_df = read_data(str(interpretability_summary_path), watch_curly_brackets=False)
            csv_contracts["interpretability_summary.csv"] = interpretability_summary_df.columns.tolist()
        save_json(selection_with_schema, self.config.evaluate_dir / "selection_summary.json")
        json_contracts = {"selection_summary.json": sorted(selection_with_schema.keys())}
        self._save_schema_manifest(
            self.config.evaluate_dir, "evaluate_step_schema_manifest", json_contracts=json_contracts, csv_contracts=csv_contracts
        )
        train_report_df = (
            read_data(str(self.config.evaluate_dir / "model_metrics_train_with_voting.csv"), watch_curly_brackets=False)
            if (self.config.evaluate_dir / "model_metrics_train_with_voting.csv").exists()
            else train_metrics_df
        )
        test_report_df = (
            read_data(str(self.config.evaluate_dir / "model_metrics_test_with_voting.csv"), watch_curly_brackets=False)
            if (self.config.evaluate_dir / "model_metrics_test_with_voting.csv").exists()
            else test_metrics_df
        )
        write_evaluate_report(
            self.config,
            train_report_df,
            test_report_df,
            selection,
            geo_train_metrics_df=geo_train_report_df,
            geo_test_metrics_df=geo_test_report_df,
            ranking_method_metrics_df=confidence.ranking_metrics if not confidence.ranking_metrics.empty else None,
            ranking_train_metrics_df=confidence.train_ranking_metrics if not confidence.train_ranking_metrics.empty else None,
            confidence_metrics_df=confidence.metrics if not confidence.metrics.empty else None,
            oof_confidence_metrics_df=confidence.oof_metrics if not confidence.oof_metrics.empty else None,
            confidence_by_class_df=confidence.by_class if not confidence.by_class.empty else None,
        )
        dashboard_path = write_pipeline_final_dashboard(self.config, train_report_df, test_report_df, selection)
        print_formatted_txt(f"Generated final dashboard: {dashboard_path}", "RESULTS")
        write_index_report(self.config)
        print_formatted_txt(f"Selected strategy: {selection['selection_type']} using {selection['selected_models']}", "RESULTS")
        return selection_with_schema

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
            test_metrics_df, self.config.evaluate_dir / "plots" / "model_comparison.png", self.config.scoring_primary
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
        return {"report": str(self.config.evaluate_dir / "report.html"), "final_dashboard": str(dashboard_path)}
