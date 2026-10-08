"""Pipeline step for full-dataset inference and final prediction artifact creation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_classification.shared.config.base import PipelineStepBase
from ml_classification.shared.reports.final_dashboard import write_pipeline_final_dashboard
from ml_classification.shared.logging import print_formatted_txt, time_decorator

# Prediction reloads persisted training artifacts so inference matches the fitted feature space.
from ml_classification.shared.models.modeling_context import load_modeling_context
from ml_classification.shared.persistence import load_joblib, load_json, save_frame_csv, save_joblib, save_json
from ml_classification.shared.reports.report_index import write_index_report
from ml_classification.step_05_predict.libraries.inspection_priority.scoring import INSPECTION_OUTPUT_COLUMNS
from ml_classification.step_05_predict.libraries.plots import save_prediction_fill_plot
from ml_classification.step_05_predict.libraries.inspection_priority.plots import save_inspection_relationship_plot
from ml_classification.step_05_predict.libraries.prediction_quality import (
    add_inspection_metrics,
    apply_rank_confidence_columns,
    calculate_rank_confidence,
    resolve_rank_confidence_contract,
)
from ml_classification.step_05_predict.libraries.refit import fit_and_predict_selected_strategy
from ml_classification.step_05_predict.libraries.report import write_predict_report
from shared.io import read_data


class PredictStep(PipelineStepBase):
    """Generate final predictions and probability outputs for all rows."""

    def _apply_rank_confidence_columns(
        self, final_df: pd.DataFrame, rank_confidence: dict[str, np.ndarray] | None
    ) -> pd.DataFrame:
        """Attach rank-confidence outputs to an existing prediction frame."""
        return apply_rank_confidence_columns(self.config, final_df, rank_confidence)

    def _resolve_rank_confidence_contract(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Resolve the frozen Step-4 confidence contract with legacy fallback."""
        return resolve_rank_confidence_contract(self.config, selection)

    def _calculate_rank_confidence(self, selection, predictions, member_probabilities):
        """Calculate confidence using the definition frozen during evaluation."""
        return calculate_rank_confidence(self.config, selection, predictions, member_probabilities)

    def _load_prediction_inputs(self):
        # Load persisted dataset and modeling artifacts needed for prediction.
        dataset = load_joblib(self.config.check_dir / "input_dataset.joblib")
        # Return dataset, modeling context, chosen selection strategy, and model specs.
        return (
            dataset,
            load_modeling_context(self.config),
            load_json(self.config.evaluate_dir / "selection_summary.json"),
            load_json(self.config.train_dir / "model_specs.json"),
        )

    def _create_prediction_frame(self, dataset, predictions, probabilities, labels, rank_confidence=None):
        # Create a copy of the dataset to avoid mutating the original input.
        final_df = dataset.copy()
        # Identify rows where target is missing (those are candidates to be filled).
        unknown_mask = final_df[self.config.target_column].isna()
        # Attach predicted labels and confidence score (max class probability).
        final_df[self.config.prediction_column] = predictions
        if self.config.prediction_column != "predicted_class":
            final_df["predicted_class"] = predictions
        final_df[self.config.prediction_confidence_column] = np.max(probabilities, axis=1)
        # Mark rows that need manual review because confidence is below threshold.
        final_df[self.config.prediction_review_column] = (
            final_df[self.config.prediction_confidence_column] < self.config.prediction_confidence_threshold
        )
        if rank_confidence is not None:
            final_df = self._apply_rank_confidence_columns(final_df, rank_confidence)
        # Produce a filled target column that uses prediction only where original was unknown.
        final_df[self.config.prediction_filled_column] = final_df[self.config.target_column].where(
            ~unknown_mask, final_df[self.config.prediction_column]
        )
        # Add class-wise probability columns for downstream analysis/plots.
        for class_index, class_label in enumerate(labels):
            final_df[f"{self.config.probability_prefix}_{class_label}"] = probabilities[:, class_index]
        return final_df, unknown_mask

    def _persist_predictions(self, final_df, summary):
        preview_columns = [
            self.config.id_column,
            self.config.target_column,
            self.config.prediction_column,
            self.config.prediction_confidence_column,
            self.config.prediction_review_column,
            self.config.prediction_filled_column,
        ]
        if "predicted_class" in final_df and "predicted_class" not in preview_columns:
            preview_columns.insert(3, "predicted_class")
        optional_confidence_columns = [
            self.config.prediction_confidence_level_column,
            "prediction_confidence_valid",
            "prediction_confidence_reason",
            "prediction_mean_borda",
            "prediction_rank_range",
            "prediction_mean_rank",
            "prediction_median_rank",
            "prediction_rank_std",
            "prediction_rank_iqr",
            "prediction_borda_winner",
            "prediction_borda_winner_tied",
            "prediction_rank_agrees_with_final",
            "prediction_rank_confidence_level",
            "prediction_class_oof_precision",
            "prediction_class_oof_support",
            "prediction_class_reliability_level",
            "prediction_class_reliability_valid",
            "prediction_class_reliability_reason",
            "prediction_borda_margin",
            "prediction_top1_agreement",
            "prediction_top2_agreement",
            "prediction_top3_agreement",
            "prediction_runner_up_class",
            "prediction_runner_up_borda",
            "prediction_rank_models_used",
            "prediction_rank_confidence_valid",
            "prediction_rank_confidence_reason",
            self.config.inspection_data_reliability_column,
            self.config.inspection_geometry_complexity_column,
            *INSPECTION_OUTPUT_COLUMNS,
        ]
        preview_columns.extend(column for column in optional_confidence_columns if column in final_df.columns)
        # Persist full predictions, a compact CSV preview, and the summary JSON.
        save_joblib(final_df, self.config.predict_dir / "final_predictions.joblib")
        save_frame_csv(final_df[preview_columns], self.config.predict_dir / "final_predictions_preview.csv")
        save_json(summary, self.config.predict_dir / "predict_summary.json")
        # Record expected schema/contracts for downstream validation and reporting.
        self._save_schema_manifest(
            self.config.predict_dir,
            "predict_step_schema_manifest",
            json_contracts={"predict_summary.json": sorted(summary)},
            csv_contracts={"final_predictions_preview.csv": preview_columns},
        )

    def _save_prediction_plots(self, final_df):
        plots = self.config.predict_dir / "plots"
        # Save distribution plot comparing original vs filled target values.
        save_prediction_fill_plot(
            final_df[self.config.target_column],
            final_df[self.config.prediction_filled_column],
            plots / "filled_target_distribution.png",
        )
        save_inspection_relationship_plot(final_df, plots / "inspection_risk_relationships.png", self.config)

    def _add_inspection_metrics(self, final_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Attach inspection risks when all three independent inputs exist."""
        return add_inspection_metrics(self.config, final_df)

    def calculate_prediction_quality(
        self,
        dataset: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        selection: dict[str, Any],
        member_probabilities: dict[str, np.ndarray],
    ) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
        """Build the enriched prediction frame without running model fitting again."""
        rank_confidence, rank_contract = self._calculate_rank_confidence(selection, predictions, member_probabilities)
        final_df, unknown_mask = self._create_prediction_frame(
            dataset, predictions, probabilities, selection["labels"], rank_confidence=rank_confidence
        )
        final_df, inspection_summary = self._add_inspection_metrics(final_df)
        return final_df, unknown_mask, rank_confidence, rank_contract, inspection_summary

    @time_decorator
    def run_predict(self) -> dict[str, Any]:
        """Run full-dataset inference and persist final prediction artifacts."""
        # Load required context and execute selected prediction strategy.
        print_formatted_txt("Classify all data...", "SUBSECTION")
        # Load all the initial data
        dataset, context, selection, model_specs = self._load_prediction_inputs()
        active_features = context["active_features"]
        numeric_features = context["numeric_features"]
        categorical_features = context["categorical_features"]
        label_encoder = context["label_encoder"]

        # Build a labeled subset to fit any selected strategies that require re-fitting.
        labeled_df = dataset.loc[dataset[self.config.target_column].notna(), :].copy()
        labeled_df[self.config.target_column] = labeled_df[self.config.target_column].astype(str)

        # Execute the chosen selection strategy (may refit models using all labeled data) to obtain predictions and probabilities.
        predictions, probabilities, member_probabilities = fit_and_predict_selected_strategy(
            config=self.config,
            X_fit=labeled_df[active_features].copy(),
            y_fit=label_encoder.transform(labeled_df[self.config.target_column].copy()),
            X_all=dataset[active_features].copy(),
            selection=selection,
            model_specs=model_specs,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Save the selected-model member probabilities so confidence can be recalculated later
        # without refitting the models.
        save_joblib(member_probabilities, self.config.predict_dir / "member_probabilities.joblib")

        # Augment the dataset with confidence and inspection outputs.
        final_df, unknown_mask, rank_confidence, rank_contract, inspection_summary = self.calculate_prediction_quality(
            dataset=dataset,
            predictions=predictions,
            probabilities=probabilities,
            selection=selection,
            member_probabilities=member_probabilities,
        )
        unknown_rows_before = int(unknown_mask.sum())
        # Fill all unknown rows with predictions and mark those needing review based on confidence.
        rows_filled = int(final_df.loc[unknown_mask, self.config.prediction_filled_column].notna().sum())
        summary_values = {
            "selection_type": selection["selection_type"],
            "selected_models": selection["selected_models"],
            "rows_filled": rows_filled,
            "rows_unknown_original": unknown_rows_before,
            "rows_needing_review": int(final_df[self.config.prediction_review_column].sum()),
            "prediction_max_probability_column": self.config.prediction_confidence_column,
            "review_decision_basis": "rank_confidence" if rank_confidence is not None else "maximum_probability",
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
            "rank_confidence_level_counts": (
                final_df[self.config.prediction_confidence_level_column].value_counts().to_dict()
                if self.config.prediction_confidence_level_column in final_df
                else {}
            ),
            "output_path": str(self.config.predict_dir / "final_predictions.joblib"),
            **inspection_summary,
        }
        if rank_confidence is None:
            summary_values["maximum_probability_review_threshold"] = self.config.prediction_confidence_threshold
        # Save the final predictions, a preview CSV, and the summary JSON with schema manifest.
        predict_summary = self._with_schema(summary_values, "predict_summary")
        self._persist_predictions(final_df, predict_summary)
        self._save_prediction_plots(final_df)

        write_predict_report(self.config, final_df, selection)
        write_pipeline_final_dashboard(
            self.config,
            pd.read_csv(self.config.evaluate_dir / "model_metrics_train_with_voting.csv"),
            pd.read_csv(self.config.evaluate_dir / "model_metrics_test_with_voting.csv"),
            selection,
            prediction_df=final_df,
        )
        write_index_report(self.config)
        print_formatted_txt(f"Filled {rows_filled} unknown labels.", "RESULTS")
        return predict_summary

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
            self.config, train_metrics_df, test_metrics_df, selection, prediction_df=final_df
        )
        return {"report": str(self.config.predict_dir / "report.html"), "final_dashboard": str(dashboard_path)}

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
