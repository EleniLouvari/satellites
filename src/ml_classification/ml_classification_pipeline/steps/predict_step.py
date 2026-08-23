"""Pipeline step for full-dataset inference and final prediction artifact creation."""

from __future__ import annotations

from typing import Any

import numpy as np

# Prediction reloads persisted training artifacts so inference matches the fitted feature space.
from ..core.metrics import load_modeling_context
from ..core.persistence import load_joblib, load_json, print_formatted_txt, save_frame_csv, save_joblib, save_json, time_decorator
from ..core.rank_confidence import classify_ensemble_rank_based, rank_confidence_thresholds_from_config
from ..core.rank_confidence_calibration import apply_class_aware_confidence_calibration
from ..core.selection import fit_and_predict_selected_strategy
from ..reporting import write_index_report, write_predict_report
from ..core import PipelineStepBase
from ..visuals import save_predicted_labels_map, save_prediction_fill_plot


class PredictStep(PipelineStepBase):
    """Generate final predictions and probability outputs for all rows."""

    def _resolve_rank_confidence_contract(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Resolve the frozen Step-4 confidence contract with legacy fallback."""
        # Read the confidence definition persisted alongside the selected strategy.
        frozen = selection.get("rank_confidence")
        if frozen is not None:
            # Reject corrupted or manually edited summaries with an invalid object type.
            if not isinstance(frozen, dict):
                raise RuntimeError("selection_summary rank_confidence must be an object.")
            # Step 4's enabled flag is authoritative even if the live config later changes.
            enabled = bool(frozen.get("enabled", False))
            if not enabled:
                # Return an explicit disabled contract so downstream summary fields stay stable.
                return {
                    "enabled": False,
                    "thresholds": None,
                    "minimum_models": None,
                    "calibration": None,
                    "source": "selection_summary",
                }
            # Require the complete frozen rank contract before running production inference.
            if "thresholds" not in frozen or "minimum_models" not in frozen:
                raise RuntimeError(
                    "The frozen rank-confidence contract is incomplete; thresholds and minimum_models are required."
                )
            # Carry both the fallback rank rules and optional OOF empirical table forward.
            return {
                "enabled": True,
                "thresholds": frozen["thresholds"],
                "minimum_models": int(frozen["minimum_models"]),
                "calibration": frozen.get("calibration"),
                "source": "selection_summary",
            }

        # Backward compatibility for selection summaries created before the
        # confidence contract was persisted by Step 4.
        return {
            "enabled": bool(self.config.rank_confidence_enabled),
            "thresholds": rank_confidence_thresholds_from_config(self.config),
            "minimum_models": int(self.config.rank_confidence_minimum_models),
            "calibration": None,
            "source": "live_config_legacy_fallback",
        }

    def _calculate_rank_confidence(self, selection, predictions, member_probabilities):
        """Calculate confidence using the definition frozen during evaluation."""
        # Resolve Step 4's contract before examining any production probabilities.
        contract = self._resolve_rank_confidence_contract(selection)
        # Skip all rank-confidence work when the frozen strategy disabled it.
        if not contract["enabled"]:
            return None, contract

        # Preserve the selected model order used by voting and the frozen calibration.
        selected_models = list(selection["selected_models"])
        # Stack raw member probabilities as [models, rows, classes] for within-model ranking.
        stacked_probabilities = np.stack(
            [member_probabilities[model_name] for model_name in selected_models], axis=0
        )
        # Map canonical string labels to encoded probability-column indices.
        label_to_index = {str(label): index for index, label in enumerate(selection["labels"])}
        # Identify the actual final soft-voting class whose confidence must be explained.
        predicted_indices = np.asarray([label_to_index[str(label)] for label in predictions], dtype=int)
        # Calculate scale-invariant rank features and structural safeguard outcomes first.
        result = classify_ensemble_rank_based(
            stacked_probabilities,
            class_names=list(selection["labels"]),
            model_names=selected_models,
            predicted_class_indices=predicted_indices,
            confidence_thresholds=contract["thresholds"],
            minimum_models=contract["minimum_models"],
        )
        # Replace provisional H/M/L with the frozen OOF empirical category when available.
        if contract["calibration"] is not None:
            # Update only confidence outputs; all underlying rank diagnostics remain unchanged.
            result.update(
                apply_class_aware_confidence_calibration(
                    predicted_classes=predictions,
                    top1_agreement=result["top1_agreement"],
                    n_models_used=result["n_models_used"],
                    confidence_valid=result["confidence_valid"],
                    confidence_reason=result["confidence_reason"],
                    rank_agrees_with_prediction=result["rank_agrees_with_prediction"],
                    rank_winner_tied=result["rank_winner_tied"],
                    calibration=contract["calibration"],
                )
            )
        # Return both parcel results and contract provenance for output summaries.
        return result, contract

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
        final_df[self.config.prediction_confidence_column] = np.max(probabilities, axis=1)
        # Mark rows that need manual review because confidence is below threshold.
        final_df[self.config.prediction_review_column] = (
            final_df[self.config.prediction_confidence_column] < self.config.prediction_confidence_threshold
        )
        if rank_confidence is not None:
            rank_columns = {
                self.config.prediction_confidence_level_column: "confidence_level",
                "rank_aggregate_score": "aggregate_rank_score",
                "rank_margin": "rank_margin",
                "rank_top1_agreement": "top1_agreement",
                "rank_top2_agreement": "top2_agreement",
                "rank_top3_agreement": "top3_agreement",
                "rank_mean": "mean_rank",
                "rank_median": "median_rank",
                "rank_std": "rank_std",
                "rank_iqr": "rank_iqr",
                "rank_runner_up_class": "runner_up_class",
                "rank_runner_up_score": "runner_up_rank_score",
                "rank_prediction": "rank_prediction",
                "rank_winner_tied": "rank_winner_tied",
                "rank_agrees_with_prediction": "rank_agrees_with_prediction",
                "rank_n_models_used": "n_models_used",
                "rank_confidence_valid": "confidence_valid",
                "rank_confidence_reason": "confidence_reason",
                "rank_confidence_empirical_accuracy": "confidence_empirical_accuracy",
                "rank_confidence_calibration_support": "confidence_calibration_support",
                "rank_confidence_calibration_valid": "confidence_calibration_valid",
                "rank_confidence_calibration_source": "confidence_calibration_source",
            }
            for output_column, result_key in rank_columns.items():
                if result_key in rank_confidence:
                    final_df[output_column] = rank_confidence[result_key]
            final_df[self.config.prediction_review_column] = (
                (final_df[self.config.prediction_confidence_level_column] == "LOW")
                | ~final_df["rank_confidence_valid"].astype(bool)
            )
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
            self.config.id_column, self.config.target_column, self.config.prediction_column,
            self.config.prediction_confidence_column, self.config.prediction_review_column,
            self.config.prediction_filled_column,
        ]
        optional_confidence_columns = [
            self.config.prediction_confidence_level_column,
            "rank_aggregate_score",
            "rank_margin",
            "rank_top1_agreement",
            "rank_top2_agreement",
            "rank_top3_agreement",
            "rank_mean",
            "rank_median",
            "rank_std",
            "rank_iqr",
            "rank_runner_up_class",
            "rank_runner_up_score",
            "rank_prediction",
            "rank_winner_tied",
            "rank_agrees_with_prediction",
            "rank_n_models_used",
            "rank_confidence_valid",
            "rank_confidence_reason",
            "rank_confidence_empirical_accuracy",
            "rank_confidence_calibration_support",
            "rank_confidence_calibration_valid",
            "rank_confidence_calibration_source",
        ]
        preview_columns.extend(column for column in optional_confidence_columns if column in final_df.columns)
        # Persist full predictions, a compact CSV preview, and the summary JSON.
        save_joblib(final_df, self.config.predict_dir / "final_predictions.joblib")
        save_frame_csv(final_df[preview_columns], self.config.predict_dir / "final_predictions_preview.csv")
        save_json(summary, self.config.predict_dir / "predict_summary.json")
        # Record expected schema/contracts for downstream validation and reporting.
        self._save_schema_manifest(
            self.config.predict_dir, "predict_step_schema_manifest",
            json_contracts={"predict_summary.json": sorted(summary)},
            csv_contracts={"final_predictions_preview.csv": preview_columns},
        )

    def _save_prediction_plots(self, final_df):
        plots = self.config.predict_dir / "plots"
        # Save distribution plot comparing original vs filled target values.
        save_prediction_fill_plot(
            final_df[self.config.target_column], final_df[self.config.prediction_filled_column],
            plots / "filled_target_distribution.png",
        )
        # Save a spatial map of predicted/final labels for visual inspection.
        save_predicted_labels_map(
            final_df, self.config.prediction_filled_column, plots / "predicted_labels_map.png",
            title="Classified Predicted Labels Map", max_geometries=self.config.max_map_geometries,
        )

    @time_decorator
    def run_predict(self) -> dict[str, Any]:
        """Run full-dataset inference and persist final prediction artifacts."""
        # Load required context and execute selected prediction strategy.
        print_formatted_txt("Classify all data...", "SUBSECTION")
        dataset, context, selection, model_specs = self._load_prediction_inputs()
        active_features = context["active_features"]
        numeric_features = context["numeric_features"]
        categorical_features = context["categorical_features"]
        label_encoder = context["label_encoder"]

        # Build a labeled subset to fit any selected strategies that require re-fitting.
        labeled_df = dataset.loc[dataset[self.config.target_column].notna(), :].copy()
        labeled_df[self.config.target_column] = labeled_df[self.config.target_column].astype(str)

        # Execute the chosen selection strategy (may refit models) to obtain predictions and probabilities.
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

        rank_confidence, rank_contract = self._calculate_rank_confidence(
            selection, predictions, member_probabilities
        )

        # Augment the dataset with predictions, probabilities and filled labels.
        final_df, unknown_mask = self._create_prediction_frame(
            dataset,
            predictions,
            probabilities,
            selection["labels"],
            rank_confidence=rank_confidence,
        )
        unknown_rows_before = int(unknown_mask.sum())
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
            "rank_confidence_calibration_method": (
                rank_contract["calibration"].get("method") if rank_contract["calibration"] is not None else None
            ),
            "rank_confidence_minimum_models": rank_contract["minimum_models"],
            "rank_confidence_level_counts": (
                final_df[self.config.prediction_confidence_level_column].value_counts().to_dict()
                if self.config.prediction_confidence_level_column in final_df
                else {}
            ),
            "output_path": str(self.config.predict_dir / "final_predictions.joblib"),
        }
        if rank_confidence is None:
            summary_values["maximum_probability_review_threshold"] = self.config.prediction_confidence_threshold
        predict_summary = self._with_schema(
            summary_values,
            "predict_summary",
        )
        self._persist_predictions(final_df, predict_summary)
        self._save_prediction_plots(final_df)

        write_predict_report(self.config, final_df, selection)
        write_index_report(self.config)
        print_formatted_txt(f"Filled {rows_filled} unknown labels.", "RESULTS")
        return predict_summary
