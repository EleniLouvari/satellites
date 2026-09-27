"""Pipeline step for full-dataset inference and final prediction artifact creation."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

# Prediction reloads persisted training artifacts so inference matches the fitted feature space.
from ..core.metrics import load_modeling_context
from ..core.inspection import INSPECTION_OUTPUT_COLUMNS, calculate_inspection_metrics
from ..core.persistence import load_joblib, load_json, print_formatted_txt, save_frame_csv, save_joblib, save_json, time_decorator
from ..core.class_reliability import combine_confidence_components, get_class_reliability
from ..core.rank_confidence import classify_ensemble_rank_based, rank_confidence_thresholds_from_config
from ..core.selection import fit_and_predict_selected_strategy
from ..reporting import write_index_report, write_predict_report
from ..reporting.final_dashboard import write_pipeline_final_dashboard
from ..core import PipelineStepBase
from ..visuals import save_inspection_relationship_plot, save_prediction_fill_plot


class PredictStep(PipelineStepBase):
    """Generate final predictions and probability outputs for all rows."""

    def _apply_rank_confidence_columns(
        self,
        final_df: pd.DataFrame,
        rank_confidence: dict[str, np.ndarray] | None,
    ) -> pd.DataFrame:
        """Attach rank-confidence outputs to an existing prediction frame."""
        if rank_confidence is None:
            return final_df

        rank_columns = {
            self.config.prediction_confidence_level_column: "prediction_confidence_level",
            "prediction_confidence_valid": "prediction_confidence_valid",
            "prediction_confidence_reason": "prediction_confidence_reason",
            "prediction_confidence_source": "prediction_confidence_source",
            "prediction_class_reliability_applied": "prediction_class_reliability_applied",
            "prediction_mean_borda": "prediction_mean_borda",
            "prediction_rank_range": "prediction_rank_range",
            "prediction_mean_rank": "prediction_mean_rank",
            "prediction_median_rank": "prediction_median_rank",
            "prediction_rank_std": "prediction_rank_std",
            "prediction_rank_iqr": "prediction_rank_iqr",
            "prediction_borda_winner": "borda_prediction",
            "prediction_borda_winner_tied": "rank_winner_tied",
            "prediction_rank_agrees_with_final": "rank_agrees_with_prediction",
            "prediction_rank_confidence_level": "rank_confidence_level",
            "prediction_class_oof_precision": "class_oof_precision",
            "prediction_class_oof_support": "class_oof_support",
            "prediction_class_reliability_level": "class_reliability_level",
            "prediction_class_reliability_valid": "class_reliability_valid",
            "prediction_class_reliability_reason": "class_reliability_reason",
            "prediction_borda_margin": "borda_margin",
            "prediction_top1_agreement": "top1_agreement",
            "prediction_top2_agreement": "top2_agreement",
            "prediction_top3_agreement": "top3_agreement",
            "prediction_runner_up_class": "runner_up_class",
            "prediction_runner_up_borda": "runner_up_borda",
            "prediction_rank_models_used": "n_models_used",
            "prediction_rank_confidence_valid": "rank_confidence_valid",
            "prediction_rank_confidence_reason": "rank_confidence_reason",
        }
        for output_column, result_key in rank_columns.items():
            if result_key in rank_confidence:
                final_df[output_column] = rank_confidence[result_key]
        final_df[self.config.prediction_review_column] = (
            (final_df[self.config.prediction_confidence_level_column] == "LOW")
            | ~final_df["prediction_confidence_valid"].astype(bool)
        )
        return final_df

    def _resolve_rank_confidence_contract(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Resolve the frozen Step-4 confidence contract with legacy fallback."""
        frozen = selection.get("confidence")
        legacy_rank = selection.get("rank_confidence")
        supported_methods = {"rank_consensus_with_class_reliability_guard", "rank_consensus_only"}
        if frozen is None and isinstance(legacy_rank, dict) and legacy_rank.get("method") in supported_methods:
            frozen = legacy_rank
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
                    "class_reliability": None,
                    "class_reliability_enabled": False,
                    "source": "selection_summary",
                }
            if frozen.get("method") not in supported_methods:
                raise RuntimeError(
                    "The frozen confidence contract uses an obsolete method. Rerun Step 4 to create the "
                    "rank-consensus-with-class-reliability contract."
                )
            rank = frozen.get("rank")
            reliability = frozen.get("class_reliability")
            requires_reliability = frozen["method"] == "rank_consensus_with_class_reliability_guard"
            if not isinstance(rank, dict) or (requires_reliability and not isinstance(reliability, dict)):
                raise RuntimeError("The frozen confidence contract is incomplete; rerun Step 4.")
            return {
                "enabled": True,
                "method": frozen["method"],
                "class_reliability_enabled": bool(frozen.get("class_reliability_enabled", True)),
                "thresholds": {
                    "high_min_borda": rank["high_min_borda"],
                    "high_max_range": rank["high_max_range"],
                    "medium_min_borda": rank["medium_min_borda"],
                    "medium_max_range": rank["medium_max_range"],
                },
                "minimum_models": int(rank["minimum_models"]),
                "class_reliability": reliability,
                "source": "selection_summary",
            }

        if selection.get("rank_confidence") is not None and self.config.rank_confidence_enabled:
            raise RuntimeError(
                "selection_summary.json contains the obsolete class x Top-1 confidence method. "
                "Rerun Step 4 before production prediction."
            )
        if self.config.rank_confidence_enabled and getattr(self.config, "class_reliability_enabled", True):
            raise RuntimeError(
                "selection_summary.json has no frozen class-reliability contract. Rerun Step 4 before prediction."
            )
        return {
            "enabled": bool(self.config.rank_confidence_enabled),
            "method": "rank_consensus_only" if self.config.rank_confidence_enabled else None,
            "class_reliability_enabled": bool(getattr(self.config, "class_reliability_enabled", True)),
            "thresholds": rank_confidence_thresholds_from_config(self.config),
            "minimum_models": int(self.config.rank_confidence_minimum_models),
            "class_reliability": None,
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
        if contract["class_reliability"] is not None:
            reliability = get_class_reliability(predictions, contract["class_reliability"])
            result.update(reliability)
            class_reliability_applied = bool(contract.get("class_reliability_enabled", True))
            result["prediction_class_reliability_applied"] = class_reliability_applied
            result["prediction_confidence_source"] = (
                "rank_plus_class_reliability" if class_reliability_applied else "rank_only"
            )
            if class_reliability_applied:
                final = combine_confidence_components(
                    result["rank_confidence_level"],
                    result["rank_confidence_valid"],
                    result["rank_confidence_reason"],
                    reliability["class_reliability_level"],
                    reliability["class_reliability_valid"],
                    reliability["class_reliability_reason"],
                )
                result.update(final)
            else:
                result["prediction_confidence_level"] = result["rank_confidence_level"]
                result["prediction_confidence_valid"] = result["rank_confidence_valid"]
                result["prediction_confidence_reason"] = result["rank_confidence_reason"]
        else:
            result["prediction_class_reliability_applied"] = False
            result["prediction_confidence_source"] = "rank_only"
            result["prediction_confidence_level"] = result["rank_confidence_level"]
            result["prediction_confidence_valid"] = result["rank_confidence_valid"]
            result["prediction_confidence_reason"] = result["rank_confidence_reason"]
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
            self.config.id_column, self.config.target_column, self.config.prediction_column,
            self.config.prediction_confidence_column, self.config.prediction_review_column,
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
        save_inspection_relationship_plot(
            final_df,
            plots / "inspection_risk_relationships.png",
            self.config,
        )

    def _add_inspection_metrics(self, final_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Attach inspection risks when all three independent inputs exist."""
        # Allow deployments to disable the complete inspection layer without changing prediction.
        if not self.config.inspection_scoring_enabled:
            return final_df, {"inspection_scoring_enabled": False, "inspection_scoring_available": False}

        # Declaration columns are optional; only the three label-independent risk inputs are required.
        required = {
            self.config.prediction_confidence_level_column,
            self.config.inspection_data_reliability_column,
            self.config.inspection_geometry_complexity_column,
        }
        missing = sorted(required.difference(final_df.columns))
        if missing:
            warnings.warn(
                f"Inspection scoring skipped because required columns are missing: {missing}",
                RuntimeWarning,
                stacklevel=2,
            )
            return final_df, {
                "inspection_scoring_enabled": True,
                "inspection_scoring_available": False,
                "inspection_missing_columns": missing,
            }

        # The helper adds score, label agreement, final need, check type, and reasons per parcel.
        scored = calculate_inspection_metrics(final_df, self.config)
        mean_score = scored["inspection_score"].mean()
        # Persist aggregate counts in predict_summary.json for fast operational monitoring.
        return scored, {
            "inspection_scoring_enabled": True,
            "inspection_scoring_available": True,
            "inspection_weights": {
                "model": self.config.inspection_model_weight,
                "data": self.config.inspection_data_weight,
                "geometry": self.config.inspection_geometry_weight,
            },
            "inspection_need_counts": scored["inspection_need"].value_counts(dropna=False).to_dict(),
            "inspection_check_type_counts": scored["inspection_check_type"].value_counts(dropna=False).to_dict(),
            "label_prediction_status_counts": scored["label_prediction_status"].value_counts(dropna=False).to_dict(),
            "inspection_score_mean": None if pd.isna(mean_score) else float(mean_score),
        }

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
            dataset,
            predictions,
            probabilities,
            selection["labels"],
            rank_confidence=rank_confidence,
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
            "class_reliability_applied_to_final_confidence": bool(final_df.get(
                "prediction_class_reliability_applied", pd.Series([False])
            ).iloc[0]),
            "class_reliability_method": (
                rank_contract["class_reliability"].get("method")
                if rank_contract["class_reliability"] is not None else None
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
