"""Pipeline step for full-dataset inference and final prediction artifact creation."""

from __future__ import annotations

from typing import Any

import numpy as np

# Prediction reloads persisted training artifacts so inference matches the fitted feature space.
from ..core.metrics import load_modeling_context
from ..core.persistence import load_joblib, load_json, print_formatted_txt, save_frame_csv, save_joblib, save_json, time_decorator
from ..core.selection import fit_and_predict_selected_strategy
from ..reporting import write_index_report, write_predict_report
from ..core import PipelineStepBase
from ..visuals import save_predicted_labels_map, save_prediction_fill_plot


class PredictStep(PipelineStepBase):
    """Generate final predictions and probability outputs for all rows."""

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

    def _create_prediction_frame(self, dataset, predictions, probabilities, labels):
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
        predictions, probabilities = fit_and_predict_selected_strategy(
            config=self.config,
            X_fit=labeled_df[active_features].copy(),
            y_fit=label_encoder.transform(labeled_df[self.config.target_column].copy()),
            X_all=dataset[active_features].copy(),
            selection=selection,
            model_specs=model_specs,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Augment the dataset with predictions, probabilities and filled labels.
        final_df, unknown_mask = self._create_prediction_frame(dataset, predictions, probabilities, selection["labels"])
        unknown_rows_before = int(unknown_mask.sum())
        rows_filled = int(final_df.loc[unknown_mask, self.config.prediction_filled_column].notna().sum())
        predict_summary = self._with_schema(
            {
                "selection_type": selection["selection_type"],
                "selected_models": selection["selected_models"],
                "rows_filled": rows_filled,
                "rows_unknown_original": unknown_rows_before,
                "rows_needing_review": int(final_df[self.config.prediction_review_column].sum()),
                "prediction_confidence_threshold": self.config.prediction_confidence_threshold,
                "output_path": str(self.config.predict_dir / "final_predictions.joblib"),
            },
            "predict_summary",
        )
        self._persist_predictions(final_df, predict_summary)
        self._save_prediction_plots(final_df)

        write_predict_report(self.config, final_df, selection)
        write_index_report(self.config)
        print_formatted_txt(f"Filled {rows_filled} unknown labels.", "RESULTS")
        return predict_summary
