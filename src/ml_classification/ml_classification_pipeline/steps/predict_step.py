"""Pipeline step for full-dataset inference and final prediction artifact creation."""

from __future__ import annotations

from typing import Any

from ..core.metrics import load_modeling_context
from ..core.persistence import (
    load_joblib,
    load_json,
    print_formatted_txt,
    save_frame_csv,
    save_joblib,
    save_json,
    time_decorator,
)
from ..core.selection import fit_and_predict_selected_strategy
from ..reporting import write_index_report, write_predict_report
from ..core import PipelineStepBase
from ..visuals import save_predicted_labels_map, save_prediction_fill_plot


class PredictStep(PipelineStepBase):
    """Generate final predictions and probability outputs for all rows."""

    @time_decorator
    def run_predict(self) -> dict[str, Any]:
        """Run full-dataset inference and persist final prediction artifacts."""
        # Load required context and execute selected prediction strategy.
        print_formatted_txt("Classify all data...", "SUBSECTION")
        dataset = load_joblib(self.config.check_dir / "input_dataset.joblib")
        context = load_modeling_context(self.config)
        selection = load_json(self.config.evaluate_dir / "selection_summary.json")
        model_specs = load_json(self.config.train_dir / "model_specs.json")
        active_features = context["active_features"]
        numeric_features = context["numeric_features"]
        categorical_features = context["categorical_features"]
        label_encoder = context["label_encoder"]

        labeled_df = dataset.loc[dataset[self.config.target_column].notna(), :].copy()
        labeled_df[self.config.target_column] = labeled_df[self.config.target_column].astype(str)

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

        final_df = dataset.copy()
        unknown_mask = final_df[self.config.target_column].isna()
        unknown_rows_before = int(unknown_mask.sum())
        final_df[self.config.prediction_column] = predictions
        final_df[self.config.prediction_filled_column] = final_df[self.config.target_column].where(
            final_df[self.config.target_column].notna(),
            final_df[self.config.prediction_column],
        )
        rows_filled = int(final_df.loc[unknown_mask, self.config.prediction_filled_column].notna().sum())
        for class_index, class_label in enumerate(selection["labels"]):
            final_df[f"{self.config.probability_prefix}_{class_label}"] = probabilities[:, class_index]

        save_joblib(final_df, self.config.predict_dir / "final_predictions.joblib")
        save_frame_csv(
            final_df[
                [
                    self.config.id_column,
                    self.config.target_column,
                    self.config.prediction_column,
                    self.config.prediction_filled_column,
                ]
            ],
            self.config.predict_dir / "final_predictions_preview.csv",
        )
        save_prediction_fill_plot(
            final_df[self.config.target_column],
            final_df[self.config.prediction_filled_column],
            self.config.predict_dir / "plots" / "filled_target_distribution.png",
        )
        save_predicted_labels_map(
            final_df,
            self.config.prediction_filled_column,
            self.config.predict_dir / "plots" / "predicted_labels_map.png",
            title="Classified Predicted Labels Map",
            max_geometries=self.config.max_map_geometries,
        )
        predict_summary = self._with_schema(
            {
                "selection_type": selection["selection_type"],
                "selected_models": selection["selected_models"],
                "rows_filled": rows_filled,
                "rows_unknown_original": unknown_rows_before,
                "output_path": str(self.config.predict_dir / "final_predictions.joblib"),
            },
            "predict_summary",
        )
        save_json(predict_summary, self.config.predict_dir / "predict_summary.json")
        self._save_schema_manifest(
            self.config.predict_dir,
            "predict_step_schema_manifest",
            json_contracts={"predict_summary.json": sorted(predict_summary.keys())},
            csv_contracts={
                "final_predictions_preview.csv": [
                    self.config.id_column,
                    self.config.target_column,
                    self.config.prediction_column,
                    self.config.prediction_filled_column,
                ],
            },
        )

        write_predict_report(self.config, final_df, selection)
        write_index_report(self.config)
        print_formatted_txt(f"Filled {rows_filled} unknown labels.", "RESULTS")
        return predict_summary
