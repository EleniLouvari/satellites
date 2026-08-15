"""Pipeline step for validating and profiling input datasets before preparation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from ..core.iqr import apply_iqr_filter
from ..core.metrics import build_feature_profile, optimize_dataframe, validate_input
from ..core.longitudinal import reshape_longitudinal_dataset
from ..core.spatial_interpolation import interpolate_spatial_nulls

# The check stage validates inputs before any expensive fitting or artifact generation occurs.
from ..core.persistence import (
    print_formatted_txt,
    reset_project_outputs,
    save_frame_csv,
    save_joblib,
    save_json,
    time_decorator,
)
from ..reporting import write_check_report, write_index_report
from ..core import PipelineStepBase
from ..visuals import (
    save_label_map,
    save_label_map_with_osm_basemap,
    save_missing_values_plot,
    save_target_distribution_plot,
)


class CheckStep(PipelineStepBase):
    """Validate inputs, derive active features, and persist check artifacts."""

    @time_decorator
    def run_check(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate the raw dataframe and persist check-stage artifacts and plots."""
        # Start step-level validation and artifact generation.
        print_formatted_txt("Checking input data...", "SUBSECTION")
        if self.config.reset_project_dir_on_run_check:
            print_formatted_txt(f"Resetting project output folder: {self.config.project_dir}", "INFO")
            reset_project_outputs(self.config)
            self.config.ensure_directories()
            print_formatted_txt(f"Recreated project output folders under: {self.config.project_dir}", "INFO")

        raw_dataset = df.copy()
        if self.config.id_column not in raw_dataset.columns:
            if self.config.reshape_time_series:
                raise ValueError(f"id_column={self.config.id_column!r} is required when reshape_time_series=True.")
            raw_dataset[self.config.id_column] = raw_dataset.index.astype(str)
        else:
            raw_dataset[self.config.id_column] = raw_dataset[self.config.id_column].astype(str)

        validate_input(raw_dataset, self.config)
        # Reshape longitudinal dataset using the time column.
        dataset, requested_features, longitudinal_summary = reshape_longitudinal_dataset(raw_dataset, self.config)
        print_formatted_txt(f"Reshaped longitudinal dataset: {longitudinal_summary}", "INFO")

        # Replace configurable IQR outliers with nulls before optional interpolation.
        dataset, iqr_summary = apply_iqr_filter(dataset, requested_features, self.config)
        print_formatted_txt(f"IQR feature filtering: {iqr_summary}", "INFO")

        # Perform spatial interpolation of missing values in requested features.
        dataset, interpolation_summary = interpolate_spatial_nulls(dataset, requested_features, self.config)
        print_formatted_txt(
            "Feature nulls before spatial interpolation: "
            f"{interpolation_summary['null_counts_before_interpolation']}",
            "INFO",
        )
        print_formatted_txt(
            "Feature nulls after spatial interpolation: "
            f"{interpolation_summary['null_counts_after_interpolation']}",
            "INFO",
        )
        print_formatted_txt(f"Requested features: {requested_features}", "INFO")

        # Validate that the dataset has unique IDs and sufficient class representation for cross-validation.
        if self.config.enforce_unique_ids and dataset[self.config.id_column].duplicated().any():
            examples = dataset.loc[
                dataset[self.config.id_column].duplicated(keep=False),
                self.config.id_column,
            ].astype(str).unique()[:10]
            raise ValueError(
                f"Modeling data must contain one row per {self.config.id_column}. "
                f"Duplicate examples: {examples.tolist()}. Use reshape_time_series=True for parcel-period data."
            )
        entity_class_counts = dataset.loc[dataset[self.config.target_column].notna(), self.config.target_column].astype(str).value_counts()
        insufficient_classes = entity_class_counts[entity_class_counts < self.config.cv_folds]
        if not insufficient_classes.empty:
            raise ValueError(
                "Each class needs at least cv_folds distinct parcels after reshaping. "
                f"Found: {insufficient_classes.to_dict()}"
            )
        dataset = optimize_dataframe(dataset, self.config, requested_features)

        # Derive active features and summarize dataset characteristics.
        numeric_features = [column for column in requested_features if pd.api.types.is_numeric_dtype(dataset[column])]
        categorical_features = [column for column in requested_features if column not in numeric_features]
        labeled_mask = dataset[self.config.target_column].notna()
        labeled_df = dataset.loc[labeled_mask].copy()

        # Identify all-null and constant features, and derive the set of active features.
        all_null_features = [column for column in requested_features if dataset[column].isna().all()]
        constant_features = [column for column in requested_features if labeled_df[column].nunique(dropna=True) <= 1]
        active_features = [
            column
            for column in requested_features
            if column not in set(all_null_features + constant_features)
        ]
        if not active_features:
            raise ValueError("No active features remain after removing all-null and constant columns.")

        summary = {
            "rows_total": int(len(dataset)),
            "rows_labeled": int(labeled_mask.sum()),
            "rows_unknown_target": int((~labeled_mask).sum()),
            "n_features_requested": int(len(requested_features)),
            "n_features_active": int(len(active_features)),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "all_null_features": all_null_features,
            "constant_features": constant_features,
            "active_features": active_features,
            "longitudinal_reshape": longitudinal_summary,
            "iqr_filter": iqr_summary,
            "spatial_interpolation": interpolation_summary,
            "class_distribution": labeled_df[self.config.target_column].astype(str).value_counts().to_dict(),
            "config": asdict(self.config),
        }

        save_joblib(dataset, self.config.check_dir / "input_dataset.joblib")
        check_summary_with_schema = self._with_schema(summary, "check_summary")
        save_json(check_summary_with_schema, self.config.check_dir / "check_summary.json")
        feature_profile = build_feature_profile(dataset, active_features, self.config, requested_features)
        save_frame_csv(feature_profile, self.config.check_dir / "feature_profile.csv")
        self._save_schema_manifest(
            self.config.check_dir,
            "check_step_schema_manifest",
            json_contracts={
                "check_summary.json": sorted(check_summary_with_schema.keys()),
            },
            csv_contracts={
                "feature_profile.csv": feature_profile.columns.tolist(),
            },
        )

        plots_dir = self.config.check_dir / "plots"
        save_missing_values_plot(
            dataset[requested_features + [self.config.target_column]],
            plots_dir / "missing_values.png",
        )
        save_target_distribution_plot(
            labeled_df[self.config.target_column],
            plots_dir / "target_distribution.png",
            "Known Target Distribution",
        )
        save_label_map(
            labeled_df,
            self.config.target_column,
            plots_dir / "known_labels_map.png",
            title="Known Labels Map",
            max_geometries=self.config.max_map_geometries,
        )
        save_label_map_with_osm_basemap(
            labeled_df,
            self.config.target_column,
            plots_dir / "known_labels_map_osm.html",
            title="Known Labels Map with OSM Basemap",
            max_geometries=self.config.max_map_geometries,
        )
        write_check_report(self.config, summary, feature_profile)
        write_index_report(self.config)
        print_formatted_txt(
            f"Checked {summary['rows_total']} rows with {summary['n_features_active']} active features.",
            "RESULTS",
        )
        return summary
