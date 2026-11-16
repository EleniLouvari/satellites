"""Pipeline step for validating and profiling input datasets before preparation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from ml_classification.shared.config.base import PipelineStepBase

# The check stage validates inputs before any expensive fitting or artifact generation occurs.
from ml_classification.shared.logging import print_formatted_txt, time_decorator
from ml_classification.shared.mapping import save_label_map
from ml_classification.shared.models.modeling_context import EXCLUDED_FROM_TRAINING_COLUMN, OUTLIER_COLUMN, training_rows_mask
from ml_classification.shared.persistence import (
    load_joblib,
    load_json,
    reset_project_outputs,
    save_frame_csv,
    save_joblib,
    save_json,
)
from ml_classification.shared.reports.report_index import write_index_report
from ml_classification.step_01_check.libraries.feature_profile import build_feature_profile
from ml_classification.step_01_check.libraries.iqr import apply_iqr_filter
from ml_classification.step_01_check.libraries.maps import save_label_map_with_osm_basemap
from ml_classification.step_01_check.libraries.plots import save_missing_values_plot, save_target_distribution_plot
from ml_classification.step_01_check.libraries.report import write_check_report
from ml_classification.step_01_check.libraries.spatial_interpolation import interpolate_spatial_nulls
from ml_classification.step_01_check.libraries.validation import optimize_dataframe, validate_input
from shared.io import read_data


class CheckStep(PipelineStepBase):
    """Validate inputs, derive active features, and persist check artifacts."""

    def _reset_outputs_if_requested(self) -> None:
        if not self.config.reset_project_dir_on_run_check:
            return
        # If requested, clear the project's output directories to start fresh.
        print_formatted_txt(f"Resetting project output folder: {self.config.project_dir}", "INFO")
        reset_project_outputs(self.config)
        # Recreate required directory structure after reset.
        self.config.ensure_directories()
        print_formatted_txt(f"Recreated project output folders under: {self.config.project_dir}", "INFO")

    def _normalize_and_validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        # Make a safe copy and ensure the ID column exists as string type.
        dataset = df.copy()
        if self.config.id_column not in dataset:
            dataset[self.config.id_column] = dataset.index.astype(str)
        else:
            dataset[self.config.id_column] = dataset[self.config.id_column].astype(str)
        # Run library-level validation for required fields and types.
        validate_input(dataset, self.config)
        return dataset

    def _validate_entities(self, dataset: pd.DataFrame) -> None:
        # Ensure ID uniqueness when configured; surface example duplicates.
        if self.config.enforce_unique_ids and dataset[self.config.id_column].duplicated().any():
            duplicates = (
                dataset.loc[dataset[self.config.id_column].duplicated(keep=False), self.config.id_column]
                .astype(str)
                .unique()[:10]
            )
            raise ValueError(
                f"Error: Modeling data must contain one row per {self.config.id_column}. Duplicate examples: {duplicates.tolist()}."
            )
        # Check that each class has at least `cv_folds` labeled examples for cross-validation.
        counts = dataset.loc[training_rows_mask(dataset, self.config), self.config.target_column].astype(str).value_counts()
        insufficient = counts[counts < self.config.cv_folds]
        if not insufficient.empty:
            raise ValueError(f"Error: Each class needs at least cv_folds distinct parcels. Found: {insufficient.to_dict()}")

    def _flag_null_rows(self, dataset: pd.DataFrame, requested_features: list[str]) -> tuple[pd.DataFrame, int]:
        # All-null features are rejected later in the summary, so they must not flag every row here.
        checked = [name for name in requested_features if not dataset[name].isna().all()]
        null_mask = dataset[checked].isna()
        rows_with_nulls = null_mask.any(axis=1)
        # Flagged rows stay in the dataset so the final model still predicts them.
        dataset = dataset.copy()
        dataset[EXCLUDED_FROM_TRAINING_COLUMN] = rows_with_nulls & (not self.config.fill_nulls)
        if not rows_with_nulls.any():
            print_formatted_txt("No nulls found in feature columns.", "INFO")
            return dataset, 0
        print_formatted_txt(f"{int(rows_with_nulls.sum())} {self.config.id_column} values have null features:", "WARNING")
        for row_id, row_nulls in zip(dataset.loc[rows_with_nulls, self.config.id_column], null_mask[rows_with_nulls].to_numpy()):
            null_columns = [name for name, is_null in zip(checked, row_nulls) if is_null]
            print_formatted_txt(f"{self.config.id_column}={row_id}: {null_columns}", "INFO")
        if self.config.fill_nulls:
            return dataset, 0
        excluded = int(rows_with_nulls.sum())
        print_formatted_txt(
            f"fill_nulls=False: excluding {excluded} rows with null features from train/validation/test; "
            "they are still predicted by the final model.",
            "WARNING",
        )
        return dataset, excluded

    def _flag_outlier_rows(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Exclude ``is_outlier=True`` rows from training when configured; they are still predicted."""
        if not self.config.exclude_outliers_from_training:
            return dataset, 0
        if OUTLIER_COLUMN not in dataset.columns:
            raise ValueError(
                f"Error: exclude_outliers_from_training=True requires a boolean '{OUTLIER_COLUMN}' column in the input data."
            )
        # A missing flag means "not an outlier", matching the inspection-priority adjustment.
        outliers = dataset[OUTLIER_COLUMN].eq(True).fillna(False).astype(bool)
        # Count only labeled rows not already excluded: these are the rows this option actually removes.
        excluded = int((outliers & training_rows_mask(dataset, self.config)).sum())
        # Flagged rows stay in the dataset so the final model still predicts them.
        dataset = dataset.copy()
        dataset[EXCLUDED_FROM_TRAINING_COLUMN] = dataset[EXCLUDED_FROM_TRAINING_COLUMN].astype(bool) | outliers
        print_formatted_txt(
            f"exclude_outliers_from_training=True: excluding {excluded} labeled {OUTLIER_COLUMN}=True rows from "
            "train/validation/test; they are still predicted by the final model.",
            "WARNING" if excluded else "INFO",
        )
        return dataset, excluded

    def _build_summary(self, dataset, requested_features, iqr_summary, interpolation_summary):
        # Separate numeric vs categorical requested features by dtype.
        numeric = [name for name in requested_features if pd.api.types.is_numeric_dtype(dataset[name])]
        categorical = [name for name in requested_features if name not in numeric]
        # Build labeled subset mask (training-eligible rows only) and frame for class statistics.
        labeled_mask = training_rows_mask(dataset, self.config)
        labeled = dataset.loc[labeled_mask]
        # Detect features that are entirely null or constant among labeled rows.
        all_null = [name for name in requested_features if dataset[name].isna().all()]
        constant = [name for name in requested_features if labeled[name].nunique(dropna=True) <= 1]
        rejected = set(all_null + constant)
        active = [name for name in requested_features if name not in rejected]
        if not active:
            raise ValueError("Error: No active features remain after removing all-null and constant columns.")
        # Assemble summary metadata used for downstream steps and reporting.
        summary = {
            "rows_total": len(dataset),
            "rows_labeled": int(labeled_mask.sum()),
            "rows_unknown_target": int(dataset[self.config.target_column].isna().sum()),
            "n_features_requested": len(requested_features),
            "n_features_active": len(active),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "all_null_features": all_null,
            "constant_features": constant,
            "active_features": active,
            "iqr_filter": iqr_summary,
            "spatial_interpolation": interpolation_summary,
            "class_distribution": labeled[self.config.target_column].astype(str).value_counts().to_dict(),
            "config": asdict(self.config),
        }
        return summary, labeled

    def _persist_check_artifacts(self, dataset, summary):
        # Persist normalized dataset and a validated summary (with schema) for later stages.
        save_joblib(dataset, self.config.check_dir / "input_dataset.joblib")
        persisted_summary = self._with_schema(summary, "check_summary")
        save_json(persisted_summary, self.config.check_dir / "check_summary.json")
        # Build a per-feature profile and save as CSV for exploratory analysis.
        profile = build_feature_profile(dataset, summary["active_features"], self.config, self.config.feature_columns)
        save_frame_csv(profile, self.config.check_dir / "feature_profile.csv")
        # Export schema contract describing produced artifacts.
        self._save_schema_manifest(
            self.config.check_dir,
            "check_step_schema_manifest",
            json_contracts={"check_summary.json": sorted(persisted_summary)},
            csv_contracts={"feature_profile.csv": profile.columns.tolist()},
        )
        return profile

    def _save_check_plots(self, dataset, labeled_df, requested_features):
        plots_dir = self.config.check_dir / "plots"
        # Create diagnostics: missing values heatmap, target distribution, and spatial label maps.
        save_missing_values_plot(dataset[requested_features + [self.config.target_column]], plots_dir / "missing_values.png")
        save_target_distribution_plot(
            labeled_df[self.config.target_column], plots_dir / "target_distribution.png", "Known Target Distribution"
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

    @time_decorator
    def run_check(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate the raw dataframe and persist check-stage artifacts and plots."""
        # Start step-level validation and artifact generation.
        print_formatted_txt("Checking input data...", "SUBSECTION")
        self._reset_outputs_if_requested()
        # Normalize and validate the raw DataFrame (IDs, required columns, types).
        dataset = self._normalize_and_validate_input(df)
        requested_features = list(self.config.feature_columns)

        # Optionally apply IQR-based filtering to remove outlier extreme values per feature.
        dataset, iqr_summary = apply_iqr_filter(dataset, requested_features, self.config)
        print_formatted_txt(f"IQR feature filtering: {iqr_summary}", "INFO")
        # Interpolate spatially-localized nulls when configured.
        dataset, interpolation_summary = interpolate_spatial_nulls(dataset, requested_features, self.config)
        print_formatted_txt(
            f"Feature nulls before spatial interpolation: {interpolation_summary['null_counts_before_interpolation']}", "INFO"
        )
        print_formatted_txt(
            f"Feature nulls after spatial interpolation: {interpolation_summary['null_counts_after_interpolation']}", "INFO"
        )
        print_formatted_txt(f"Requested features: {requested_features}", "INFO")
        # Report parcels with remaining nulls; exclude them from training unless null filling is enabled.
        dataset, interpolation_summary["rows_excluded_from_training_with_nulls"] = self._flag_null_rows(
            dataset, requested_features
        )
        # Optionally leave EDA outliers out of training as well; they also remain in the final predictions.
        dataset, outlier_rows_excluded = self._flag_outlier_rows(dataset)

        # Validate entity-level constraints (unique IDs, class counts) and optimize memory/dtypes.
        self._validate_entities(dataset)
        dataset = optimize_dataframe(dataset, self.config, requested_features)
        # Build summary metadata and labeled subset for reporting.
        summary, labeled_df = self._build_summary(dataset, requested_features, iqr_summary, interpolation_summary)
        summary["rows_excluded_from_training_as_outliers"] = outlier_rows_excluded
        # Persist artifacts and create exploratory feature profile CSV.
        feature_profile = self._persist_check_artifacts(dataset, summary)
        # Save visual diagnostics for the check stage.
        self._save_check_plots(dataset, labeled_df, requested_features)
        write_check_report(self.config, summary, feature_profile)
        write_index_report(self.config)
        print_formatted_txt(
            f"Checked {summary['rows_total']} rows with {summary['n_features_active']} active features.", "RESULTS"
        )
        return summary

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
        labeled_df = dataset.loc[training_rows_mask(dataset, self.config)].copy()
        self._save_check_plots(dataset, labeled_df, requested_features)
        write_check_report(self.config, summary, feature_profile)
        return str(self.config.check_dir / "report.html")
