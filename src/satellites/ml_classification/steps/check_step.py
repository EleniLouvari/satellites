"""Pipeline step for validating and profiling input datasets before preparation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from ..core.iqr import apply_iqr_filter
from ..core.metrics import build_feature_profile, optimize_dataframe, validate_input
from ..core.spatial_interpolation import interpolate_spatial_nulls

# The check stage validates inputs before any expensive fitting or artifact generation occurs.
from ..core.persistence import print_formatted_txt, reset_project_outputs, save_frame_csv, save_joblib, save_json, time_decorator
from ..reporting import write_check_report, write_index_report
from ..core import PipelineStepBase
from ..visuals import save_label_map, save_label_map_with_osm_basemap, save_missing_values_plot, save_target_distribution_plot


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
            duplicates = dataset.loc[dataset[self.config.id_column].duplicated(keep=False), self.config.id_column].astype(str).unique()[:10]
            raise ValueError(
                f"Modeling data must contain one row per {self.config.id_column}. "
                f"Duplicate examples: {duplicates.tolist()}."
            )
        # Check that each class has at least `cv_folds` labeled examples for cross-validation.
        counts = dataset.loc[dataset[self.config.target_column].notna(), self.config.target_column].astype(str).value_counts()
        insufficient = counts[counts < self.config.cv_folds]
        if not insufficient.empty:
            raise ValueError(f"Each class needs at least cv_folds distinct parcels. Found: {insufficient.to_dict()}")

    def _build_summary(self, dataset, requested_features, iqr_summary, interpolation_summary):
        # Separate numeric vs categorical requested features by dtype.
        numeric = [name for name in requested_features if pd.api.types.is_numeric_dtype(dataset[name])]
        categorical = [name for name in requested_features if name not in numeric]
        # Build labeled subset mask and frame for class statistics.
        labeled_mask = dataset[self.config.target_column].notna()
        labeled = dataset.loc[labeled_mask]
        # Detect features that are entirely null or constant among labeled rows.
        all_null = [name for name in requested_features if dataset[name].isna().all()]
        constant = [name for name in requested_features if labeled[name].nunique(dropna=True) <= 1]
        rejected = set(all_null + constant)
        active = [name for name in requested_features if name not in rejected]
        if not active:
            raise ValueError("No active features remain after removing all-null and constant columns.")
        # Assemble summary metadata used for downstream steps and reporting.
        summary = {
            "rows_total": int(len(dataset)), "rows_labeled": int(labeled_mask.sum()),
            "rows_unknown_target": int((~labeled_mask).sum()), "n_features_requested": len(requested_features),
            "n_features_active": len(active), "numeric_features": numeric, "categorical_features": categorical,
            "all_null_features": all_null, "constant_features": constant, "active_features": active,
            "iqr_filter": iqr_summary, "spatial_interpolation": interpolation_summary,
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
            self.config.check_dir, "check_step_schema_manifest",
            json_contracts={"check_summary.json": sorted(persisted_summary)},
            csv_contracts={"feature_profile.csv": profile.columns.tolist()},
        )
        return profile

    def _save_check_plots(self, dataset, labeled_df, requested_features):
        plots_dir = self.config.check_dir / "plots"
        # Create diagnostics: missing values heatmap, target distribution, and spatial label maps.
        save_missing_values_plot(dataset[requested_features + [self.config.target_column]], plots_dir / "missing_values.png")
        save_target_distribution_plot(labeled_df[self.config.target_column], plots_dir / "target_distribution.png", "Known Target Distribution")
        save_label_map(labeled_df, self.config.target_column, plots_dir / "known_labels_map.png", title="Known Labels Map",
                       max_geometries=self.config.max_map_geometries)
        save_label_map_with_osm_basemap(labeled_df, self.config.target_column, plots_dir / "known_labels_map_osm.html",
                                        title="Known Labels Map with OSM Basemap", max_geometries=self.config.max_map_geometries)

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

        # Validate entity-level constraints (unique IDs, class counts) and optimize memory/dtypes.
        self._validate_entities(dataset)
        dataset = optimize_dataframe(dataset, self.config, requested_features)
        # Build summary metadata and labeled subset for reporting.
        summary, labeled_df = self._build_summary(dataset, requested_features, iqr_summary, interpolation_summary)
        # Persist artifacts and create exploratory feature profile CSV.
        feature_profile = self._persist_check_artifacts(dataset, summary)
        # Save visual diagnostics for the check stage.
        self._save_check_plots(dataset, labeled_df, requested_features)
        write_check_report(self.config, summary, feature_profile)
        write_index_report(self.config)
        print_formatted_txt(f"Checked {summary['rows_total']} rows with {summary['n_features_active']} active features.", "RESULTS")
        return summary
