"""Pipeline step for train/test splitting and cross-validation fold preparation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml_classification.shared.config.base import PipelineStepBase
from ml_classification.shared.logging import print_formatted_txt, time_decorator
from ml_classification.shared.persistence import load_joblib, load_json, save_frame_csv, save_joblib, save_json
from ml_classification.shared.reports.report_index import write_index_report
from ml_classification.step_02_prepare.libraries.cross_validation import (
    balanced_spatial_cv_splits,
    balanced_spatial_row_cv_splits,
    build_cv_folds,
)
from ml_classification.step_02_prepare.libraries.maps import save_spatial_split_plot
from ml_classification.step_02_prepare.libraries.plots import (
    save_correlation_heatmap,
    save_split_distribution_plot,
    save_train_test_feature_distributions,
)
from ml_classification.step_02_prepare.libraries.report import write_prepare_report
from ml_classification.step_02_prepare.libraries.spatial_split import SpatialTrainTestSplitter


class PrepareStep(PipelineStepBase):
    """Prepare labeled data splits, CV folds, and preparation artifacts."""

    def _load_labeled_data(self):
        # Load the validated dataset and active feature list from the check step artifacts.
        dataset = load_joblib(self.config.check_dir / "input_dataset.joblib")
        active_features = load_json(self.config.check_dir / "check_summary.json")["active_features"]
        # Keep only rows with known target labels for splitting and encoding.
        labeled = dataset.loc[dataset[self.config.target_column].notna()].copy().reset_index(drop=True)
        labeled[self.config.target_column] = labeled[self.config.target_column].astype(str)
        # Require at least two classes for classification tasks.
        if labeled[self.config.target_column].nunique() < 2:
            raise ValueError("Error: At least two classes are required for classification.")
        # Fit a label encoder and store encoded targets used during training.
        encoder = LabelEncoder()
        labeled["_target_encoded"] = encoder.fit_transform(labeled[self.config.target_column])
        return labeled, active_features, encoder

    def _split_labeled_data(self, labeled_df):
        # Use either random stratified split or a spatial splitter based on config.
        if not self.config.spatial_split:
            train_df, test_df = train_test_split(
                labeled_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labeled_df[self.config.target_column],
            )
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True), "random_stratified", None
        # Row mode spreads both sets across the grid; group mode holds out cells.
        splitter = SpatialTrainTestSplitter(random_state=self.config.random_state, grid_size=self.config.spatial_split_grid_size)
        train_df, test_df = splitter.split(
            labeled_df=labeled_df,
            target_column=self.config.target_column,
            test_size=self.config.test_size,
            method=self.config.spatial_split_method,
        )
        return (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            f"spatial_{self.config.spatial_split_method}",
            splitter,
        )

    def _build_cv_folds(self, train_df, active_features, spatial_splitter):
        # Disjoint CV targets 1 / cv_folds of the training rows per validation fold.
        return build_cv_folds(self.config, train_df, active_features, spatial_splitter)

    def _balanced_spatial_cv_splits(self, train_df, active_features, groups):
        """Choose row-balanced whole-cell folds with every class retained in training."""
        return balanced_spatial_cv_splits(self.config, train_df, active_features, groups)

    def _balanced_spatial_row_cv_splits(self, train_df, active_features, groups):
        """Jointly spread validation and training coverage, retaining class support."""
        return balanced_spatial_row_cv_splits(self.config, train_df, active_features, groups)

    def _persist_prepare_artifacts(self, labeled_df, train_df, test_df, summary, label_encoder):
        # Persist train/test/labeled datasets and a prepare summary with schema.
        persisted = self._with_schema(summary, "prepare_summary")
        save_joblib(train_df, self.config.prepare_dir / "train_dataset.joblib")
        save_joblib(test_df, self.config.prepare_dir / "test_dataset.joblib")
        save_joblib(labeled_df, self.config.prepare_dir / "labeled_dataset.joblib")
        save_json(persisted, self.config.prepare_dir / "prepare_summary.json")
        # Also persist a mapping of labels to encoded integers for reproducibility.
        mapping = pd.DataFrame(
            {"target_label": label_encoder.classes_, "target_encoded": list(range(len(label_encoder.classes_)))}
        )
        save_frame_csv(mapping, self.config.prepare_dir / "label_mapping.csv")
        # Save schema manifest describing produced files and their expected columns.
        self._save_schema_manifest(
            self.config.prepare_dir,
            "prepare_step_schema_manifest",
            json_contracts={"prepare_summary.json": sorted(persisted)},
            csv_contracts={"label_mapping.csv": mapping.columns.tolist()},
        )

    def _save_prepare_plots(self, train_df, test_df, active_features):
        plots = self.config.prepare_dir / "plots"
        # Save diagnostic plots: class distribution, numeric correlations, feature distributions, and spatial split.
        save_split_distribution_plot(
            train_df[self.config.target_column], test_df[self.config.target_column], plots / "train_test_distribution.png"
        )
        save_correlation_heatmap(train_df[active_features], plots / "numeric_correlation_heatmap.png")
        save_train_test_feature_distributions(
            train_df,
            test_df,
            active_features + [self.config.target_column],
            plots / "feature_distributions_train_vs_test.png",
            max_features=self.config.max_distribution_features,
        )
        save_spatial_split_plot(
            train_df,
            test_df,
            plots / "spatial_train_test_split.png",
            max_geometries=self.config.max_map_geometries,
            random_state=self.config.random_state,
        )

    @time_decorator
    def run_prepare(self) -> dict[str, Any]:
        """Create train/test splits, CV folds, and persisted preparation outputs."""
        # Load validated input data and begin split preparation.
        print_formatted_txt("Preparing data...", "SUBSECTION")
        labeled_df, active_features, label_encoder = self._load_labeled_data()
        train_df, test_df, split_strategy, spatial_splitter = self._split_labeled_data(labeled_df)
        folds = self._build_cv_folds(train_df, active_features, spatial_splitter)

        # Keep the split metadata and fold definitions together as the contract consumed by training.
        prepare_summary = {
            "active_features": active_features,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_strategy": split_strategy,
            "target_labels": label_encoder.classes_.tolist(),
            "cv_folds": folds,
        }

        # Persist artifacts, produce plots, and write reports used by later stages.
        self._persist_prepare_artifacts(labeled_df, train_df, test_df, prepare_summary, label_encoder)
        self._save_prepare_plots(train_df, test_df, active_features)
        # Refresh both the stage report and the pipeline index after all artifacts are available.
        write_prepare_report(self.config, prepare_summary, train_df, test_df)
        write_index_report(self.config)
        print_formatted_txt(
            f"Prepared train/test split: {prepare_summary['train_rows']} train rows, {prepare_summary['test_rows']} test rows.",
            "RESULTS",
        )
        return prepare_summary

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
