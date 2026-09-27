"""Pipeline step for train/test splitting and cross-validation fold preparation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from ..core.persistence import load_joblib, load_json, print_formatted_txt, save_frame_csv, save_joblib, save_json, time_decorator
from ..reporting import write_index_report, write_prepare_report
from ..core import PipelineStepBase
from ..core.spatial_split import SpatialTrainTestSplitter
from ..visuals import (
    save_correlation_heatmap,
    save_spatial_split_plot,
    save_split_distribution_plot,
    save_train_test_feature_distributions,
)


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
            raise ValueError("At least two classes are required for classification.")
        # Fit a label encoder and store encoded targets used during training.
        encoder = LabelEncoder()
        labeled["_target_encoded"] = encoder.fit_transform(labeled[self.config.target_column])
        return labeled, active_features, encoder

    def _split_labeled_data(self, labeled_df):
        # Use either random stratified split or a spatial splitter based on config.
        if not self.config.spatial_split:
            train_df, test_df = train_test_split(
                labeled_df, test_size=self.config.test_size, random_state=self.config.random_state,
                stratify=labeled_df[self.config.target_column],
            )
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True), "random_stratified", None
        # Spatial splitting preserves geographic separation using a grid-based splitter.
        splitter = SpatialTrainTestSplitter(
            random_state=self.config.random_state, grid_size=self.config.spatial_split_grid_size
        )
        train_df, test_df = splitter.split(
            labeled_df=labeled_df, target_column=self.config.target_column,
            test_size=self.config.test_size, method=self.config.spatial_split_method,
        )
        return (
            train_df.reset_index(drop=True), test_df.reset_index(drop=True),
            f"spatial_{self.config.spatial_split_method}", splitter,
        )

    def _build_cv_folds(self, train_df, active_features, spatial_splitter):
        # Disjoint CV targets 1 / cv_folds of the training rows per validation fold.
        if spatial_splitter is None:
            splitter = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            split_iterator = splitter.split(train_df[active_features], train_df["_target_encoded"])
        else:
            # Keep cells intact while prioritizing row balance over class stratification.
            groups = spatial_splitter.build_spatial_groups(train_df)
            split_iterator = self._balanced_spatial_cv_splits(train_df, active_features, groups)
        folds = []
        all_labels = set(train_df["_target_encoded"])
        for fold_id, (train_idx, valid_idx) in enumerate(split_iterator):
            # For spatial folds, ensure every class appears in the training portion of each fold.
            if spatial_splitter is not None:
                missing = sorted(all_labels - set(train_df.iloc[train_idx]["_target_encoded"]))
                if missing:
                    raise ValueError(
                        f"Spatial CV fold {fold_id} has no training parcels for encoded classes {missing}. "
                        "Merge unsupported rare classes or reduce cv_folds."
                    )
            folds.append({"fold": fold_id, "train_index": train_idx.tolist(), "valid_index": valid_idx.tolist()})
        return folds

    def _balanced_spatial_cv_splits(self, train_df, active_features, groups):
        """Choose row-balanced whole-cell folds with every class retained in training."""
        if groups.nunique() < self.config.cv_folds:
            raise ValueError("Spatial CV requires at least cv_folds occupied grid cells. Increase spatial_split_grid_size.")
        target = train_df["_target_encoded"]
        overall_distribution = target.value_counts(normalize=True)
        all_labels = set(overall_distribution.index)
        candidates = []
        # GroupKFold balances row counts; the stratified candidate can rescue class
        # coverage or improve label balance when its fold sizes are equally good.
        splitters = (
            GroupKFold(n_splits=self.config.cv_folds),
            StratifiedGroupKFold(
                n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state
            ),
        )
        for splitter in splitters:
            splits = list(splitter.split(train_df[active_features], target, groups=groups))
            if any(set(target.iloc[train_idx]) != all_labels for train_idx, _ in splits):
                continue
            size_errors = [abs(len(valid_idx) / len(train_df) - 1 / self.config.cv_folds) for _, valid_idx in splits]
            class_error = sum(
                float((target.iloc[valid_idx].value_counts(normalize=True)
                       .reindex(overall_distribution.index, fill_value=0.0) - overall_distribution).abs().mean())
                for _, valid_idx in splits
            )
            score = (max(size_errors), sum(error ** 2 for error in size_errors), class_error)
            candidates.append((score, splits))
        if not candidates:
            raise ValueError(
                "Spatial CV could not retain every class in every training fold. "
                "Increase spatial_split_grid_size, merge unsupported rare classes, or reduce cv_folds."
            )
        return min(candidates, key=lambda candidate: candidate[0])[1]

    def _persist_prepare_artifacts(self, labeled_df, train_df, test_df, summary, label_encoder):
        # Persist train/test/labeled datasets and a prepare summary with schema.
        persisted = self._with_schema(summary, "prepare_summary")
        save_joblib(train_df, self.config.prepare_dir / "train_dataset.joblib")
        save_joblib(test_df, self.config.prepare_dir / "test_dataset.joblib")
        save_joblib(labeled_df, self.config.prepare_dir / "labeled_dataset.joblib")
        save_json(persisted, self.config.prepare_dir / "prepare_summary.json")
        # Also persist a mapping of labels to encoded integers for reproducibility.
        mapping = pd.DataFrame({
            "target_label": label_encoder.classes_,
            "target_encoded": list(range(len(label_encoder.classes_))),
        })
        save_frame_csv(mapping, self.config.prepare_dir / "label_mapping.csv")
        # Save schema manifest describing produced files and their expected columns.
        self._save_schema_manifest(
            self.config.prepare_dir, "prepare_step_schema_manifest",
            json_contracts={"prepare_summary.json": sorted(persisted)},
            csv_contracts={"label_mapping.csv": mapping.columns.tolist()},
        )

    def _save_prepare_plots(self, train_df, test_df, active_features):
        plots = self.config.prepare_dir / "plots"
        # Save diagnostic plots: class distribution, numeric correlations, feature distributions, and spatial split.
        save_split_distribution_plot(train_df[self.config.target_column], test_df[self.config.target_column], plots / "train_test_distribution.png")
        save_correlation_heatmap(train_df[active_features], plots / "numeric_correlation_heatmap.png")
        save_train_test_feature_distributions(
            train_df, test_df, active_features + [self.config.target_column],
            plots / "feature_distributions_train_vs_test.png", max_features=self.config.max_distribution_features,
        )
        save_spatial_split_plot(
            train_df, test_df, plots / "spatial_train_test_split.png",
            max_geometries=self.config.max_map_geometries, random_state=self.config.random_state,
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
