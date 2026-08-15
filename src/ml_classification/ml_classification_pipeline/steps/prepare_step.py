"""Pipeline step for train/test splitting and cross-validation fold preparation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from ..core.persistence import (
    load_joblib,
    load_json,
    print_formatted_txt,
    save_frame_csv,
    save_joblib,
    save_json,
    time_decorator,
)
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

    @time_decorator
    def run_prepare(self) -> dict[str, Any]:
        """Create train/test splits, CV folds, and persisted preparation outputs."""
        # Load validated input data and begin split preparation.
        print_formatted_txt("Preparing data...", "SUBSECTION")
        dataset = load_joblib(self.config.check_dir / "input_dataset.joblib")
        check_summary = load_json(self.config.check_dir / "check_summary.json")
        # Reuse the validated feature list so preparation cannot silently reintroduce rejected columns.
        active_features = check_summary["active_features"]

        # Unlabeled rows are retained by the check stage but cannot participate in supervised training.
        labeled_df = dataset.loc[dataset[self.config.target_column].notna(), :].copy().reset_index(drop=True)
        # Normalize labels to strings before encoding to support mixed scalar label types consistently.
        labeled_df[self.config.target_column] = labeled_df[self.config.target_column].astype(str)
        if labeled_df[self.config.target_column].nunique() < 2:
            raise ValueError("At least two classes are required for classification.")
        # Store a numeric target for estimators while preserving the original target column for reports.
        label_encoder = LabelEncoder()
        labeled_df["_target_encoded"] = label_encoder.fit_transform(labeled_df[self.config.target_column])

        # Default to a reproducible stratified holdout unless spatial isolation is explicitly requested.
        split_strategy = "random_stratified"
        spatial_splitter = None
        if self.config.spatial_split:
            split_strategy = f"spatial_{self.config.spatial_split_method}"
            # Assign nearby geometries to common blocks to reduce geographic leakage into the test set.
            spatial_splitter = SpatialTrainTestSplitter(random_state=self.config.random_state, grid_size=self.config.spatial_split_grid_size)
            train_df, test_df = spatial_splitter.split(
                labeled_df=labeled_df,
                target_column=self.config.target_column,
                test_size=self.config.test_size,
                method=self.config.spatial_split_method,
            )
        else:
            # Stratification keeps class proportions comparable between the random train and test sets.
            train_df, test_df = train_test_split(labeled_df, test_size=self.config.test_size, random_state=self.config.random_state, stratify=labeled_df[self.config.target_column])
        # Reset indices because persisted fold positions below are relative to these prepared frames.
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Persist explicit fold indices so every candidate model is evaluated on identical partitions.
        folds = []
        if self.config.spatial_split:
            # Group-aware folds keep all observations from a spatial block on the same side of a fold.
            groups = spatial_splitter.build_spatial_groups(train_df)
            cv_splitter = StratifiedGroupKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            for fold_id, (train_idx, valid_idx) in enumerate(cv_splitter.split(train_df[active_features], train_df["_target_encoded"], groups=groups)):
                # Every training fold must see every class or its fitted estimator cannot predict it reliably.
                fold_train_labels = set(train_df.iloc[train_idx]["_target_encoded"])
                all_train_labels = set(train_df["_target_encoded"])
                if fold_train_labels != all_train_labels:
                    missing = sorted(all_train_labels - fold_train_labels)
                    raise ValueError(
                        f"Spatial CV fold {fold_id} has no training parcels for encoded classes {missing}. "
                        "Merge unsupported rare classes or reduce cv_folds."
                    )
                folds.append(
                    {
                        "fold": fold_id,
                        "train_index": train_idx.tolist(),
                        "valid_index": valid_idx.tolist(),
                    }
                )
        else:
            # Ordinary stratified folds are sufficient when spatial blocking is disabled.
            cv_splitter = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            for fold_id, (train_idx, valid_idx) in enumerate(
                cv_splitter.split(train_df[active_features], train_df["_target_encoded"])
            ):
                folds.append(
                    {
                        "fold": fold_id,
                        "train_index": train_idx.tolist(),
                        "valid_index": valid_idx.tolist(),
                    }
                )

        # Keep the split metadata and fold definitions together as the contract consumed by training.
        prepare_summary = {
            "active_features": active_features,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_strategy": split_strategy,
            "target_labels": label_encoder.classes_.tolist(),
            "cv_folds": folds,
        }

        # Embed schema metadata only in the persisted form; callers receive the concise runtime summary.
        prepare_summary_with_schema = self._with_schema(prepare_summary, "prepare_summary")

        # Joblib preserves dataframe dtypes and geometry objects needed by later pipeline stages.
        save_joblib(train_df, self.config.prepare_dir / "train_dataset.joblib")
        save_joblib(test_df, self.config.prepare_dir / "test_dataset.joblib")
        save_joblib(labeled_df, self.config.prepare_dir / "labeled_dataset.joblib")
        save_json(prepare_summary_with_schema, self.config.prepare_dir / "prepare_summary.json")
        # Export the encoder mapping in a human-readable format for decoding predictions independently.
        label_mapping_df = pd.DataFrame(
            {
                "target_label": label_encoder.classes_,
                "target_encoded": list(range(len(label_encoder.classes_))),
            }
        )
        save_frame_csv(label_mapping_df, self.config.prepare_dir / "label_mapping.csv")
        self._save_schema_manifest(
            self.config.prepare_dir,
            "prepare_step_schema_manifest",
            json_contracts={"prepare_summary.json": sorted(prepare_summary_with_schema.keys())},
            csv_contracts={"label_mapping.csv": label_mapping_df.columns.tolist()},
        )

        # Generate diagnostics that reveal class, feature, or spatial drift introduced by the split.
        plots_dir = self.config.prepare_dir / "plots"
        save_split_distribution_plot(
            train_df[self.config.target_column],
            test_df[self.config.target_column],
            plots_dir / "train_test_distribution.png",
        )
        save_correlation_heatmap(train_df[active_features], plots_dir / "numeric_correlation_heatmap.png")
        save_train_test_feature_distributions(
            train_df,
            test_df,
            active_features + [self.config.target_column],
            plots_dir / "feature_distributions_train_vs_test.png",
            max_features=self.config.max_distribution_features,
        )
        save_spatial_split_plot(
            train_df,
            test_df,
            plots_dir / "spatial_train_test_split.png",
            max_geometries=self.config.max_map_geometries,
            random_state=self.config.random_state,
        )
        # Refresh both the stage report and the pipeline index after all artifacts are available.
        write_prepare_report(self.config, prepare_summary, train_df, test_df)
        write_index_report(self.config)
        print_formatted_txt(
            f"Prepared train/test split: {prepare_summary['train_rows']} train rows, {prepare_summary['test_rows']} test rows.",
            "RESULTS",
        )
        return prepare_summary
