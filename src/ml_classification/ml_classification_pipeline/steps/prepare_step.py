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
        active_features = check_summary["active_features"]

        labeled_df = dataset.loc[dataset[self.config.target_column].notna(), :].copy().reset_index(drop=True)
        labeled_df[self.config.target_column] = labeled_df[self.config.target_column].astype(str)
        if labeled_df[self.config.target_column].nunique() < 2:
            raise ValueError("At least two classes are required for classification.")
        label_encoder = LabelEncoder()
        labeled_df["_target_encoded"] = label_encoder.fit_transform(labeled_df[self.config.target_column])

        split_strategy = "random_stratified"
        spatial_splitter = None
        if self.config.spatial_split:
            split_strategy = "spatial_uniform"
            spatial_splitter = SpatialTrainTestSplitter(
                random_state=self.config.random_state,
                grid_size=self.config.spatial_split_grid_size,
            )
            train_df, test_df = spatial_splitter.split(
                labeled_df=labeled_df,
                target_column=self.config.target_column,
                test_size=self.config.test_size,
            )
        else:
            train_df, test_df = train_test_split(
                labeled_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labeled_df[self.config.target_column],
            )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        folds = []
        if self.config.spatial_split:
            groups = spatial_splitter.build_spatial_groups(train_df)
            cv_splitter = StratifiedGroupKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            for fold_id, (train_idx, valid_idx) in enumerate(
                cv_splitter.split(
                    train_df[active_features],
                    train_df["_target_encoded"],
                    groups=groups,
                )
            ):
                folds.append(
                    {
                        "fold": fold_id,
                        "train_index": train_idx.tolist(),
                        "valid_index": valid_idx.tolist(),
                    }
                )
        else:
            cv_splitter = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
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

        prepare_summary = {
            "active_features": active_features,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_strategy": split_strategy,
            "target_labels": label_encoder.classes_.tolist(),
            "cv_folds": folds,
        }

        prepare_summary_with_schema = self._with_schema(prepare_summary, "prepare_summary")

        save_joblib(train_df, self.config.prepare_dir / "train_dataset.joblib")
        save_joblib(test_df, self.config.prepare_dir / "test_dataset.joblib")
        save_joblib(labeled_df, self.config.prepare_dir / "labeled_dataset.joblib")
        save_json(prepare_summary_with_schema, self.config.prepare_dir / "prepare_summary.json")
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
            json_contracts={
                "prepare_summary.json": sorted(prepare_summary_with_schema.keys()),
            },
            csv_contracts={
                "label_mapping.csv": label_mapping_df.columns.tolist(),
            },
        )

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
        write_prepare_report(self.config, prepare_summary, train_df, test_df)
        write_index_report(self.config)
        print_formatted_txt(
            f"Prepared train/test split: {prepare_summary['train_rows']} train rows, {prepare_summary['test_rows']} test rows.",
            "RESULTS",
        )
        return prepare_summary
