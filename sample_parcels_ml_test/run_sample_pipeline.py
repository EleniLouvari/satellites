"""Run the corrected parcel classifier against the compact sample dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


sample_dir = Path(__file__).resolve().parent
repository_root = sample_dir.parent
source_candidates = [
    repository_root / "src",
    repository_root / "_satellite_src_review" / "src",
]
source_root = next((path for path in source_candidates if path.exists()), None)
if source_root is None:
    raise FileNotFoundError("Could not find the project src directory next to the sample folder.")
sys.path.insert(0, str(source_root))

from ml_classification.ml_classification_pipeline import (  # noqa: E402
    ClassificationPipelineConfig,
    GeospatialClassificationPipeline,
)


observations = gpd.read_parquet(sample_dir / "sample_parcels_ml_features_long.parquet")
features = pd.read_csv(sample_dir / "ml_feature_columns.csv")["feature"].tolist()

config = ClassificationPipelineConfig(
    project_dir=sample_dir / "pipeline_output",
    target_column="class",
    feature_columns=features,
    id_column="parcel_id",
    time_column="period_start",
    reshape_time_series=True,
    prediction_cutoff="2024-07-30",
    cv_folds=3,
    test_size=0.25,
    random_state=42,
    n_jobs=1,
    max_search_candidates=8,
    scoring_primary="f1_macro",
    selection_type="soft_voting",
    top_voting_models=3,
    spatial_split=True,
    spatial_split_grid_size=4,
    selected_models=(
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    ),
    prediction_confidence_threshold=0.60,
    max_map_geometries=1000,
    open_html_report=False,
)

pipeline = GeospatialClassificationPipeline(config)
summary = pipeline.run_all(observations)
print(summary)
