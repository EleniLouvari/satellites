# ML Classification Pipeline

Restartable classification package designed for dataframe or GeoDataFrame projects.

Pipeline steps:

1. `run_check(df)`
2. `run_prepare()`
3. `run_train()`
4. `run_evaluate()`
5. `run_predict()`

Each step also writes an HTML report:

- `01_check/report.html`
- `02_prepare/report.html`
- `03_train/report.html`
- `04_evaluate/report.html`
- `05_predict/report.html`

A top-level index page is written to:

- `report_index.html`

Schema metadata:

- JSON summaries include `_schema` with `artifact` and `schema_version`.
- Each step writes `schema_manifest.json` with JSON/CSV output contracts.
- Prediction step also writes `05_predict/predict_summary.json`.

Minimal usage:

```python
from ml_classification_pipeline import (
    GeospatialClassificationPipeline,
    ClassificationPipelineConfig,
)

config = ClassificationPipelineConfig(
    project_dir="src/my_geo_classifier_project",
    target_column="label",
    feature_columns=["f1", "f2", "f3"],
    id_column="row_id",
    apply_iqr=False,
    iqr_lower_quantile=0.25,
    iqr_upper_quantile=0.75,
    iqr_multiplier=1.5,
    # Works with the original input table as well as reshaped time-series data.
    spatial_interpolation_method="nearest",  # None, nearest, idw, or kriging
    spatial_interpolation_max_distance_in_meters=None,  # required for idw
    cv_ranking_method="score_minus_std",
    spatial_split=False,
    spatial_split_method="by_group",
    spatial_split_grid_size=10,
    label_balancing_method="none",  # options: none, random_oversample, smote
    smote_k_neighbors=5,
    interpretability_top_models=3,
    interpretability_include_shap=False,
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()
pipeline.run_evaluate()
pipeline.run_predict()
```

When `apply_iqr=True`, numeric values outside the configured IQR fences are
replaced with nulls. Spatial interpolation, when configured, runs afterward and
can fill those nulls. IQR bounds and replacement counts are written to the
check summary.

Parcel time-series usage:

```python
non_features = {"class", "parcel_id", "period_start", "period_end", "geometry", "batch_number"}
features = [column for column in observations.columns if column not in non_features]

config = ClassificationPipelineConfig(
    project_dir="src/my_crop_classifier",
    target_column="class",
    id_column="parcel_id",
    time_column="period_start",
    reshape_time_series=True,
    prediction_cutoff="2024-09-30",
    # Optional spatial filling of the generated period features.
    spatial_interpolation_method="nearest",
    feature_columns=features,
    spatial_split=True,
    selection_type="soft_voting",
    top_voting_models=3,
    prediction_confidence_threshold=0.60,
)
```

This mode pivots parcel-period observations to one row per parcel, rejects
identifier/time leakage, holds out complete spatial grid cells, selects the
voting members from spatial cross-validation, and reserves the test split for
final reporting only.

Artifacts are saved inside:

- `01_check`
- `02_prepare`
- `03_train`
- `04_evaluate`
- `05_predict`

Training artifacts are organized by model under:

- `03_train/models/<model_name>/best_model.joblib`
- `03_train/models/<model_name>/cv_results.csv`
- `03_train/models/<model_name>/search_results.png`

Cross-validation ranking:

- `cv_ranking_method="score_minus_std"` ranks models by `mean CV score - std CV score`, which favors more stable models across folds.
- `cv_ranking_method="mean_score"` ranks models by mean CV score only.

Interpretability:

- `interpretability_top_models=3` adds a lightweight interpretability section in Step 4 for the top-ranked base models.
- Tree-based models in that top group get feature-importance CSV and plot outputs.
- `interpretability_include_shap=False` keeps SHAP off by default.
- Set `interpretability_include_shap=True` to generate SHAP summaries as an opt-in diagnostic.

Label balancing (optional):

- `label_balancing_method="none"` keeps the current behavior.
- `label_balancing_method="random_oversample"` applies random oversampling during CV/model fitting.
- `label_balancing_method="smote"` applies SMOTE during CV/model fitting.
- `smote_k_neighbors` controls SMOTE neighborhood size.

Spatial train-test split (optional):

- `spatial_split=False` keeps standard stratified random splitting.
- `spatial_split=True` enables a spatial train-test split.
- `spatial_split_method="by_group"` holds out complete grid cells so train and test are spatially disjoint.
- `spatial_split_method="by_row"` samples rows from across the grid while preserving every class in both sets; grid cells can occur in both sets.
- `spatial_split_grid_size` controls the grid granularity used to enforce spatial coverage.
