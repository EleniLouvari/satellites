# ML Classification Pipeline

Restartable classification package designed for dataframe or GeoDataFrame projects.

For a complete source tree and generated-artifact reference, see
[STRUCTURE.md](STRUCTURE.md).

Pipeline steps:

1. `run_check(df)`
2. `run_prepare()`
3. `run_train()`
4. `run_evaluate()`
5. `run_predict()`

## New: Incremental Training & Filtered Evaluation

**Want to add new models without retraining everything?** Or evaluate only a subset of trained models?

See [INCREMENTAL_WORKFLOW.md](INCREMENTAL_WORKFLOW.md) for:
- **Incremental Training**: Skip already-trained models when re-running with new model candidates
- **Filtered Evaluation**: Evaluate only selected models without full retraining
- **Forced Retraining**: Option to retrain specific models when needed
- Complete workflow examples and API reference

## Reports

- `01_check/report.html`
- `02_prepare/report.html`
- `03_train/report.html`
- `04_evaluate/report.html`
- `final_dashboard/report.html` (created automatically when `run_evaluate()` finishes)
- `05_predict/report.html`

A top-level index page is written to:

- `report_index.html`

The final dashboard combines the OpenStreetMap-backed study-area map,
one combined parcel-level box-plot figure with one row per month, NDVI on the
left, NDWI on the right, and class/label on the shared x-axis; monthly
highest/lowest index lines faceted by modeled class; CV fold stability;
train-versus-test performance; and the selected best model or voting ensemble.
The static map PNG is not included.
Supporting plots and CSV tables are exported under `final_dashboard/plots` and
`final_dashboard/data`.

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
    # A supplied model entry replaces that model's complete default search space.
    # Models omitted from tune_params retain their default search spaces.
    tune_params={
        "gradient_boosting": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [1, 2],
            "model__max_features": ["sqrt", 0.3],
        },
    },
    optimize_class_probabilities=False,  # learn class multipliers from out-of-fold probabilities
    probability_multiplier_grid=(0.8, 1.0, 1.2, 1.5, 2.0),
    probability_optimization_iterations=2,
    probability_optimization_max_accuracy_drop=0.02,
    rank_confidence_enabled=True,
    rank_confidence_minimum_models=3,
    rank_confidence_high_min_borda=90.0,
    rank_confidence_high_max_range=2.0,
    rank_confidence_medium_min_borda=75.0,
    rank_confidence_medium_max_range=4.0,
    class_reliability_enabled=True,
    class_reliability_minimum_oof_support=100,
    class_reliability_high_min_precision=0.80,
    class_reliability_medium_min_precision=0.60,
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

ML-ready satellite GeoParquet usage:

```python
non_features = {"class", "parcel_id", "geometry", "batch_number"}
features = [column for column in parcels.columns if column not in non_features]

config = ClassificationPipelineConfig(
    project_dir="src/my_crop_classifier",
    target_column="class",
    id_column="parcel_id",
    # Optional spatial filling of dated period features.
    spatial_interpolation_method="nearest",
    feature_columns=features,
    spatial_split=True,
    selection_type="soft_voting",
    top_voting_models=3,
    prediction_confidence_threshold=0.60,  # Used only when rank confidence is disabled/unavailable.
)
```

When rank confidence is enabled, the pipeline retains every selected model's
probabilities long enough to compare its within-model class ordering. It keeps
Top-1/Top-2/Top-3 agreement, a 0-100 mean Borda score, and rank dispersion as
parcel diagnostics. H/M/L rank confidence itself uses only mean Borda, the
predicted-class rank range, Borda-winner agreement/ties, and model count. The numeric
`prediction_max_probability` is retained as a diagnostic only; it is not
treated as calibrated statistical confidence and does not control the review
flag when rank confidence is available.

By default, Step 4 uses selected-member OOF soft-voting predictions to calculate
one precision value per predicted class. No Top-1 bins, neighboring-bin pooling,
or class pooling are used. Classes with fewer than
`class_reliability_minimum_oof_support` predictions remain invalid and `LOW`.
The defaults map OOF class precision `>=0.80` to `HIGH`, `>=0.60` to `MEDIUM`,
and lower precision to `LOW`. Final confidence is the lower of parcel rank
confidence and this frozen class reliability level.

For soft voting, the predicted label remains probability-based. If the
rank-aggregated winner disagrees with that label, the parcel receives `LOW`
rank confidence and the disagreement is retained in the diagnostic columns.
Class reliability is never learned from final prediction rows or recalculated
on the test set. Step 4 freezes the OOF class table and rank thresholds in
`selection_summary.json`; Step 5 applies that exact contract even if the live
configuration later changes. The untouched holdout confidence artifacts verify
the frozen contract. Old selection summaries using class x Top-1 calibration
must rerun Step 4 before production prediction.

The zonal-statistics pipeline performs the time-series pivot before writing the
GeoParquet. The ML pipeline validates the one-row-per-parcel grain, holds out
complete spatial grid cells, selects the voting members from spatial
cross-validation, and reserves the test split for final reporting only.

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
