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

The preparation report's correlation heatmap uses numeric features whose names
contain the `_median` statistic token (for example, `NDVI_median__20240701`).
It shows the first 20 matching features in input order. The limit is the
`max_columns=20` argument of `save_correlation_heatmap` in
`src/ml_classification/step_02_prepare/libraries/plots.py`, not a pipeline
configuration field. This filter affects only the heatmap, not model inputs.
Regenerate an existing report with `pipeline.create_reports(step="prepare")`.

## New: Incremental Training & Filtered Evaluation

**Want to add new models without retraining everything?** Or evaluate only a subset of trained models?

See [INCREMENTAL_WORKFLOW.md](INCREMENTAL_WORKFLOW.md) for:
- **Incremental Training**: Skip already-trained models when re-running with new model candidates
- **Filtered Evaluation**: Evaluate only selected models without full retraining
- **Forced Retraining**: Option to retrain specific models when needed
- Complete workflow examples and API reference

## Support vector machine (SVM)

Use `selected_models=("support_vector_machine",)` to train SVM alone, or add
`"support_vector_machine"` to your existing model selection. It is also included
when `selected_models=None`.

The SVM uses numeric imputation and standardization, categorical imputation and
one-hot encoding, and the configured optional label balancing. Its class
probabilities support evaluation and soft voting. The existing default tuning
space searches `C`, kernel (`linear`, `rbf`, `poly`), and `gamma`; override it via
`tune_params["support_vector_machine"]`. Existing flat keys such as `model__C`
and `model__kernel` remain supported. `model__probability=True` is retained for
compatibility; disabling probabilities is rejected.

Probabilities use `CalibratedClassifierCV(SVC(), ensemble=False)` with sigmoid
calibration instead of the deprecated `SVC(probability=True)` option. Internal
stratified calibration uses up to five shuffled folds, seeded by `random_state`,
and reduces the fold count for small classes. Each fit requires at least two
classes. Calibration needs two samples per class; when a class has a single row
(common in the small, unstratified subsamples of the successive-halving search,
especially with `label_balancing_method="none"`), that fit skips calibration and
returns softmax-normalised SVC decision scores instead of failing the model. The
fitted estimator records the path taken in `calibrated_`. The final refit on the
full training data is calibrated whenever every class has at least two rows.
The base SVC is then refitted on all rows supplied to that fit.
Predicted labels follow the highest calibrated probability. Newly trained model
probabilities and labels can therefore differ from the previous libsvm calibration.
Existing saved models are not converted; retrain them to use this implementation.

[Scikit-learn's SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
notes that kernel SVM training scales at least quadratically with sample count,
and probability estimation adds internal cross-validation, so large parcel
datasets can take substantially longer to train.

## Reports

- `01_check/report.html`
- `02_prepare/report.html`
- `03_train/report.html`
- `04_evaluate/report.html`
- `final_dashboard/report.html` (created after `run_evaluate()` and refreshed with confidence diagnostics after `run_predict()`)
- `05_predict/report.html`

A top-level index page is written to:

- `report_index.html`

With `open_html_report=True`, the pipeline opens this index through a local
HTTP server. Use its links to view the step reports. Opening HTML files directly
with `file://` can cause OpenStreetMap to reject tile requests because the browser
does not send a web referrer. Tiles still require internet access.

To view an existing report from a notebook without rerunning any pipeline step:

```python
from ml_classification.shared.reports.report_server import open_report

open_report(config.check_dir / "report.html", root=config.project_dir)
```

The local viewer runs only on `127.0.0.1` and lasts for the lifetime of the Python
process/notebook kernel. For reports from a finished script, run the standalone
viewer from the repository root and leave it running while viewing:

```console
python -I src/ml_classification/shared/reports/report_server.py /path/to/project/report_index.html
```

The final dashboard combines the OpenStreetMap-backed study-area map,
one combined parcel-level box-plot figure with one row per month, NDVI on the
left, NDWI on the right, and class/label on the shared x-axis; monthly
highest/lowest index lines faceted by modeled class; CV fold stability;
train-versus-test performance; and the selected best model or voting ensemble.
After prediction, it also includes the Borda/correctness overview, predicted-class
need-to-check rates, and the class-reliability versus rank-confidence matrix.
The static map PNG is not included.
Supporting plots and CSV tables are exported under `final_dashboard/plots` and
`final_dashboard/data`.

Schema metadata:

- JSON summaries include `_schema` with `artifact` and `schema_version`.
- Each step writes `schema_manifest.json` with JSON/CSV output contracts.
- Prediction step also writes `05_predict/predict_summary.json`.

The Step 5 HTML report describes prediction confidence and class reliability as
separate signals. Its summary shows the confidence method, class reliability's
inspection-policy status, and the number of parcels where class reliability was
applied. Enabled with zero applied parcels is distinguished from disabled or
unrecorded application. These values come from saved predictions and the frozen
selection contract, so changing live configuration does not relabel a saved run.
The JSON summary retains its existing fields for compatibility.

To refresh an existing report after updating the code, restart the notebook kernel,
recreate `config` and `pipeline` for the same run directory, then call
`pipeline.create_reports(step="predict")`. This reads saved predictions without
training models or recalculating prediction and inspection decisions.

Minimal usage:

```python
from ml_classification import (
    GeospatialClassificationPipeline,
    ClassificationPipelineConfig,
)

config = ClassificationPipelineConfig(
    project_dir="outputs/example/run_001/ml_classification",
    target_column="label",
    feature_columns=["f1", "f2", "f3"],
    id_column="row_id",
    apply_iqr=False,
    iqr_lower_quantile=0.25,
    iqr_upper_quantile=0.75,
    iqr_multiplier=1.5,
    # Works with the original input table as well as reshaped time-series data.
    fill_nulls=False,  # False excludes rows with null features from training; True fills them
    exclude_outliers_from_training=False,  # True leaves is_outlier=True rows out of training; they are still predicted
    spatial_interpolation_method="nearest",  # None, nearest, idw, or kriging; used when fill_nulls=True
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
replaced with nulls. Spatial interpolation runs afterward only when
`fill_nulls=True` (which requires `spatial_interpolation_method`) and can fill
those nulls. `run_check` prints each `id_column` value that still has nulls and
the null columns. With `fill_nulls=False` (the default), rows with nulls that
were already in the input get `excluded_from_training=True`: they are left out
of the train/validation/test split, cross-validation, and the final refit, but
are still predicted by the final model in `run_predict`. Values blanked by the
IQR filter do not exclude a row: they stay null and every model pipeline imputes
them with statistics from its training folds (median for numeric features). All-null
feature columns are ignored for this check because they are rejected as features.
IQR bounds, replacement counts, `nulls_left_for_model_imputation`, and
`rows_excluded_from_training_with_nulls` are written to the check summary.

Set `exclude_outliers_from_training=True` to treat EDA outliers the same way:
rows with `is_outlier=True` get `excluded_from_training=True`, so they are left
out of the train/test split, cross-validation, and the final refit, but
`run_predict` still predicts them and they appear in `final_predictions` with
all other parcels. The input must contain a boolean `is_outlier` column (a
missing flag counts as not an outlier); otherwise `run_check` raises. The number
of labeled rows removed is written to the check summary as
`rows_excluded_from_training_as_outliers`. Every class must still have at least
`cv_folds` non-outlier parcels. If `is_outlier` is also selected as a feature it
becomes constant among the training rows and is dropped as a constant feature.
The default is `False`; the option is independent of `inspection_include_outlier`.

ML-ready satellite GeoParquet usage:

```python
non_features = {"class", "parcel_id", "geometry", "batch_number", "is_outlier"}
features = [column for column in parcels.columns if column not in non_features]

config = ClassificationPipelineConfig(
    project_dir="outputs/example/run_001/ml_classification",
    target_column="class",
    id_column="parcel_id",
    # Optional spatial filling of dated period features.
    fill_nulls=True,
    spatial_interpolation_method="nearest",
    feature_columns=features,
    spatial_split=True,
    selection_type="soft_voting",
    top_voting_models=3,
    prediction_confidence_threshold=0.60,  # Review cut-off without rank confidence; MEDIUM limit of the fallback.
    prediction_high_confidence_threshold=0.80,  # HIGH limit of the probability fallback.
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

Rank agreement needs at least `rank_confidence_minimum_models` voting models
(default 3). With fewer (a single-model run, or a vote of two), the level is
graded from the mean member probability of the predicted class instead: HIGH from
`prediction_high_confidence_threshold` (0.80), MEDIUM from
`prediction_confidence_threshold` (0.60), otherwise LOW. These predictions stay
valid, so they still receive an inspection priority. Their reasons start with
`probability_fallback_`, `prediction_confidence_source` is `probability_fallback`,
and `prediction_mean_probability` holds the graded value. Both limits are frozen
in Step 4 with the rank thresholds.

By default, Step 4 uses selected-member OOF soft-voting predictions to calculate
one precision value per predicted class. No Top-1 bins, neighboring-bin pooling,
or class pooling are used. Classes with fewer than
`class_reliability_minimum_oof_support` predictions remain invalid and `LOW`.

The defaults map OOF class precision `>=0.80` to `HIGH`, `>=0.60` to `MEDIUM`,
and lower precision to `LOW`. Final confidence uses parcel rank evidence only.
When enabled, class reliability enters the inspection decision table without
changing confidence or the confidence-only review flag.

For soft voting, the predicted label remains probability-based. If the
rank-aggregated winner disagrees with that label, the parcel receives `LOW`
rank confidence and the disagreement is retained in the diagnostic columns.
Class reliability is never learned from final prediction rows or recalculated
on the test set. Step 4 freezes the OOF class table and rank thresholds in
`selection_summary.json`; Step 5 applies that exact contract even if the live
configuration later changes. The untouched holdout confidence artifacts verify
the frozen contract. Old selection summaries using class x Top-1 calibration
must rerun Step 4 before production prediction.

Step 5 also creates an independent inspection-priority assessment when
`data_reliability_score` and `geom_shape_complexity_score` are available:

```text
confidence_risk = HIGH: 0.0, MEDIUM: 0.5, LOW: 1.0
data_risk = 1 - data_reliability_score
geometry_risk = percentile_rank(geom_shape_complexity_score)
inspection_score = 100 * (0.60*confidence_risk + 0.25*data_risk + 0.15*geometry_risk)
```

The score never sets the priority band, but it orders parcels inside a band in
the queue; `inspection_need_base` is a diagnostic only. Final priority
comes from a base rule table using class reliability, declared-versus-predicted
agreement, rank confidence, data reliability, and geometry risk, followed by the
optional `is_outlier` adjustment. When `inspection_include_outlier=True`, a true
flag raises the base priority one band, capped at Very High; Unknown stays unscored. See the
[complete 20-rule decision table](INSPECTION_PRIORITY_DECISION_TABLE.md) for every
combination, exact boundaries, disabled/missing-input behavior, and queue ordering.

The seven priorities are Very Low, Low, Low–Medium, Medium, Medium–High, High,
and Very High. `UNKNOWN` is an unscored status. Outputs include `inspection_need`,
`inspection_priority_label`, `inspection_rule`, `inspection_class_reliability_applied`,
`inspection_need_before_outlier`, `inspection_outlier_adjusted`,
`inspection_check_type`, and `inspection_reasons`. Base Very Low/Low need no check,
and neither do Low–Medium/Medium parcels whose declaration matches the prediction
(set `inspection_check_matching_declarations=True` to queue them);
base Medium–High/High/Very High disagreements are suspected declaration conflicts;
remaining cases need evidence verification. The data-reliability thresholds are
applied as written by default (`inspection_data_reliability_threshold_mode="absolute"`);
set `"percentile"` to calibrate them to the predicted parcels. The applied values are
saved as `inspection_data_reliability_low_threshold` and
`inspection_data_reliability_medium_threshold`. Outlier adjustment preserves check
type except NONE raised to Low–Medium or above enters evidence review; it never creates
a declaration conflict. The queue uses final priority first; within a band, parcels
with the higher `inspection_score` come first (unscored rows last, remaining ties
in input order), so the score weights set the visiting order inside a band.
By default geometry complexity uses only the absolute complexity thresholds
(`inspection_geometry_threshold_mode="absolute"`); with `"percentile"` the parcel must
pass both its percentile threshold and the complexity threshold. The result is
saved as `geometry_complexity_level`.

With existing saved member probabilities and an OOF reliability contract,
`pipeline.recalculate_prediction_quality()` updates prediction outputs without
retraining. Legacy guard contracts now supply independent policy evidence.
Rerun Step 4 to refresh evaluation artifacts or change its frozen reliability flag.
Recreating reports alone does not recalculate saved predictions.

The report keeps all existing model-confidence diagnostics in a
`Confidence Levels` tab and places the composite operational workflow in a separate
`Inspection Priority` tab. Inspection charts are exported one per full-width
image for readability. The operational tab includes check-type and need
summaries, EO evidence-quality and declaration-agreement matrices,
predicted-crop inspection rates, reliable declaration conflicts, a comparison
with the confidence-only review flag, the EDA outlier before/after priority
matrix, and the ordered parcel inspection queue. Operational charts use final
priority; diagnostic score and confidence charts are unaffected by outliers.

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

Row-wise allocation uses one shared grid for all classes and jointly chooses train
and test counts within each occupied class/cell. Every class must have at least two
rows. Both sets retain every class; the requested test count (`ceil(test_size * N)`)
is adjusted, with a warning, only when needed for class presence. Within those
sizes, allocation maximizes the number of occupied regions across the fine grid
and progressively coarser grids. This encourages broad area coverage even when
there are too few samples to visit every cell. Sparse cells may appear in only one
set. Class proportions break coverage ties, so they may differ from the full-data
proportions when broader coverage requires it. Rows without usable geometry retain
their class support but do not contribute to geographic coverage. The seed controls
row selection within the optimized class/cell counts. Optimization works on counts,
not individual rows; runtime depends on occupied class/cell combinations and folds.
This evaluates predictions within the sampled area, not transfer to unseen regions.

Cross-validation uses `cv_folds` validation subsets of the training data,
so the target validation fraction is `1 / cv_folds`. For `test_size=0.2`, use
`cv_folds=5` to match the holdout fraction; `test_size` itself controls the outer
train/test split. With the default `cv_validation_method="disjoint"`, every
training row appears in validation exactly once, as required for out-of-fold
probabilities.

With `cv_validation_method="overlapping"`, each of the `cv_folds` validation
folds is drawn independently (about `1 / cv_folds` of the training rows each),
so a row can be validated in several folds or in none. The draw follows the
configured split mode: stratified random sampling without `spatial_split`,
spatially spread rows for `by_row`, and whole grid cells for `by_group`; each
fold uses seed `random_state + fold`. These folds are used for the
hyperparameter search and CV scores. Out-of-fold probabilities (probability
optimization, rank confidence) still need one prediction per row, so the
prepare step also saves disjoint `oof_folds` in `prepare_summary.json` and
training uses them for OOF predictions.

With `by_row`, CV jointly allocates disjoint validation folds with sizes differing
by at most one row, maximizing multiscale coverage in both validation and its
training complement. Every class is retained in every training fold. Classes with
at least `cv_folds` training samples also appear in every validation fold; rarer
classes are spread across as many validation folds as possible, with a warning.
A class with only one training sample cannot support CV and raises an error.

With `by_group`, CV keeps whole grid cells together.
It compares row-balanced and class-stratified grouped assignments,
rejects assignments missing a class in any training fold, and prioritizes the
smallest worst-case deviation from the target validation fraction. Class balance
breaks ties in row balance. Exact sizes are not guaranteed: a cell with more than
20% of the training rows makes five balanced folds impossible. Increase
`spatial_split_grid_size` to create finer cells if appropriate for the intended
geographic separation. The preparation report shows actual and target validation
fractions.

### Include EDA outliers in inspection ranking

Keep `is_outlier` in the input parcel table as metadata, outside `feature_columns`.
Join EDA results by parcel ID with `validate="one_to_one"`, and retain the flag
when selecting input columns. The Neuro notebook does both and enables the option.
Set `inspection_include_outlier=True` in `ClassificationPipelineConfig` to raise
each flagged parcel's matrix priority by one level, capped at Very High.
Unknown stays unscored; missing flags leave priority unchanged. The default is
`False`. See the [outlier adjustment table](INSPECTION_PRIORITY_DECISION_TABLE.md#optional-eda-outlier-adjustment)
for auditing fields and recalculation behavior.

The report exports `data/inspection_outlier_priority_transitions.csv` and
`plots/inspection_outlier_priority_transitions.png`: rows are saved base priorities,
columns are final priorities, and cells count parcels. Diagonal cells are unchanged.
Older runs without `inspection_need_before_outlier` have no transition plot;
reports do not infer or recalculate missing decisions.
