# ML Classification

This folder contains a restartable supervised-classification workflow for pandas
DataFrames and GeoPandas GeoDataFrames, a ready-made output-directory template,
and a runner for measuring sensitivity to the random seed.

## Package layout

- `ml_classification_pipeline/` contains the five-step training and inference
  pipeline. Each step persists its inputs and outputs, so later steps can be
  rerun without repeating the whole workflow.
- `ml_classification_project_template/` shows the expected runtime artifact
  layout (`01_check` through `05_predict`). Using the template is optional: the
  pipeline creates these directories automatically beneath `project_dir`.
- `ml_classification_sensitivity/` repeats the complete pipeline for multiple
  random seeds and produces a consolidated CSV, plot, statistics, and HTML
  report.

## Pipeline flow

| Step | Method | Purpose |
| --- | --- | --- |
| 1 | `run_check(df)` | Validate one-row-per-entity input, remove unusable features, profile it, and persist it. |
| 2 | `run_prepare()` | Encode labels and build the train/test split and reproducible CV folds. |
| 3 | `run_train()` | Tune candidate estimators and persist each fitted best model and its CV results. |
| 4 | `run_evaluate()` | Evaluate on the reserved test set, create diagnostics and interpretability outputs, and select a single model or soft-voting ensemble. |
| 5 | `run_predict()` | Refit the selected strategy on all labeled rows and predict every row, including probability, confidence, and review columns. |

`run_all(df)` executes the same five methods in order. Step methods after
`run_check` read the preceding artifacts from disk rather than accepting the
dataframe again.

## Basic usage

Run from an environment where the repository's `src` directory is on
`PYTHONPATH`:

```python
from ml_classification.ml_classification_pipeline import (
    ClassificationPipelineConfig,
    GeospatialClassificationPipeline,
)

config = ClassificationPipelineConfig(
    project_dir="src/my_classifier_run",
    target_column="class",
    id_column="parcel_id",
    feature_columns=["ndvi", "elevation", "soil_type"],
    random_state=42,
)

pipeline = GeospatialClassificationPipeline(config)
summaries = pipeline.run_all(df)
```

The target may be missing for rows that need classification, but the labeled
subset must contain at least two classes and at least `cv_folds` observations
per class. Numeric and categorical features are supported. The ID, target,
and geometry columns must not be included in `feature_columns`.

If `id_column` is absent, the dataframe index is used. A GeoDataFrame is
required only for spatial splitting and meaningful map outputs; ordinary
dataframe projects can use the default stratified split.

### Important output-directory behavior

`reset_project_dir_on_run_check=True` is the default. Consequently,
`run_check(df)` clears prior pipeline outputs under `project_dir` before writing
the new run. Use a dedicated run directory, or explicitly set
`reset_project_dir_on_run_check=False` when retaining existing outputs is
intentional. The five template directories are artifact destinations, not
places for source data or code.

## Common configuration

- `selected_models`: optional tuple of model names; `None` considers all
  available models. Optional XGBoost, LightGBM, and TensorFlow candidates
  require their respective libraries. The new TensorFlow option is exposed as
  `tensorflow_neural_network` and uses a separate Keras-based dense-network
  module so hidden layers, activations, dropout, and batch normalization can be
  configured independently of sklearn's `MLPClassifier`.
- `selection_type`: `"soft_voting"` (default) or `"single_model"`.
- `cv_ranking_method`: `"score_minus_std"` (default, rewards stability) or
  `"mean_score"`.
- `scoring_primary`: primary CV and comparison metric; defaults to `f1_macro`.
- `label_balancing_method`: `"none"`, `"random_oversample"`, or `"smote"`.
- `spatial_split=True`: enables geometry-based splitting. `by_group` holds out
  whole grid cells; `by_row` samples rows while retaining class coverage.
- `interpretability_top_models`: number of leading base models included in the
  interpretability section. SHAP is opt-in with
  `interpretability_include_shap=True`.
- `prediction_confidence_threshold`: rows below this probability are marked in
  `prediction_needs_review` (default `0.60`).
- `open_html_report=True`: open generated HTML reports after writing them.

Satellite zonal statistics already writes one ML-ready GeoParquet row per
parcel. Select its dated `feature__YYYYMMDD` columns directly:

Sentinel-2 index features in that artifact are calculated after IQR cleaning
and null filling of their physical source bands. This keeps indices sharing a
source band consistent and prevents independently interpolated index ratios.
The zonal-statistics pipeline also retains observation-only index state, so
valid-pixel counts exclude values that depend on an imputed source band.

```python
config = ClassificationPipelineConfig(
    project_dir="src/crop_classifier_run",
    target_column="class",
    id_column="parcel_id",
    feature_columns=[
        "NDVI_median__20240101",
        "NDVI_median__20240201",
        "VV_mean__20240101",
    ],
    spatial_split=True,
    spatial_split_method="by_group",
)
```

## Outputs

Every step writes `report.html` and `schema_manifest.json`; JSON summaries carry
an `_schema` block with an artifact name and schema version. The run root also
contains `report_index.html` and `pipeline.log`.

The principal artifacts are:

```text
project_dir/
+-- 01_check/       # persisted input, data profile, validation plots
+-- 02_prepare/     # train/test datasets, label mapping, split diagnostics
+-- 03_train/       # training summary and models/<name>/best_model.joblib
+-- 04_evaluate/    # metrics, selection summary, diagnostics, interpretation
+-- 05_predict/     # final_predictions.joblib and a CSV preview
+-- pipeline.log
`-- report_index.html
```

The complete prediction dataframe, including per-class probability columns, is
stored in `05_predict/final_predictions.joblib`. The CSV is intentionally a
compact preview.

## Sensitivity analysis

The sensitivity runner executes an independent full run for each seed. Its
`config_builder` must return a unique `project_dir` for each seed; otherwise the
default check-step reset will overwrite earlier runs.

```python
from pathlib import Path

from ml_classification.ml_classification_pipeline import ClassificationPipelineConfig
from ml_classification.ml_classification_sensitivity import ClassificationSensitivityRunner

run_root = Path("src/sensitivity_runs")

def build_config(seed: int) -> ClassificationPipelineConfig:
    return ClassificationPipelineConfig(
        project_dir=run_root / f"seed_{seed}",
        target_column="class",
        id_column="parcel_id",
        feature_columns=["ndvi", "elevation", "soil_type"],
        random_state=seed,
    )

runner = ClassificationSensitivityRunner(run_root / "summary")
results = runner.run(df, seeds=[7, 21, 42, 84], config_builder=build_config)
```

A failed seed is recorded rather than stopping the remaining experiments. The
summary directory contains:

- `sensitivity_results.csv` with success, selected model, test accuracy,
  project directory, and error for every seed;
- `accuracy_by_seed.png` and `accuracy_summary_stats.csv` when at least one run
  succeeds; and
- `sensitivity_report.html` with aggregate and per-seed results.

For a staged workflow, use `run_seeds(...)` first and pass its dataframe to
`create_report(...)` later.

## Detailed reference

See [`ml_classification_pipeline/README.md`](ml_classification_pipeline/README.md)
for pipeline-specific notes and
[`ml_classification_project_template/README.md`](ml_classification_project_template/README.md)
for the template contract.
