# ML Classification Pipeline Structure

This document describes the source-code layout of `ml_classification`,
the responsibility of every maintained file, and the output directory created
when the pipeline runs. Runtime caches such as `__pycache__` are intentionally
omitted.

## Execution flow

```text
Input DataFrame or GeoDataFrame
        |
        v
01_check -> 02_prepare -> 03_train -> 04_evaluate -> 05_predict
 validate      split       search       compare       refit selected
 clean         encode      fit          select        strategy and
 profile       CV folds    persist      explain       predict all rows
```

Each step persists the artifacts required by later steps. This makes the
workflow restartable: a later step can load earlier artifacts without keeping
all intermediate objects in memory.

## Source package and ownership

The maintained source is organized by workflow step:

```text
src/ml_classification/
  pipeline.py
  step_01_check/       check.py + libraries/
  step_02_prepare/     prepare.py + libraries/
  step_03_train/       train.py + libraries/
  step_04_evaluate/    evaluate.py + libraries/
  step_05_predict/     predict.py + libraries/
  shared/
    config/           base.py + config.py + tune_params.py
    models/           models.py + tensorflow_models.py + tensorflow_lstm_models.py
      modeling_context.py  load saved feature schema and label encoder for steps 3-5
    reports/          figure utilities, HTML, formatting, report index/server, diagnostics, dashboard
    probabilities.py  apply saved adjustments in evaluation and prediction
    ...               other modules used across steps
  sensitivity/        runner.py + libraries/
  to_delete/          previous wrappers and original source snapshots
```

Each step owns its algorithms, report assembly, plots, and report regeneration.
Its `libraries/report.py` uses the common rendering and report tools in `shared/reports/`.
`pipeline.py` coordinates the existing step classes. Shared modules never import
step implementations; private step libraries never import another step.

Evaluation owns `libraries/metrics.py`, `libraries/probability_optimization.py`,
and OOF reliability fitting in `libraries/class_reliability.py`. Applying frozen
class reliability and combining confidence remain in `shared/class_reliability.py`.
Prediction owns `libraries/inspection_priority/`, containing `scoring.py`,
`diagnostics.py`, and `plots.py`. The root `pipeline.py` remains the orchestrator.

- [Source guide and artifact lineage](../../src/ml_classification/README.md)
- [Generated file responsibilities and direct importers](../../src/ml_classification/FILE_USAGE.md)
- [Configuration](../../src/ml_classification/shared/config/config.py)

The former `core/`, `steps/`, `reporting/`, and `visuals/` directories are held in
`to_delete/` as compatibility wrappers until testing is complete. Original source
snapshots are in `to_delete/_archive/`. Old imports and saved class paths still resolve;
new code imports the defining numbered-step or `shared/` module directly.

## Dependency direction

```text
pipeline.py -> numbered step entry files -> their libraries -> shared modules
                         |---------------------------------> shared modules
sensitivity/runner.py -> pipeline.py + sensitivity/libraries
```

The workflow remains restartable through persisted artifacts. Source folder
numbers do not change the existing generated run directories below.

## Generated project output tree

The following tree is created below `ClassificationPipelineConfig.project_dir`.
Files marked **optional** are produced only when the corresponding data,
configuration, model capability, or dependency is available.

```text
project_dir/
|-- pipeline.log
|-- report_index.html
|-- 01_check/
|   |-- input_dataset.joblib
|   |-- check_summary.json
|   |-- feature_profile.csv
|   |-- schema_manifest.json
|   |-- report.html
|   `-- plots/
|       |-- missing_values.png
|       |-- target_distribution.png
|       |-- known_labels_map.png                 [optional]
|       `-- known_labels_map_osm.html            [optional]
|-- 02_prepare/
|   |-- train_dataset.joblib
|   |-- test_dataset.joblib
|   |-- labeled_dataset.joblib
|   |-- prepare_summary.json
|   |-- label_mapping.csv
|   |-- schema_manifest.json
|   |-- report.html
|   `-- plots/
|       |-- train_test_distribution.png
|       |-- numeric_correlation_heatmap.png
|       |-- feature_distributions_train_vs_test.png
|       `-- spatial_train_test_split.png         [optional]
|-- 03_train/
|   |-- training_summary.csv
|   |-- training_failed_models.csv
|   |-- best_cv_fold_scores.csv
|   |-- model_specs.json
|   |-- schema_manifest.json
|   |-- report.html
|   |-- plots/
|   |   `-- best_cv_fold_scores.png
|   `-- models/
|       `-- <model_name>/
|           |-- best_model.joblib
|           |-- cv_results.csv
|           |-- search_results.png
|           |-- oof_probabilities.joblib         [optional]
|           `-- temporal_schema.json             [Keras LSTM only]
|-- 04_evaluate/
|   |-- model_metrics_train.csv
|   |-- model_metrics_test.csv
|   |-- model_metrics_train_with_voting.csv      [optional]
|   |-- model_metrics_test_with_voting.csv       [optional]
|   |-- geo_classifier_style_train_metrics.csv
|   |-- geo_classifier_style_test_metrics.csv
|   |-- selection_summary.json
|   |-- schema_manifest.json
|   |-- report.html
|   |-- reports/
|   |   `-- <model_name>_classification_report.csv
|   |-- confusion_matrices/
|   |   `-- <model_name>_confusion_matrix.png
|   |-- panels/
|   |   `-- <model_name>_roc_confusion.png
|   |-- curves/                                  [binary ROC/PR outputs]
|   |-- plots/
|   |   |-- model_comparison.png
|   |   `-- model_comparison_with_voting.png     [optional]
|   |-- ranking/
|   |   |-- parcel_best_class_by_ranking.csv
|   |   |-- parcel_best_class_by_ranking_train.csv
|   |   |-- ranking_method_metrics.csv
|   |   `-- ranking_method_metrics_train.csv
|   |-- confidence/
|   |   |-- rank_confidence_test.csv
|   |   |-- rank_confidence_oof.csv
|   |   |-- class_reliability_oof.csv
|   |   |-- confidence_level_metrics_test.csv
|   |   |-- confidence_level_metrics_oof.csv
|   |   `-- confidence_by_class_test.csv
|   `-- interpretability/                        [optional]
|       |-- interpretability_summary.csv
|       |-- <model_name>_feature_importance.csv
|       |-- <model_name>_feature_importance.png
|       `-- <model_name>_shap_summary.png         [optional]
`-- 05_predict/
    |-- member_probabilities.joblib
    |-- final_predictions.joblib
    |-- final_predictions_preview.csv
    |-- predict_summary.json
    |-- schema_manifest.json
    |-- report.html
    |-- data/
    |   |-- inspection_need_summary.csv
    |   |-- inspection_check_type_summary.csv
    |   |-- inspection_priority_by_predicted_class.csv
    |   |-- inspection_declaration_conflict_matrix.csv
    |   |-- inspection_evidence_quality_matrix.csv
    |   |-- inspection_confidence_review_comparison.csv
    |   |-- inspection_need_by_declaration_status.csv
    |   |-- inspection_score_by_confidence_data.csv
    |   `-- inspection_top_priority_parcels.csv
    `-- plots/
        |-- filled_target_distribution.png
        |-- inspection_risk_relationships.png
        |-- inspection_check_type_distribution.png
        |-- inspection_priority_by_predicted_class.png
        |-- inspection_need_by_declaration_status.png
        |-- inspection_score_confidence_data_heatmap.png
        |-- inspection_declaration_conflicts.png   [when conflicts exist]
        |-- inspection_evidence_quality_heatmap.png
        `-- inspection_confidence_review_comparison.png
```

## Generated-file reference

### Project-level files

| File | Contents |
|---|---|
| `pipeline.log` | Timestamped messages, warnings, errors, durations, and returned step summaries. |
| `report_index.html` | Links to the five step reports. |

### Step 1: check

| File | Contents |
|---|---|
| `input_dataset.joblib` | Validated and optimized input, including configured cleaning and interpolation changes. |
| `check_summary.json` | Row counts, active features, data-quality diagnostics, configuration snapshot, and schema metadata. |
| `feature_profile.csv` | Feature dtype, missingness, uniqueness, and descriptive profile. |
| `plots/*` | Missing-value, target-distribution, and optional spatial label visualizations. |

### Step 2: prepare

| File | Contents |
|---|---|
| `train_dataset.joblib` / `test_dataset.joblib` | Frozen holdout split used for fitting/CV and final unbiased evaluation. |
| `labeled_dataset.joblib` | All rows with known target labels, used when refitting the final strategy. |
| `prepare_summary.json` | Active features, labels, split metadata, and CV fold indices. |
| `label_mapping.csv` | Original labels and their encoded integer values. |
| `plots/*` | Split balance, correlation, feature distribution, and optional spatial split diagnostics. |

### Step 3: train

| File | Contents |
|---|---|
| `training_summary.csv` | One row per model with mean CV score, fold standard deviation, selection score, ranking method, and probability support. This is the authoritative ranking input for Step 4. |
| `best_cv_fold_scores.csv` | Best candidate's validation score for every model and fold. |
| `training_failed_models.csv` | Candidate failures isolated so remaining models can continue. |
| `model_specs.json` | Best parameters and metadata needed to reconstruct estimators. |
| `models/<model>/best_model.joblib` | Best fitted search estimator for evaluation and incremental reuse. |
| `models/<model>/cv_results.csv` | Full hyperparameter-search results for the model. |
| `models/<model>/search_results.png` | Visual summary of that model's hyperparameter search. |
| `models/<model>/oof_probabilities.joblib` | Out-of-fold probabilities used to learn class multipliers without holdout leakage. |
| `models/keras_lstm/temporal_schema.json` | Temporal feature-to-timestep mapping required by the LSTM. |

### Step 4: evaluate

| File | Contents |
|---|---|
| `model_metrics_train.csv` / `model_metrics_test.csv` | Standard metrics for every evaluated base model. |
| `*_with_voting.csv` | Base-model metrics plus the soft-voting aggregate when an ensemble is available. |
| `geo_classifier_style_*_metrics.csv` | Legacy-compatible analytical layout with per-class values. |
| `selection_summary.json` | Frozen selection type, selected models, CV metric, labels, optional probability optimization, rank thresholds, and OOF class-reliability contract. Step 5 treats this as authoritative. |
| `reports/<model>_classification_report.csv` | Per-class precision, recall, F1, support, and aggregate rows. |
| `confusion_matrices/*`, `panels/*`, `curves/*`, `plots/*` | Model evaluation graphics referenced by the HTML report. |
| `ranking/parcel_best_class_by_ranking*.csv` | Per-row predictions from probability average/median and rank average/median. |
| `ranking/ranking_method_metrics*.csv` | Train/test performance summaries of the ranking aggregation methods. |
| `confidence/rank_confidence_*.csv` | Parcel-level holdout/OOF rank evidence, confidence levels, agreement, margin, dispersion, runner-up, and model-count diagnostics. |
| `confidence/class_reliability_oof.csv` | Frozen OOF prediction support, correct count, precision, level, and validity for each exact predicted class. |
| `confidence/confidence_*.csv` | Overall and per-class empirical correctness summaries for HIGH, MEDIUM, and LOW parcels. |
| `interpretability/*` | Feature importance and optional SHAP diagnostics for top CV-ranked models. These explain models but do not affect selection or prediction. |

### Step 5: predict

| File | Contents |
|---|---|
| `final_predictions.joblib` | Complete output with predictions, filled target, class probabilities, rank-confidence diagnostics, independent model/data/geometry risks, declaration agreement, label-aware inspection need/check type, and explainable reasons. |
| `member_probabilities.joblib` | Selected-model probabilities used to recalculate confidence without refitting. |
| `final_predictions_preview.csv` | Full-row lightweight CSV with identifiers, target/prediction fields, rank-confidence diagnostics, declaration agreement, and inspection-priority fields. |
| `predict_summary.json` | Prediction counts, confidence diagnostics, selected strategy, output path, and schema metadata. |
| `data/inspection_*.csv` | Operational inspection summaries, matrices, predicted-crop rates, and the parcel audit queue used by the report. |
| `plots/*` | Filled-target distribution, enlarged inspection diagnostics, and optional predicted-label map. |

Every step-level `schema_manifest.json` records the expected JSON and CSV
contracts for that step. JSON summary files also include an `_schema` object
with the artifact name and configured schema version.

## Recommended reading order

1. Read the [source guide](../../src/ml_classification/README.md).
2. Follow `pipeline.py` and the five numbered step entry files.
3. Follow explicit imports into each step's `libraries/` or `shared/`.
4. Use [FILE_USAGE.md](../../src/ml_classification/FILE_USAGE.md) to find consumers of any file.
5. Read `INCREMENTAL_WORKFLOW.md` before reusing artifacts across training runs.
