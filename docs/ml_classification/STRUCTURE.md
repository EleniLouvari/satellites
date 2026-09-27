# ML Classification Pipeline Structure

This document describes the source-code layout of `satellites.ml_classification`,
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

## Source package tree

```text
src/satellites/ml_classification/
|-- core/
|   |-- __init__.py
|   |-- base.py
|   |-- class_reliability.py
|   |-- config.py
|   |-- inspection.py
|   |-- iqr.py
|   |-- metrics.py
|   |-- models.py
|   |-- persistence.py
|   |-- rank_confidence.py
|   |-- selection.py
|   |-- spatial_interpolation.py
|   |-- spatial_split.py
|   |-- tensorflow_lstm_models.py
|   `-- tensorflow_models.py
|-- reporting/
|   |-- __init__.py
|   |-- confidence_diagnostics.py
|   |-- final_dashboard.py
|   |-- html.py
|   |-- inspection_diagnostics.py
|   `-- reports.py
|-- sensitivity/
|   |-- __init__.py
|   `-- runner.py
|-- steps/
|   |-- __init__.py
|   |-- check_step.py
|   |-- evaluate_step.py
|   |-- predict_step.py
|   |-- prepare_step.py
|   `-- train_step.py
|-- visuals/
|   |-- __init__.py
|   |-- interpretability.py
|   |-- maps.py
|   `-- plots.py
|-- __init__.py
`-- pipeline.py
```

## Entry points and accompanying documentation

| File | Responsibility |
|---|---|
| `__init__.py` | Defines the small public API: `ClassificationPipelineConfig` and `GeospatialClassificationPipeline`. |
| `pipeline.py` | Composes the five step mixins into `GeospatialClassificationPipeline` and provides `run_all(df)`. |
| `docs/ml_classification/README.md` | Quick-start configuration, main concepts, supported options, and common usage examples. |
| `docs/ml_classification/STRUCTURE.md` | This package and generated-artifact reference. |
| `docs/ml_classification/INCREMENTAL_WORKFLOW.md` | Explains model reuse, adding models without retraining existing ones, forced retraining, and filtered evaluation. |

## `core/`: shared modeling logic

| File | Responsibility |
|---|---|
| `core/__init__.py` | Re-exports common core classes and helpers for concise internal imports. |
| `core/base.py` | Defines `PipelineStepBase`, which stores validated configuration and standardizes schema metadata and manifest persistence. |
| `core/config.py` | Defines and validates `ClassificationPipelineConfig`; normalizes options, applies safe caps, derives output columns, and exposes paths for every step directory. |
| `core/iqr.py` | Applies optional IQR outlier filtering to numeric features and returns bounds and replacement diagnostics. |
| `core/metrics.py` | Validates input, reduces dataframe memory use, builds feature profiles, loads modeling context, calculates metrics, ranks models for interpretability, and extracts feature importance. |
| `core/models.py` | Defines model candidates, sklearn-compatible wrappers, preprocessing, optional balancing, search spaces, and estimator reconstruction by model name. |
| `core/persistence.py` | Centralizes logging, timing, directory management, output reset behavior, and JSON/joblib/CSV serialization. |
| `core/rank_confidence.py` | Validates selected-member outputs, converts within-model probabilities to average ranks and 0-100 Borda scores, and assigns parcel rank confidence from Borda support, predicted-class rank range, and winner agreement. |
| `core/class_reliability.py` | Fits one OOF soft-voting precision record per predicted class, applies the frozen table, and combines it with parcel rank confidence using the lower level. |
| `core/selection.py` | Ranks probability models from CV, applies `top_voting_models`, evaluates soft voting, optimizes optional class multipliers, freezes selection, and refits the selected strategy. |
| `core/spatial_interpolation.py` | Fills missing spatial feature values with the configured nearest, IDW, or kriging method. |
| `core/spatial_split.py` | Implements spatial train/test splitting using grid intersections, class-preservation checks, and spatial grouping. |
| `core/tensorflow_models.py` | Implements the optional dense TensorFlow classifier and its sklearn-compatible wrapper. |
| `core/tensorflow_lstm_models.py` | Parses temporal features, constructs parcel time-series tensors, and implements the sklearn-compatible Keras LSTM classifier. |

## `steps/`: pipeline stages

| File | Responsibility |
|---|---|
| `steps/__init__.py` | Exports the five step mixins in workflow order. |
| `steps/check_step.py` | Step 1. Resets outputs when configured, validates input, optimizes dtypes, applies IQR/interpolation, profiles features, and persists checked data. |
| `steps/prepare_step.py` | Step 2. Encodes labels, creates train/test partitions, builds stratified or spatial CV folds, and persists modeling context. |
| `steps/train_step.py` | Step 3. Builds candidates, performs halving random search using `scoring_primary`, stores estimators and CV results, creates optional OOF probabilities, and supports incremental training. |
| `steps/evaluate_step.py` | Step 4. Evaluates base models, produces holdout diagnostics, freezes selection from training CV, computes voting/ranking ensembles, and creates interpretability artifacts. |
| `steps/predict_step.py` | Step 5. Refits the frozen strategy on all labeled rows, predicts the full dataset, adds probabilities and confidence fields, persists final output, and refreshes the final dashboard with validation-backed confidence diagnostics. |

## `reporting/`: HTML report construction

| File | Responsibility |
|---|---|
| `reporting/__init__.py` | Exports step-report and report-index writers. |
| `reporting/confidence_diagnostics.py` | Builds shared Borda-consensus correctness summaries, class-risk tables, confidence-component matrices, confusion views, and the interactive confidence-flow diagram. |
| `reporting/html.py` | Renders standalone styled HTML with key/value blocks, tables, semantic row colors, images, embeds, and links. |
| `reporting/reports.py` | Assembles all step reports and the index, including metrics, CV rankings, selection explanations, Voting highlights, and artifact links. |

## `visuals/`: generated figures and maps

| File | Responsibility |
|---|---|
| `visuals/__init__.py` | Re-exports plotting, mapping, and interpretability helpers. |
| `visuals/plots.py` | Creates data-quality, class-balance, distribution, correlation, CV, comparison, confusion-matrix, ROC, and prediction-fill plots. |
| `visuals/maps.py` | Creates static and optional interactive maps for known labels, spatial splits, and predictions, with geometry sampling for large data. |
| `visuals/interpretability.py` | Creates feature-importance charts and optional SHAP summaries, including multiclass per-class grids. |

## Main dependency direction

```text
pipeline.py
    -> steps/*
        -> core/*
        -> visuals/*
        -> reporting/*
            -> reporting/html.py
```

`core/` contains reusable computation. `steps/` coordinate core helpers and
persistence. `reporting/` and `visuals/` format results but do not decide which
model is selected.

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
| `final_predictions_preview.csv` | Full-row lightweight CSV with identifiers, target/prediction fields, rank-confidence diagnostics, declaration agreement, and inspection-priority fields. |
| `predict_summary.json` | Prediction counts, confidence diagnostics, selected strategy, output path, and schema metadata. |
| `data/inspection_*.csv` | Operational inspection summaries, matrices, predicted-crop rates, and the parcel audit queue used by the report. |
| `plots/*` | Filled-target distribution, enlarged inspection diagnostics, and optional predicted-label map. |

Every step-level `schema_manifest.json` records the expected JSON and CSV
contracts for that step. JSON summary files also include an `_schema` object
with the artifact name and configured schema version.

## Recommended reading order

1. Start with `README.md` to configure and run the pipeline.
2. Read `pipeline.py` and `steps/` to follow execution order.
3. Use `core/config.py` as the configuration reference.
4. Use `core/models.py`, `core/selection.py`, and `core/metrics.py` for model behavior and selection rules.
5. Use `reporting/` and `visuals/` when changing reports or figures.
6. Read `INCREMENTAL_WORKFLOW.md` before reusing artifacts across training runs.
