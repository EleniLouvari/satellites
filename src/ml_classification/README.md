# ML classification: reading and changing the code

Start with `pipeline.py`, then follow the numbered steps in order. Each step's
main file coordinates loading, processing, persistence, and reporting. Its
`libraries/` folder owns the detailed calculations, plots, and report assembly.
The public API remains:

```python
from ml_classification import ClassificationPipelineConfig, GeospatialClassificationPipeline
```

## Structure and reading order

| Order | Entry file | Helpers owned by the step |
| --- | --- | --- |
| 1 | [step_01_check/check.py](step_01_check/check.py) | Validation, feature profiles, IQR filtering, interpolation, check plots/maps, report |
| 2 | [step_02_prepare/prepare.py](step_02_prepare/prepare.py) | Spatial splitting/allocation, cross-validation folds, preparation plots/maps, report |
| 3 | [step_03_train/train.py](step_03_train/train.py) | Search construction and summaries, out-of-fold probabilities, training plots, report |
| 4 | [step_04_evaluate/evaluate.py](step_04_evaluate/evaluate.py) | Model evaluation/selection, ranking, confidence evidence, interpretability, plots, report |
| 5 | [step_05_predict/predict.py](step_05_predict/predict.py) | Final refitting, prediction quality, inspection diagnostics, prediction plots, report |
| Repeated experiments | [sensitivity/runner.py](sensitivity/runner.py) | Result records, tabular summaries, seed comparison plot, report |

`pipeline.py` composes the five existing step classes and dispatches `run_all()`
and `create_reports()`. Report regeneration methods live on the owning step.
`recalculate_prediction_quality()` lives on `PredictStep` and remains available
through the pipeline. Thin forwarding methods preserve existing step interfaces;
the calculation itself is in the explicitly imported library function.

## Shared code and dependency rules

`shared/` contains configuration, model, and report packages, plus distinct modules
for persistence, logging, probability adjustments, confidence
calculations, and map primitives. Common step behavior lives under `config/`.

```text
pipeline.py                         main orchestrator
shared/
  config/
    base.py                         common step initialization and artifact helpers
    config.py                       settings and validation
    tune_params.py                  default hyperparameter tuning grids
  models/
    modeling_context.py             load saved feature schema and label encoder
    models.py                       model builders, wrappers, preprocessing
    tensorflow_models.py            dense neural network implementation
    tensorflow_lstm_models.py       temporal LSTM implementation
  reports/
    plotting.py                     common figure styling and saving
    report_html.py                  shared HTML rendering
    report_helpers.py               CV tables and report formatting
    report_index.py                 links to step reports and the dashboard
    report_server.py                local HTTP report viewer
    confidence_diagnostics.py       confidence summaries and diagnostic figures
    final_dashboard.py              dashboard shared by evaluation and prediction
  probabilities.py                  apply saved probability adjustments
  class_reliability.py               apply saved reliability and combine confidence
step_04_evaluate/libraries/
  class_reliability.py               learn per-class reliability from training OOF predictions
  metrics.py                        score individual and voting models
  probability_optimization.py       learn adjustments from training OOF predictions
step_05_predict/libraries/
  inspection_priority/
    scoring.py                      inspection score, policy, explanations
    diagnostics.py                  inspection summaries and diagnostic figures
    plots.py                        inspection relationship plot
```

Training creates OOF probabilities in `step_03_train/libraries/oof_predictions.py`.
Training, evaluation, and prediction load their saved feature schema and label
encoder through `shared/models/modeling_context.py`; `models.py` builds estimators.
Evaluation uses those training predictions to learn probability adjustments;
evaluation and prediction both apply them through `shared/probabilities.py`.
Inspection priority consumes shared confidence evidence but is owned by prediction.

Each step retains its own `libraries/report.py`, which assembles that step's
report using `shared/reports/`. Plot and map helpers use the common figure saver
in `shared/reports/plotting.py`. Step-specific plots, maps, and inspection
diagnostics remain in their owning libraries. Shared report code never imports
step implementations.

- Code used by one step belongs in that step's `libraries/`, including its report.
- Code used by multiple steps belongs in `shared/`, with a responsibility-specific filename.
- Model definitions are shared because training builds candidates and prediction reconstructs them.
- The final dashboard is shared because evaluation creates it and prediction refreshes it.
- Shared code must not import a step or the pipeline. Step libraries must not import another step.
- Maintained code imports definitions directly; package initializers do not hide wildcard exports.
- General repository utilities live in `src/shared/`. Classification-specific utilities live here.

For example, `step_04_evaluate/libraries/confidence.py` calls
`step_04_evaluate/libraries/class_reliability.py` to learn the OOF reliability
contract. `shared/class_reliability.py` owns `get_class_reliability()` and
`combine_confidence_components()`, used by both evaluation and prediction to
apply the frozen contract. The contract method identifier also stays shared.
Cross-step data flow uses the artifacts below rather than imports between
private step libraries.

## Trace a file or output

[FILE_USAGE.md](FILE_USAGE.md) lists every Python module, its responsibility,
and its direct importers, including tests and notebooks. Follow an entry file's
imports to the exact implementation. Search a helper's name to find individual
calls within those consumers.

Regenerate the index after moving files or changing imports:

```powershell
python scripts/generate_ml_code_map.py
python scripts/generate_ml_code_map.py --check
```

The index is a static import map. Inheritance, CLI invocation, serialization,
and external notebooks can introduce callers that do not appear as imports.

## Artifact producers and consumers

Source directory names have changed; generated run directories retain their
existing `01_check/` through `05_predict/` names and schemas.

| Artifact relative to the configured project directory | Producer | Consumers |
| --- | --- | --- |
| `01_check/input_dataset.joblib` | Check | Prepare, predict, check report regeneration, final dashboard |
| `01_check/check_summary.json` | Check | Shared modeling-context loader, check report regeneration |
| `01_check/feature_profile.csv` | Check | Check report regeneration |
| `02_prepare/train_dataset.joblib`, `test_dataset.joblib` | Prepare | Train/evaluate, preparation reports |
| `02_prepare/labeled_dataset.joblib` | Prepare | Downstream users; current prediction code refits from the checked input |
| `02_prepare/prepare_summary.json`, `label_mapping.csv` | Prepare | Modeling-context loader, train/evaluate/predict, reports |
| `03_train/models/<model>/best_model.joblib`, `cv_results.csv` | Train | Incremental training, evaluate, training reports |
| `03_train/models/<model>/oof_probabilities.joblib` | Train | Evaluation probability optimization and confidence evidence |
| `03_train/model_specs.json`, `training_summary.csv` | Train | Incremental training, evaluate, predict, reports/dashboard |
| `03_train/best_cv_fold_scores.csv` | Train | Training report regeneration, final dashboard |
| `04_evaluate/selection_summary.json` | Evaluate | Predict, prediction-quality recalculation, report regeneration |
| `04_evaluate/model_metrics_*.csv`, `geo_classifier_style_*_metrics.csv` | Evaluate | Evaluation reports, final dashboard, sensitivity results |
| `04_evaluate/ranking/*.csv`, `confidence/*.csv` | Evaluate | Evaluation/prediction reports and confidence diagnostics |
| `05_predict/final_predictions.joblib` | Predict | Prediction report regeneration, final dashboard, quality recalculation |
| `05_predict/member_probabilities.joblib` | Predict | Quality recalculation without fitting models again |
| `05_predict/predict_summary.json`, `final_predictions_preview.csv` | Predict | Run inspection and downstream users |
| Step `report.html`, plots, diagnostic CSVs, `schema_manifest.json` | Owning step | Report readers and downstream artifact validation |
| `report_index.html`, final dashboard | Shared writers called by steps/pipeline | Report readers |

The detailed output reference is in
[docs/ml_classification/STRUCTURE.md](../../docs/ml_classification/STRUCTURE.md).
