# Incremental Training and Filtered Evaluation Workflow

## Overview

The ML classification pipeline now supports **incremental training** and **filtered evaluation**, enabling efficient model exploration and experimentation without redundant computation.

### Key Features

1. **Incremental Training**: Skip retraining models that were already trained in a previous pipeline run
2. **Forced Retraining**: Option to force retraining of existing models when needed
3. **Filtered Evaluation**: Run evaluation on only a subset of trained models by changing `selected_models` in config

## Incremental Training

### Use Case

You trained models A, B, and C in your first run. Now you want to add model D to compare. Instead of retraining A, B, and C, the pipeline automatically reuses them and only trains D.

### How It Works

When `run_train()` is called:
1. The pipeline checks for an existing `model_specs.json` in the train directory
2. It compares the requested model candidates against previously trained models
3. Models that already exist are **skipped** and their specs are reused
4. Only new models are trained
5. Results are merged and reported

### Configuration

```python
from satellites.ml_classification import GeospatialClassificationPipeline, ClassificationPipelineConfig

# First run: Train 3 models
config = ClassificationPipelineConfig(
    project_dir="output_project",
    target_column="label",
    feature_columns=["feature1", "feature2"],
    selected_models=("random_forest", "extra_trees", "gradient_boosting"),
    # ... other config options
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()  # Trains all 3 models
```

#### Second run: Add a new model

```python
# Second run: Add keras_lstm without retraining existing models
config = ClassificationPipelineConfig(
    project_dir="output_project",  # Same project directory
    target_column="label",
    feature_columns=["feature1", "feature2"],
    selected_models=("random_forest", "extra_trees", "gradient_boosting", "keras_lstm"),
    # ... other config options
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)  # Note: run_check resets project directory by default
pipeline.run_prepare()
pipeline.run_train()  # Skips random_forest, extra_trees, gradient_boosting; trains only keras_lstm
```

### Force Retraining

Use `force_retrain_models=True` to retrain all models, even if they were previously trained:

```python
config = ClassificationPipelineConfig(
    project_dir="output_project",
    target_column="label",
    feature_columns=["feature1", "feature2"],
    selected_models=("random_forest", "extra_trees"),
    force_retrain_models=True,  # Force retraining all models
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()  # All models are retrained regardless of prior existence
```

### Console Output Example

```
Training models...
Incremental training enabled: checking for previously trained models
Skipping already-trained model 'random_forest' (set force_retrain_models=True to retrain)
Skipping already-trained model 'extra_trees' (set force_retrain_models=True to retrain)
__________________________________________________________________________________________________
Training model gradient_boosting
Model gradient_boosting finished in duration: 0:00:42
Incremental training: 2 models reused, 1 newly trained
```

## Filtered Evaluation

### Use Case

You trained 5 models but decided you only want to compare 3 of them in the evaluation step. Instead of re-running the entire pipeline, change `selected_models` and evaluation will only report on those 3.

### How It Works

When `run_evaluate()` is called:
1. The pipeline loads all trained models from `model_specs.json`
2. If `selected_models` is configured (not None), only models in that list are evaluated
3. Trained models not in `selected_models` are logged as skipped
4. Reports and metrics are generated only for the selected subset

### Configuration

```python
# Original config with 4 models trained
config_train = ClassificationPipelineConfig(
    project_dir="output_project",
    target_column="label",
    feature_columns=["feature1", "feature2"],
    selected_models=("random_forest", "extra_trees", "gradient_boosting", "knn"),
)

pipeline = GeospatialClassificationPipeline(config_train)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()      # Trains 4 models
pipeline.run_evaluate()   # Evaluates all 4 models
```

#### Now evaluate only a subset

```python
# New config with only 2 models selected
config_eval = ClassificationPipelineConfig(
    project_dir="output_project",  # Same project directory
    target_column="label",
    feature_columns=["feature1", "feature2"],
    selected_models=("random_forest", "gradient_boosting"),  # Only these 2
)

# Reuse the same pipeline instance or create a new one with updated config
pipeline = GeospatialClassificationPipeline(config_eval)
# Skip run_check and run_prepare; they would reset the directory
# Instead, proceed directly to evaluation
pipeline.run_evaluate()  # Evaluates only random_forest and gradient_boosting
```

### Console Output Example

```
Evaluating models...
Filtered evaluation: 2 models selected, 2 trained models skipped: ['knn', 'extra_trees']
```

### When selected_models is None

When `selected_models=None` (the default), all trained models are evaluated:

```python
config = ClassificationPipelineConfig(
    project_dir="output_project",
    target_column="label",
    feature_columns=["feature1", "feature2"],
    selected_models=None,  # Evaluate all trained models
)

pipeline = GeospatialClassificationPipeline(config)
# All trained models will be evaluated
```

## Workflow Examples

### Example 1: Progressive Model Addition

Add models one at a time to compare their contribution:

```python
df = pd.read_parquet("data.parquet")

# Phase 1: Train tree-based models
config = ClassificationPipelineConfig(
    project_dir="experiment_1",
    target_column="label",
    selected_models=("random_forest", "extra_trees", "gradient_boosting"),
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()       # Trains 3 tree models
pipeline.run_evaluate()    # Evaluates all 3
pipeline.run_predict()     # Predicts with all 3

# Phase 2: Add neural network for comparison
config = ClassificationPipelineConfig(
    project_dir="experiment_1",
    target_column="label",
    selected_models=("random_forest", "extra_trees", "gradient_boosting", "neural_network"),
    reset_project_dir_on_run_check=False,  # Prevent re-initialization
)

# Manually construct pipeline without run_check to preserve train artifacts
pipeline = GeospatialClassificationPipeline(config)
# Directly call run_train to add neural_network (skip check/prepare)
pipeline.run_train()       # Trains only neural_network
pipeline.run_evaluate()    # Evaluates all 4 models
pipeline.run_predict()     # Predicts with all 4
```

### Example 2: Model Comparison without Full Retraining

Train multiple models, then compare subsets:

```python
# Train comprehensive model set
config = ClassificationPipelineConfig(
    project_dir="comprehensive_model_run",
    target_column="label",
    selected_models=("random_forest", "extra_trees", "gradient_boosting",
                     "knn", "neural_network", "keras_lstm"),
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()           # Trains all 6 models
pipeline.run_evaluate()        # Evaluates all 6
pipeline.run_predict()

# Compare neural approaches only
config_neural = ClassificationPipelineConfig(
    project_dir="comprehensive_model_run",
    target_column="label",
    selected_models=("neural_network", "keras_lstm"),  # Only neural models
)

pipeline_neural = GeospatialClassificationPipeline(config_neural)
pipeline_neural.run_evaluate() # Evaluates only 2 neural models
```

### Example 3: Forced Recalibration of Specific Models

Retrain problematic models without retraining everything:

```python
# Original training
config = ClassificationPipelineConfig(
    project_dir="production_model",
    target_column="label",
    selected_models=("random_forest", "gradient_boosting", "keras_lstm"),
)

pipeline = GeospatialClassificationPipeline(config)
pipeline.run_check(df)
pipeline.run_prepare()
pipeline.run_train()  # Trains all 3 models

# Keras LSTM needs recalibration
config_retrain = ClassificationPipelineConfig(
    project_dir="production_model",
    target_column="label",
    selected_models=("keras_lstm",),  # Only keras_lstm
    force_retrain_models=True,         # Force retraining
)

pipeline_retrain = GeospatialClassificationPipeline(config_retrain)
pipeline_retrain.run_train()  # Only retrains keras_lstm
```

## Important Notes

### Backward Compatibility

- If `model_specs.json` does not exist (first run), all models are trained normally
- If `selected_models=None`, all trained models are evaluated (default behavior unchanged)
- Existing notebooks and configurations continue to work without modification

### Directory Structure

```
project_dir/
├── 01_check/
├── 02_prepare/
├── 03_train/
│   ├── model_specs.json              # Updated with each run_train()
│   ├── random_forest/
│   │   ├── best_model.joblib
│   │   └── cv_results.csv
│   ├── extra_trees/
│   │   ├── best_model.joblib
│   │   └── cv_results.csv
│   └── ...
├── 04_evaluate/
└── 05_predict/
```

The `model_specs.json` is the key artifact. It tracks:
- All trained models and their hyperparameters
- Feature mappings
- Artifact locations
- Prediction capability flags

### When Artifacts are Skipped

When a model is skipped (not retrained):
- Its CV results are reused from the previous `best_model.joblib`
- Its training summary and fold scores are preserved from the previous run
- Its hyperparameters remain unchanged

When a model is newly trained:
- New CV results overwrite previous data (if any)
- New training summary is computed
- Previous artifacts for that model are replaced

## Troubleshooting

### Q: My model isn't being retrained even though I expect it to

**A:** Check if it already exists in `model_specs.json`. Either:
1. Set `force_retrain_models=True` to force retraining
2. Delete the model's subdirectory in `03_train/` and the entry in `model_specs.json`

### Q: Models are being skipped but I need to retrain them

**A:** Use `force_retrain_models=True`:

```python
config = ClassificationPipelineConfig(
    # ... other config ...
    force_retrain_models=True,
)
```

### Q: Evaluation is not showing all my models

**A:** Check if `selected_models` is configured. Set it to `None` to evaluate all:

```python
config = ClassificationPipelineConfig(
    # ... other config ...
    selected_models=None,  # None means all trained models
)
```

### Q: How do I preserve training artifacts when changing selected_models?

**A:** Use `reset_project_dir_on_run_check=False` in your second config to prevent cleanup:

```python
config = ClassificationPipelineConfig(
    # ... other config ...
    reset_project_dir_on_run_check=False,  # Preserve previous artifacts
)
```

However, note that `run_check()` will still validate the current dataset and config, so it's safer to skip `run_check()` entirely when continuing from a previous run.

## API Reference

### ClassificationPipelineConfig

**New Configuration Options:**

#### `force_retrain_models: bool = False`

When `True`, forces retraining of all candidate models even if they exist in previous training runs.

```python
config = ClassificationPipelineConfig(
    # ... other options ...
    force_retrain_models=True,  # Default is False
)
```

#### `selected_models: tuple[str, ...] | None = None`

Restricts training and evaluation to specified model names. When `None`, all available models are included.

**Existing feature**, but now better integrated with:
- Training: Only new models not in previous `model_specs.json` are trained
- Evaluation: Only models in this list are loaded and evaluated

```python
config = ClassificationPipelineConfig(
    # ... other options ...
    selected_models=("random_forest", "gradient_boosting"),  # Only these 2
)
```

### TrainStep.run_train()

**Behavior Changes:**

1. Loads existing `model_specs.json` if present (and `force_retrain_models=False`)
2. Skips models already in `model_specs.json`
3. Logs which models are skipped vs. newly trained
4. Merges new results with existing specs before persisting

### EvaluateStep.run_evaluate()

**Behavior Changes:**

1. Filters loaded models to intersection of trained models and `selected_models`
2. Logs skipped trained models
3. Generates reports only for selected models

## See Also

- [ML Classification Pipeline README](README.md)
- [Configuration Guide](../../src/satellites/ml_classification/core/config.py)
- Example notebooks: `4_kozani_ml_classification.ipynb`, `4_axios_ml_classification.ipynb`
