# Reorganization validation

## Direct `src/` layout validation — 2026-10-07

The four import packages now live directly under `src/`: `data_preparation`,
`eda`, `ml_classification`, and `shared`. Source imports, dynamic import strings,
notebooks, tests, packaging, coverage settings, scripts, and documentation were
updated. The enclosing namespace and references to deleted compatibility
packages are no longer part of package discovery.

Validation results:

- All Python source/test/script files and code cells in all eight notebooks parse.
- Every absolute internal source import resolves to an existing module path.
- The generated ML file-usage index matches the current source tree.
- The built wheel contains all 139 package Python modules under the four new
  package roots. Public openEO, Planet, EDA, ML, and shared I/O imports work
  directly from the wheel in an isolated interpreter, without the source tree.
- The collectable suite reports **328 passed and 4 failed**. Package-location,
  model/configuration round-trip, spawned-worker, and import-isolation tests pass.

The four failures are previously documented outdated tests in
`tests/ml_classification/test_train_evaluate_steps.py`: an instance method called
with the old signature, the old `probability_cache_test` keyword, and two mock
configurations missing `log_path`. The already uncollectable
`tests/ml_classification/test_rank_confidence.py` was explicitly excluded because
its `core.rank_confidence_calibration` implementation is absent. No test skip was
added to project configuration.

Command used with the existing `eo_spatial` Python environment:

```powershell
python -m pytest tests -q --ignore=tests/ml_classification/test_rank_confidence.py -p no:cacheprovider --basetemp outputs/quality/pytest_flat_src --junitxml=outputs/quality/flat_src_tests.xml
```

The wheel was built without downloading dependencies using the existing
`gdal3133_python3147` environment. The `eo_spatial` build attempt exposed an
unrelated installed `pbr` plugin requiring missing `pkg_resources`; that
environment was not modified. Runtime wheel imports were verified in `eo_spatial`.

Reinstall editable installations and restart kernels after this move. Existing
pickle/joblib artifacts storing previous custom-class paths may need regeneration;
external saved models were not rewritten. See the [import guide](import_migration.md).

## Previous reorganization validation

Validation used Python 3.13 from the existing `axhub_spatial` Conda environment,
through a workspace-local `.venv-checks` environment sharing its installed
dependencies. The external Conda environment was not modified.

## Original versus reorganized code

An isolated copy of the original tracked Python files was reconstructed under
`tmp/reorganization-baseline/` for comparison. Both runs excluded only
`test_rank_confidence.py`, which cannot collect because it imports the already
missing `core.rank_confidence_calibration` module.

| Run | Passed | Failed |
| --- | ---: | ---: |
| Original code, collectable suite | 195 | 5 |
| Reorganized code, including migration tests | 211 | 5 |

All five failures are the same pre-existing tests in `test_train_evaluate_steps.py`:

- `test_every_model_candidate_uses_its_configured_tune_params`: expects a support-vector candidate absent from the implementation.
- `test_summarize_search_returns_summary_and_per_fold`: calls an instance method with the former signature.
- `test_build_parcel_ranking_outputs_generates_rows_and_metrics`: uses the former `probability_cache_test` keyword.
- `test_run_train_resumes_unfinished_models`: its mock configuration lacks `log_path`.
- `test_run_train_recovers_trained_models_without_model_specs`: its mock configuration lacks `log_path`.

These failures and the missing-module collection issue were not hidden, skipped
in test configuration, or changed as part of a directory reorganization.

## Migration-specific checks

All 16 migration checks pass. They verify canonical/legacy module identity,
old nested and short-form ML imports, private module compatibility, fitted-model
and configuration pickles containing old class paths, shared Planet processing,
spawned-worker serialization, and pipeline imports without TensorFlow/notebook
initialization or an openEO dependency for Planet.

The full collectable suite also exercises local Planet manifests and downloads
with synthetic data/mocks, raster cleaning checkpoints, EDA exports, ML reports,
and openEO job-manager helpers. No remote downloads or real-data pipeline runs
were needed for verification.

The package builds as both an editable installation and a wheel. The wheel was
installed into a separate workspace folder and its canonical and compatibility
imports verified without relying on the source tree. Python source
compilation and notebook code-cell parsing pass. Ruff undefined-name checks pass
for all notebooks and maintained source outside the pre-existing legacy raster
utility. That utility's unresolved helper references are documented in the
[migration notes](repository_structure.md).

All 145 moved-file destinations were checked. The reference GeoPackage, Word
manuals, images and retained coverage data were compared with their originals.
All 10 notebooks retain their saved outputs, execution counts and other metadata;
their code cells parse successfully. Relative Markdown links resolve.

Detailed local evidence is retained under `outputs/quality/`, including
`original-tests.xml`, `reorganization-final.xml`, their text logs, the built wheel,
and `verification-requirements.txt`. Generated evidence is ignored by Git.
