# Project Template

This is the runtime artifact layout expected by `ml_classification`.

Folders:

- `01_check`
- `02_prepare`
- `03_train`
- `04_evaluate`
- `05_predict`

Set `ClassificationPipelineConfig.project_dir` to a dedicated folder such as
`outputs/<project>/<run>/ml_classification`. The pipeline creates these five
directories automatically. This template is documentation; do not use its
directory as a runtime output destination.
