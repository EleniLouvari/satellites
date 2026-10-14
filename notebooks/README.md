# Notebooks

Select a kernel where the repository has been installed in editable mode. The
notebooks import `data_preparation`, `eda`, `ml_classification`, and `shared`
directly and do not modify `sys.path`. Reinstall the editable package and restart
the kernel after the source-layout change.

| Folder | Contents |
| --- | --- |
| `projects/neuro/` | openEO parcel statistics, feature enrichment, EDA, and classification |
| `experiments/` | ML inspection and multi-seed sensitivity experiments |
| `maintenance/` | openEO job inspection and repository quality reports |

The numbered neuro notebooks stay together so the full workflow is easy to
follow: `1_parcel_stats_openEO_fill.ipynb` → `2_parcel_features.ipynb` →
`3_eda_report.ipynb` → `4_ml_classification.ipynb`.
Existing external input and output paths and saved notebook outputs are retained.
Review the path/configuration cell before running a notebook against your data.

The Neuro classification notebook joins EDA `is_outlier` by parcel ID with a
one-to-one check, retains it in metadata when selecting columns, and excludes it
from model features. It enables `inspection_include_outlier=True`: Step 5 raises
true outliers one priority band after the base rules, capped at Very High;
Unknown remains unscored. Reports show before/after priority transitions and
sort the inspection queue by final priority. See the
[inspection decision table](../docs/ml_classification/INSPECTION_PRIORITY_DECISION_TABLE.md).

`projects/neuro/1_parcel_stats_openEO_fill.ipynb` loads CDSE accounts from PostgreSQL
with `load_openeo_users_from_db()`. Install the `database` extra and provide
`DB_NAME`, `TBL_USERS`, and `POSTGRES_*` in the kernel environment. `.env` is not
loaded automatically. See [account setup and legacy resume ordering](../docs/data_preparation/openeo/CREDENTIALS.md).

New code uses explicit imports. `shared.notebook.configure_notebook`
provides opt-in display/reproducibility settings. Only notebooks requesting
`tensorflow=True` need TensorFlow for that setup.
