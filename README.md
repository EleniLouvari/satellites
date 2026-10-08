# Satellites

Satellite imagery acquisition, parcel statistics, exploratory analysis, and
restartable geospatial classification.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `src/data_preparation/sources/` | openEO cube acquisition, Planet delivery downloads, HUB catalog access |
| `src/data_preparation/parcel_stats/` | Shared local statistics engine and source-specific workflows |
| `src/data_preparation/features/` | Temporal reduction, elevation, soil, and nearest-lake features |
| `src/eda/` | EDA, feature-screening diagnostics, plots, and HTML reports |
| `src/ml_classification/` | Check, prepare, train, evaluate, predict, and seed sensitivity |
| `src/shared/` | Reusable I/O, logging, geometry, and data-cleaning utilities |
| `notebooks/projects/volvi/` | Numbered project workflow notebooks |
| `notebooks/experiments/` | Exploratory ML and sensitivity notebooks |
| `notebooks/maintenance/` | openEO job inspection and code-quality notebooks |
| `tests/` | Tests grouped by source-code responsibility |
| `configs/` | Example project configuration; machine-specific overrides stay local |
| `docs/` | Architecture, pipeline guides, manuals, and diagrams |
| `scripts/` | Development checks |
| `data/reference/` | Reference data such as the Planet tile grid |
| `outputs/` | Generated run artifacts and quality reports, ignored by Git |

Classification source is organized into `step_01_check/` through `step_05_predict/`,
each with its own `libraries/`, plus classification-specific `shared/` modules.
See the [source guide](src/ml_classification/README.md) and
[file usage index](src/ml_classification/FILE_USAGE.md) to trace implementations.

## Installation

Use the Python environment containing your geospatial dependencies. Install the
repository into that same environment (and select its kernel in Jupyter):

```powershell
python -m pip install --no-deps --no-build-isolation -e .
```

This command requires setuptools in that environment and does not upgrade the
installed scientific stack. It enables imports from `data_preparation`, `eda`,
`ml_classification`, and `shared` from any working directory, without notebook
`sys.path` edits. Reinstall after moving the source packages, then restart the
notebook kernel or Python process to load the new module paths.

For a new Python 3.12+ environment, install the dependency groups you need:

```powershell
python -m pip install -e ".[acquisition,database,ml,interpolation,notebooks,gis-utilities,dev]"
```

`neural` adds TensorFlow; `gis-utilities` supports the geometry/tabular helpers
used by the project notebooks. Legacy notebook convenience imports and older GIS
utilities have additional dependencies from the historical environment; they
are not loaded by the maintained pipeline entry points. See
[migration notes](docs/repository_structure.md) before changing existing environments.

`database` adds the PostgreSQL driver and SQLAlchemy for database-backed openEO
accounts. The Neuro fill notebook uses `load_openeo_users_from_db()` with
`DB_NAME`, `TBL_USERS`, and `POSTGRES_*` settings in the kernel environment.
See [openEO account setup](docs/data_preparation/openeo/CREDENTIALS.md).

## Public APIs

```python
from data_preparation.parcel_stats import (
    OpenEOZonalStats,
    OpenEOJobManagerZonalStats,
    PlanetBasemapZonalStats,
)
from data_preparation.features import append_parcel_elevation
from eda import EDAConfig, EDAPipeline
from ml_classification import (
    ClassificationPipelineConfig,
    GeospatialClassificationPipeline,
)
from ml_classification.sensitivity import ClassificationSensitivityRunner
```

The data flow is imagery acquisition → parcel statistics → environmental and
temporal features → EDA → classification. EDA and ML also accept independently
prepared DataFrames or GeoDataFrames; neither imports an acquisition workflow.

## Project data and outputs

Keep source data outside pipeline output directories. A recommended new run is:

```text
outputs/<project>/<run>/
  parcel_stats/
  features/
  eda/
  ml_classification/
    01_check/
    02_prepare/
    03_train/
    04_evaluate/
    05_predict/
  sensitivity/
```

The existing notebooks retain their external data and result paths to preserve
their current runs. Large imagery can remain in your external work directory.
The example in [configs](configs/README.md) shows how to configure new runs.

EDA and ML can reset their configured output directories; each pipeline should
receive its own dedicated run subdirectory. They create output folders themselves.

## Development

```powershell
python -m pytest
scripts\run_code_checks.bat
scripts\run_quality_checks.bat
```

Tests use `tests/`; generated reports go to `outputs/quality/`. The check scripts
use the active environment. See the [documentation index](docs/README.md) and
[migration notes](docs/repository_structure.md) for compatibility and validation details.
