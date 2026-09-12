# Satellites

Satellite imagery acquisition, parcel statistics, exploratory analysis, and
restartable geospatial classification.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `src/satellites/data_preparation/sources/` | openEO cube acquisition, Planet delivery downloads, HUB catalog access |
| `src/satellites/data_preparation/parcel_stats/` | Shared local statistics engine and source-specific workflows |
| `src/satellites/data_preparation/features/` | Temporal reduction, elevation, soil, and nearest-lake features |
| `src/satellites/eda/` | EDA, feature-screening diagnostics, plots, and HTML reports |
| `src/satellites/ml_classification/` | Check, prepare, train, evaluate, predict, and seed sensitivity |
| `src/satellites/shared/` | Reusable I/O, logging, geometry, and data-cleaning utilities |
| `notebooks/projects/volvi/` | Numbered project workflow notebooks |
| `notebooks/experiments/` | Exploratory ML and sensitivity notebooks |
| `notebooks/maintenance/` | openEO job inspection and code-quality notebooks |
| `tests/` | Tests grouped by source-code responsibility |
| `configs/` | Example project configuration; machine-specific overrides stay local |
| `docs/` | Architecture, pipeline guides, manuals, and diagrams |
| `scripts/` | Development checks |
| `data/reference/` | Reference data such as the Planet tile grid |
| `outputs/` | Generated run artifacts and quality reports, ignored by Git |
| `src/_compat/` | Old import paths forwarding to maintained implementations |

## Installation

Use the Python environment containing your geospatial dependencies. Install the
repository into that same environment (and select its kernel in Jupyter):

```powershell
python -m pip install --no-deps --no-build-isolation -e .
```

This command requires setuptools in that environment and does not upgrade the
installed scientific stack. It enables both canonical and compatibility imports
from any working directory, without notebook `sys.path` edits.

For a new Python 3.12+ environment, install the dependency groups you need:

```powershell
python -m pip install -e ".[acquisition,ml,interpolation,notebooks,gis-utilities,dev]"
```

`neural` adds TensorFlow; `gis-utilities` supports the geometry/tabular helpers
used by the project notebooks. Legacy notebook convenience imports and older GIS
utilities have additional dependencies from the historical environment; they
are not loaded by the maintained pipeline entry points. See
[migration notes](docs/repository_structure.md) before changing existing environments.

## Public APIs

```python
from satellites.data_preparation.parcel_stats import (
    OpenEOZonalStats,
    OpenEOJobManagerZonalStats,
    PlanetBasemapZonalStats,
)
from satellites.data_preparation.features import append_parcel_elevation
from satellites.eda import EDAConfig, EDAPipeline
from satellites.ml_classification import (
    ClassificationPipelineConfig,
    GeospatialClassificationPipeline,
)
from satellites.ml_classification.sensitivity import ClassificationSensitivityRunner
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
