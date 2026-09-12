# Project configuration

Keep reusable example settings here. Copy `project.example.toml` to
`project.local.toml` for machine-specific settings (ignored by Git).

Python 3.12+ includes a TOML reader:

```python
from pathlib import Path
import tomllib

config_file = Path("/path/to/satellites/configs/project.local.toml")
settings = tomllib.loads(config_file.read_text(encoding="utf-8"))
data_root = (config_file.parent / settings["paths"]["data_root"]).resolve()
output_root = (config_file.parent / settings["paths"]["output_root"]).resolve()
run_root = output_root / settings["project"]["name"] / settings["project"]["run"]

eda_output = run_root / "eda"
ml_output = run_root / "ml_classification"
```

Absolute data roots can point to existing external work directories. Keep
credentials in your existing environment, not in committed configuration.
The existing project notebooks retain their current paths so their inputs and
saved runs remain usable; this example is for creating new runs.
