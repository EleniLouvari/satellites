"""Editable EDA metadata and strict web input validation."""

import json
from dataclasses import asdict, fields

from eda.core.config import EDAConfig

MANAGED = {"output_dir", "reset_output_dir", "artifact_names", "schema_version"}
GROUPS = {
    "basic": ("Report and column roles", ["report_title", "target_column", "target_task", "id_column"]),
    "quality": (
        "Data quality and outliers",
        [
            "high_missing_percent_threshold",
            "quasi_constant_threshold",
            "normality_sample_size",
            "outlier_zscore_threshold",
            "iqr_multiplier",
        ],
    ),
    "associations": (
        "Associations and feature screening",
        [
            "correlation_threshold",
            "categorical_association_max_unique_ratio",
            "feature_selection_alpha",
            "feature_selection_redundancy_threshold",
            "feature_selection_max_features",
            "feature_selection_exclude_columns",
        ],
    ),
    "display": (
        "Plots and analysis limits",
        [
            "max_features_per_plot",
            "max_categories",
            "max_target_levels_for_plots",
            "max_multivariate_features",
            "include_plots",
            "include_geospatial",
        ],
    ),
    "output": ("Output and reproducibility", ["include_html_report", "print_feature_selection_summary", "random_state"]),
}


def eda_defaults():
    """Return JSON-compatible options without a user-supplied filesystem location."""
    defaults = asdict(EDAConfig(output_dir="managed-per-analysis", reset_output_dir=False))
    defaults["output_dir"] = "A separate directory is allocated for every EDA analysis."
    return json.loads(json.dumps(defaults))


def eda_schema(columns):
    """Expose every dataclass option as editable metadata or explicit managed values."""
    defaults = eda_defaults()
    names = [column["name"] for column in columns if not column.get("geometry")]
    choices = {
        "target_task": ["auto", "classification", "regression"],
        "target_column": [None, *names],
        "id_column": [None, *names],
        "feature_selection_exclude_columns": names,
    }
    metadata = []
    for option in fields(EDAConfig):
        name = option.name
        if name in MANAGED:
            continue
        value = defaults[name]
        kind = (
            "boolean"
            if isinstance(value, bool)
            else "number"
            if isinstance(value, (int, float))
            else "array"
            if isinstance(value, list)
            else "string"
        )
        metadata.append(
            {
                "name": name,
                "type": kind,
                "default": value,
                "choices": choices.get(name),
                "group": next((group for group, (_, members) in GROUPS.items() if name in members), "output"),
                "description": "Leave blank for no feature-count limit." if name == "feature_selection_max_features" else "",
            }
        )
    return {
        "defaults": {k: v for k, v in defaults.items() if k not in MANAGED},
        "fields": metadata,
        "groups": [{"id": group, "label": title, "description": ""} for group, (title, _) in GROUPS.items()],
        "managed": {k: defaults[k] for k in MANAGED},
        "limits": {"upload_limit_bytes": 0, "max_batch_workers": 1},
    }


def validate_eda(settings, selected, available):
    """Reject unknown paths, invalid types and roles absent from the selected input."""
    defaults = eda_defaults()
    if not isinstance(settings, dict) or set(settings) - (set(defaults) - MANAGED):
        raise ValueError("Error: Unsupported or server-managed EDA setting.")
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(available):
        raise ValueError("Error: Select unique columns from the completed result.")
    options = {k: v for k, v in defaults.items() if k not in MANAGED} | settings
    for name, value in options.items():
        original = defaults[name]
        if value is None and name in {"target_column", "id_column", "feature_selection_max_features"}:
            continue
        if isinstance(original, bool):
            valid = type(value) is bool
        elif type(original) is int:
            valid = type(value) is int
        elif type(original) is float:
            valid = type(value) in {int, float}
        elif isinstance(original, list):
            valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
        else:
            valid = isinstance(value, str) and len(value) <= 200
        if not valid:
            raise ValueError(f"Error: Invalid value for {name}.")
    for name in ("target_column", "id_column"):
        if options[name] is not None and options[name] not in selected:
            raise ValueError(f"Error: Include the {name} in the selected columns.")
    if set(options["feature_selection_exclude_columns"]) - set(selected):
        raise ValueError("Error: Feature-screening exclusions must belong to the selected columns.")
    if not 0 <= options["random_state"] <= 2**32 - 1:
        raise ValueError("Error: random_state must be between 0 and 4294967295.")
    json.dumps(options, allow_nan=False)
    EDAConfig(output_dir="managed-per-analysis", reset_output_dir=False, **options)
    return options
