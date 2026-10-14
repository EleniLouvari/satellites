"""Exercise the Neuro notebook's EDA join and feature selection without running the pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from tests.utils import expect_equal, expect_in, expect_true


def _notebook_context(eda):
    """Supply synthetic inputs to the actual notebook cells without external I/O."""
    notebook = Path(__file__).resolve().parents[2] / "notebooks/projects/neuro/4_ml_classification.ipynb"
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]}
    parcels = gpd.GeoDataFrame({
        "parcel_id": [1, 2, 3], "crop_category": [10, 20, 10],
        "data_reliability_score": [0.9] * 3, "geom_shape_complexity_score": [1.0] * 3,
        "NDVI_median__20240101": [0.3, 0.6, 0.4],
    }, geometry=[Point(500000 + index * 10, 4200000) for index in range(3)], crs=2100)
    inputs = {"parcels": parcels, "eda": eda}
    context = {
        "io_l": SimpleNamespace(read_data=lambda path: inputs[path].copy()),
        "PARCELS_PATH": "parcels", "EDA_REPORT_PATH": "eda",
        "parcel_id_column": "parcel_id", "class_column": "crop_category",
        "proposed_features": ["NDVI_median__20240101", "is_outlier"],
        "display": lambda *_: None,
    }
    return cells, context


def test_neuro_retains_outlier_metadata_outside_features():
    """Left join keeps unmatched parcels and column selection preserves the EDA flag."""
    cells, context = _notebook_context(pd.DataFrame({"parcel_id": [2, 1], "is_outlier": [False, True]}))
    exec(compile(cells["8fd58efc"], "neuro_eda_join", "exec"), context)
    exec(compile(cells["8519505d"], "neuro_feature_selection", "exec"), context)
    selected = context["gdf_parcels"]
    expect_equal(selected["parcel_id"].tolist(), [1, 2, 3])
    expect_equal(selected["is_outlier"].iloc[:2].tolist(), [True, False])
    expect_true(pd.isna(selected["is_outlier"].iloc[2]))
    expect_in("is_outlier", context["base_cols"])
    expect_true("is_outlier" not in context["features"])


def test_neuro_rejects_duplicate_eda_parcels():
    """An ambiguous EDA join must not duplicate classification entities."""
    cells, context = _notebook_context(pd.DataFrame({"parcel_id": [1, 1], "is_outlier": [True, False]}))
    with pytest.raises(pd.errors.MergeError, match="not a one-to-one merge"):
        exec(compile(cells["8fd58efc"], "neuro_eda_join", "exec"), context)
