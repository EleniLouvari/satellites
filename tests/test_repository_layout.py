"""Regression checks for package relocation and saved Python object compatibility."""
from concurrent.futures import ProcessPoolExecutor
import importlib
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from sklearn.linear_model import LogisticRegression


@pytest.mark.parametrize(("old", "new"), [
    ("common_libraries.io_library", "satellites.shared.io"),
    ("common_libraries.data_cleaning.outliers", "satellites.shared.data_cleaning.outliers"),
    ("common_libraries.catalog_s3_checks", "satellites.data_preparation.sources.hub.catalog_s3_checks"),
    ("eda_pipeline.pipeline", "satellites.eda.pipeline"),
    ("ml_classification.ml_classification_pipeline.core.models", "satellites.ml_classification.core.models"),
    ("ml_classification_pipeline.core.models", "satellites.ml_classification.core.models"),
    ("ml_classification.ml_classification_sensitivity.runner", "satellites.ml_classification.sensitivity.runner"),
    ("ml_classification_sensitivity.runner", "satellites.ml_classification.sensitivity.runner"),
    ("openeo_parcel_stats_pipeline.zonal_stats", "satellites.data_preparation.parcel_stats.openeo"),
    ("openeo_parcel_stats_pipeline.zonal_stats_job_manager", "satellites.data_preparation.parcel_stats.job_manager"),
    ("hub_catalog.planet_parcel_stats", "satellites.data_preparation.parcel_stats.planet"),
    ("hub_catalog.planet_delivery", "satellites.data_preparation.sources.planet.delivery"),
])
def test_historical_modules_share_identity(old, new):
    """Old monkeypatch targets and private symbols resolve to the same module."""
    assert importlib.import_module(old) is importlib.import_module(new)


def test_old_model_pickle_loads_and_predicts(monkeypatch):
    """A payload carrying the historical class path still loads a fitted model."""
    from satellites.ml_classification.core.models import ContiguousLabelClassifier

    data = pd.DataFrame({"value": [-2.0, -1.0, 1.0, 2.0]})
    model = ContiguousLabelClassifier(LogisticRegression(), num_classes=2).fit(data, np.array([0, 0, 1, 1]))
    expected = model.predict_proba(data)
    old_module = "ml_classification.ml_classification_pipeline.core.models"
    with monkeypatch.context() as context:
        context.setattr(ContiguousLabelClassifier, "__module__", old_module)
        payload = pickle.dumps(model)
    assert old_module.encode() in payload
    restored = pickle.loads(payload)
    assert type(restored) is ContiguousLabelClassifier
    np.testing.assert_allclose(restored.predict_proba(data), expected)


def test_old_config_pickle_preserves_run_paths(monkeypatch, tmp_path):
    from satellites.ml_classification import ClassificationPipelineConfig

    config = ClassificationPipelineConfig(project_dir=tmp_path / "ml", target_column="label", feature_columns=["value"])
    with monkeypatch.context() as context:
        context.setattr(ClassificationPipelineConfig, "__module__", "ml_classification.ml_classification_pipeline.core.config")
        payload = pickle.dumps(config)
    restored = pickle.loads(payload)
    assert type(restored) is ClassificationPipelineConfig
    assert restored.train_dir == config.train_dir
    assert restored.predict_dir == config.predict_dir


def test_planet_uses_shared_engine_and_survives_spawn(tmp_path):
    from satellites.data_preparation.parcel_stats import OpenEOZonalStats, ParcelStatsBase, PlanetBasemapZonalStats

    assert issubclass(PlanetBasemapZonalStats, ParcelStatsBase)
    assert not issubclass(PlanetBasemapZonalStats, OpenEOZonalStats)
    parcels = gpd.GeoDataFrame({"parcel_id": ["one"]}, geometry=[box(500000, 4500000, 500010, 4500010)], crs=2100)
    extractor = PlanetBasemapZonalStats(parcels, "2024-01-01", "2024-01-31", tmp_path, 2100)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(type, extractor).result(timeout=60) is PlanetBasemapZonalStats


def test_pipeline_imports_do_not_initialize_notebook_or_openeo(tmp_path):
    """A fresh interpreter checks imports without pytest's preloaded modules."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    code = f"""
import sys
sys.path.insert(0, {str(source_root)!r})
from satellites.eda import EDAPipeline
from satellites.ml_classification import GeospatialClassificationPipeline
from satellites.data_preparation.parcel_stats import PlanetBasemapZonalStats
assert 'tensorflow' not in sys.modules
assert 'openeo' not in sys.modules
assert 'satellites.shared.legacy.imports' not in sys.modules
assert 'satellites.shared.legacy.notebook_settings' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
