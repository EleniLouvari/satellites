"""Regression checks for top-level packages and serialized worker/model objects."""
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


@pytest.mark.parametrize("name", ["data_preparation", "eda", "ml_classification", "shared"])
def test_packages_are_direct_children_of_src(name):
    module = importlib.import_module(name)
    source_root = Path(__file__).resolve().parents[1] / "src"
    assert Path(module.__file__).resolve() == source_root / name / "__init__.py"


def test_model_pickle_loads_and_predicts():
    """A model saved with its current module path retains predictions."""
    from ml_classification.shared.models.models import ContiguousLabelClassifier

    data = pd.DataFrame({"value": [-2.0, -1.0, 1.0, 2.0]})
    model = ContiguousLabelClassifier(LogisticRegression(), num_classes=2).fit(data, np.array([0, 0, 1, 1]))
    expected = model.predict_proba(data)
    payload = pickle.dumps(model)
    assert b"ml_classification.shared.models.models" in payload
    restored = pickle.loads(payload)
    assert type(restored) is ContiguousLabelClassifier
    np.testing.assert_allclose(restored.predict_proba(data), expected)


def test_config_pickle_preserves_run_paths(tmp_path):
    from ml_classification import ClassificationPipelineConfig

    config = ClassificationPipelineConfig(project_dir=tmp_path / "ml", target_column="label", feature_columns=["value"])
    restored = pickle.loads(pickle.dumps(config))
    assert type(restored) is ClassificationPipelineConfig
    assert restored.train_dir == config.train_dir
    assert restored.predict_dir == config.predict_dir


def test_planet_uses_shared_engine_and_survives_spawn(tmp_path):
    from data_preparation.parcel_stats import OpenEOZonalStats, ParcelStatsBase, PlanetBasemapZonalStats

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
from eda import EDAPipeline
from ml_classification import GeospatialClassificationPipeline
from data_preparation.parcel_stats import PlanetBasemapZonalStats
assert 'tensorflow' not in sys.modules
assert 'openeo' not in sys.modules
assert 'shared.legacy.imports' not in sys.modules
assert 'shared.legacy.notebook_settings' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, capture_output=True, text=True, timeout=60)
