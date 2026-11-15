"""Check opt-in notebook configuration without leaking global settings to other tests."""

import importlib.util
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from shared import notebook


@pytest.fixture
def notebook_state(monkeypatch):
    """Restore display options, random generators, and environment after configuration."""
    display = pytest.importorskip("IPython.display")
    pytest.importorskip("seaborn")
    random_state, numpy_state = random.getstate(), np.random.get_state()
    shown = Mock()
    monkeypatch.setattr(display, "display", shown)
    for key in ("PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "TF_CUDNN_USE_AUTOTUNE", "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL"):
        monkeypatch.setenv(key, "original")
    with mpl.rc_context(), pd.option_context("display.float_format", None, "display.max_columns", 20, "display.max_colwidth", 50):
        try:
            yield shown
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)


def test_configuration_sets_display_and_reproducible_seeds(notebook_state):
    """Repeated setup reproduces both random streams and applies notebook display options."""
    notebook.configure_notebook(seed=17)
    first = (random.random(), np.random.random())
    notebook.configure_notebook(seed=17)
    assert (random.random(), np.random.random()) == first
    assert os.environ["PYTHONHASHSEED"] == "17"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"
    assert pd.options.display.max_columns is None
    assert pd.options.display.max_colwidth == 20
    assert pd.options.display.float_format(1.25) == "1.2500000000"
    assert "width:100%" in notebook_state.call_args.args[0].data


def test_tensorflow_setup_is_explicit_and_configures_cpu(monkeypatch, notebook_state):
    """The optional backend receives the seed after CPU-only environment settings are applied."""
    logger, seed = Mock(), Mock()
    backend = SimpleNamespace(get_logger=Mock(return_value=logger), random=SimpleNamespace(set_seed=seed))
    monkeypatch.setitem(sys.modules, "tensorflow", backend)
    notebook.configure_notebook(seed=23)
    seed.assert_not_called()
    notebook.configure_notebook(seed=23, tensorflow=True)
    seed.assert_called_once_with(23)
    logger.setLevel.assert_called_once_with("ERROR")
    assert {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "TF_CUDNN_USE_AUTOTUNE", "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL")
    } == {"CUDA_VISIBLE_DEVICES": "-1", "TF_CUDNN_USE_AUTOTUNE": "0", "OMP_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}


def test_constants_resolve_environment_without_notebook_side_effects(monkeypatch, notebook_state):
    """Reading constants does not configure displays or reseed the caller's generators."""
    monkeypatch.setenv("GIS_LIBRARY", "test-gis")
    monkeypatch.setenv("GLOBAL_WORK_DIR", "test-work")
    random_state, numpy_state = random.getstate(), np.random.get_state()
    spec = importlib.util.spec_from_file_location("shared.constants", Path(notebook.__file__).with_name("constants.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constants = vars(module)
    assert constants["gis_library_dir"] == "test-gis"
    assert constants["global_work_dir"] == "test-work"
    assert constants["SEED_NUMBER"] == 42
    assert constants["null_values_list"] == constants["numeric_null_values_list"] + constants["string_null_values_list"]
    for settings in constants["dict_sat_index_params"].values():
        assert settings["clip_values"][0] < settings["clip_values"][1]
        assert all(mpl.colors.is_color_like(color) for color in settings["colors"])
    notebook_state.assert_not_called()
    assert random.getstate() == random_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert pd.options.display.max_columns == 20
