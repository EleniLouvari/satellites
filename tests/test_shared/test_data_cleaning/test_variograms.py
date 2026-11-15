"""Numerical and failure-path tests for variogram estimation and kriging."""

from unittest.mock import Mock

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import Point

from shared.data_cleaning import variograms as vg


@pytest.fixture
def donors():
    """Provide ordered projected donor points with known values."""
    return gpd.GeoDataFrame({"value": [0.0, 2.0, 6.0]}, geometry=[Point(0, 0), Point(1, 0), Point(3, 0)], crs=3857)


@pytest.mark.parametrize(
    "model,parameters,expected",
    [
        (vg.linear_model, [2, 1], [1, 3, 5]),
        (vg.power_model, [2, 2, 1], [1, 3, 9]),
        (vg.gaussian_model, [2, 1, 1], [1, 3 - 2 * np.exp(-1), 3 - 2 * np.exp(-4)]),
        (vg.exponential_model, [2, 1, 1], [1, 3 - 2 * np.exp(-1), 3 - 2 * np.exp(-2)]),
        (vg.spherical_model, [2, 1, 1], [1, 3, 3]),
    ],
)
def test_models_have_expected_nugget_and_lag_response(model, parameters, expected):
    """Supported models reproduce analytically calculated semivariances."""
    np.testing.assert_allclose(model(np.array([0.0, 1.0, 2.0]), *parameters), expected)


@pytest.mark.parametrize("distance,expected_lags,expected_values", [(None, [1, 2.5], [2, 13]), (2, [1, 2], [2, 8])])
def test_experimental_variogram_bins_pairs_and_includes_maximum_distance(distance, expected_lags, expected_values):
    """Pairwise squared differences are halved and averaged only in populated bins."""
    lags, values = vg.compute_variogram([0, 1, 3], [0, 0, 0], [0, 2, 6], n_lags=3, max_distance=distance)
    np.testing.assert_allclose(lags, expected_lags)
    np.testing.assert_allclose(values, expected_values)


@pytest.mark.parametrize(
    "x,n_lags,max_distance,message",
    [([0, 1], 0, None, "n_lags"), ([0], 2, None, "At least two"), ([0, 1], 2, 0.5, "At least two")],
)
def test_experimental_variogram_rejects_unusable_inputs(x, n_lags, max_distance, message):
    """Invalid lag counts and absent donor pairs produce actionable errors."""
    with pytest.raises(ValueError, match=message):
        vg.compute_variogram(x, np.zeros(len(x)), x, n_lags=n_lags, max_distance=max_distance)


def test_fit_recovers_linear_parameters():
    """Actual least-squares fitting recovers a known line with negligible residual error."""
    lags = np.arange(1.0, 6.0)
    parameters, error = vg.fit_variogram_model(lags, 3 * lags + 2, vg.linear_model, [1.0, 0.1])
    np.testing.assert_allclose(parameters, [3, 2])
    assert error < 1e-20


@pytest.mark.parametrize("values,message", [(["a", "b", "c"], "not numeric"), ([1.0, np.nan, np.nan], "two non-null")])
def test_model_selection_rejects_invalid_donors(donors, values, message):
    """Model selection requires numeric values and enough non-null donors."""
    donors["value"] = values
    with pytest.raises(ValueError, match=message):
        vg.calculate_variogram(donors, "value", 3, None)


def test_model_selection_skips_failed_fits_and_plots_successes(donors, monkeypatch):
    """A failed candidate cannot prevent selecting the lowest-error successful model."""
    fits = Mock(
        side_effect=[
            RuntimeError("no fit"),
            (np.array([1, 1, 0]), 4),
            (np.array([2, 1, 0]), 1),
            ValueError("no fit"),
            TypeError("no fit"),
        ]
    )
    monkeypatch.setattr(vg, "fit_variogram_model", fits)
    shown = Mock()
    monkeypatch.setattr(plt, "show", shown)
    before = set(plt.get_fignums())
    try:
        name, parameters = vg.calculate_variogram(donors, "value", 3, None, plot=True)
        assert name == "gaussian"
        np.testing.assert_array_equal(parameters, [2, 1, 0])
        assert fits.call_count == 5
        shown.assert_called_once_with()
        assert len(plt.gca().lines) == 3
    finally:
        for figure in set(plt.get_fignums()) - before:
            plt.close(figure)


def test_model_selection_reports_when_all_candidates_fail(donors, monkeypatch):
    """An entirely failed fit cannot silently yield an arbitrary model."""
    monkeypatch.setattr(vg, "fit_variogram_model", Mock(side_effect=RuntimeError("no fit")))
    with pytest.raises(ValueError, match="No supported variogram model"):
        vg.calculate_variogram(donors, "value", 3, None)


def test_kriging_empty_donors_and_invalid_model(donors):
    """Missing donors give aligned NaNs while unsupported models are rejected."""
    result = vg.fill_nulls_kriging(donors, donors.iloc[:0], "value", "linear")
    assert result.shape == (3,)
    assert np.isnan(result).all()
    with pytest.raises(ValueError, match="Invalid variogram model"):
        vg.fill_nulls_kriging(donors, donors, "value", "unknown")


def test_kriging_predicts_known_donor_locations(donors):
    """Real ordinary kriging reproduces observed values at their original coordinates."""
    pytest.importorskip("pykrige")
    result = vg.fill_nulls_kriging(donors.iloc[[2, 0]], donors, "value", "linear")
    np.testing.assert_allclose(result, [6, 0], atol=1e-8)
