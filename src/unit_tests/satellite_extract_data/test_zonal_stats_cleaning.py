"""Focused regression tests for parcel-isolated zonal-statistics cleaning."""

from __future__ import annotations

import numpy as np
import pytest

from satellite_extract_data.zonal_stats import SatelliteZonalStats


def test_explicit_openeo_credentials_use_password_oidc_flow() -> None:
    calls = []

    class Connection:
        def authenticate_oidc_resource_owner_password_credentials(self, **kwargs):
            calls.append(("password", kwargs))

        def authenticate_oidc(self):
            calls.append(("default", {}))

    extractor = object.__new__(SatelliteZonalStats)
    extractor.openeo_username = "user@example.com"
    extractor.openeo_password = "secret"

    returned = extractor._authenticate_openeo_connection(Connection())

    assert isinstance(returned, Connection)
    assert calls == [("password", {"username": "user@example.com", "password": "secret"})]


def test_missing_openeo_credentials_use_default_oidc_flow() -> None:
    calls = []

    class Connection:
        def authenticate_oidc(self):
            calls.append("default")

    extractor = object.__new__(SatelliteZonalStats)
    extractor.openeo_username = None
    extractor.openeo_password = None

    extractor._authenticate_openeo_connection(Connection())

    assert calls == ["default"]


def test_openeo_credentials_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        SatelliteZonalStats._validate_openeo_credentials("user@example.com", None)


def _extractor(**overrides) -> SatelliteZonalStats:
    """Build a lightweight instance for testing pure array operations."""
    extractor = object.__new__(SatelliteZonalStats)
    defaults = {
        "sentinel2_bands": ("B02",),
        "sentinel2_indices": (),
        "sentinel1_bands": (),
        "temporal_fill_mode": "past_only",
        "remove_outliers": True,
        "iqr_quantiles": (0.25, 0.75),
        "iqr_multiplier": 1.5,
        "iqr_min_valid_pixels": 5,
        "minimum_observed_fraction_for_fill": 0.20,
        "interpolation_method": "nearest",
        "interpolation_max_distance_in_meters": None,
        "interpolation_variogram_lags": 15,
        "interpolation_variogram_max_distance_in_meters": None,
    }
    for name, value in {**defaults, **overrides}.items():
        setattr(extractor, name, value)
    return extractor


def test_past_only_fill_does_not_use_future_observation() -> None:
    extractor = _extractor(temporal_fill_mode="past_only")
    values = np.array([np.nan, 1.0, np.nan], dtype=float).reshape(3, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    assert np.isnan(filled[0, 0, 0])
    np.testing.assert_allclose(filled[1:, 0, 0], [1.0, 1.0])


def test_bidirectional_fill_is_explicitly_available_for_retrospective_runs() -> None:
    extractor = _extractor(temporal_fill_mode="bidirectional")
    values = np.array([np.nan, 1.0, np.nan], dtype=float).reshape(3, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    np.testing.assert_allclose(filled[:, 0, 0], [1.0, 1.0, 1.0])


def test_bidirectional_fill_averages_bracketing_values_across_consecutive_gaps() -> None:
    extractor = _extractor(temporal_fill_mode="bidirectional")
    values = np.array([0.0, np.nan, np.nan, 9.0], dtype=float).reshape(4, 1, 1)

    filled = extractor._fill_temporal_neighbors(values)

    np.testing.assert_allclose(filled[:, 0, 0], [0.0, 4.5, 4.5, 9.0])


def test_spatial_fill_ignores_pixels_outside_current_parcel() -> None:
    extractor = _extractor()
    values = np.array([[[np.nan, 100.0], [2.0, 100.0]]])
    parcel_mask = np.array([[True, False], [True, False]])

    filled = extractor._fill_spatial_neighbors(values, parcel_mask)

    assert filled[0, 0, 0] == 2.0
    assert np.isnan(filled[0, 0, 1]) or filled[0, 0, 1] == 100.0


def test_iqr_filter_is_skipped_for_unreliable_small_samples() -> None:
    extractor = _extractor(iqr_min_valid_pixels=5)
    values = np.array([[[[1.0, 1.0], [1.0, 100.0]]]])
    parcel_mask = np.ones((2, 2), dtype=bool)

    cleaned, audit = extractor._remove_iqr_outliers(values, parcel_mask, 1, ["2024-01-01"])

    assert cleaned[0, 0, 1, 1] == 100.0
    assert audit[(0, 0)]["iqr_applied"] is False


def test_low_coverage_period_is_not_imputed_or_used_as_fill_source() -> None:
    extractor = _extractor(minimum_observed_fraction_for_fill=0.50)
    extractor._interpolate_interval_layer = lambda layer, *_args: layer
    values = np.array(
        [
            [[1.0, np.nan], [np.nan, np.nan]],
            [[2.0, 2.0], [np.nan, np.nan]],
        ]
    )
    parcel_mask = np.ones((2, 2), dtype=bool)

    filled = extractor._fill_variable_cube(values, parcel_mask, None, None)

    assert np.isfinite(filled[0]).sum() == 1
    assert np.isfinite(filled[1]).sum() == 4


def test_added_crop_indices_request_their_source_bands() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ("B02", "B03", "B04", "B08")
    extractor.sentinel2_indices = ("MSAVI", "MNDWI", "PSRI")

    required = extractor._required_sentinel2_bands()

    assert {"B02", "B03", "B04", "B06", "B08", "B11"}.issubset(required)


def test_explicit_empty_sentinel2_bands_supports_sentinel1_only_runs() -> None:
    extractor = _extractor()

    assert extractor._validate_sentinel2_bands([]) == ()

    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ("VV", "VH")
    extractor._validate_sensor_selection()
    assert extractor._output_sensor_variables() == ("VV", "VH")


def test_sensor_selection_rejects_a_run_without_any_output_variables() -> None:
    extractor = _extractor()
    extractor.sentinel2_bands = ()
    extractor.sentinel2_indices = ()
    extractor.sentinel1_bands = ()

    with pytest.raises(ValueError, match="Select at least one"):
        extractor._validate_sensor_selection()
