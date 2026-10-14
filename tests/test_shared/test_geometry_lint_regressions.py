"""Regression checks for geometry units and spatial writer validation."""

import pandas as pd
import pytest

from shared.io import write_geopackage, write_geoparquet
from tests.utils import expect_equal, expect_false


@pytest.mark.parametrize("writer", [write_geopackage, write_geoparquet])
def test_spatial_writers_reject_plain_dataframes(writer, tmp_path):
    """Spatial-only writers signal a type error before writing any file."""
    output = tmp_path / "invalid"
    with pytest.raises(TypeError, match="Error:"):
        writer(pd.DataFrame({"value": [1]}), str(output))
    expect_false(output.exists())


def test_create_unit_conversion_factor_supports_known_pairs_and_rejects_unknown():
    """Unit conversion should stay deterministic for known pairs and fail otherwise."""
    from shared.geometry import create_unit_conversion_factor

    expect_equal(create_unit_conversion_factor("m", "ft"), pytest.approx(3.28084))
    expect_equal(create_unit_conversion_factor("ft", "m"), pytest.approx(0.3048))
    expect_equal(create_unit_conversion_factor("degree", "m"), pytest.approx(111000.0))
    with pytest.raises(ValueError, match="Error: Cannot handle units"):
        create_unit_conversion_factor("yard", "m")
