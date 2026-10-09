"""Regression checks for geometry validation and coordinate extraction."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point, box

from shared.geometry import create_polygon_from_bounds, get_line_coordinates
from shared.io import write_geopackage, write_geoparquet
from tests.utils import expect_equal, expect_false, expect_true


def test_nan_buffer_is_treated_as_zero():
    """A missing buffer must leave finite bounds unchanged."""
    expect_true(create_polygon_from_bounds([0, 0, 10, 20], np.nan).equals(box(0, 0, 10, 20)))


@pytest.mark.parametrize("reverse", [False, True])
def test_line_coordinate_extraction_preserves_multipart_order(reverse):
    """The retained implementation flattens parts before optionally reversing."""
    line = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
    expected = [(0, 0), (1, 1), (2, 2), (3, 3)]
    expect_equal(get_line_coordinates(line, reverse=reverse), (expected[::-1] if reverse else expected))
    expect_equal(get_line_coordinates(LineString(expected), reverse=reverse), (expected[::-1] if reverse else expected))


@pytest.mark.parametrize("writer", [write_geopackage, write_geoparquet])
def test_spatial_writers_reject_plain_dataframes(writer, tmp_path):
    """Spatial-only writers signal a type error before writing any file."""
    output = tmp_path / "invalid"
    with pytest.raises(TypeError, match="Error:"):
        writer(pd.DataFrame({"value": [1]}), str(output))
    expect_false(output.exists())


@pytest.mark.parametrize(
    ("p1", "p2", "expected_degrees", "expected_distance"),
    [
        (Point(0, 0), Point(0, 0), 0.0, 0.0),
        (Point(0, 0), Point(0, 5), 0.0, 5.0),
        (Point(0, 0), Point(5, 0), 90.0, 5.0),
        (Point(0, 0), Point(0, -5), 180.0, 5.0),
        (Point(0, 0), Point(-5, 0), 270.0, 5.0),
    ],
)
def test_calc_vector_azimuth_dist_preserves_cardinal_directions(p1, p2, expected_degrees, expected_distance):
    """Azimuth and distance stay compatible with the original orientation convention."""
    from shared.geometry import calc_vector_azimuth_dist

    degrees, distance = calc_vector_azimuth_dist(p1, p2)
    expect_equal(degrees, pytest.approx(expected_degrees))
    expect_equal(distance, pytest.approx(expected_distance))


def test_create_unit_conversion_factor_supports_known_pairs_and_rejects_unknown():
    """Unit conversion should stay deterministic for known pairs and fail otherwise."""
    from shared.geometry import create_unit_conversion_factor

    expect_equal(create_unit_conversion_factor("m", "ft"), pytest.approx(3.28084))
    expect_equal(create_unit_conversion_factor("ft", "m"), pytest.approx(0.3048))
    expect_equal(create_unit_conversion_factor("degree", "m"), pytest.approx(111000.0))
    with pytest.raises(ValueError, match="Error: Cannot handle units"):
        create_unit_conversion_factor("yard", "m")
