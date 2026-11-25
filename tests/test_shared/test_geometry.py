"""Unit tests for shared geometry helpers (explode, point conversion, units, validity, duplicates)."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

from shared import geometry
from tests.utils import expect_equal, expect_in, expect_isinstance, expect_true

# Projected CRS without an EPSG code exercises the proj4 "+units=" parsing branch.
_CUSTOM_FT_CRS = "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=us-ft +no_defs"
_CUSTOM_M_CRS = "+proj=tmerc +lat_0=0 +lon_0=21 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"


def _frame(geometries, crs="EPSG:2100", **columns):
    """Build a small GeoDataFrame with an ``id`` column and optional extra columns."""
    data = {"id": list(range(1, len(geometries) + 1)), **columns}
    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)


def test_explode_multigeometries_splits_parts_and_resets_index():
    """Each multi-part feature becomes one row per part with a fresh RangeIndex."""
    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    df = _frame([multi, box(5, 5, 6, 6)])
    df.index = [10, 20]

    exploded = geometry.explode_multigeometries(df)

    expect_equal(len(exploded), 3)
    expect_equal(list(exploded.index), [0, 1, 2])
    expect_equal(list(exploded["id"]), [1, 1, 2])
    expect_true(set(exploded.geom_type) == {"Polygon"}, "Exploded geometries must be single polygons")


def test_explode_multigeometries_rejects_missing_column():
    """A missing geometry column is reported with the project error prefix."""
    with pytest.raises(ValueError, match="Error: shape column"):
        geometry.explode_multigeometries(_frame([Point(0, 0)]), geom_column="shape")


def test_convert_multipolygons_to_polygons_keeps_largest_part_only():
    """MultiPolygons collapse to their largest part while polygons stay unchanged."""
    small, large = box(0, 0, 1, 1), box(10, 10, 14, 14)
    single = box(20, 20, 21, 21)
    df = _frame([MultiPolygon([small, large]), single])

    converted = geometry.convert_multipolygons_to_polygons(df)

    expect_equal(len(converted), 2)
    expect_true(converted.geometry.iloc[0].equals(large), "The largest part must be retained")
    expect_true(converted.geometry.iloc[1].equals(single), "Polygons must be returned unchanged")
    expect_isinstance(df.geometry.iloc[0], MultiPolygon, "The input frame must not be mutated")


def test_convert_multipolygons_to_polygons_rejects_missing_column():
    """The function validates the requested geometry column."""
    with pytest.raises(ValueError, match="Error: geom column"):
        geometry.convert_multipolygons_to_polygons(_frame([box(0, 0, 1, 1)]), geom_column="geom")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Point(3, 4), Point(3, 4)),
        (LineString([(0, 0), (10, 0)]), Point(5, 0)),
        # The midpoint of the longest member line is used for multi-lines.
        (MultiLineString([[(0, 0), (1, 0)], [(0, 5), (20, 5)]]), Point(10, 5)),
        (box(0, 0, 4, 4), Point(2, 2)),
    ],
)
def test_geometry_to_point_returns_expected_location(value, expected):
    """Points pass through, lines use their midpoint and polygons their centroid."""
    result = geometry.geometry_to_point(value)
    expect_true(result.equals(expected), f"Expected {expected.wkt}, got {result.wkt}")


def test_geometry_to_point_falls_back_to_representative_point_for_hollow_shapes():
    """A U-shaped polygon whose centroid lies outside gets an interior representative point."""
    u_shape = Polygon([(0, 0), (30, 0), (30, 30), (25, 30), (25, 5), (5, 5), (5, 30), (0, 30)])
    expect_true(not u_shape.contains(u_shape.centroid), "Fixture centroid must lie outside the shape")

    result = geometry.geometry_to_point(u_shape)

    expect_true(u_shape.contains(result), "The fallback point must lie inside the polygon")


def test_convert_geometries_to_points_converts_every_row_without_mutation():
    """All rows are converted to points on a copy of the input."""
    df = _frame([box(0, 0, 2, 2), LineString([(0, 0), (4, 0)])])

    points = geometry.convert_geometries_to_points(df)

    expect_equal(list(points.geom_type), ["Point", "Point"])
    expect_equal(list(df.geom_type), ["Polygon", "LineString"])


def test_convert_geometries_to_points_requires_geometry_column():
    """The data-check helper fails loudly without a geometry field."""
    with pytest.raises(AssertionError, match="there is no geometry field"):
        geometry.convert_geometries_to_points(_frame([Point(0, 0)]).rename(columns={"geometry": "geom"}))


@pytest.mark.parametrize(
    ("crs", "expected"),
    [("EPSG:2100", "m"), ("EPSG:2227", "ft"), (_CUSTOM_M_CRS, "m"), (_CUSTOM_FT_CRS, "ft")],
)
def test_get_unit_of_length_normalizes_epsg_and_proj4_units(crs, expected):
    """EPSG axis units and proj4 ``+units=`` tokens normalize to ``m`` or ``ft``."""
    expect_equal(geometry.get_unit_of_length(_frame([Point(0, 0)], crs=crs)), expected)


def test_get_unit_of_length_rejects_geographic_units_and_missing_geometry():
    """Degree-based CRSs and frames without geometry are rejected."""
    with pytest.raises(ValueError, match="Error: Unrecognized projection unit"):
        geometry.get_unit_of_length(_frame([Point(0, 0)], crs="EPSG:4326"))
    plain = _frame([Point(0, 0)]).rename_geometry("geom")
    with pytest.raises(ValueError, match="Error: There is no 'geometry' field"):
        geometry.get_unit_of_length(plain)


def test_create_unit_conversion_factor_identity_is_case_insensitive():
    """Identical units always convert with a factor of one, regardless of case."""
    expect_equal(geometry.create_unit_conversion_factor("M", "m"), 1.0)


def test_value_conversions_follow_frame_units():
    """The four convenience conversions use the frame's CRS unit."""
    metric = _frame([Point(0, 0)], crs="EPSG:2100")
    feet = _frame([Point(0, 0)], crs="EPSG:2227")

    expect_equal(geometry.convert_value_in_ft_to_df_units(metric, 10), pytest.approx(3.048))
    expect_equal(geometry.convert_value_in_m_to_df_units(metric, 10), 10.0)
    expect_equal(geometry.convert_value_in_m_to_df_units(feet, 1), pytest.approx(3.28084))
    expect_equal(geometry.convert_value_in_df_units_from_ft(feet, 7), 7.0)
    expect_equal(geometry.convert_value_in_df_units_from_ft(metric, 1), pytest.approx(3.28084))
    expect_equal(geometry.convert_value_in_df_units_from_m(feet, 10), pytest.approx(3.048))


def _mixed_validity_frame():
    """Return a frame with one valid, one self-intersecting, one empty and one missing geometry."""
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    return _frame([box(0, 0, 1, 1), bowtie, Polygon(), None], name=["ok", "bowtie", "empty", "none"])


def test_get_ids_of_invalid_geoms_reports_empty_and_missing_geometries():
    """Buffering repairs self-intersections, so only empty and missing geometries remain invalid."""
    invalid = geometry.get_ids_of_invalid_geoms(_mixed_validity_frame(), colid="name")
    expect_equal(sorted(invalid), ["empty", "none"])


def test_return_valid_and_invalid_geometries_partition_rows():
    """Valid and invalid outputs are complementary, reindexed and free of the helper column."""
    df = _mixed_validity_frame()

    valid = geometry.return_valid_geometries(df)
    invalid = geometry.return_invalid_geometries(df)

    expect_equal(list(valid["name"]), ["ok", "bowtie"])
    expect_equal(list(invalid["name"]), ["empty", "none"])
    expect_equal(list(valid.index), [0, 1])
    for result in (valid, invalid):
        expect_true("tmp_id" not in result.columns, "The temporary id column must be removed")


@pytest.mark.parametrize("func", [geometry.return_valid_geometries, geometry.return_invalid_geometries])
def test_validity_filters_require_geometry_column(func):
    """Validity filters fail when the requested geometry column is absent."""
    with pytest.raises(AssertionError, match="there is no geometry field"):
        func(_frame([Point(0, 0)]), geom_column="shape")


def test_get_duplicate_geometries_returns_sorted_points_and_prints_count(capsys):
    """Duplicated geometries are reported as points with x/y columns."""
    df = _frame([box(0, 0, 1, 1), box(5, 5, 6, 6), box(0, 0, 1, 1)])

    duplicates = geometry.get_duplicate_geometries(df, col_id="id", verbose=True)

    expect_equal(sorted(duplicates["id"]), [1, 3])
    expect_equal(list(duplicates.geom_type.unique()), ["Point"])
    expect_equal(list(duplicates[["x", "y"]].iloc[0]), [0.5, 0.5])
    expect_in("Duplicate geometries: 2", capsys.readouterr().out)


def test_get_duplicate_geometries_reports_when_none_found(capsys):
    """A frame without duplicates yields an empty result and an explanatory message."""
    duplicates = geometry.get_duplicate_geometries(_frame([box(0, 0, 1, 1), box(2, 2, 3, 3)]), col_id="id")

    expect_true(duplicates.empty, "No duplicates were expected")
    expect_in("There aren't any duplicate geometries", capsys.readouterr().out)
