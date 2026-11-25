"""Unit tests for parcel preparation, spatial batching and per-partition extractor configuration."""

import logging

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from data_preparation.parcel_stats import partition_scheduler
from data_preparation.parcel_stats.core.parcel_batches import ParcelBatchPlanner
from tests.utils import expect_equal, expect_in, expect_is_none, expect_isinstance, expect_true


class _Planner(ParcelBatchPlanner):
    """Minimal host providing the attributes the mixin reads from the extractor."""

    PARCEL_ID_FIELD = "parcel_id"
    GRID_SIZE_METRES = 1_000
    MAX_FEATURES_PER_JOB = 2

    def __init__(self, working_epsg=2100):
        """Store the projected CRS used for batching and a quiet logger."""
        self.working_epsg = working_epsg
        self.logger = logging.getLogger("test_parcel_batches")


def _parcels(geometries, ids=None, crs="EPSG:2100"):
    """Return parcels with optional identifiers and a label attribute."""
    data = {"label": [f"l{i}" for i in range(len(geometries))]}
    if ids is not None:
        data["parcel_id"] = ids
    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)


# --------------------------------------------------------------------------------
# _extract_polygonal_geometry
# --------------------------------------------------------------------------------


def test_extract_polygonal_geometry_handles_empty_and_non_polygonal_inputs():
    """Missing, empty and purely linear geometries yield None."""
    planner = _Planner()
    expect_is_none(planner._extract_polygonal_geometry(None))
    expect_is_none(planner._extract_polygonal_geometry(Polygon()))
    expect_is_none(planner._extract_polygonal_geometry(LineString([(0, 0), (1, 1)])))
    expect_is_none(planner._extract_polygonal_geometry(GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 0)])])))


def test_extract_polygonal_geometry_keeps_polygons_and_repairs_bowties():
    """Valid polygons pass through; self-intersections are repaired into polygonal output."""
    planner = _Planner()
    square = box(0, 0, 1, 1)
    expect_true(planner._extract_polygonal_geometry(square).equals(square))
    repaired = planner._extract_polygonal_geometry(Polygon([(0, 0), (2, 2), (2, 0), (0, 2)]))
    expect_isinstance(repaired, MultiPolygon)
    expect_true(repaired.is_valid)


def test_extract_polygonal_geometry_keeps_only_polygon_parts_of_collections():
    """Collections retain polygon and multipolygon members and drop lines and points."""
    collection = GeometryCollection(
        [box(0, 0, 1, 1), MultiPolygon([box(5, 5, 6, 6), box(8, 8, 9, 9)]), LineString([(0, 0), (3, 3)]), Point(4, 4)]
    )

    result = _Planner()._extract_polygonal_geometry(collection)

    expect_equal(result.geom_type, "MultiPolygon")
    expect_equal(len(result.geoms), 3)
    expect_equal(result.area, 3.0)


# --------------------------------------------------------------------------------
# _prepare_parcels
# --------------------------------------------------------------------------------


def test_prepare_parcels_validates_type_and_crs():
    """Only GeoDataFrames with a CRS are accepted."""
    planner = _Planner()
    with pytest.raises(TypeError, match="Error: parcels must be a geopandas.GeoDataFrame"):
        planner._prepare_parcels([box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="Error: The parcels GeoDataFrame must have a CRS"):
        planner._prepare_parcels(_parcels([box(0, 0, 1, 1)], crs=None))


def test_prepare_parcels_creates_string_ids_from_index_and_reprojects():
    """Missing identifiers come from the index as strings and output is WGS84."""
    parcels = _parcels([box(400_000, 4_200_000, 400_100, 4_200_100)])
    parcels.index = [42]

    prepared = _Planner()._prepare_parcels(parcels)

    expect_equal(list(prepared.columns[:1]), ["parcel_id"])
    expect_equal(prepared["parcel_id"].tolist(), ["42"])
    expect_equal(prepared.crs.to_epsg(), 4326)
    expect_in("label", prepared.columns)


def test_prepare_parcels_drops_non_polygonal_rows_and_casts_ids():
    """Line parcels are dropped and numeric identifiers become strings."""
    parcels = _parcels([box(0, 0, 10, 10), LineString([(0, 0), (5, 5)])], ids=[1, 2])

    prepared = _Planner()._prepare_parcels(parcels)

    expect_equal(prepared["parcel_id"].tolist(), ["1"])


def test_prepare_parcels_rejects_empty_and_duplicate_results():
    """No surviving polygons, or duplicate identifiers, are reported."""
    planner = _Planner()
    with pytest.raises(ValueError, match="Error: No valid Polygon or MultiPolygon parcels"):
        planner._prepare_parcels(_parcels([LineString([(0, 0), (1, 1)])], ids=["a"]))
    with pytest.raises(ValueError, match=r"Example duplicates: \['a'\]"):
        planner._prepare_parcels(_parcels([box(0, 0, 1, 1), box(2, 2, 3, 3)], ids=["a", "a"]))


# --------------------------------------------------------------------------------
# _iter_spatial_batches
# --------------------------------------------------------------------------------


def test_iter_spatial_batches_groups_by_grid_cell_and_caps_batch_size():
    """Parcels are grouped per metric grid cell and split into capped, ordered batches."""
    planner = _Planner()
    # Three parcels in one 1 km cell (south to north) and one parcel in a distant cell.
    metric = _parcels(
        [
            box(400_100, 4_200_500, 400_110, 4_200_510),
            box(400_100, 4_200_100, 400_110, 4_200_110),
            box(400_100, 4_200_300, 400_110, 4_200_310),
            box(410_100, 4_200_100, 410_110, 4_200_110),
        ],
        ids=["north", "south", "middle", "far"],
    )
    planner.parcels = planner._prepare_parcels(metric)

    batches = list(planner._iter_spatial_batches())

    expect_equal(sorted(len(batch) for batch in batches), [1, 1, 2])
    ids = [batch["parcel_id"].tolist() for batch in batches]
    expect_in(["south", "middle"], ids)
    expect_in(["north"], ids)
    expect_in(["far"], ids)
    for batch in batches:
        expect_equal(list(batch.index), list(range(len(batch))))
        expect_equal(batch.crs.to_epsg(), 4326)


# --------------------------------------------------------------------------------
# run_partition_extractor
# --------------------------------------------------------------------------------


def _partition_config(tmp_path):
    """Return the required keys plus one pass-through option."""
    return {
        "output_dir": str(tmp_path),
        "run_identifier": "run",
        "parcel_id_field": "parcel_id",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "working_epsg": 2100,
        "sentinel2_bands": ["B04"],
        "sentinel1_bands": ["VV"],
        "spatial_statistics": ["median"],
        "tile_size_metres": 10_000,
        "tile_buffer_metres": 100,
        "fill_nulls": True,
    }


def test_run_partition_extractor_assigns_users_round_robin_and_forwards_options(tmp_path, monkeypatch, capsys):
    """Each partition uses its round-robin account, its own output folder and any extra options."""
    created = []

    class FakeExtractor:
        """Record constructor arguments and return the input parcels."""

        def __init__(self, **kwargs):
            created.append(kwargs)
            self.parcels = kwargs["parcels"]

        def run(self):
            return self.parcels

    monkeypatch.setattr(partition_scheduler, "OpenEOJobManagerZonalStats", FakeExtractor)
    parcels = _parcels([box(0, 0, 1, 1), box(2, 2, 3, 3)], ids=["a", "b"])
    users = [("user1", "pw1"), ("user2", "pw2")]

    part_idx, result = partition_scheduler.run_partition_extractor(3, parcels, users, 4, _partition_config(tmp_path))

    expect_equal(part_idx, 3)
    expect_true(result is parcels, "The extractor result must be returned unchanged")
    kwargs = created[0]
    expect_equal((kwargs["openeo_username"], kwargs["openeo_password"]), ("user1", "pw1"))
    expect_equal(kwargs["output_dir"], tmp_path / "partition_3")
    expect_equal(kwargs["run_identifier"], "run_part3")
    expect_equal((kwargs["sentinel2_indices"], kwargs["sentinel1_indices"]), ([], []))
    expect_equal(kwargs["fill_nulls"], True)
    output = capsys.readouterr().out
    expect_in("PARTITION 3/4: 2 parcels", output)
    expect_in("Partition 3 complete: 2 parcels", output)
