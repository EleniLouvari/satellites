"""Offline tests for HUB ARD item normalisation, filtering, grouping and manifests."""

from unittest.mock import Mock

import pandas as pd
import pytest

from data_preparation.sources.hub import ard_items as hub


def feature(item_id, when, *, collection_props=None, assets=("B04", "SCL", "MASK"), checksum="abc"):
    """Build a minimal STAC feature resembling deployed HUB ARD items."""
    properties = {"datetime": when, "processing:datetime": "2026-11-10T06:37:10.000Z", "processing:version": "1.0.5"}
    properties.update(collection_props or {})
    return {
        "id": item_id,
        "properties": properties,
        "assets": {
            key: {"href": f"https://gw.test/{item_id}/{key}", "hub:checksum": checksum, "raster:bands": [{"nodata": 0.0}]}
            for key in assets
        },
    }


def s2_item(item_id, when="2024-07-26T09:15:59Z", **props):
    """Return a normalised Sentinel-2 ARD item."""
    defaults = {"platform": "sentinel-2b", "s2:datatake_id": "GS2B_1", "eo:cloud_cover": 5.0, "proj:code": "EPSG:2100"}
    defaults.update(props)
    return hub.normalise_item(feature(item_id, when, collection_props=defaults), hub.SENTINEL2_ARD_COLLECTION)


def s1_item(item_id, when="2024-06-09T16:24:34Z", **props):
    """Return a normalised Sentinel-1 ARD item."""
    defaults = {"platform": "SENTINEL-1A", "sat:orbit_state": "ascending", "sat:absolute_orbit": 54249, "proj:code": 2100}
    defaults.update(props)
    return hub.normalise_item(
        feature(item_id, when, collection_props=defaults, assets=("VV", "VH", "MASK")), hub.SENTINEL1_ARD_COLLECTION
    )


def test_normalise_item_harmonises_inconsistent_deployed_metadata():
    """Projection codes, platform and orbit spellings differ between deployed S1 and S2 items."""
    s2 = s2_item("a")
    s1 = s1_item("b", platform="sentinel-1a")
    assert (s2.epsg, s1.epsg) == (2100, 2100)
    assert s1.platform == "sentinel-1a" == s1_item("c").platform
    assert s1.orbit_state == "ASCENDING"
    assert s2.datetime == pd.Timestamp("2024-07-26T09:15:59Z")
    assert s2.assets["B04"].checksum == "abc" and s2.assets["B04"].nodata == 0.0


def test_normalise_item_rejects_missing_datetime_and_skips_assets_without_href():
    """Items need a datetime; assets without an href are ignored rather than failing later reads."""
    with pytest.raises(ValueError, match="without id or datetime"):
        hub.normalise_item({"id": "x", "properties": {}}, hub.SENTINEL2_ARD_COLLECTION)
    raw = feature("y", "2024-01-01T00:00:00Z")
    raw["assets"]["B04"]["href"] = " "
    assert "B04" not in hub.normalise_item(raw, hub.SENTINEL2_ARD_COLLECTION).assets


def test_build_search_body_uses_inclusive_end_day_and_rejects_unknown_collections():
    """The configured end date is inclusive, matching the parcel-statistics contract."""
    geometry = {"type": "Point", "coordinates": [22.4, 39.6]}
    body = hub.build_search_body("ARD_S2", geometry, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"))
    start, end, spatial = body["filter"]["args"]
    assert start["args"][1] == "2024-06-01T00:00:00Z"
    assert end == {"op": "<", "args": [{"property": "datetime"}, "2024-07-01T00:00:00Z"]}
    assert spatial == {"op": "s_intersects", "args": [{"property": "geometry"}, geometry]}
    assert body["filter-lang"] == "cql2-json" and len(body["filter"]["args"]) == 3
    cloudy = hub.build_search_body("ARD_S2", geometry, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"), max_cloud_cover=40)
    # Only queryables in the AND, so the HUB proxy evaluates the whole filter in the catalog database.
    assert cloudy["filter"]["args"][-1] == {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 40.0]}
    assert "sat:orbit_state" not in str(cloudy)
    with pytest.raises(ValueError, match="Unsupported"):
        hub.build_search_body("S2L2", geometry, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30"))
    with pytest.raises(ValueError, match="precedes"):
        hub.build_search_body("ARD_S2", geometry, pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-01"))


def test_search_ard_items_posts_once_and_removes_duplicate_pages():
    """Overlapping result pages must not create duplicate observations."""
    catalog = Mock()
    catalog.fetch_features.return_value = [
        feature("b", "2024-06-02T00:00:00Z"),
        feature("a", "2024-06-01T00:00:00Z"),
        feature("b", "2024-06-02T00:00:00Z"),
    ]
    items = hub.search_ard_items(
        catalog, "ARD_S2", {"type": "Point", "coordinates": [0, 0]}, pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30")
    )
    assert [item.item_id for item in items] == ["a", "b"]
    assert catalog.fetch_features.call_args.kwargs["method"] == "POST"


def test_filter_items_applies_assets_cloud_and_orbit_filters():
    """Missing required assets, cloudy scenes and the wrong pass direction are dropped."""
    clear, cloudy, unknown = (
        s2_item("clear"),
        s2_item("cloudy", **{"eo:cloud_cover": 90.0}),
        s2_item("unknown", **{"eo:cloud_cover": None}),
    )
    kept = hub.filter_items([clear, cloudy, unknown], required_assets=("B04", "SCL"), max_cloud_cover=50)
    assert [item.item_id for item in kept] == ["clear", "unknown"]
    assert hub.filter_items([clear], required_assets=("B08",)) == []
    ascending, descending = s1_item("asc"), s1_item("desc", **{"sat:orbit_state": "descending"})
    assert [
        i.item_id for i in hub.filter_items([ascending, descending], required_assets=("VV",), orbit_direction="descending")
    ] == ["desc"]
    with pytest.raises(ValueError, match="orbit_direction"):
        hub.filter_items([ascending], required_assets=(), orbit_direction="LEFT")


def test_group_observations_unites_duplicate_s2_generations_and_orders_by_quality():
    """Two ESA generations of one datatake form one observation; the complete one is used first."""
    partial = s2_item("partial", **{"statistics:nodata_pixel": 86.5, "eo:cloud_cover": 16.4})
    complete = s2_item("complete", **{"statistics:nodata_pixel": 1.2, "eo:cloud_cover": 5.9})
    other_day = s2_item("other", when="2024-07-31T09:15:59Z", **{"s2:datatake_id": "GS2B_2"})
    observations = hub.group_observations([partial, other_day, complete])
    assert [o.key for o in observations] == ["s2|GS2B_1", "s2|GS2B_2"]
    assert [item.item_id for item in observations[0].items] == ["complete", "partial"]


def test_group_observations_unites_s1_slices_of_one_pass_with_earliest_timestamp():
    """Consecutive slices share an absolute orbit and receive one time label."""
    first = s1_item("slice7", when="2024-06-09T16:24:34Z")
    second = s1_item("slice8", when="2024-06-09T16:24:59Z")
    later = s1_item("next", when="2024-06-21T16:24:33Z", **{"sat:absolute_orbit": 54424})
    observations = hub.group_observations([second, later, first])
    assert len(observations) == 2
    assert observations[0].datetime == pd.Timestamp("2024-06-09T16:24:34Z")
    assert {item.item_id for item in observations[0].items} == {"slice7", "slice8"}


def test_observation_key_keeps_items_separate_without_acquisition_evidence():
    """Missing datatake, platform or absolute orbit never merges same-day items under None values."""
    first = s2_item("x", **{"s2:datatake_id": None, "sat:relative_orbit": 93})
    second = s2_item("y", **{"s2:datatake_id": None, "sat:relative_orbit": 93})
    assert hub.observation_key(first) == "s2|item|x"
    assert len(hub.group_observations([first, second])) == 2
    radar = s1_item("r", platform=None)
    assert hub.observation_key(radar) == "s1|item|r"


def test_planet_basemap_quads_are_searchable_and_grouped_by_month():
    """Every quad of one month mosaics together; other months, and an unknown prefix, stay separate."""
    geometry = {"type": "Point", "coordinates": [22.4, 39.6]}
    body = hub.build_search_body(hub.PLANET_BASEMAP_COLLECTION, geometry, pd.Timestamp("2024-04-01"), pd.Timestamp("2024-05-31"))
    assert body["collections"] == ["PL_BSM"]
    quads = [
        hub.normalise_item(feature(item_id, when, assets=("quad_bandmath",)), hub.PLANET_BASEMAP_COLLECTION)
        for item_id, when in (
            ("apr_a", "2024-04-01T00:00:00Z"),
            ("apr_b", "2024-04-01T00:00:00Z"),
            ("may_a", "2024-05-01T00:00:00Z"),
        )
    ]
    assert hub.observation_key(quads[0]) == "pl|2024-04"
    observations = hub.group_observations(quads)
    assert [(item.key, len(item.items)) for item in observations] == [("pl|2024-04", 2), ("pl|2024-05", 1)]


def test_manifest_digest_tracks_versions_locations_and_nodata_but_not_signed_queries():
    """Reprocessing, a moved asset or changed nodata alter the digest; query strings do not."""
    keys = {hub.SENTINEL2_ARD_COLLECTION: ("B04", "SCL")}

    def digest_for(edit=None, **kwargs):
        raw = feature("a", "2024-07-26T09:15:59Z", collection_props={"s2:datatake_id": "GS2B_1"}, **kwargs)
        if edit:
            edit(raw["assets"]["B04"])
        frame = hub.manifest_frame(hub.group_observations([hub.normalise_item(raw, "ARD_S2")]), keys)
        return frame, hub.manifest_digest(frame)

    base, digest = digest_for()
    assert len(base) == 2 and base["asset_location"].iloc[0] == "https://gw.test/a/B04"
    assert digest_for(lambda asset: asset.update(href=asset["href"] + "?X-Amz-Signature=s"))[1] == digest
    assert digest_for(lambda asset: asset.update(href="https://other.test/a/B04"))[1] != digest
    assert digest_for(lambda asset: asset.update({"raster:bands": [{"nodata": 255}]}))[1] != digest
    assert digest_for(checksum="new")[1] != digest
    assert hub.asset_location("https://gw.test/x/B04?sig=1#frag") == "https://gw.test/x/B04"


def test_manifest_digest_expires_unversioned_inventory_by_search_date():
    """Items without checksum and processing datetime cannot be versioned, so the digest includes the search day."""
    raw = feature("u", "2024-07-26T09:15:59Z", checksum=None, collection_props={"processing:datetime": None})
    manifest = hub.manifest_frame(hub.group_observations([hub.normalise_item(raw, "ARD_S2")]), {"ARD_S2": ("B04",)})
    assert hub.unversioned_items(manifest) == ["u"]
    with pytest.raises(ValueError, match="search_date"):
        hub.manifest_digest(manifest)
    assert hub.manifest_digest(manifest, search_date="2026-11-21") != hub.manifest_digest(manifest, search_date="2026-11-22")
    assert hub.manifest_digest(hub.manifest_frame([], {})) == hub.manifest_digest(hub.manifest_frame([], {}))
