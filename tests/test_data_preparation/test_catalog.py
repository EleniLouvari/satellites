"""Offline catalog tests for authentication, retries, search paging, and feature conversion."""

import json
import runpy
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import requests
from shapely.geometry import Point

pytest.importorskip("pystac_client")

from data_preparation.sources.hub import catalog_library as catalog


def response(status=200, payload=None):
    """Build a real requests response so status and JSON handling remain under test."""
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps({} if payload is None else payload).encode()
    result.url = "https://catalog.test/search"
    return result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Prevent live HTTP and real sleeps even if a test forgets an explicit transport mock."""
    monkeypatch.setattr(requests.sessions.Session, "request", Mock(side_effect=AssertionError("Unexpected live HTTP")))
    monkeypatch.setattr(catalog.time, "sleep", Mock())
    monkeypatch.setattr(catalog, "MAX_CATALOG_CONNECTION_RETRIES", 2)
    monkeypatch.setattr(catalog, "MAX_CATALOG_REQUEST_RETRIES", 3)
    monkeypatch.setattr(catalog, "RETRY_BASE_SLEEP_SECONDS", 1)
    monkeypatch.setattr(catalog, "RETRY_MAX_SLEEP_SECONDS", 5)
    monkeypatch.setattr(catalog, "RETRY_JITTER_SECONDS", 0.5)
    monkeypatch.setattr(catalog.SYSTEM_RANDOM, "uniform", Mock(return_value=0.25))
    monkeypatch.setenv("CATALOG_URL", "https://catalog.test/")
    monkeypatch.setenv("CATALOG_CLIENT_ID", "internal")
    monkeypatch.setenv("KEYCLOAK_URL", "https://auth.test/realm/")
    monkeypatch.setattr(catalog, "USERS", {"internal": "test-secret", "admin": None})


@pytest.fixture
def search(monkeypatch):
    """Use real catalog orchestration with an isolated mock HTTP session."""
    session = Mock(spec=requests.Session)
    session.headers = {}
    monkeypatch.setattr(catalog, "get_auth_session", Mock(return_value=session))
    return catalog.CatalogSearchUtils("https://catalog.test/", "internal", verbose=True), session


def test_catalog_constants_read_typed_environment_settings(monkeypatch):
    """Environment strings become typed timeout and retry settings without reading real credentials."""
    settings = {
        "CATALOG_CLIENT_SECRET": "test-secret",
        "STAGEOUT_SECRET": "stage-test",
        "ADMIN_SECRET": "admin-test",
        "CATALOG_REQUEST_TIMEOUT_SECONDS": "2.5",
        "MAX_CATALOG_CONNECTION_RETRIES": "4",
        "MAX_CATALOG_REQUEST_RETRIES": "6",
        "RETRY_BASE_SLEEP_SECONDS": "0.2",
        "RETRY_MAX_SLEEP_SECONDS": "3.5",
        "RETRY_JITTER_SECONDS": "0.1",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    values = runpy.run_path(str(Path(catalog.__file__).with_name("catalog_constants.py")))
    assert values["BASE_URL"] == "https://catalog.test/"
    assert values["USERS"] == {"internal": "test-secret", "stageout": "stage-test", "admin": "admin-test"}
    assert values["CATALOG_REQUEST_TIMEOUT_SECONDS"] == 2.5
    assert values["MAX_CATALOG_CONNECTION_RETRIES"] == 4
    assert values["MAX_CATALOG_REQUEST_RETRIES"] == 6
    assert values["RETRY_MAX_SLEEP_SECONDS"] == 3.5
    assert values["RETRY_BASE_SLEEP_SECONDS"] == 0.2
    assert values["RETRY_JITTER_SECONDS"] == 0.1


@pytest.mark.parametrize(
    "status,payload,message",
    [(403, {}, "Token request failed"), (200, {}, "No access_token"), (200, {"access_token": 123}, "No access_token")],
)
def test_token_rejects_failed_or_malformed_response(monkeypatch, status, payload, message):
    """HTTP success alone is insufficient when no nonempty token is present."""
    monkeypatch.setattr(requests, "post", Mock(return_value=response(status, payload)))
    with pytest.raises(ValueError, match=message):
        catalog.get_token("internal")


def test_token_request_uses_configured_realm_and_credentials(monkeypatch):
    """Client credentials and explicit timeout reach the token endpoint."""
    post = Mock(return_value=response(payload={"access_token": "fresh-token"}))
    monkeypatch.setattr(requests, "post", post)
    assert catalog.get_token("internal", timeout=7) == "fresh-token"
    assert post.call_args.args == ("https://auth.test/realm/protocol/openid-connect/token",)
    assert post.call_args.kwargs["data"] == {
        "grant_type": "client_credentials",
        "client_id": "internal",
        "client_secret": "test-secret",
        "scope": "openid",
    }
    assert post.call_args.kwargs["timeout"] == 7


@pytest.mark.parametrize("client,message", [("unknown", "Not valid client_id"), ("admin", "Missing client secret")])
def test_token_rejects_unknown_or_unconfigured_client(client, message):
    """Invalid credentials fail before making a network request."""
    with pytest.raises(ValueError, match=message):
        catalog.get_token(client)


def test_token_requires_keycloak_url(monkeypatch):
    """Missing endpoint configuration produces a clear error without attempting HTTP."""
    monkeypatch.delenv("KEYCLOAK_URL")
    with pytest.raises(ValueError, match="KEYCLOAK_URL"):
        catalog.get_token("internal")


@pytest.mark.parametrize("succeeds", [True, False])
def test_auth_session_retries_and_preserves_failure_cause(monkeypatch, succeeds):
    """Authentication retries are bounded and successful sessions carry the bearer token."""
    failure = requests.ConnectionError("offline")
    get_token = Mock(side_effect=[failure, "fresh"] if succeeds else [failure, failure])
    monkeypatch.setattr(catalog, "get_token", get_token)
    if succeeds:
        session = catalog.get_auth_session("internal")
        assert session.headers["Authorization"] == "Bearer fresh"
        session.close()
    else:
        with pytest.raises(RuntimeError, match="after 2 attempts") as error:
            catalog.get_auth_session("internal")
        assert error.value.__cause__ is failure
    assert get_token.call_count == 2
    catalog.time.sleep.assert_called_once_with(1)


@pytest.mark.parametrize("attempt,expected", [(0, 1.25), (1, 1.25), (2, 2.25), (20, 5)])
def test_retry_delay_is_exponential_with_bounded_jitter(attempt, expected):
    """Backoff grows from the base delay and never exceeds the configured cap."""
    assert catalog._compute_retry_delay(attempt) == expected


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout("timeout"),
        requests.ConnectionError("offline"),
        requests.exceptions.ChunkedEncodingError("partial"),
        response(429),
        response(503),
    ],
)
def test_get_retries_transient_failures(monkeypatch, failure):
    """Transport failures and retryable responses are replayed until success."""
    success = response(payload={"ok": True})
    request = Mock(side_effect=[failure, success])
    monkeypatch.setattr(requests, "request", request)
    assert catalog.make_request("GET", "https://catalog.test/search", token="token", timeout=8, verify=True) is success
    assert request.call_count == 2
    assert request.call_args.kwargs["timeout"] == 8
    assert request.call_args.kwargs["verify"] is True
    catalog.time.sleep.assert_called_once_with(1.25)


@pytest.mark.parametrize("method,status", [("POST", 503), ("GET", 400), ("GET", 404)])
def test_nonretryable_requests_are_not_replayed(monkeypatch, method, status):
    """Mutations and permanent client errors consume one request only."""
    reply = response(status)
    request = Mock(return_value=reply)
    monkeypatch.setattr(requests, "request", request)
    assert catalog.make_request(method, "https://catalog.test/items", token="token", data={"id": "x"}) is reply
    request.assert_called_once()
    catalog.time.sleep.assert_not_called()


@pytest.mark.parametrize("failure", [requests.Timeout("timeout"), response(503)])
def test_request_retry_budget_is_exhausted(monkeypatch, failure):
    """Exhaustion returns the last HTTP response or raises the last transport exception."""
    request = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr(requests, "request", request)
    if isinstance(failure, Exception):
        with pytest.raises(requests.Timeout):
            catalog.make_request("GET", "https://catalog.test/search", token="token")
    else:
        assert catalog.make_request("GET", "https://catalog.test/search", token="token") is failure
    assert request.call_count == 3
    assert catalog.time.sleep.call_args_list == [call(1.25), call(2.25)]


@pytest.mark.parametrize("refresh", [True, False])
def test_request_refreshes_unauthorized_token_at_most_once(monkeypatch, refresh):
    """A 401 refresh is optional and cannot cause an endless authentication loop."""
    token = Mock(side_effect=["old", "new"])
    monkeypatch.setattr(catalog, "get_token", token)
    snapshots = []

    def unauthorized(**kwargs):
        """Capture headers before the retry mutates them."""
        snapshots.append(dict(kwargs["headers"]))
        return response(401)

    monkeypatch.setattr(requests, "request", unauthorized)
    assert catalog.make_request("GET", "https://catalog.test/search", retry_on_401=refresh).status_code == 401
    assert [headers["Authorization"] for headers in snapshots] == (["Bearer old", "Bearer new"] if refresh else ["Bearer old"])
    assert token.call_count == (2 if refresh else 1)


@pytest.mark.parametrize("status,expected", [(200, True), (404, False)])
def test_item_existence_distinguishes_found_and_missing(monkeypatch, status, expected):
    """The item URL includes collection and item identifiers and uses the configured client."""
    request = Mock(return_value=response(status))
    monkeypatch.setattr(catalog, "make_request", request)
    assert catalog.item_id_exists_incatalog("sentinel", "item-1", timeout=3, verify_ssl=True) is expected
    request.assert_called_once_with(
        "GET", "https://catalog.test/collections/sentinel/items/item-1", client_id="internal", timeout=3, verify=True
    )


@pytest.mark.parametrize(
    "failure,message", [(response(500), "Unexpected response"), (requests.ConnectionError("offline"), "connectivity error")]
)
def test_item_existence_does_not_treat_outages_as_missing(monkeypatch, failure, message):
    """Server and transport errors propagate rather than masquerading as an absent item."""
    request = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    monkeypatch.setattr(catalog, "make_request", request)
    with pytest.raises(RuntimeError, match=message):
        catalog.item_id_exists_incatalog("sentinel", "item-1")


@pytest.mark.parametrize("succeeds", [True, False])
def test_stac_connection_retries_with_authenticated_headers(monkeypatch, succeeds):
    """Client initialization retries failures with normalized URLs and bearer headers."""
    session = Mock(headers={"Authorization": "Bearer test"})
    monkeypatch.setattr(catalog, "get_auth_session", Mock(return_value=session))
    expected, failure = object(), ValueError("invalid catalog")
    opening = Mock(side_effect=[failure, expected] if succeeds else [failure, failure])
    monkeypatch.setattr(catalog.Client, "open", opening)
    if succeeds:
        assert catalog.create_stac_client_catalog() is expected
    else:
        with pytest.raises(ConnectionError, match="after 2 attempts") as error:
            catalog.create_stac_client_catalog()
        assert error.value.__cause__ is failure
    assert opening.call_args == call("https://catalog.test", headers={"Authorization": "Bearer test"})
    assert opening.call_count == 2


def test_stac_connection_requires_endpoint(monkeypatch):
    """Missing catalog configuration fails before authentication."""
    monkeypatch.delenv("CATALOG_URL")
    with pytest.raises(ValueError, match="CATALOG_URL"):
        catalog.create_stac_client_catalog()


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_search_pagination_keeps_filters_and_annotates_page_order(search, method):
    """GET paging preserves base parameters and POST paging retains its filter body."""
    client, session = search
    transport = session.get if method == "GET" else session.post
    transport.side_effect = [
        response(payload={"features": [{"id": "a"}], "links": [{"rel": "next", "href": "search?offset=1&limit=2"}]}),
        response(payload={"features": [{"id": "b"}, {"id": "c"}]}),
    ]
    body = {"collections": ["s1", "s2"], "bbox": [0, 1, 2, 3]}
    original = deepcopy(body)
    result = client.fetch_features(body, method=method, limit=2)
    assert body == original
    assert [(item["id"], item["page"], item["aa"]) for item in result] == [("a", 1, 0), ("b", 2, 0), ("c", 2, 1)]
    assert transport.call_args.args[0] == "https://catalog.test/search?offset=1&limit=2"
    if method == "GET":
        assert transport.call_args.kwargs["params"] == {
            "collections": "s1,s2",
            "bbox": "0.0,1.0,2.0,3.0",
            "limit": "2",
            "offset": "1",
        }
    else:
        assert transport.call_args.kwargs["json"] == {**body, "limit": 2}


def test_search_max_items_stops_before_next_page(search):
    """Truncation returns exactly the requested count without downloading another page."""
    client, session = search
    session.post.return_value = response(
        payload={"features": [{"id": "a"}, {"id": "b"}], "links": [{"rel": "next", "href": "search?offset=2"}]}
    )
    assert client.fetch_item_ids({}, max_items=1) == ["a"]
    session.post.assert_called_once()


def test_search_refreshes_session_and_retries_transient_page_errors(search, monkeypatch):
    """A search can refresh authentication, back off after a server failure, and then succeed."""
    client, session = search
    monkeypatch.setattr(catalog, "get_token", Mock(return_value="renewed"))
    session.get.side_effect = [response(401), response(503), response(payload={"features": [{"id": "ok"}]})]
    assert client.fetch_item_ids({}, method="GET") == ["ok"]
    assert session.headers["Authorization"] == "Bearer renewed"
    assert session.get.call_count == 3
    catalog.time.sleep.assert_called_once_with(1.25)


@pytest.mark.parametrize(
    "body,kwargs,message",
    [
        ({}, {"page_retry_attempts": 0}, "page_retry_attempts"),
        ({}, {"method": "DELETE"}, "method must"),
        ({"op": "=", "args": [{"property": "id"}, "x"]}, {"method": "GET"}, "does not support CQL2"),
        ({"bbox": [1, 2, 3]}, {"method": "GET"}, "4-length"),
        ({"intersects": 12}, {"method": "GET"}, "GeoJSON"),
    ],
)
def test_search_rejects_invalid_requests_before_transport(search, body, kwargs, message):
    """Invalid methods, retry budgets, and geometry parameters cannot produce a request."""
    client, session = search
    with pytest.raises(ValueError, match=message):
        client.fetch_features(body, **kwargs)
    session.get.assert_not_called()
    session.post.assert_not_called()


def test_post_search_wraps_bare_cql_filter_without_modifying_input(search):
    """Bare expressions are normalized to a STAC CQL2 JSON request."""
    client, session = search
    session.post.return_value = response(payload={"features": []})
    expression = {"op": "=", "args": [{"property": "platform"}, "sentinel-2"]}
    before = deepcopy(expression)
    assert client.fetch_features(expression, limit=7) == []
    assert session.post.call_args.kwargs["json"] == {"filter-lang": "cql2-json", "filter": expression, "limit": 7}
    assert expression == before


@pytest.mark.parametrize("geometry", [{"type": "Point", "coordinates": [1, 2]}, '{"type":"Point","coordinates":[1,2]}'])
def test_get_encodes_geometry_and_ignores_unsupported_keys(search, geometry):
    """Simple search parameters are serialized without forwarding unsupported fields."""
    client, session = search
    session.get.return_value = response()
    assert (
        client.fetch_features(
            {
                "ids": "a",
                "intersects": geometry,
                "datetime": "2024-01-01",
                "offset": 2,
                "sortby": "datetime",
                "collections": None,
            },
            method="GET",
        )
        == []
    )
    assert session.get.call_args.kwargs["params"] == {
        "ids": "a",
        "intersects": '{"type":"Point","coordinates":[1,2]}',
        "datetime": "2024-01-01",
        "offset": "2",
        "limit": "100",
    }


def test_search_propagates_failed_page_instead_of_partial_success(search):
    """A later-page HTTP failure must not return a misleading complete result."""
    client, session = search
    session.post.side_effect = [
        response(payload={"features": [{"id": "first"}], "links": [{"rel": "next", "href": "search?offset=1"}]}),
        response(400),
    ]
    with pytest.raises(requests.HTTPError):
        client.fetch_features({})


def test_collection_and_item_iterators_follow_next_links(search):
    """Collection and item listings omit initial query parameters on subsequent next URLs."""
    client, session = search
    next_url = "https://catalog.test/collections?offset=1"
    session.get.side_effect = [
        response(payload={"collections": [{"id": "s1"}], "links": [{"rel": "next", "href": next_url}]}),
        response(payload={"collections": [{"id": "s2"}]}),
    ]
    assert client.get_all_collections(limit=1) == [{"id": "s1"}, {"id": "s2"}]
    assert session.get.call_args.kwargs["params"] == {}
    assert session.get.call_args.args[0] == next_url
    session.get.side_effect = [
        response(payload={"features": [{"id": "a"}, {}], "links": [{"rel": "next", "href": next_url}]}),
        response(payload={"features": [{"id": "b"}]}),
    ]
    assert list(client.iter_item_ids("s1")) == ["a", "b"]
    assert session.get.call_args.kwargs["params"] == {}


@pytest.mark.parametrize("value,expected", [("12", 12), (None, 0), ("bad", 0)])
def test_collection_count_coerces_numeric_values_and_defaults_invalid_values(search, value, expected):
    """Malformed count metadata does not break collection summaries."""
    client, session = search
    session.post.return_value = response(payload={"numberMatched": value})
    assert client.get_collection_item_count("s1") == expected
    assert session.post.call_args.kwargs["json"] == {"collections": ["s1"], "limit": 1}


def test_collection_summary_retains_successful_counts_when_one_fails(search, capsys):
    """Partial count failures become zero and do not discard successful collections."""
    client, session = search
    session.get.return_value = response(payload={"collections": [{"id": "s1"}, {}, {"id": "s2"}]})
    session.post.side_effect = [response(payload={"numberMatched": 5}), response(400)]
    assert client.get_all_collection_item_counts() == {"s1": 5, "s2": 0}
    output = capsys.readouterr().out
    assert "Failed to get item count for collection s2" in output
    assert "Total items" in output


def test_empty_collection_summary(search):
    """An empty catalog produces an empty summary without formatting errors."""
    client, session = search
    session.get.return_value = response(payload={"collections": []})
    assert client.get_all_collection_item_counts() == {}


def test_feature_conversion_preserves_schema_and_tolerates_missing_geometry(search):
    """STAC properties map to stable columns while malformed geometries remain null."""
    client, _ = search
    features = [
        {
            "id": "a",
            "collection": "s1",
            "geometry": {"type": "Point", "coordinates": [1, 2]},
            "properties": {"eo:cloud_cover": 12, "sat:orbit_state": "ascending", "instruments": ["SAR", "C"]},
        },
        {"id": "bad", "geometry": {"type": "not-a-geometry"}},
        {"id": "missing"},
    ]
    result = client.features_to_geodataframe(features)
    assert result.crs.to_epsg() == 4326
    assert result["item_id"].tolist() == ["a", "bad", "missing"]
    assert result.geometry.iloc[0].equals(Point(1, 2))
    assert result.geometry.isna().tolist() == [False, True, True]
    assert result.loc[0, "instruments"] == "SAR, C"
    assert result.loc[0, "cloud_cover"] == 12
    assert result.loc[0, "sat_orbit_state"] == "ascending"
    empty = client.features_to_geodataframe([])
    assert empty.empty
    assert empty.columns.tolist() == result.columns.tolist()
    assert empty.crs == result.crs
