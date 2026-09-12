"""Catalog search utilities for querying STAC API endpoints with authentication.

This module provides the CatalogSearchUtils class for performing authenticated
search queries against STAC (SpatioTemporal Asset Catalog) API endpoints.
It handles token management, pagination, and support for both GET and POST methods.
"""

import json
import os
import secrets
import time
import warnings
from copy import deepcopy
from typing import Any, Callable, Literal, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import geopandas as gpd
import requests
import urllib3
from pystac_client import Client
from shapely.geometry import shape
from urllib3.exceptions import InsecureRequestWarning

from satellites.data_preparation.sources.hub.catalog_constants import (
    CATALOG_REQUEST_TIMEOUT_SECONDS,
    MAX_CATALOG_CONNECTION_RETRIES,
    MAX_CATALOG_REQUEST_RETRIES,
    RETRY_BASE_SLEEP_SECONDS,
    RETRY_JITTER_SECONDS,
    RETRY_MAX_SLEEP_SECONDS,
    RETRYABLE_HTTP_STATUS_CODES,
    RETRYABLE_REQUEST_EXCEPTIONS,
    USERS,
)

_KEY_EO_CLOUD_COVER = "eo:cloud_cover"
_KEY_SAT_ORBIT_STATE = "sat:orbit_state"
_KEY_VIEW_AZIMUTH = "view:azimuth"
_KEY_VIEW_INCIDENCE_ANGLE = "view:incidence_angle"
APPLICATION_JSON = "application/json"
EPSG_4326 = "EPSG:4326"

warnings.filterwarnings("ignore", category=UserWarning, message=".*CPLE_NotSupported.*")
urllib3.disable_warnings(InsecureRequestWarning)
SYSTEM_RANDOM = secrets.SystemRandom()


def _compute_retry_delay(attempt: int) -> float:
    """Compute exponential backoff with jitter, capped to a max delay."""
    exponential_delay = RETRY_BASE_SLEEP_SECONDS * (2 ** max(attempt - 1, 0))
    jitter = SYSTEM_RANDOM.uniform(0.0, RETRY_JITTER_SECONDS)
    return min(RETRY_MAX_SLEEP_SECONDS, exponential_delay + jitter)


def _run_retry_backoff(
    *,
    attempt: int,
    message: str,
    on_retry: Optional[Callable[[int, str], None]] = None,
) -> None:
    """Execute retry callback when provided; otherwise apply default backoff sleep."""
    if on_retry is not None:
        on_retry(attempt, message)
        return
    time.sleep(_compute_retry_delay(attempt))


def _build_transient_retry_message(
    *,
    method_u: str,
    url: str,
    attempt: int,
    max_attempts: int,
    error: Exception,
) -> str:
    """Build retry log message for transient request exceptions."""
    return f"Transient request error on {method_u} {url} (attempt {attempt}/{max_attempts}): {error}. Retrying..."


def _build_retryable_status_message(
    *,
    method_u: str,
    url: str,
    attempt: int,
    max_attempts: int,
    status_code: int,
) -> str:
    """Build retry log message for retryable HTTP status codes."""
    return f"Catalog returned HTTP {status_code} for {method_u} {url} (attempt {attempt}/{max_attempts}). Retrying..."


def _should_retry_status(status_code: int, *, attempt: int, max_attempts: int) -> bool:
    """Return True when response status is retryable and attempts remain."""
    return status_code in RETRYABLE_HTTP_STATUS_CODES and attempt < max_attempts


def _execute_with_optional_401_refresh(
    request_once: Callable[[], requests.Response],
    *,
    retry_on_401: bool,
    request_after_refresh: Optional[Callable[[], requests.Response]] = None,
) -> requests.Response:
    """Execute request and optionally retry once after auth refresh on HTTP 401."""
    resp = request_once()
    if resp.status_code == 401 and retry_on_401 and request_after_refresh is not None:
        return request_after_refresh()
    return resp


def _request_with_retries(
    request_func: Callable[[], requests.Response],
    *,
    method_u: str,
    url: str,
    max_attempts: int,
    on_retry: Optional[Callable[[int, str], None]] = None,
) -> requests.Response:
    """Execute a request callable with retry/backoff for transient failures."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = request_func()
        except RETRYABLE_REQUEST_EXCEPTIONS as exc:
            if attempt >= max_attempts:
                raise
            _run_retry_backoff(
                attempt=attempt,
                message=_build_transient_retry_message(
                    method_u=method_u,
                    url=url,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=exc,
                ),
                on_retry=on_retry,
            )
            continue

        if _should_retry_status(resp.status_code, attempt=attempt, max_attempts=max_attempts):
            _run_retry_backoff(
                attempt=attempt,
                message=_build_retryable_status_message(
                    method_u=method_u,
                    url=url,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    status_code=resp.status_code,
                ),
                on_retry=on_retry,
            )
            continue

        return resp

    raise RuntimeError("Unexpected retry loop exit in _request_with_retries")


# -----------------------------------------------------------------------
# Auth / Keycloak functions
# -----------------------------------------------------------------------
def _require_secret(client_id: str) -> str:
    """Get the secret key for the user.

    Parameters
    ----------
    client_id : str
        One of the configured client IDs (e.g. "internal").

    Returns
    -------
    str
        The secret key for the specified client ID.

    """
    secret = USERS.get(client_id)
    if not secret:
        raise ValueError(f"Missing client secret for '{client_id}'. Ensure env var is set (e.g., ADMIN, AGRICULTURE, etc.).")
    return secret


def fetch_keycloak_token(
    client_id: str,
    client_secret: str,
    timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Get the keycloak token for the client.

    Parameters
    ----------
    client_id : str
        One of the configured client IDs (e.g. "internal").
    client_secret : str
        The secret key for the client.
    timeout : float, optional
        Request timeout in seconds, by default 60.0.

    Returns
    -------
    str
        The access token retrieved from Keycloak.

    """
    keycloack_url = os.getenv("KEYCLOAK_URL")
    if not keycloack_url:
        raise ValueError("KEYCLOAK_URL environment variable not set")
    url = f"{keycloack_url.rstrip('/')}/protocol/openid-connect/token"

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "openid",
    }
    response = requests.post(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=timeout)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("No access_token in Keycloak response")
        return access_token
    raise ValueError(f"Token request failed: {response.status_code} {response.text}")


def get_token(client_id: str, timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS) -> str:
    """Get the token for the client.

    Parameters
    ----------
    client_id : str
        One of the configured client IDs (e.g. "internal").
    timeout : float, optional
        Request timeout in seconds, by default 60.0.

    Returns
    -------
    str
        The access token for the specified client ID.

    """
    if client_id not in USERS:
        raise ValueError(f"Not valid client_id: {client_id}; must be one of {list(USERS.keys())}")
    client_secret = _require_secret(client_id)
    return fetch_keycloak_token(client_id, client_secret, timeout=timeout)


def get_auth_session(client_id: str, verbose=True) -> requests.Session:
    """Create an authenticated requests session with OAuth2 bearer token.

    Returns
    -------
    requests.Session
        Authenticated session with Authorization header.

    Raises
    ------
    RuntimeError
        If token acquisition fails.

    """
    last_exception = None
    for attempt in range(1, MAX_CATALOG_CONNECTION_RETRIES + 1):
        try:
            token = get_token(client_id)
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {token}"})
            return session
        except Exception as e:
            last_exception = e
            msg = (
                f"Failed to acquire OAuth2 token for client_id={client_id!r} "
                f"(attempt {attempt}/{MAX_CATALOG_CONNECTION_RETRIES}): {e}"
            )
            if verbose:
                print(msg)
            if attempt < MAX_CATALOG_CONNECTION_RETRIES:
                time.sleep(RETRY_BASE_SLEEP_SECONDS * attempt)

    raise RuntimeError(
        f"Failed to acquire OAuth2 token for client_id={client_id!r} after {MAX_CATALOG_CONNECTION_RETRIES} attempts"
    ) from last_exception


def make_request(
    method: str,
    url: str,
    token: Optional[str] = None,
    client_id: Optional[str] = None,
    data: Optional[dict] = None,
    timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS,
    verify: bool = False,
    retry_on_401: bool = True,
) -> requests.Response:
    """Create an HTTP request to ``url`` with basic retry for idempotent methods.

    Parameters
    ----------
    method : str
        HTTP method to use (e.g., "GET").
    url : str
        The URL to send the request to.
    token : Optional[str], optional
        Bearer token for authorization, by default None.
    client_id : Optional[str], optional
        OAuth2 client ID used to obtain/refresh token when needed.
    data : Optional[dict], optional
        JSON data to include in the request body, by default None.
    timeout : float, optional
        Request timeout in seconds, by default 60.0.
    verify : bool, optional
        Whether to verify TLS certificates, by default False.
    retry_on_401 : bool, optional
        Whether to refresh token and retry once on HTTP 401, by default True.

    Returns
    -------
    requests.Response
        The HTTP response from the request.

    """
    method_u = method.upper()
    retryable_methods = {"GET", "HEAD", "OPTIONS"}
    max_attempts = MAX_CATALOG_REQUEST_RETRIES if method_u in retryable_methods else 1

    # Resolve client identity for auth flows when token is not provided
    resolved_client_id = client_id or os.getenv("CATALOG_CLIENT_ID", "internal")
    if token is None:
        token = get_token(resolved_client_id)

    def _request_once_with_401_refresh() -> requests.Response:
        nonlocal token
        headers = {"Accept": APPLICATION_JSON}
        if data is not None:
            headers["Content-Type"] = APPLICATION_JSON
        if token:
            headers["Authorization"] = f"Bearer {token}"

        def _request_once() -> requests.Response:
            return requests.request(method=method, url=url, headers=headers, json=data, timeout=timeout, verify=verify)

        def _request_after_refresh() -> requests.Response:
            nonlocal token
            token = get_token(resolved_client_id)
            headers["Authorization"] = f"Bearer {token}"
            return requests.request(method=method, url=url, headers=headers, json=data, timeout=timeout, verify=verify)

        return _execute_with_optional_401_refresh(
            _request_once,
            retry_on_401=retry_on_401,
            request_after_refresh=_request_after_refresh,
        )

    return _request_with_retries(
        _request_once_with_401_refresh,
        method_u=method_u,
        url=url,
        max_attempts=max_attempts,
    )


def item_id_exists_incatalog(
    collection_id: str,
    item_id: str,
    timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS,
    verify_ssl: bool = False,
) -> bool:
    """Check if a STAC Item exists in the catalog (HTTP 200 means exists, 404 not found).

    Parameters
    ----------
    collection_id : str
        The STAC Collection ID to check.
    item_id : str
        The STAC Item ID to check.
    timeout : float, optional
        Request timeout in seconds, by default 60.0.
    verify_ssl : bool, optional
        Whether to verify TLS certificates, by default False.

    Returns
    -------
    bool
        True if item exists, False if not. Raises on unexpected HTTP errors.

    """
    from urllib.parse import urljoin

    # Build URL robustly
    base_url = os.getenv("CATALOG_URL")
    client_id = os.getenv("CATALOG_CLIENT_ID", "internal")

    catalog_url = base_url.rstrip("/") + "/"
    url = urljoin(catalog_url, f"collections/{collection_id}/items/{item_id}")

    # HEAD can be used too, but GET is safer across proxies
    try:
        resp = make_request("GET", url, client_id=client_id, timeout=timeout, verify=verify_ssl)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Catalog connectivity error while checking item existence: collection={collection_id}, item={item_id}: {exc}"
        ) from exc

    if resp.status_code == 200:
        return True
    if resp.status_code == 404:
        return False
    # Anything else is unexpected → debug it
    raise RuntimeError(f"Unexpected response checking item existence: {resp.status_code} - {resp.text[:500]}")


# -----------------------------------------------------------------------
# Catalog functions using stac-auth client
# -----------------------------------------------------------------------


def create_stac_client_catalog(client_id: str | None = None, catalog_url: str | None = None, verbose: bool = True) -> Client:
    """Connect to an EOPKA catalog with stac-auth.

    Parameters
    ----------
    client_id : str, optional
        One of the configured client IDs (e.g. "internal"), by default None.
    catalog_url : str, optional
        The URL of the catalog to connect to, by default None.
    verbose : bool, optional
        Whether to print connection status, by default True.

    Returns
    -------
    Client
        The connected STAC Client instance.

    """
    if not catalog_url:
        catalog_url = os.getenv("CATALOG_URL")
    if not catalog_url:
        raise ValueError("CATALOG_URL is not configured")
    catalog_url = catalog_url.rstrip("/")
    if verbose:
        print(f"Connecting to catalog: {catalog_url}")
    if not client_id:
        client_id = os.getenv("CATALOG_CLIENT_ID", "internal")

    last_exception = None
    for attempt in range(1, MAX_CATALOG_CONNECTION_RETRIES + 1):
        try:
            session = get_auth_session(client_id, verbose=verbose)
            headers: dict[str, str] = {k: str(v) for k, v in session.headers.items()}
            return Client.open(catalog_url, headers=headers)
        except Exception as e:
            last_exception = e
            if verbose:
                print(f"Failed to connect to catalog (attempt {attempt}/{MAX_CATALOG_CONNECTION_RETRIES}): {e}")
            if attempt < MAX_CATALOG_CONNECTION_RETRIES:
                time.sleep(RETRY_BASE_SLEEP_SECONDS * attempt)

    raise ConnectionError(
        f"Error: Could not connect with the catalog after {MAX_CATALOG_CONNECTION_RETRIES} attempts: {last_exception}"
    ) from last_exception


class CatalogSearchUtils:
    """Self-contained helper to run authenticated /search queries and return features.

    This class provides methods to query a STAC catalog endpoint with authentication,
    including support for pagination, filtering, and collection management.
    It automatically handles token refresh on 401 responses.

    Attributes
    ----------
        base_url (str): The base URL of the STAC catalog endpoint.
        client_id (str): The OAuth2 client ID for authentication.
        verbose (bool): Whether to print verbose output during operations.
    """

    def __init__(self, catalog_endpoint: str, client_id: str, verbose: bool = True):
        """Initialize the CatalogSearchUtils.

        Parameters
        ----------
        catalog_endpoint : str
            Base URL of the STAC catalog API endpoint.
        client_id : str
            OAuth2 client ID for authentication.
        verbose : bool, optional
            Whether to print debug/info messages. Defaults to True.

        """
        self.base_url = catalog_endpoint.rstrip("/")
        self.client_id = client_id
        self.verbose = verbose
        # Allowed query parameters for GET /search requests
        self._GET_SEARCH_ALLOWED = {"collections", "ids", "datetime", "bbox", "intersects", "limit", "offset"}

    def _refresh_token_and_retry(self, session: requests.Session) -> requests.Session:
        """Refresh the OAuth2 token and update the session's Authorization header.

        This method is called when a 401 Unauthorized response is received,
        allowing the session to continue with a fresh token.

        Parameters
        ----------
        session : requests.Session
            The existing session to update.

        Returns
        -------
        requests.Session
            The session with updated Authorization header.

        """
        token = get_token(self.client_id)
        session.headers.update({"Authorization": f"Bearer {token}"})
        return session

    def _get_authenticated_session(self) -> requests.Session:
        """Create an authenticated session using this utility's client id."""
        return get_auth_session(self.client_id, verbose=self.verbose)

    def _request_once(
        self,
        session: requests.Session,
        method_u: Literal["GET", "POST"],
        url: str,
        timeout: int,
        **kwargs: Any,
    ) -> requests.Response:
        """Execute one HTTP request with the specified method."""
        if method_u == "GET":
            return session.get(url, timeout=timeout, **kwargs)
        return session.post(url, timeout=timeout, **kwargs)

    def _retry_with_backoff(self, attempt: int, message: str) -> None:
        """Log retry message (if verbose) and sleep with exponential backoff + jitter."""
        delay_seconds = _compute_retry_delay(attempt)
        if self.verbose:
            print(f"{message} Backing off for {delay_seconds:.2f}s")
        time.sleep(delay_seconds)

    def _make_request(
        self,
        session: requests.Session,
        method: Literal["GET", "POST"],
        url: str,
        *,
        timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS,
        max_attempts: int | None = None,
        retry_on_401: bool = True,
        **kwargs: Any,
    ) -> tuple[requests.Response, requests.Session]:
        """Make an HTTP request with automatic token refresh on 401 Unauthorized.

        Parameters
        ----------
        session : requests.Session
            The requests session to use for the request.
        method : Literal["GET", "POST"]
            HTTP method to use ('GET' or 'POST').
        url : str
            Request URL.
        timeout : int, optional
            Request timeout in seconds. Defaults to 60.
        max_attempts : int | None, optional
            Maximum attempts for transient transport errors and retryable HTTP
            responses. ``None`` uses ``MAX_CATALOG_REQUEST_RETRIES``.
        retry_on_401 : bool, optional
            If True, refresh token and retry on 401 response. Defaults to True.
        **kwargs
            Additional arguments to pass to the request method (json, params, etc.)

        Returns
        -------
        tuple[requests.Response, requests.Session]
            Tuple of (response object, updated session with refreshed token if needed).

        Raises
        ------
        ValueError
            If an unsupported HTTP method is provided.

        """
        if session is None:
            raise RuntimeError("Authenticated session is None.")

        method_u = method.upper()
        if method_u not in {"GET", "POST"}:
            raise ValueError(f"Unsupported HTTP method: {method}")
        if max_attempts is None:
            max_attempts = MAX_CATALOG_REQUEST_RETRIES
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        def _request_once_with_401_refresh() -> requests.Response:
            nonlocal session

            def _request_once() -> requests.Response:
                return self._request_once(session, method_u, url, timeout, **kwargs)

            def _request_after_refresh() -> requests.Response:
                nonlocal session
                session = self._refresh_token_and_retry(session)
                return self._request_once(session, method_u, url, timeout, **kwargs)

            return _execute_with_optional_401_refresh(
                _request_once,
                retry_on_401=retry_on_401,
                request_after_refresh=_request_after_refresh,
            )

        resp = _request_with_retries(
            _request_once_with_401_refresh,
            method_u=method_u,
            url=url,
            max_attempts=max_attempts,
            on_retry=self._retry_with_backoff,
        )
        return resp, session

    def _extract_properties_from_filter(self, filter_obj: Any) -> set[str]:
        """Recursively extract property names from a CQL2 filter object.

        This method traverses nested filter structures (dictionaries, lists) to find
        all property names referenced in the filter, which is useful for validation.

        Parameters
        ----------
        filter_obj : Any
            A filter object (dict, list, or other type) to extract properties from.

        Returns
        -------
        set[str]
            Set of all property names found in the filter.

        """
        properties = set()
        if isinstance(filter_obj, dict):
            # Check if this dict has a "property" key with a string value
            if "property" in filter_obj:
                prop = filter_obj["property"]
                if isinstance(prop, str):
                    properties.add(prop)
            # Recursively check all values in the dict
            for value in filter_obj.values():
                properties.update(self._extract_properties_from_filter(value))
        elif isinstance(filter_obj, (list, tuple)):
            # Recursively check all items in the list/tuple
            for item in filter_obj:
                properties.update(self._extract_properties_from_filter(item))
        return properties

    def _encode_collections_ids_param(self, value: Any) -> str:
        """Encode collections or ids parameter as comma-separated string.

        Parameters
        ----------
        value : Any
            List, tuple, or string value.

        Returns
        -------
        str
            Comma-separated string of values.

        """
        if isinstance(value, (list, tuple)):
            return ",".join(str(x) for x in value)
        return str(value)

    def _encode_bbox_param(self, value: Any) -> str:
        """Encode bbox parameter as comma-separated coordinate string.

        Parameters
        ----------
        value : Any
            Must be a 4-element list/tuple (minx, miny, maxx, maxy).

        Returns
        -------
        str
            Comma-separated bbox coordinates.

        Raises
        ------
        ValueError
            If bbox is not a 4-element list/tuple.

        """
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"GET bbox must be a 4-length list/tuple; got: {value!r}")
        return ",".join(str(float(x)) for x in value)

    def _encode_intersects_param(self, value: Any) -> str:
        """Encode intersects parameter as GeoJSON string.

        Parameters
        ----------
        value : Any
            Must be a dict (GeoJSON geometry) or JSON string.

        Returns
        -------
        str
            JSON-encoded geometry.

        Raises
        ------
        ValueError
            If intersects format is invalid.

        """
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value, separators=(",", ":"))
        raise ValueError(f"GET intersects must be a dict GeoJSON geometry or JSON string; got: {type(value)}")

    def _encode_get_search_params(self, body: dict[str, Any]) -> dict[str, str]:
        """Encode a search body dictionary into GET query parameters.

        Converts search parameters to URL-encoded format suitable for GET requests,
        with special handling for collections, bbox, datetime, intersects, limit, and offset.

        Parameters
        ----------
        body : dict[str, Any]
            Search request body with parameters.

        Returns
        -------
        dict[str, str]
            Dictionary of query parameters encoded as strings.

        Raises
        ------
        ValueError
            If bbox is not a 4-element list/tuple or intersects format is invalid.

        """
        params: dict[str, str] = {}
        # Process only the allowed GET search parameters
        for k in self._GET_SEARCH_ALLOWED:
            if k not in body or body[k] is None:
                continue
            v = body[k]

            # Handle comma-separated lists for collections and ids
            if k in {"collections", "ids"}:
                params[k] = self._encode_collections_ids_param(v)
            # Handle bounding box as 4-element list (minx, miny, maxx, maxy)
            elif k == "bbox":
                params[k] = self._encode_bbox_param(v)
            # Handle ISO 8601 datetime string
            elif k == "datetime":
                params[k] = str(v)
            # Handle spatial intersects as GeoJSON geometry
            elif k == "intersects":
                params[k] = self._encode_intersects_param(v)
            # Handle pagination parameters as integers
            elif k in {"limit", "offset"}:
                params[k] = str(int(v))
            else:
                params[k] = str(v)
        return params

    def _extract_paging_params_from_href(self, href: str) -> dict[str, str]:
        """Extract pagination parameters (offset, limit) from a URL.

        Parameters
        ----------
        href : str
            URL containing query parameters.

        Returns
        -------
        dict[str, str]
            Dictionary with 'offset' and/or 'limit' if present in the URL.

        """
        q = parse_qs(urlparse(href).query)
        out: dict[str, str] = {}
        # Extract the first value for offset and limit query parameters
        for k in ("offset", "limit"):
            if k in q and q[k]:
                out[k] = q[k][0]
        return out

    def _format_composite_filter(self, obj: dict[str, Any], level: int) -> str:
        """Format a composite CQL2 filter (and/or/not) with indentation."""
        indent = "    " * level
        op = obj.get("op")

        result = "{\n"
        result += f'{indent}    "op": "{op}"'

        args = obj.get("args")
        if "args" not in obj:
            result += "\n"
            result += f"{indent}}}"
            return result

        if not isinstance(args, list):
            result += "\n"
            result += f"{indent}}}"
            return result

        result += ",\n"
        result += f'{indent}    "args": [\n'
        for i, arg in enumerate(args):
            comma = "," if i < len(args) - 1 else ""
            arg_str = self._format_json_filter_node(arg, level + 2)
            result += f"{indent}        {arg_str}{comma}\n"

        result += f"{indent}    ]\n"
        result += f"{indent}}}"
        return result

    def _format_json_filter_node(self, obj: Any, level: int = 0) -> str:
        """Recursively format one CQL2 filter node for pretty-print output."""
        if not isinstance(obj, dict) or "op" not in obj:
            return json.dumps(obj)

        composite_ops = {"and", "or", "not"}
        if obj.get("op") in composite_ops:
            return self._format_composite_filter(obj, level)

        return json.dumps(obj)

    def _print_json_filter(self, filter_dict: dict[str, Any]) -> None:
        """Pretty-print a CQL2 filter object with proper indentation.

        Recursively formats composite filter operations (and, or, not) with indentation
        for readability. Non-composite filters are printed as compact JSON.

        Parameters
        ----------
        filter_dict : dict[str, Any]
            The filter dictionary to print.

        """
        payload = deepcopy(filter_dict)
        if "filter" in payload:
            payload = payload["filter"]
        print(self._format_json_filter_node(payload))

    def _normalize_search_body(self, search_body: dict[str, Any]) -> dict[str, Any]:
        """Normalize a search body to standard STAC search format.

        If the body contains CQL2 filter operators at the top level (like 'op', 'args')
        without explicit 'filter' or 'filter-lang' keys, wraps them in the standard
        STAC format with 'filter-lang': 'cql2-json'.

        Parameters
        ----------
        search_body : dict[str, Any]
            The search request body.

        Returns
        -------
        dict[str, Any]
            Normalized search body with proper filter wrapping.

        """
        body = dict(search_body)
        # Check if this is a bare CQL2 filter that needs wrapping
        if ("filter" not in body and "filter-lang" not in body) and ("op" in body or "args" in body):
            body = {"filter-lang": "cql2-json", "filter": body}
        return body

    def _check_filter_properties_allowed(self, filter_obj: Any) -> None:
        """Check if filter contains only GET-allowed properties.

        Parameters
        ----------
        filter_obj : Any
            The filter object to validate.

        Raises
        ------
        ValueError
            If filter contains unsupported properties.

        """
        extracted_props = self._extract_properties_from_filter(filter_obj)
        unsupported_props = extracted_props - self._GET_SEARCH_ALLOWED
        if unsupported_props:
            raise ValueError(
                f"GET /search does not support CQL2 filter properties: {sorted(unsupported_props)}. "
                f"Supported properties for GET are: {sorted(self._GET_SEARCH_ALLOWED)}. "
                f"Use method='POST' if you need complex filters with properties like "
                f"'parentidentifier', 'identifier', 'title', etc."
            )

    def _has_filter_expression(self, body: dict[str, Any]) -> bool:
        """Check if body contains filter expressions.

        Parameters
        ----------
        body : dict[str, Any]
            The search request body.

        Returns
        -------
        bool
            True if filter expressions are present.

        """
        return "filter" in body or any(k in body for k in ("op", "args"))

    def _validate_get_filter_properties(self, body: dict[str, Any]) -> dict[str, Any]:
        """Validate and filter body for GET request compatibility.

        Checks for CQL2 filters and validates that all properties are GET-compatible.
        Raises an error if unsupported properties are found.

        Parameters
        ----------
        body : dict[str, Any]
            The search request body.

        Returns
        -------
        dict[str, Any]
            Filtered body with only allowed keys.

        Raises
        ------
        ValueError
            If filter contains unsupported properties.

        """
        # Get filter object from body
        filter_obj = body if "op" in body else body.get("filter")
        if filter_obj:
            # Validate properties are GET-compatible
            self._check_filter_properties_allowed(filter_obj)
            # Keep only supported keys and filter if present
            if "filter" not in body and "op" in body:
                return body
            return {k: v for k, v in body.items() if k in self._GET_SEARCH_ALLOWED or k == "filter"}
        return body

    def _filter_body_keys(self, body: dict[str, Any]) -> dict[str, Any]:
        """Filter body to only include GET-allowed keys.

        Parameters
        ----------
        body : dict[str, Any]
            The search request body.

        Returns
        -------
        dict[str, Any]
            Filtered body with only allowed keys.

        """
        unsupported_keys = set(body.keys()) - self._GET_SEARCH_ALLOWED
        if unsupported_keys and self.verbose:
            print(f"WARNING: GET /search does not support keys: {sorted(unsupported_keys)}. Ignored.")
        return {k: v for k, v in body.items() if k in self._GET_SEARCH_ALLOWED}

    def _prepare_get_params(
        self,
        body: dict[str, Any],
        *,
        limit: int,
        url: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Prepare and validate search parameters for a GET request.

        Validates that the search body can be expressed as GET query parameters,
        filters out unsupported keys, and encodes the parameters. Raises an error
        if complex CQL2 filters are used that cannot be expressed as GET parameters.

        Parameters
        ----------
        body : dict[str, Any]
            The search request body.
        limit : int
            Default result limit for pagination.
        url : str
            The request URL (used for logging).

        Returns
        -------
        tuple[dict[str, str], dict[str, Any]]
            Tuple of (encoded_params, filtered_body).

        Raises
        ------
        ValueError
            If the filter uses properties not supported by GET /search.

        """
        # Validate and filter body
        if "filter" in body or any(k in body for k in ("op", "args")):
            body = self._validate_get_filter_properties(body)
        else:
            body = self._filter_body_keys(body)

        # Set default limit and encode parameters
        body.setdefault("limit", limit)
        base_params = self._encode_get_search_params(body)

        if self.verbose:
            print(f"\nGET {url}\nparams={json.dumps(base_params, indent=2)}")

        return base_params, body

    def _make_initial_search_request(
        self,
        session: requests.Session,
        url: str,
        body: dict[str, Any],
        limit: int,
        timeout: int,
        method_u: str,
        page_retry_attempts: int | None = None,
    ) -> tuple[requests.Response, dict[str, str] | None, requests.Session]:
        """Make the initial search request.

        Parameters
        ----------
        session : requests.Session
            Authenticated session.
        url : str
            The request URL.
        body : dict[str, Any]
            Search request body.
        limit : int
            Results per page.
        timeout : int
            Request timeout in seconds.
        method_u : str
            HTTP method (uppercase).
        page_retry_attempts : int | None
            Maximum attempts for this search page. ``None`` uses the shared
            catalog retry setting.

        Returns
        -------
        tuple[requests.Response, dict[str, str] | None, requests.Session]
            Tuple of (response, base_params for GET or None for POST, updated session).

        """
        base_params: dict[str, str] | None = None
        if method_u == "GET":
            base_params, body = self._prepare_get_params(
                body,
                limit=limit,
                url=url,
            )
            resp, session = self._make_request(
                session, "GET", url, params=base_params, timeout=timeout, max_attempts=page_retry_attempts
            )
        else:
            # POST method: include limit and print request body if verbose
            body.setdefault("limit", limit)
            if self.verbose:
                print(f"\nPOST {url}")
                print(body)
            resp, session = self._make_request(session, "POST", url, json=body, timeout=timeout, max_attempts=page_retry_attempts)
        return resp, base_params, session

    def _log_page_info(self, page_idx: int, data: dict[str, Any], features: list) -> None:
        """Log pagination information if verbose mode is enabled.

        Parameters
        ----------
        page_idx : int
            Current page index.
        data : dict[str, Any]
            Response data.
        features : list
            Features from current page.

        """
        for idx, feature in enumerate(features):
            feature["page"] = page_idx
            feature["aa"] = idx

        if self.verbose:
            links = data.get("links") or []
            rels = [link.get("rel") for link in links]
            nm = data.get("numberMatched")
            nr = data.get("numberReturned")
            print(f"Page {page_idx}: numberMatched={nm}, numberReturned={nr}, features={len(features)}, rels={rels}")

    def _should_continue_pagination(self, max_items: int | None, total_count: int) -> bool:
        """Check if pagination should continue based on max_items limit.

        Parameters
        ----------
        max_items : int | None
            Maximum items to return.
        total_count : int
            Current total features accumulated.

        Returns
        -------
        bool
            True if pagination should continue.

        """
        return max_items is None or total_count < max_items

    def _get_next_page_link(self, links: list) -> dict[str, Any] | None:
        """Extract the next page link from HATEOAS links.

        Parameters
        ----------
        links : list
            List of HATEOAS link objects.

        Returns
        -------
        dict[str, Any] | None
            The next page link or None if not found.

        """
        return next((link for link in links if link.get("rel") == "next"), None)

    def _get_next_href(self, links: list) -> str | None:
        """Extract the href from the rel=next link, if present and valid."""
        next_link = self._get_next_page_link(links)
        if not next_link:
            return None
        href = next_link.get("href")
        if isinstance(href, str) and href:
            return href
        return None

    def _fetch_paginated_response(
        self,
        session: requests.Session,
        url: str,
        base_url: str,
        base_params: dict[str, str] | None,
        body: dict[str, Any],
        timeout: int,
        method_u: str,
        page_retry_attempts: int | None = None,
    ) -> tuple[requests.Response, requests.Session]:
        """Fetch the next page of paginated results.

        Parameters
        ----------
        session : requests.Session
            Authenticated session.
        url : str
            Base search URL.
        base_url : str
            Base catalog URL.
        base_params : dict[str, str] | None
            Base parameters for GET.
        body : dict[str, Any]
            Body for POST.
        timeout : int
            Request timeout in seconds.
        method_u : str
            HTTP method (uppercase).
        page_retry_attempts : int | None
            Maximum attempts for this search page. ``None`` uses the shared
            catalog retry setting.

        Returns
        -------
        tuple[requests.Response, requests.Session]
            Tuple of (response object, updated session).

        """
        if method_u == "POST":
            return self._make_request(session, "POST", url, json=body, timeout=timeout, max_attempts=page_retry_attempts)
        # For GET: extract pagination params from the next URL
        if base_params is None:
            raise RuntimeError("base_params not initialized for GET pagination.")
        paging = self._extract_paging_params_from_href(url)
        params = dict(base_params)
        params.update(paging)
        return self._make_request(session, "GET", url, params=params, timeout=timeout, max_attempts=page_retry_attempts)

    @staticmethod
    def _validate_fetch_features_method(method: Literal["POST", "GET"], body: dict[str, Any]) -> str:
        """Validate fetch_features method and GET filter constraints.

        Returns
        -------
        str
            Uppercase HTTP method ("POST" or "GET").

        Raises
        ------
        ValueError
            If method is unsupported or GET includes filter-based keys.

        """
        method_u = method.upper()
        if method_u not in {"POST", "GET"}:
            raise ValueError(f"method must be 'POST' or 'GET'; got: {method!r}")

        if method_u == "POST":
            return "POST"

        forbidden = {"filter", "filter-lang", "op", "args"} & set(body.keys())
        if forbidden:
            raise ValueError(
                "GET /search is restricted to simple query parameters and does not support CQL2 filters "
                f"(found keys: {sorted(forbidden)}). Use method='POST'."
            )
        return "GET"

    def fetch_features(
        self,
        search_body: dict[str, Any],
        *,
        limit: int = 100,
        max_items: int | None = None,
        timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS,
        method: Literal["POST", "GET"] = "POST",
        page_retry_attempts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search the catalog and iterate through paginated results.

        Performs authenticated search against the catalog endpoint and returns all
        matching features, following pagination links automatically. Supports both
        GET and POST methods, with automatic fallback on authentication failures.

        Parameters
        ----------
        search_body : dict[str, Any]
            STAC search request body with filters, bbox, etc.
        limit : int, optional
            Results per page for pagination. Defaults to 100.
        max_items : int | None, optional
            Maximum items to return. Defaults to None (all items).
        timeout : int, optional
            Request timeout in seconds. Defaults to 60.
        method : Literal["POST", "GET"], optional
            HTTP method to use. Defaults to "POST".
        page_retry_attempts : int | None, optional
            Maximum attempts for each individual search page when a transient
            transport error or retryable HTTP status occurs. Successfully read
            pages are retained while only the failed page is retried. ``None``
            uses ``MAX_CATALOG_REQUEST_RETRIES``.

        Returns
        -------
        list[dict[str, Any]]
            List of feature dictionaries from the search results.

        Raises
        ------
        ValueError
            If an unsupported HTTP method or invalid search parameters are provided.

        """
        if page_retry_attempts is None:
            page_retry_attempts = MAX_CATALOG_REQUEST_RETRIES
        if page_retry_attempts < 1:
            raise ValueError("page_retry_attempts must be at least 1")

        session = self._get_authenticated_session()
        base_url = self.base_url.rstrip("/")
        url = f"{base_url}/search"

        # Normalize the search body to standard STAC format
        body = self._normalize_search_body(search_body)

        # Validate and normalize the HTTP method
        method_u = self._validate_fetch_features_method(method, body)

        # Make initial request
        resp, base_params, session = self._make_initial_search_request(
            session, url, body, limit, timeout, method_u, page_retry_attempts
        )
        resp.raise_for_status()

        total_features: list[dict[str, Any]] = []
        page_idx = 0

        # Iterate through paginated results
        while True:
            data = resp.json()
            page_idx += 1
            features = data.get("features") or []
            links = data.get("links") or []

            # Log pagination information
            self._log_page_info(page_idx, data, features)

            # Accumulate features
            total_features.extend(features)

            # Check if we've reached max_items limit
            if not self._should_continue_pagination(max_items, len(total_features)):
                return total_features[:max_items]

            # Look for next page link
            next_href = self._get_next_href(links)
            if not next_href:
                # No more pages, return all accumulated features
                return total_features

            # Prepare next page request
            next_url = urljoin(base_url + "/", next_href)
            if self.verbose:
                print(f"Next Page: {next_url}")

            # Fetch next page using the same method
            resp, session = self._fetch_paginated_response(
                session, next_url, base_url, base_params, body, timeout, method_u, page_retry_attempts
            )
            resp.raise_for_status()

    def fetch_item_ids(
        self,
        search_body: dict[str, Any],
        *,
        limit: int = 1000,
        max_items: int | None = None,
        timeout: float = CATALOG_REQUEST_TIMEOUT_SECONDS,
        method: Literal["POST", "GET"] = "POST",
    ) -> list[str]:
        """Fetch only the IDs of items matching a search query.

        Performs a search and extracts the 'id' field from each matching feature.

        Parameters
        ----------
        search_body : dict[str, Any]
            STAC search request body.
        limit : int, optional
            Results per page. Defaults to 1000.
        max_items : int | None, optional
            Maximum items to return. Defaults to None (all).
        timeout : int, optional
            Request timeout in seconds. Defaults to 60.
        method : Literal["POST", "GET"], optional
            HTTP method. Defaults to "POST".

        Returns
        -------
        list[str]
            List of item IDs from matching features.

        """
        features = self.fetch_features(
            search_body,
            limit=limit,
            max_items=max_items,
            timeout=timeout,
            method=method,
        )
        # Extract only the 'id' field from features that have it
        return [feat["id"] for feat in features if "id" in feat]

    def get_all_collections(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve all collections from the catalog with pagination support.

        Fetches the complete list of available collections from the /collections
        endpoint, following pagination links automatically.

        Parameters
        ----------
        limit : int, optional
            Results per page for pagination. Defaults to 100.

        Returns
        -------
        list[dict[str, Any]]
            List of collection metadata dictionaries.

        """
        session = self._get_authenticated_session()
        base_url = self.base_url.rstrip("/")
        # Start with the initial collections URL
        collections_url: str | None = f"{base_url}/collections"
        params: dict[str, Any] = {"limit": limit}
        all_collections: list[dict[str, Any]] = []

        # Iterate through paginated collections
        while collections_url:
            # Fetch current page of collections
            resp, session = self._make_request(
                session,
                "GET",
                collections_url,
                params=params,
                timeout=CATALOG_REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()

            # Accumulate collections from this page
            collections = data.get("collections", [])
            all_collections.extend(collections)

            # Look for rel="next" link for pagination
            links = data.get("links") or []
            next_href = self._get_next_href(links)
            if next_href:
                collections_url = next_href
                params = {}  # Clear params for next URL
            else:
                break

        return all_collections

    def get_collection_item_count(self, collection_id: str) -> int:
        """Get total item count for a specific collection using numberMatched.

        Performs a minimal search request (limit=1) for the collection to retrieve
        the 'numberMatched' count without fetching actual items.

        Parameters
        ----------
        collection_id : str
            The ID of the collection to count items in.

        Returns
        -------
        int
            Total number of items in the collection, or 0 if not available.

        """
        session = self._get_authenticated_session()
        base_url = self.base_url.rstrip("/")
        url = f"{base_url}/search"

        # Minimal search request to get total count without fetching items
        body = {"collections": [collection_id], "limit": 1}
        resp, session = self._make_request(
            session,
            "POST",
            url,
            json=body,
            timeout=CATALOG_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()

        value = data.get("numberMatched", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def print_formatted_item_count(self, collection_counts) -> None:
        """Print a formatted table of item counts per collection.

        Displays a nicely aligned table showing the item count for each collection,
        with a total row at the bottom. Column widths are calculated based on content.

        Parameters
        ----------
        collection_counts : dict[str, int]
            Dictionary mapping collection IDs to item counts.

        """
        total_items = 0
        print("Item counts per collection:")

        # Calculate column widths based on the longest values
        widths1 = []
        widths2 = []
        for collection_id, count in collection_counts.items():
            widths1.append(len(collection_id))
            widths2.append(len(str(count)))

        width1, width2 = max(widths1), max(widths2)

        # Print each collection with right-aligned values
        for collection_id, count in collection_counts.items():
            print(f"{collection_id:>{width1}}: {count:>{width2}}")
            total_items += count

        # Print separator and total row
        print(f"{width1 * '-':>{width1}}: {width2 * '-':>{width2}}")
        print(f"{'Total items':>{width1}}: {total_items:>{width2}}")

    def get_all_collection_item_counts(self) -> dict[str, int]:
        """Get total item count for all collections.

        Retrieves the item count for every collection in the catalog. Handles
        errors gracefully by assigning a count of 0 to collections that fail.
        If verbose mode is enabled, prints a formatted table of results.

        Returns
        -------
        dict[str, int]
            Dictionary mapping collection IDs to item counts.

        """
        # Get all available collections
        collections = self.get_all_collections()
        if self.verbose:
            print(f"Found {len(collections)} collections.")

        # Count items for each collection
        collection_counts = {}
        for col in collections:
            col_id = col.get("id")
            if col_id:
                try:
                    count = self.get_collection_item_count(col_id)
                    collection_counts[col_id] = count
                except Exception as e:
                    # Handle errors gracefully
                    print(f"Warning: Failed to get item count for collection {col_id}: {e}")
                    collection_counts[col_id] = 0

        # Print formatted results if verbose
        if self.verbose:
            if collection_counts:
                self.print_formatted_item_count(collection_counts)
            else:
                print("No collections found or failed to retrieve counts.")

        return collection_counts

    def iter_collection_items(self, session: requests.Session, collection_id: str, *, limit: int = 100):
        """Yield all items of a collection via /collections/{cid}/items, following pagination via 'next' links.

        Parameters
        ----------
        session : requests.Session
            Authenticated session for making requests.
        collection_id : str
            The ID of the collection to iterate through.
        limit : int, optional
            Items per page for pagination. Defaults to 100.

        Yields
        ------
        dict[str, Any]
            Item feature dictionaries from the collection.

        """
        items_url: str | None = f"{self.base_url}/collections/{collection_id}/items"
        params = {"limit": limit}

        # Iterate through paginated collection items
        while items_url:
            resp, session = self._make_request(
                session,
                "GET",
                items_url,
                params=params,
                timeout=CATALOG_REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()

            # Yield each item on the current page
            features = data.get("features", [])
            for item in features:
                yield item

            # Look for rel="next" link for pagination
            links = data.get("links") or []
            next_href = self._get_next_href(links)
            if next_href:
                items_url = next_href
                # The 'next' href is usually fully formed, no need for params
                params = {}
            else:
                items_url = None

    def iter_item_ids(
        self,
        collection_id: str,
    ):
        """Yield item IDs for a given collection in a given catalog.

        Generator function that yields only the 'id' field from items in a collection.
        Logs warnings for items that don't have an 'id' field but continues iteration.

        Parameters
        ----------
        collection_id : str
            The ID of the collection to iterate through.

        Yields
        ------
        str
            Item IDs from the collection.

        """
        count = 0
        session = self._get_authenticated_session()
        # Iterate through items and yield only those with valid IDs
        for item_dict in self.iter_collection_items(session, collection_id):
            item_id = item_dict.get("id")
            count += 1
            if item_id is None:
                print(f"  - Item {count}: WARNING: item without 'id', skipping")
                continue
            yield item_id

    def _geodataframe_column_mapping(self) -> dict[str, tuple[str, str | None]]:
        """Return column mapping used for STAC feature to GeoDataFrame conversion."""
        return {
            "collection_id": ("collection", None),
            "item_id": ("id", None),
            "page": ("page", None),  # Added page number for debugging
            "aa": ("aa", None),  # Added aa for debugging
            "created": ("properties", "created"),
            "datetime": ("properties", "datetime"),
            "start_datetime": ("properties", "start_datetime"),
            "end_datetime": ("properties", "end_datetime"),
            "title": ("properties", "title"),
            "gsd": ("properties", "gsd"),
            "platform": ("properties", "platform"),
            "constellation": ("properties", "constellation"),
            "instruments": ("properties", "instruments"),
            "sat_orbit_state": ("properties", _KEY_SAT_ORBIT_STATE),
            "cloud_cover": ("properties", _KEY_EO_CLOUD_COVER),
            "keywords": ("properties", "keywords"),
            "view_azimuth": ("properties", _KEY_VIEW_AZIMUTH),
            "view_incidence_angle": ("properties", _KEY_VIEW_INCIDENCE_ANGLE),
        }

    def _parse_feature_geometry(self, feature: dict[str, Any]):
        """Parse and return geometry for a STAC feature, or None if unavailable/invalid."""
        geom_dict = feature.get("geometry")
        if not geom_dict:
            return None
        try:
            return shape(geom_dict)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to parse geometry: {e}")
            return None

    def _extract_feature_column_value(
        self,
        feature: dict[str, Any],
        path: tuple[str, str | None],
    ) -> Any:
        """Extract and normalize a single output column value from a STAC feature."""
        if path[1] is None:
            value = feature.get(path[0])
        elif path[0] == "properties" and len(path) > 1 and path[1] is not None:
            value = feature.get("properties", {}).get(path[1])
        else:
            value = None

        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return value

    def _extract_feature_row(
        self,
        feature: dict[str, Any],
        column_mapping: dict[str, tuple[str, str | None]],
    ) -> dict[str, Any]:
        """Extract one tabular row from a STAC feature according to mapping."""
        return {col_name: self._extract_feature_column_value(feature, path) for col_name, path in column_mapping.items()}

    def features_to_geodataframe(self, features: list[dict[str, Any]]) -> gpd.GeoDataFrame:
        """Convert STAC features to a GeoPandas GeoDataFrame.

        Transforms a list of STAC feature dictionaries into a GeoDataFrame with
        geometry column and properties extracted into appropriately named columns.
        List properties (instruments, keywords) are converted to comma-separated strings.
        Missing properties are filled with null values.

        Parameters
        ----------
        features : list[dict[str, Any]]
            List of STAC feature dictionaries.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with geometry and the following columns:
            - id: Feature ID
            - created: properties.created
            - datetime: properties.datetime
            - start_datetime: properties.start_datetime
            - end_datetime: properties.end_datetime
            - title: properties.title
            - gsd: properties.gsd
            - platform: properties.platform
            - constellation: properties.constellation
            - instruments: properties.instruments (comma-separated)
            - sat_orbit_state: properties.sat:orbit_state
            - keywords: properties.keywords (comma-separated)
            - view_azimuth: properties.view:azimuth
            - view_incidence_angle: properties.view:incidence_angle

        """
        # Get column mapping and extract rows of data
        column_mapping = self._geodataframe_column_mapping()

        # Extract tabular data for each feature according to the column mapping
        rows = [self._extract_feature_row(feature, column_mapping) for feature in features]

        # Create a dictionary of columns with lists of values for each column
        data = {col_name: [row[col_name] for row in rows] for col_name in column_mapping}

        # Extract geometries for each feature, handling missing or invalid geometries gracefully
        geometries = [self._parse_feature_geometry(feature) for feature in features]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=EPSG_4326)

        if self.verbose:
            print(f"Created GeoDataFrame with {len(gdf)} features and {len(gdf.columns)} columns")

        return gdf
