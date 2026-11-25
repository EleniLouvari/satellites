"""Authenticated, range-readable access to HUB raster assets.

Verified access contract (2026-11-21): an asset ``href`` points to the HUB
data-access gateway. A bearer-authenticated GET returns HTTP 302 to a signed
object-storage URL (``X-Amz-Expires=600``). That URL serves HTTP range requests
without the bearer header. GDAL therefore opens the signed URL through
``/vsicurl/``; forwarding the bearer token to storage is not part of the contract.

Signed URLs expire, so they are cached for less than their lifetime and are
never persisted, logged or hashed.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar
from urllib.parse import parse_qs, urlparse

import rasterio
import requests

from data_preparation.sources.hub import catalog_library

T = TypeVar("T")
# Seconds subtracted from the advertised signed-URL lifetime before re-resolving.
SIGNED_URL_SAFETY_SECONDS = 120
# Lifetime assumed when a signed URL does not advertise X-Amz-Expires.
DEFAULT_SIGNED_URL_SECONDS = 600
# GDAL settings for remote COG windows. Do NOT set CPL_VSIL_CURL_ALLOWED_EXTENSIONS:
# signed URLs end in a query string, so any extension filter rejects every open.
GDAL_REMOTE_COG_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}
_QUERY_PATTERN = re.compile(r"\?\S*")
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def redact_url(text: object) -> str:
    """Remove query strings, which may hold signatures, from a URL or error message."""
    return _QUERY_PATTERN.sub("?<redacted>", str(text))


def signed_url_lifetime(url: str) -> int:
    """Return the advertised lifetime in seconds of a signed storage URL."""
    query = {key.lower(): values for key, values in parse_qs(urlparse(url).query).items()}
    try:
        return int(query["x-amz-expires"][0])
    except (KeyError, IndexError, ValueError):
        return DEFAULT_SIGNED_URL_SECONDS


class HubAssetError(RuntimeError):
    """Raised when a HUB asset cannot be resolved to a readable storage URL."""


class HubAssetResolver:
    """Resolve gateway asset hrefs to signed storage URLs with token refresh and caching.

    Parameters
    ----------
    client_id:
        Catalog client role used to obtain bearer tokens.
    token_provider:
        Callable returning a fresh bearer token; defaults to ``catalog_library.get_token``.
    timeout:
        Request timeout for one gateway exchange, in seconds.
    clock:
        Monotonic clock; injectable for tests.

    """

    def __init__(
        self,
        client_id: str,
        *,
        token_provider: Callable[[str], str] | None = None,
        timeout: float = catalog_library.CATALOG_REQUEST_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Validate the client role and prepare an empty signed-URL cache."""
        if client_id not in catalog_library.USERS:
            raise ValueError(f"Error: Unsupported HUB client_id {client_id!r}; expected one of {sorted(catalog_library.USERS)}.")
        self.client_id = client_id
        self._token_provider = token_provider or catalog_library.get_token
        self._timeout = timeout
        self._clock = clock
        self._token: str | None = None
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._session = requests.Session()

    def _bearer(self, refresh: bool = False) -> str:
        """Return the cached token, fetching a new one on first use or refresh."""
        with self._lock:
            if refresh or self._token is None:
                self._token = self._token_provider(self.client_id)
            return self._token

    @staticmethod
    def _is_login_redirect(response: requests.Response) -> bool:
        """Return whether the gateway sent the client to the identity provider instead of storage."""
        location = response.headers.get("Location") or ""
        if response.status_code not in REDIRECT_STATUSES or not location:
            return False
        parsed = urlparse(location)
        keycloak_host = urlparse(os.getenv("KEYCLOAK_URL") or "").netloc
        return "/protocol/openid-connect/" in parsed.path or bool(keycloak_host and parsed.netloc == keycloak_host)

    def _request_redirect(self, href: str) -> requests.Response:
        """Request the gateway without following redirects; refresh the token once on 401 or login redirect."""
        for refresh in (False, True):
            response = self._session.get(
                href,
                headers={"Authorization": f"Bearer {self._bearer(refresh=refresh)}"},
                allow_redirects=False,
                timeout=self._timeout,
                stream=True,
            )
            response.close()
            if response.status_code != 401 and not self._is_login_redirect(response):
                return response
        return response

    def resolve(self, href: str, *, force: bool = False) -> str:
        """Return a signed storage URL for ``href``, reusing a cached one until near expiry."""
        now = self._clock()
        with self._lock:
            cached = self._cache.get(href)
        if cached and not force and cached[1] > now:
            return cached[0]
        try:
            response = self._request_redirect(href)
        except requests.RequestException as exc:
            # ``from None``: the chained exception text can contain signed URLs.
            raise HubAssetError(f"Error: HUB asset gateway request failed for {redact_url(href)}: {redact_url(exc)}") from None
        location = response.headers.get("Location")
        if self._is_login_redirect(response):
            raise HubAssetError(
                f"Error: HUB asset gateway redirected to the identity provider for {redact_url(href)}; check the client credentials."
            )
        if response.status_code not in REDIRECT_STATUSES or not location:
            raise HubAssetError(
                f"Error: HUB asset gateway returned HTTP {response.status_code} without a storage redirect for {redact_url(href)}."
            )
        if urlparse(location).scheme != "https":
            raise HubAssetError(f"Error: HUB asset gateway redirected to a non-HTTPS location for {redact_url(href)}.")
        lifetime = max(0, signed_url_lifetime(location) - SIGNED_URL_SAFETY_SECONDS)
        with self._lock:
            self._cache[href] = (location, now + lifetime)
        return location

    def invalidate(self, href: str) -> None:
        """Forget the signed URL for ``href`` so the next access re-resolves it."""
        with self._lock:
            self._cache.pop(href, None)

    @contextmanager
    def open(self, href: str) -> Iterator[rasterio.io.DatasetReader]:
        """Open an asset as a rasterio dataset, re-resolving once if the signed URL was rejected."""
        last_error: Exception | None = None
        for force in (False, True):
            url = self.resolve(href, force=force)
            try:
                with rasterio.Env(**gdal_remote_options()):
                    dataset = rasterio.open(f"/vsicurl/{url}")
            except rasterio.errors.RasterioIOError as exc:
                # Expired or revoked signatures surface as open failures; retry with a fresh URL.
                last_error = exc
                self.invalidate(href)
                continue
            try:
                with rasterio.Env(**gdal_remote_options()):
                    yield dataset
            finally:
                dataset.close()
            return
        # ``from None``: the GDAL error names the signed URL; only the redacted text is kept.
        raise HubAssetError(f"Error: Could not open HUB asset {redact_url(href)}: {redact_url(last_error)}") from None


def read_asset(resolver, href: str, operation: Callable[[rasterio.io.DatasetReader], T]) -> T:  # noqa: UP047 - pydocstyle cannot parse PEP 695
    """Run ``operation(dataset)`` on an opened asset, retrying open *and* read once with a fresh URL.

    A signed URL can expire between opening and the range requests of a read,
    so failures inside ``operation`` are retried as a whole. Errors are raised as
    :class:`HubAssetError` with redacted text and no chained cause, so full
    tracebacks never contain signed query parameters. ``resolver`` needs an
    ``open(href)`` context manager; ``invalidate(href)`` is used when present.
    """
    message = ""
    for _attempt in range(2):
        try:
            with resolver.open(href) as dataset:
                return operation(dataset)
        except rasterio.errors.RasterioError as exc:
            message = redact_url(exc)
            invalidate = getattr(resolver, "invalidate", None)
            if invalidate is not None:
                invalidate(href)
    raise HubAssetError(f"Error: Could not read HUB asset {redact_url(href)}: {message}") from None


def gdal_remote_options() -> dict[str, str]:
    """Return GDAL options for remote COG reads; environment values take precedence."""
    return {key: os.environ.get(key, value) for key, value in GDAL_REMOTE_COG_OPTIONS.items()}
