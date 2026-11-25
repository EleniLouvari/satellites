"""Offline tests for HUB asset gateway resolution, signed-URL caching and dataset opening."""

import io
import traceback
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import rasterio
import requests

from data_preparation.sources.hub import assets as hub_assets
from data_preparation.sources.hub import catalog_library

SIGNED = "https://s3.test/bucket/B04.tif?X-Amz-Expires=600&X-Amz-Signature=secret-sig"


def response(status, location=None):
    """Build a gateway response with an optional redirect location."""
    result = requests.Response()
    result.status_code = status
    result.raw = io.BytesIO(b"")
    if location:
        result.headers["Location"] = location
    return result


class Clock:
    """Controllable monotonic clock."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def resolver(monkeypatch):
    """Resolver with a fake token provider, fake clock and mocked HTTP session."""
    monkeypatch.setattr(catalog_library, "USERS", {"internal": "secret"})
    tokens = Mock(side_effect=["token-1", "token-2", "token-3"])
    clock = Clock()
    result = hub_assets.HubAssetResolver("internal", token_provider=tokens, clock=clock)
    result._session = Mock()
    return result, tokens, clock


def test_resolve_follows_gateway_redirect_without_forwarding_and_caches(resolver):
    """One gateway call yields a signed URL reused until shortly before expiry."""
    instance, tokens, clock = resolver
    instance._session.get.return_value = response(302, SIGNED)
    assert instance.resolve("https://gw.test/B04") == SIGNED
    assert instance.resolve("https://gw.test/B04") == SIGNED
    assert instance._session.get.call_count == 1
    kwargs = instance._session.get.call_args.kwargs
    assert kwargs["allow_redirects"] is False and kwargs["headers"]["Authorization"] == "Bearer token-1"
    clock.now += 600 - hub_assets.SIGNED_URL_SAFETY_SECONDS + 1
    instance.resolve("https://gw.test/B04")
    assert instance._session.get.call_count == 2
    assert tokens.call_count == 1


def test_resolve_refreshes_token_once_on_401(resolver):
    """An expired bearer token is replaced and the request repeated once."""
    instance, tokens, _clock = resolver
    instance._session.get.side_effect = [response(401), response(302, SIGNED)]
    assert instance.resolve("https://gw.test/B04") == SIGNED
    assert tokens.call_count == 2
    assert instance._session.get.call_args.kwargs["headers"]["Authorization"] == "Bearer token-2"


@pytest.mark.parametrize("status", [200, 403, 404, 401])
def test_resolve_rejects_non_redirect_responses_without_leaking_queries(resolver, status):
    """Gateway failures raise a clear error and never print query strings."""
    instance, _tokens, _clock = resolver
    instance._session.get.return_value = response(status)
    with pytest.raises(hub_assets.HubAssetError, match=f"HTTP {status}") as error:
        instance.resolve("https://gw.test/B04?token=abc")
    assert "abc" not in str(error.value)


def test_resolve_wraps_transport_errors(resolver):
    """Network failures become HubAssetError with redacted text."""
    instance, _tokens, _clock = resolver
    instance._session.get.side_effect = requests.ConnectionError(f"failed for {SIGNED}")
    with pytest.raises(hub_assets.HubAssetError) as error:
        instance.resolve("https://gw.test/B04")
    assert "secret-sig" not in str(error.value)


def test_resolver_rejects_unknown_client_id(monkeypatch):
    """Only configured catalog roles can request tokens."""
    monkeypatch.setattr(catalog_library, "USERS", {"internal": "secret"})
    with pytest.raises(ValueError, match="Unsupported HUB client_id"):
        hub_assets.HubAssetResolver("agriculture")


def test_signed_url_lifetime_and_redaction():
    """Lifetime comes from X-Amz-Expires; redaction removes every query string."""
    assert hub_assets.signed_url_lifetime(SIGNED) == 600
    assert hub_assets.signed_url_lifetime("https://s3.test/file.tif") == hub_assets.DEFAULT_SIGNED_URL_SECONDS
    assert hub_assets.redact_url(f"error opening {SIGNED} now") == "error opening https://s3.test/bucket/B04.tif?<redacted> now"


def test_gdal_options_never_filter_extensions(monkeypatch):
    """An extension filter breaks signed URLs; environment overrides still win."""
    monkeypatch.setenv("GDAL_HTTP_MAX_RETRY", "9")
    options = hub_assets.gdal_remote_options()
    assert "CPL_VSIL_CURL_ALLOWED_EXTENSIONS" not in options
    assert options["GDAL_HTTP_MAX_RETRY"] == "9"


def test_open_reresolves_once_when_signed_url_is_rejected(resolver, monkeypatch, tmp_path):
    """A rejected signed URL is invalidated and the asset re-resolved before failing."""
    instance, _tokens, _clock = resolver
    second = "https://s3.test/bucket/B04.tif?X-Amz-Expires=600&X-Amz-Signature=fresh"
    instance._session.get.side_effect = [response(302, SIGNED), response(302, second)]
    dataset = Mock()
    opener = Mock(side_effect=[rasterio.errors.RasterioIOError("expired"), dataset])
    monkeypatch.setattr(hub_assets.rasterio, "open", opener)
    with instance.open("https://gw.test/B04") as opened:
        assert opened is dataset
    assert opener.call_args_list[-1].args[0] == f"/vsicurl/{second}"
    dataset.close.assert_called_once()


def test_open_raises_after_second_rejection(resolver, monkeypatch):
    """Two consecutive open failures raise HubAssetError without the signature."""
    instance, _tokens, _clock = resolver
    instance._session.get.side_effect = [response(302, SIGNED), response(302, SIGNED)]
    monkeypatch.setattr(hub_assets.rasterio, "open", Mock(side_effect=rasterio.errors.RasterioIOError(f"bad {SIGNED}")))
    with pytest.raises(hub_assets.HubAssetError) as error, instance.open("https://gw.test/B04"):
        pass
    assert "secret-sig" not in str(error.value)


def test_login_redirect_refreshes_token_and_then_fails_clearly(resolver, monkeypatch):
    """An expired token can produce a 302 to the identity provider; refresh once, never open login HTML."""
    monkeypatch.setenv("KEYCLOAK_URL", "https://auth.test/realms/hub")
    instance, tokens, _clock = resolver
    login = "https://auth.test/realms/hub/protocol/openid-connect/auth?client_id=x"
    instance._session.get.side_effect = [response(302, login), response(302, SIGNED)]
    assert instance.resolve("https://gw.test/B04") == SIGNED
    assert tokens.call_count == 2
    instance._session.get.side_effect = [response(302, login), response(302, login)]
    with pytest.raises(hub_assets.HubAssetError, match="identity provider"):
        instance.resolve("https://gw.test/B05")


def test_non_https_storage_redirect_is_rejected(resolver):
    """Signed URLs must stay on HTTPS."""
    instance, _tokens, _clock = resolver
    instance._session.get.return_value = response(302, "http://s3.test/B04.tif?X-Amz-Signature=secret-sig")
    with pytest.raises(hub_assets.HubAssetError, match="non-HTTPS"):
        instance.resolve("https://gw.test/B04")


def test_formatted_tracebacks_never_contain_signed_queries(resolver, monkeypatch):
    """Chained causes are suppressed, so full tracebacks stay free of signatures."""
    instance, _tokens, _clock = resolver
    instance._session.get.side_effect = [response(302, SIGNED), response(302, SIGNED)]
    monkeypatch.setattr(hub_assets.rasterio, "open", Mock(side_effect=rasterio.errors.RasterioIOError(f"bad {SIGNED}")))
    with pytest.raises(hub_assets.HubAssetError) as opened, instance.open("https://gw.test/B04"):
        pass
    instance._session.get.side_effect = requests.ConnectionError(f"failed for {SIGNED}")
    with pytest.raises(hub_assets.HubAssetError) as requested:
        instance.resolve("https://gw.test/B06")
    for error in (opened.value, requested.value):
        assert "secret-sig" not in "".join(traceback.format_exception(error))


def test_read_asset_retries_failures_inside_the_read_with_a_fresh_url(resolver, monkeypatch):
    """A signed URL that expires during range reads is re-resolved and the whole read repeated."""
    instance, _tokens, _clock = resolver
    second = "https://s3.test/bucket/B04.tif?X-Amz-Expires=600&X-Amz-Signature=fresh"
    instance._session.get.side_effect = [response(302, SIGNED), response(302, second)]
    opened = []
    monkeypatch.setattr(hub_assets.rasterio, "open", lambda path: opened.append(path) or Mock())
    calls = Mock(side_effect=[rasterio.errors.RasterioIOError(f"range failed {SIGNED}"), "pixels"])
    assert hub_assets.read_asset(instance, "https://gw.test/B04", calls) == "pixels"
    assert opened[-1] == f"/vsicurl/{second}"


def test_read_asset_raises_redacted_error_after_second_read_failure(resolver, monkeypatch):
    """Persistent read failures raise HubAssetError whose formatted traceback has no signature."""
    instance, _tokens, _clock = resolver
    instance._session.get.side_effect = [response(302, SIGNED), response(302, SIGNED)]
    monkeypatch.setattr(hub_assets.rasterio, "open", lambda _path: Mock())
    failing = Mock(side_effect=rasterio.errors.RasterioIOError(f"range failed {SIGNED}"))
    with pytest.raises(hub_assets.HubAssetError, match="Could not read") as error:
        hub_assets.read_asset(instance, "https://gw.test/B04", failing)
    assert failing.call_count == 2
    assert "secret-sig" not in "".join(traceback.format_exception(error.value))


def test_read_asset_accepts_resolvers_without_invalidate():
    """Test doubles that only provide open() still work."""

    class OpenOnly:
        @contextmanager
        def open(self, _href):
            yield "dataset"

    assert hub_assets.read_asset(OpenOnly(), "x", lambda dataset: dataset.upper()) == "DATASET"
