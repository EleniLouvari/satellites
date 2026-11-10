"""Distinguish printed credentials from ordinary scientific log values."""

import logging

import pytest

from parcel_app.artifacts import sanitize_text


def test_short_credentials_do_not_change_processing_values():
    """A credential equal to 1 must not erase bands, numbers or time intervals."""
    message = (
        "Source=openEO, optical bands=['B04', 'B08', 'B01', 'B11', 'B12'], "
        "iqr_multiplier=1.5, temporal_period=1M, batch_workers=1, "
        "https://openeo.example.org/v1, metric=hunter2, token_count=1, password_client_id=1"
    )
    assert sanitize_text(message, ["1", "hunter2"]) == message


@pytest.mark.parametrize(
    "message",
    [
        "password=1",
        "POSTGRES_PASSWORD=1",
        "openeo_password: 1",
        "username=alice",
        "openeo_username=alice@example.org",
        '"password": "two words"',
        "{'password': 'two words'}",
        'password="escaped\\"quote"',
        "password='escaped\\'quote'",
        "Authorization: Bearer token-value",
        "authorization=Basic encoded-value",
        "{'Authorization': 'Bearer token-value'}",
        "access_token=token-value",
        "refresh_token: token-value",
        "client_secret=secret-value",
        "api_key=secret-value",
        "SATELLITES_CREDENTIAL_KEY=secret-value",
    ],
)
def test_credential_fields_are_redacted_completely(message):
    """Mask quoted values and complete authorization headers, including spaces."""
    result = sanitize_text(message, ["1"])
    assert "[redacted]" in result
    for value in (
        "1",
        "alice",
        "two words",
        "escaped",
        "quote",
        "Bearer",
        "Basic",
        "token-value",
        "encoded-value",
        "secret-value",
    ):
        assert value not in result


def test_known_unquoted_credential_with_spaces_is_redacted():
    """Use known values only after a credential label, never as global matches."""
    assert sanitize_text("password=two words, metric=two words", ["two words"]) == ("password=[redacted], metric=two words")


@pytest.mark.parametrize(
    "url",
    [
        "https://alice:password@example.org/path",
        "https://example.org/path?access_token=token-value&limit=1",
        "https://example.org/path?api_key=secret-value",
        "https://example.org/path?X-Amz-Signature=signed-value",
        "postgresql://alice:password@localhost/database",
        "https://example.org/#access_token=token-value",
    ],
)
def test_credential_urls_are_redacted(url):
    """Hide authentication-bearing URLs while leaving ordinary URLs readable."""
    assert sanitize_text(f"Request {url}") == "Request [redacted URL]"


def test_repeated_redaction_preserves_markers_and_safe_text():
    """Worker logs and downloadable copies can both pass through sanitization."""
    message = "password=1; bands=['B01', 'B11']; https://example.org/v1"
    once = sanitize_text(message, ["1", "redacted"])
    assert sanitize_text(once, ["1", "redacted"]) == once


def test_event_logging_uses_contextual_redaction(monkeypatch):
    """The record factory must not globally replace a short database password."""
    from parcel_app.progress import event_logging

    monkeypatch.setenv("POSTGRES_PASSWORD", "1")
    logger = logging.getLogger("redaction-regression")
    with event_logging({}, [], private_paths=["/private/run"]):
        record = logger.makeRecord(logger.name, logging.INFO, "test", 1, "B11: %s; password=1 /private/run", (1.5,), None)
    assert record.getMessage() == "B11: 1.5; password=[redacted] [redacted]"


def test_downloaded_log_keeps_scientific_values(tmp_path, monkeypatch):
    """Published diagnostics retain numbers while masking credentials and private paths."""
    from parcel_app import artifacts

    (tmp_path / "satellite_acquisition.log").write_text(f"B11 1.5 1M password=1 {tmp_path}\n", encoding="utf-8")
    registered = []
    monkeypatch.setattr(artifacts, "register_file", lambda run, kind, path, **kwargs: registered.append((kind, path)))
    artifacts.register_diagnostics({"output_dir": str(tmp_path), "config": {}}, ["1"])
    log = next(path.read_text(encoding="utf-8") for kind, path in registered if kind == "log")
    assert log == "B11 1.5 1M password=[redacted] [redacted]\n"
