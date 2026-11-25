"""Completed remote assets survive interrupted transfers without resubmission."""

import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import requests
from openeo.rest import OpenEoApiError, OpenEoApiPlainError
from openeo.rest._connection import RestApiConnection

from data_preparation.sources.openeo import downloads


class Transfer:
    """Supply streamed response chunks, including a mid-transfer exception."""

    def __init__(self, chunks, length=None):
        self.chunks = chunks
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def raise_for_status(self):
        """Represent a successful HTTP status; failures are supplied by the connection."""

    def iter_content(self, chunk_size):
        """Yield received bytes or raise the scripted stream failure."""
        assert chunk_size > 0
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def _job(events):
    """Build the SDK boundary while retaining real transfer and file operations."""
    calls = []
    responses = iter(events)

    def get(path, **kwargs):
        calls.append((path, kwargs))
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    connection = SimpleNamespace(get=get, default_timeout=1200, session=requests.Session())
    asset = SimpleNamespace(href="https://example.test/asset.nc?secret=hidden")
    job = SimpleNamespace(connection=connection, get_results=lambda: SimpleNamespace(get_asset=lambda: asset))
    return job, calls


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Keep retry tests fast without replacing any transfer behavior."""
    monkeypatch.setattr(downloads.time, "sleep", lambda _delay: None)


@pytest.mark.parametrize(
    "failure", [requests.exceptions.ChunkedEncodingError("broken"), requests.exceptions.ReadTimeout("idle")]
)
def test_interrupted_download_restarts_same_asset_and_commits(tmp_path, caplog, failure):
    """A retry replaces partial bytes rather than appending a corrupt prefix."""
    first = Transfer([b"partial", failure])
    second = Transfer([b"complete"], length=8)
    job, calls = _job([first, second])
    target, partial = tmp_path / "cube.nc", tmp_path / "cube.partial.nc"
    partial.write_bytes(b"stale")
    with caplog.at_level(logging.INFO):
        result = downloads.download_job_netcdf(job, partial, target, logging.getLogger("transfer-test"))
    assert result == target
    assert target.read_bytes() == b"complete"
    assert not partial.exists()
    assert first.closed and second.closed
    assert len(calls) == 2
    assert all(call[1]["timeout"] == (30, 120) and call[1]["stream"] for call in calls)
    assert job.connection.default_timeout == 1200
    assert "secret=hidden" not in caplog.text
    assert "8 bytes" in caplog.text


def test_short_successful_response_is_retried(tmp_path):
    """A clean HTTP EOF below Content-Length must not publish a completed cube."""
    job, calls = _job([Transfer([b"short"], 20), Transfer([b"complete"], 8)])
    target, partial = tmp_path / "cube.nc", tmp_path / "cube.partial.nc"
    downloads.download_job_netcdf(job, partial, target, logging.getLogger("transfer-test"))
    assert target.read_bytes() == b"complete"
    assert len(calls) == 2


def test_exhausted_stream_retries_leave_target_untouched(tmp_path):
    """A failed replacement preserves an earlier committed cube and the unfinished stage."""
    job, calls = _job([Transfer([b"bad", requests.exceptions.ChunkedEncodingError("broken")]) for _ in range(3)])
    target, partial = tmp_path / "cube.nc", tmp_path / "cube.partial.nc"
    target.write_bytes(b"previous")
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        downloads.download_job_netcdf(job, partial, target, logging.getLogger("transfer-test"))
    assert len(calls) == 3
    assert target.read_bytes() == b"previous"
    assert partial.read_bytes() == b"bad"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_http_failure_is_not_retried(tmp_path, status):
    """Authentication and unavailable assets need intervention rather than repeated requests."""
    error = OpenEoApiError(http_status_code=status, code="Failure", message="unavailable")
    job, calls = _job([error])
    with pytest.raises(OpenEoApiError):
        downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", tmp_path / "cube.nc", logging.getLogger("test"))
    assert len(calls) == 1
    assert not (tmp_path / "cube.nc").exists()


def test_transient_http_failure_is_retried(tmp_path):
    """A temporary backend failure retries the existing asset."""
    error = OpenEoApiError(http_status_code=503, code="Unavailable", message="busy")
    job, calls = _job([error, Transfer([b"cube"], 4)])
    target = tmp_path / "cube.nc"
    downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", target, logging.getLogger("test"))
    assert target.read_bytes() == b"cube" and len(calls) == 2


def test_transient_outage_backoff_leaves_time_for_recovery(tmp_path, monkeypatch):
    """Fast HTTP failures get finite cooldowns rather than consuming every attempt in six seconds."""
    delays = []
    monkeypatch.setattr(downloads.time, "sleep", delays.append)
    error = OpenEoApiError(http_status_code=503, code="Unavailable", message="busy")
    job, calls = _job([error, error, Transfer([b"cube"], 4)])
    target = tmp_path / "cube.nc"
    downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", target, logging.getLogger("test"))
    assert target.read_bytes() == b"cube" and len(calls) == 3
    assert delays == [30, 60]


def test_local_disk_failure_is_not_retried(tmp_path):
    """A filesystem failure must not trigger repeated remote transfers."""
    job, calls = _job([Transfer([b"cube"], 4)])
    with pytest.raises(OSError):
        downloads.download_job_netcdf(job, tmp_path, tmp_path / "cube.nc", logging.getLogger("test"))
    assert len(calls) <= 1
    assert not (tmp_path / "cube.nc").exists()


def test_empty_successful_response_is_not_published(tmp_path):
    """An empty asset can never be a valid NetCDF, even without a length header."""
    job, calls = _job([Transfer([]) for _ in range(3)])
    target = tmp_path / "cube.nc"
    with pytest.raises(requests.exceptions.ChunkedEncodingError, match="Error: Empty"):
        downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", target, logging.getLogger("test"))
    assert not target.exists()
    assert len(calls) == 3


def test_direct_entry_point_starts_remote_job_once(tmp_path):
    """Recovering the transfer must not restart the completed remote computation."""
    from data_preparation.sources.openeo.cubes import OpenEOCubePipeline

    job, calls = _job([requests.exceptions.ReadTimeout("idle"), Transfer([b"cube"], 4)])
    starts = []
    job.start_and_wait = lambda **_kwargs: starts.append("started")
    extractor = OpenEOCubePipeline()
    extractor.source_logger = logging.getLogger("test")
    target = tmp_path / "cube.nc"
    assert extractor._run_openeo_batch_job(job, tmp_path / "cube.partial.nc", target, 7) == (7, target)
    assert starts == ["started"]
    assert target.read_bytes() == b"cube" and len(calls) == 2


def test_manager_entry_point_recovers_transfer(tmp_path):
    """A finished manager job publishes the same asset after a broken stream."""
    from data_preparation.parcel_stats.job_manager import _NetCDFTileJobManager

    job, calls = _job([requests.exceptions.ChunkedEncodingError("broken"), Transfer([b"cube"], 4)])
    manager = _NetCDFTileJobManager(root_dir=tmp_path / "manager", download_results=False)
    target = tmp_path / "cube.nc"
    manager.on_job_done(job, {"target_path": target})
    assert target.read_bytes() == b"cube" and len(calls) == 2


def test_byte_progress_is_logged_during_transfer(tmp_path, monkeypatch, caplog):
    """Long transfers produce received-byte messages before completion."""
    ticks = iter([0, 31, 62])
    monkeypatch.setattr(downloads.time, "monotonic", lambda: next(ticks))
    job, _ = _job([Transfer([b"ab", b"cd"], 4)])
    with caplog.at_level(logging.INFO):
        downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", tmp_path / "cube.nc", logging.getLogger("test"))
    assert "received 2 bytes" in caplog.text
    assert "received 4 bytes" in caplog.text


def test_real_idle_stream_times_out_and_recovers(tmp_path, monkeypatch):
    """Exercise actual Requests streaming and SDK timeout forwarding against a local server."""
    attempts = []
    release = Event()

    class Handler(BaseHTTPRequestHandler):
        """Stop sending data on the first request; serve the next request normally."""

        def do_GET(self):
            """Return a cube body or intentionally leave the response unfinished."""
            attempts.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            if len(attempts) == 1:
                self.wfile.write(b"cu")
                self.wfile.flush()
                release.wait(2)
            else:
                self.wfile.write(b"cube")

        def log_message(self, *_args):
            """Suppress local HTTP server console output."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}"
    connection = RestApiConnection(root_url=root_url, default_timeout=1200)
    job = SimpleNamespace(
        connection=connection,
        get_results=lambda: SimpleNamespace(get_asset=lambda: SimpleNamespace(href=f"{root_url}/cube.nc")),
    )
    monkeypatch.setattr(downloads, "DOWNLOAD_TIMEOUT", (1, 0.05))
    monkeypatch.setattr(downloads, "DOWNLOAD_CHUNK_BYTES", 1)
    target = tmp_path / "cube.nc"
    try:
        downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", target, logging.getLogger("test"))
        assert target.read_bytes() == b"cube"
        assert attempts == ["/cube.nc", "/cube.nc"]
        assert connection.default_timeout == 1200
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("status, expected_requests", [(503, 3), (401, 1)])
def test_real_sdk_http_failure_obeys_transfer_attempt_budget(tmp_path, status, expected_requests):
    """SDK adapter retries cannot multiply the transfer's bounded attempts."""
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        """Always return the configured HTTP failure with a complete response body."""

        def do_GET(self):
            """Count actual HTTP requests rather than mocked retry calls."""
            attempts.append(self.path)
            body = b'{"code":"Unavailable","message":"failed"}'
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            """Suppress local server console messages."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root_url = f"http://127.0.0.1:{server.server_port}"
    connection = RestApiConnection(root_url=root_url, default_timeout=1200)
    original_adapters = dict(connection.session.adapters)
    job = SimpleNamespace(
        connection=connection,
        get_results=lambda: SimpleNamespace(get_asset=lambda: SimpleNamespace(href=f"{root_url}/cube.nc")),
    )
    target = tmp_path / "cube.nc"
    try:
        with pytest.raises((requests.exceptions.RequestException, OpenEoApiError, OpenEoApiPlainError)):
            downloads.download_job_netcdf(job, tmp_path / "cube.partial.nc", target, logging.getLogger("test"))
        assert len(attempts) == expected_requests
        assert not target.exists()
        assert connection.session.adapters == original_adapters
        assert all(adapter.max_retries.total == 3 for adapter in original_adapters.values())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        connection.session.close()
