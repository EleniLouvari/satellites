"""Serve local reports over loopback HTTP so online maps send a Referer."""

from __future__ import annotations

import argparse
import atexit
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import quote


class _ReportHandler(SimpleHTTPRequestHandler):
    """Serve files within the report root without directory listings."""

    def send_head(self):
        target = Path(self.translate_path(self.path)).resolve()
        if not target.is_relative_to(Path(self.directory).resolve()) or target.is_dir():
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        # Send only the local origin to external tile providers, not file paths.
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()

    def log_message(self, format, *args):
        """Keep background asset requests out of notebook output."""


_servers: dict[Path, ThreadingHTTPServer] = {}
_server_lock = Lock()
_opened_reports: set[Path] = set()
_browser_lock = Lock()


def _report_location(report_path: str | Path, root: str | Path | None) -> tuple[Path, str]:
    report = Path(report_path).resolve(strict=True)
    if not report.is_file():
        raise ValueError(f"Report is not a file: {report}")
    directory = Path(root).resolve(strict=True) if root is not None else report.parent
    relative = report.relative_to(directory)
    return directory, quote(relative.as_posix(), safe="/")


def report_url(report_path: str | Path, *, root: str | Path | None = None) -> str:
    """Start/reuse a local server; it lives until this Python process exits.

    Set root to the pipeline project directory to support links between steps.
    """
    directory, relative = _report_location(report_path, root)
    with _server_lock:
        server = _servers.get(directory)
        if server is None:
            handler = partial(_ReportHandler, directory=str(directory))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            Thread(target=server.serve_forever, name="ml-report-server", daemon=True).start()
            _servers[directory] = server
    return f"http://127.0.0.1:{server.server_port}/{relative}"


def open_report(report_path: str | Path, *, root: str | Path | None = None, once: bool = False) -> str:
    """Open a report over HTTP, optionally only once per path in this process."""
    url = report_url(report_path, root=root)
    report = Path(report_path).resolve()
    with _browser_lock:
        if not once or report not in _opened_reports:
            if webbrowser.open(url) is not False:
                _opened_reports.add(report)
    return url


@atexit.register
def _stop_servers() -> None:
    """Release the background servers when the notebook kernel exits."""
    with _server_lock:
        for server in _servers.values():
            server.shutdown()
            server.server_close()
        _servers.clear()
    with _browser_lock:
        _opened_reports.clear()


def main() -> None:
    """View an existing report without importing or rerunning the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--root", type=Path, help="Pipeline project directory for links between steps")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    directory, relative = _report_location(args.report, args.root)
    handler = partial(_ReportHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/{relative}"
        print(f"Report: {url}\nKeep this process running while viewing the report. Ctrl+C stops it.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
