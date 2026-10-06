"""Regression checks for HTTP report viewing and cross-step map links."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

import pytest

from satellites.ml_classification.shared.reports import report_server as server
from satellites.ml_classification.shared.reports.report_html import _to_report_relative_path, write_html_report
from satellites.ml_classification.shared.reports.report_index import write_index_report


def test_standalone_viewer_starts_without_pipeline_imports():
    result = subprocess.run(
        [sys.executable, "-I", server.__file__, "--help"], capture_output=True, text=True, timeout=10, check=True
    )
    assert "--no-browser" in result.stdout


@pytest.fixture(autouse=True)
def stop_report_servers():
    yield
    server._stop_servers()


def test_http_report_serves_embedded_map_and_reuses_project_server(tmp_path):
    check = tmp_path / "01_check"
    plots = check / "plots"
    plots.mkdir(parents=True)
    map_path = plots / "map # ελληνικά.html"
    map_path.write_text("<html>map asset</html>", encoding="utf-8")
    report = check / "report.html"
    write_html_report(report, "Check", "", [{"title": "Map", "embeds": [{"path": map_path, "title": "Map"}]}])
    url = server.report_url(report, root=tmp_path)
    with urlopen(url, timeout=5) as response:
        html = response.read().decode("utf-8")
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "plots/map%20%23%20" in html
    asset_url = urljoin(url, _to_report_relative_path(map_path, check))
    with urlopen(asset_url, timeout=5) as response:
        assert response.read() == b"<html>map asset</html>"
    assert urlsplit(server.report_url(map_path, root=tmp_path)).netloc == urlsplit(url).netloc


def test_http_server_does_not_expose_parent_or_directory_listing(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    report = project / "report.html"
    report.write_text("report", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    url = server.report_url(report)
    assert urlsplit(url).hostname == "127.0.0.1"
    for relative in ("/", "/../outside.txt", "/%2e%2e/outside.txt"):
        with pytest.raises(HTTPError) as exc:
            urlopen(urljoin(url, relative), timeout=5)
        assert exc.value.code == 404


def test_http_server_requires_report_inside_root(tmp_path):
    report = tmp_path / "report.html"
    report.write_text("report", encoding="utf-8")
    root = tmp_path / "other"
    root.mkdir()
    with pytest.raises(ValueError):
        server.report_url(report, root=root)
    with pytest.raises(FileNotFoundError):
        server.report_url(tmp_path / "missing.html")


def test_cross_step_map_link_works_over_http(tmp_path):
    check = tmp_path / "01_check" / "plots"
    check.mkdir(parents=True)
    map_path = check / "map.html"
    map_path.write_text("map", encoding="utf-8")
    dashboard = tmp_path / "final_dashboard"
    dashboard.mkdir()
    report = dashboard / "report.html"
    write_html_report(report, "Dashboard", "", [{"title": "Map", "embeds": [{"path": map_path, "title": "Map"}]}])
    relative = _to_report_relative_path(map_path, dashboard)
    assert relative == "../01_check/plots/map.html"
    assert f"src='{relative}'" in report.read_text(encoding="utf-8")
    url = server.report_url(report, root=tmp_path)
    with urlopen(urljoin(url, relative), timeout=5) as response:
        assert response.read() == b"map"


def test_relative_project_directory_produces_report_relative_links(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _to_report_relative_path(Path("project/01_check/report.html"), Path("project")) == "01_check/report.html"


@pytest.mark.parametrize("auto_open", [True, False])
def test_index_respects_auto_open_and_uses_http(monkeypatch, tmp_path, auto_open):
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    config = SimpleNamespace(
        project_dir=tmp_path,
        check_dir=tmp_path / "01_check",
        prepare_dir=tmp_path / "02_prepare",
        train_dir=tmp_path / "03_train",
        evaluate_dir=tmp_path / "04_evaluate",
        predict_dir=tmp_path / "05_predict",
        final_dashboard_dir=tmp_path / "final_dashboard",
        open_html_report=auto_open,
    )
    write_index_report(config)
    config.final_dashboard_dir.mkdir()
    (config.final_dashboard_dir / "report.html").write_text("Dashboard", encoding="utf-8")
    write_index_report(config)
    assert "Final Dashboard" in (tmp_path / "report_index.html").read_text(encoding="utf-8")
    assert len(opened) == int(auto_open)
    if auto_open:
        assert opened[0].startswith("http://127.0.0.1:")
        with urlopen(opened[0], timeout=5) as response:
            assert b"ML Classification Pipeline Report Index" in response.read()


def test_open_report_once_is_per_path_and_allows_manual_reopening(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(server.webbrowser, "open", opened.append)
    first = tmp_path / "report_index.html"
    second = tmp_path / "other.html"
    first.write_text("First", encoding="utf-8")
    second.write_text("Second", encoding="utf-8")
    first_url = server.open_report(first, once=True)
    assert server.open_report(first, once=True) == first_url
    second_url = server.open_report(second, once=True)
    server.open_report(first)
    assert opened == [first_url, second_url, first_url]


def test_open_report_once_retries_failed_browser_launch(monkeypatch, tmp_path):
    report = tmp_path / "report_index.html"
    report.write_text("Report", encoding="utf-8")
    attempts = []

    def open_browser(url):
        attempts.append(url)
        return len(attempts) > 1

    monkeypatch.setattr(server.webbrowser, "open", open_browser)
    for _ in range(3):
        server.open_report(report, once=True)
    assert len(attempts) == 2


def test_report_still_saved_when_viewer_cannot_start(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("Port unavailable")

    monkeypatch.setattr("satellites.ml_classification.shared.reports.report_html.open_report", fail)
    report = tmp_path / "report.html"
    write_html_report(report, "Saved", "", [{"title": "Summary", "open_html_report": True}])
    assert "Saved" in report.read_text(encoding="utf-8")
