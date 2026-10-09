"""Tests for ZIP extraction utilities in shared.tabular."""

from __future__ import annotations

import zipfile

import pytest

from shared.tabular import unzip_gis_file
from tests.utils import expect_equal, expect_true


def test_unzip_gis_file_returns_first_matching_extension(tmp_path):
    """When multiple matches exist, the first path is returned for filtered extraction."""
    archive = tmp_path / "layers.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.geojson", "{}")
        handle.writestr("b.gpkg", "dummy")

    extracted = unzip_gis_file(str(archive), [".geojson", ".gpkg"], str(tmp_path))
    expect_true(str(extracted).endswith("a.geojson"))


def test_unzip_gis_file_extracts_gdb_folder_as_single_match(tmp_path):
    """A .gdb folder marker should return the directory path, not inner files."""
    archive = tmp_path / "fgdb.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("roads.gdb/", "")
        handle.writestr("roads.gdb/a00000001.gdbtable", "x")

    extracted = unzip_gis_file(str(archive), [".gdb"], str(tmp_path))
    expect_true(str(extracted).endswith("roads.gdb"))


def test_unzip_gis_file_returns_empty_when_not_zip_and_raise_error_false(tmp_path):
    """Non-zip inputs should return [] when raise_error is disabled."""
    invalid = tmp_path / "not_archive.txt"
    invalid.write_text("not a zip", encoding="utf-8")

    expect_equal(unzip_gis_file(str(invalid), [".geojson"], raise_error=False), [])


def test_unzip_gis_file_raises_when_no_matching_extensions(tmp_path):
    """Filtered extraction must fail when no requested extension is present."""
    archive = tmp_path / "single.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("only.txt", "x")

    with pytest.raises(ValueError, match="No files with extensions"):
        unzip_gis_file(str(archive), [".geojson"], str(tmp_path))
