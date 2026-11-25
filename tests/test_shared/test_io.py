"""Round-trip and validation tests for the shared read/write helpers."""

import os
import time

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from shared import io
from tests.utils import expect_equal, expect_false, expect_in, expect_isinstance, expect_true


def _plain_frame():
    """Return a small frame covering integer, float, text and datetime columns."""
    return pd.DataFrame(
        {
            "parcel_id": [1, 2],
            "area": [1.5, 2.5],
            "crop": ["wheat", "maize"],
            "observed": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        }
    )


def _geo_frame():
    """Return a projected two-point GeoDataFrame."""
    return gpd.GeoDataFrame({"parcel_id": [1, 2], "crop": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:2100")


def test_file_exists_reports_files_directories_and_missing_paths(tmp_path, capsys):
    """Existing paths return True; missing paths explain why when verbose."""
    file_path = tmp_path / "data.csv"
    file_path.write_text("a\n1\n", encoding="utf-8")

    expect_true(io.file_exists(str(file_path)))
    expect_false(io.file_exists(str(tmp_path / "missing.csv")))
    expect_in("does not exist", capsys.readouterr().out)
    expect_false(io.file_exists(str(tmp_path / "missing.csv"), verbose=False))
    expect_equal(capsys.readouterr().out, "")


def test_read_types_from_csvt_separates_datetime_field(tmp_path):
    """Datetime fields are returned separately and removed from the dtype map."""
    csvt = tmp_path / "data.csvt"
    csvt.write_text("parcel_id,int64\ncrop,object\nobserved,datetime64[ns]\n", encoding="utf-8")

    dates, types = io.read_types_from_csvt(str(csvt))

    expect_equal(dates, ["observed"])
    expect_equal(types, {"parcel_id": "int64", "crop": "object"})


@pytest.mark.parametrize(
    ("dtype", "label"), [("str", "object"), ("string", "object"), ("string[python]", "object"), ("int64", "int64")]
)
def test_typed_csv_dtype_label_normalizes_text_dtypes(dtype, label):
    """Text dtypes share the ``object`` label so headers stay stable across pandas versions."""
    expect_equal(io._typed_csv_dtype_label(dtype), label)


def test_convert_df_col_datetime_to_string_only_touches_datetimes():
    """Datetime columns become strings on a copy; other columns keep their dtype."""
    df = _plain_frame()

    converted = io.convert_df_col_datetime_to_string(df)

    expect_equal(list(converted["observed"]), ["2024-01-01", "2024-02-01"])
    expect_equal(converted["area"].dtype, df["area"].dtype)
    expect_true(pd.api.types.is_datetime64_any_dtype(df["observed"]), "The input must not be mutated")


def test_typed_csv_round_trip_restores_names_types_and_dates(tmp_path):
    """Typed CSV headers carry dtypes that ``read_data`` restores."""
    path = tmp_path / "typed.csv"
    io.write_data(_plain_frame(), str(path))

    header = path.read_text(encoding="utf-8").splitlines()[0]
    expect_in("parcel_id {int64}", header)
    result = io.read_data(str(path))

    expect_equal(list(result.columns), ["parcel_id", "area", "crop", "observed"])
    expect_equal(str(result["parcel_id"].dtype), "int64")
    expect_true(pd.api.types.is_datetime64_any_dtype(result["observed"]), "Datetime column must be parsed")


def test_typed_csv_can_be_read_without_interpreting_brackets(tmp_path):
    """Disabling bracket handling keeps the raw typed header names."""
    path = tmp_path / "typed.csv"
    io.write_data(_plain_frame(), str(path))

    result = io.read_data(str(path), watch_curly_brackets=False)

    expect_in("parcel_id {int64}", list(result.columns))


def test_read_csv_rejects_malformed_typed_header(tmp_path):
    """A typed header with trailing text after the type is reported."""
    path = tmp_path / "bad.csv"
    path.write_text("a {int64}x,b {object}\n1,x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Error: Header format issue"):
        io.read_csv(str(path), watch_curly_brackets=True, encoding=None, delimiter=",")


def test_read_csv_uses_csvt_sidecar_types(tmp_path):
    """A ``.csvt`` sidecar defines column dtypes and datetime parsing."""
    path = tmp_path / "sidecar.csv"
    path.write_text("code,observed\n007,2024-03-01\n", encoding="utf-8")
    (tmp_path / "sidecar.csvt").write_text("code,str\nobserved,datetime\n", encoding="utf-8")

    result = io.read_data(str(path))

    expect_equal(result.loc[0, "code"], "007")
    expect_true(pd.api.types.is_datetime64_any_dtype(result["observed"]), "Sidecar datetime must be parsed")


def test_plain_csv_round_trip_with_custom_delimiter(tmp_path):
    """Plain CSV files are read with the requested delimiter."""
    path = tmp_path / "plain.csv"
    path.write_text("a;b\n1;x\n", encoding="utf-8")

    result = io.read_data(str(path), delimiter=";")

    expect_equal(result.to_dict("list"), {"a": [1], "b": ["x"]})


def test_write_plain_csv_keeps_bare_column_names(tmp_path):
    """``plain_csv=True`` omits dtype annotations from the header."""
    path = tmp_path / "plain.csv"
    io.write_data(_plain_frame(), str(path), plain_csv=True)

    expect_equal(path.read_text(encoding="utf-8").splitlines()[0], "parcel_id,area,crop,observed")


@pytest.mark.parametrize("extension", [".pkl", ".parquet"])
def test_tabular_binary_round_trip(tmp_path, extension):
    """Pickle and Parquet outputs read back to equal frames."""
    path = tmp_path / f"frame{extension}"
    io.write_data(_plain_frame(), str(path))

    pd.testing.assert_frame_equal(io.read_data(str(path)), _plain_frame())


def test_write_data_without_extension_defaults_by_frame_type(tmp_path):
    """Tabular frames default to ``.pkl.xz``; geospatial frames default to ``.gpkg``."""
    io.write_data(_plain_frame(), str(tmp_path), file_name="table")
    io.write_data(_geo_frame(), str(tmp_path), file_name="layer")

    expect_true((tmp_path / "table.pkl.xz").exists(), "Tabular default must be a compressed pickle")
    expect_true((tmp_path / "layer.gpkg").exists(), "Geospatial default must be a GeoPackage")
    pd.testing.assert_frame_equal(io.read_data(str(tmp_path / "table.pkl.xz")), _plain_frame())
    expect_isinstance(io.read_data(str(tmp_path / "layer.gpkg")), gpd.GeoDataFrame)


def test_write_data_replaces_existing_file(tmp_path, capsys):
    """An existing output is removed before the new content is written."""
    path = tmp_path / "frame.pkl"
    io.write_data(pd.DataFrame({"a": [1]}), str(path))
    io.write_data(pd.DataFrame({"a": [2]}), str(path))

    expect_in("Removing existing file", capsys.readouterr().out)
    expect_equal(io.read_data(str(path))["a"].tolist(), [2])


def test_write_data_rejects_unknown_extension(tmp_path):
    """Unsupported writer extensions are reported."""
    with pytest.raises(ValueError, match="Error: Unsupported file extension: .txt"):
        io.write_data(_plain_frame(), str(tmp_path / "frame.txt"))


def test_geoparquet_and_parquet_round_trip_preserve_geometry(tmp_path):
    """GeoDataFrames written to Parquet or GeoParquet read back as GeoDataFrames."""
    for name in ("parcels.parquet", "parcels.geoparquet"):
        io.write_data(_geo_frame(), str(tmp_path / name))
        result = io.read_data(str(tmp_path / name))
        expect_isinstance(result, gpd.GeoDataFrame)
        expect_equal(result.crs.to_epsg(), 2100)


def test_read_parquet_falls_back_to_pandas_without_geo_metadata(tmp_path):
    """Plain Parquet files without geo metadata are read with pandas."""
    path = tmp_path / "plain.parquet"
    _plain_frame().to_parquet(path, index=False)

    result = io.read_parquet(str(path))

    expect_false(isinstance(result, gpd.GeoDataFrame), "Plain Parquet must stay a pandas DataFrame")


def test_read_parquet_reraises_unrelated_value_errors(monkeypatch):
    """Only missing geo metadata triggers the pandas fallback."""

    def broken_reader(_path):
        raise ValueError("corrupt footer")

    monkeypatch.setattr(io.gpd, "read_parquet", broken_reader)
    with pytest.raises(ValueError, match="corrupt footer"):
        io.read_parquet("unused.parquet")


def test_geopackage_layers_are_preserved_when_layer_is_given(tmp_path):
    """Writing a named layer keeps other layers in the same GeoPackage."""
    path = tmp_path / "parcels.gpkg"
    io.write_data(_geo_frame(), str(path), layer="first")
    io.write_data(_geo_frame().iloc[:1], str(path), layer="second")

    expect_equal(len(io.read_data(str(path), layer="first")), 2)
    expect_equal(len(io.read_data(str(tmp_path), file_name={"file": "parcels.gpkg", "layer": "second"})), 1)


def test_shapefile_round_trip_converts_datetimes_to_text(tmp_path):
    """Shapefiles are written with datetimes as text and read back with all sidecars present."""
    gdf = _geo_frame().assign(observed=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    path = tmp_path / "parcels.shp"
    io.write_data(gdf, str(path))

    result = io.read_data(str(path))

    expect_equal(list(result["observed"]), ["2024-01-01", "2024-01-02"])


def test_shapefile_writer_falls_back_to_csv_for_plain_frames(tmp_path, capsys):
    """A plain DataFrame requested as shapefile is stored as CSV next to the target."""
    io.write_shapefile(pd.DataFrame({"a": [1]}), str(tmp_path / "plain.shp"))

    expect_true((tmp_path / "plain.csv").exists(), "CSV fallback must be written")
    expect_in("converting to pandas", capsys.readouterr().out)


def test_shapefile_helper_requires_all_sidecars(tmp_path):
    """Missing shapefile sidecars are detected before reading; non-shapefiles are ignored."""
    (tmp_path / "broken.shp").write_bytes(b"")
    (tmp_path / "broken.dbf").write_bytes(b"")

    with pytest.raises(ValueError, match="Error: Some required Shapefile extensions are missing"):
        io.shapefile_helper(str(tmp_path / "broken.shp"), [".shp", ".shx", ".prj", ".dbf"])
    expect_equal(io.shapefile_helper(str(tmp_path / "other.gpkg"), [".shp"]), None)


def test_excel_round_trip_drops_geometry(tmp_path):
    """Excel output drops geometry and reads the requested sheet back."""
    path = tmp_path / "parcels.xlsx"
    io.write_data(_geo_frame(), str(path))

    result = io.read_data(str(path))

    expect_equal(list(result.columns), ["parcel_id", "crop"])
    expect_equal(len(result), 2)


def test_read_data_without_extension_picks_latest_allowed_file(tmp_path):
    """An extension-less name resolves to the most recently modified supported file."""
    older, newer = tmp_path / "frame.pkl", tmp_path / "frame.parquet"
    pd.DataFrame({"source": ["pickle"]}).to_pickle(older)
    pd.DataFrame({"source": ["parquet"]}).to_parquet(newer, index=False)
    (tmp_path / "frame.txt").write_text("ignored", encoding="utf-8")
    past = time.time() - 60
    os.utime(older, (past, past))

    result = io.read_data(str(tmp_path / "frame"))

    expect_equal(result["source"].tolist(), ["parquet"])


def test_read_data_validates_paths_and_extensions(tmp_path):
    """Spaces, missing matches and unsupported extensions are reported."""
    with pytest.raises(ValueError, match="Error: The file path contains space characters"):
        io.read_data(str(tmp_path / "with space.csv"))
    with pytest.raises(FileNotFoundError, match="Error: No matching files found"):
        io.read_data(str(tmp_path / "absent"))
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Error: Unsupported file extension: .txt"):
        io.read_data(str(unsupported))


@pytest.mark.parametrize("separate_filename", [False, True])
def test_write_data_accepts_explicit_pkl_xz_extension(tmp_path, separate_filename):
    """An explicit ``.pkl.xz`` path should be writable just like the extension-less default."""
    path = tmp_path / "frame.pkl.xz"
    if separate_filename:
        io.write_data(_plain_frame(), str(tmp_path), file_name=path.name)
    else:
        io.write_data(_plain_frame(), str(path))
    expect_true(path.exists())
    pd.testing.assert_frame_equal(pd.read_pickle(path, compression="xz"), _plain_frame())


@pytest.mark.parametrize("plain_csv", [False, True])
def test_write_csv_drops_geometry_from_geodataframes(tmp_path, plain_csv):
    """Writing a GeoDataFrame to CSV should drop geometry and keep attributes."""
    path = tmp_path / "parcels.csv"
    frame = _geo_frame()
    original = frame.copy()
    io.write_csv(frame, str(path), plain_csv=plain_csv)
    pd.testing.assert_frame_equal(io.read_data(str(path)), pd.DataFrame(original.drop(columns="geometry")))
    pd.testing.assert_frame_equal(frame, original)
