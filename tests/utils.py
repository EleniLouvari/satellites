"""
Util functions used in tests.

This module provides assertion utility functions for unit testing,
including equality checks, type validation, and exception testing.
"""

import contextlib
import json

import rasterio
from rasterio.transform import from_origin


def expect_true(
    cond,
    msg: str = "Expected condition to be true",
) -> None:
    """Fail test if condition is falsy.

    Parameters
    ----------
    cond : any
        The condition to check for truthiness.
    msg : str, optional
        Custom error message if condition is falsy.

    Raises
    ------
    AssertionError
        If condition evaluates to False.

    Returns
    -------
    None

    """
    if not cond:
        raise AssertionError(msg)


def expect_false(
    cond,
    msg: str = "Expected condition to be false",
) -> None:
    """Fail test if condition is truthy.

    Parameters
    ----------
    cond : any
        The condition to check for falsiness.
    msg : str, optional
        Custom error message if condition is truthy.

    Raises
    ------
    AssertionError
        If condition evaluates to True.

    Returns
    -------
    None

    """
    if cond:
        raise AssertionError(msg)


def expect_equal(
    actual,
    expected,
    msg: str | None = None,
) -> None:
    """Fail test if values differ.

    Parameters
    ----------
    actual : any
        The actual value obtained.
    expected : any
        The expected value.
    msg : str, optional
        Custom error message if values differ.

    Raises
    ------
    AssertionError
        If actual != expected.

    Returns
    -------
    None

    """
    if actual != expected:
        raise AssertionError(msg or f"Expected {expected!r}, got {actual!r}")


def expect_is_none(
    value,
    msg: str = "Expected value to be None",
) -> None:
    """Fail test if value is not None.

    Parameters
    ----------
    value : any
        The value to check for None.
    msg : str, optional
        Custom error message if value is not None.

    Raises
    ------
    AssertionError
        If value is not None.

    Returns
    -------
    None

    """
    if value is not None:
        raise AssertionError(msg)


def expect_is_not_none(
    value,
    msg: str = "Expected value not to be None",
) -> None:
    """Fail test if value is None.

    Parameters
    ----------
    value : any
        The value to check for non-None.
    msg : str, optional
        Custom error message if value is None.

    Raises
    ------
    AssertionError
        If value is None.

    Returns
    -------
    None

    """
    if value is None:
        raise AssertionError(msg)


def expect_isinstance(
    obj,
    typ,
    msg: str | None = None,
) -> None:
    """Fail test if obj is not instance of typ.

    Parameters
    ----------
    obj : any
        The object to check type of.
    typ : type or tuple of types
        The expected type(s).
    msg : str, optional
        Custom error message if type doesn't match.

    Raises
    ------
    AssertionError
        If obj is not an instance of typ.

    Returns
    -------
    None

    """
    if not isinstance(obj, typ):
        raise AssertionError(msg or f"Expected instance of {typ}, got {type(obj)}")


def expect_in(
    member,
    container,
    msg: str | None = None,
) -> None:
    """Fail test if member not in container.

    Parameters
    ----------
    member : any
        The item to look for in container.
    container : container
        The container to search in.
    msg : str, optional
        Custom error message if member not found.

    Raises
    ------
    AssertionError
        If member not in container.

    Returns
    -------
    None

    """
    if member not in container:
        raise AssertionError(msg or f"Expected {member!r} in container")


def expect_not_in(
    member,
    container,
    msg: str | None = None,
) -> None:
    """Fail test if member in container.

    Parameters
    ----------
    member : any
        The item to look for in container.
    container : container
        The container to search in.
    msg : str, optional
        Custom error message if member found.

    Raises
    ------
    AssertionError
        If member in container.

    Returns
    -------
    None

    """
    if member in container:
        raise AssertionError(msg or f"Expected {member!r} not in container")


def expect_startswith(
    container,
    member,
    msg: str | None = None,
) -> None:
    """Fail test if container does not start with member.

    Parameters
    ----------
    container : container
        The container to search.
    member : any
        The string to search in the container.startswith.
    msg : str, optional
        Custom error message if member not found.

    Raises
    ------
    AssertionError
        If member not in container.

    Returns
    -------
    None

    """
    if not container.startswith(member):
        raise AssertionError(msg or f"Expected container starts with {member!r}")


def expect_subset(
    subset,
    superset,
    msg: str | None = None,
) -> None:
    """Fail test if subset is not subset of superset.

    Parameters
    ----------
    subset : iterable
        The supposed subset.
    superset : iterable
        The supposed superset.
    msg : str, optional
        Custom error message if not a subset.

    Raises
    ------
    AssertionError
        If subset is not a subset of superset.

    Returns
    -------
    None

    """
    if not set(subset).issubset(set(superset)):
        raise AssertionError(msg or f"Expected {set(subset)} ⊆ {set(superset)}")


@contextlib.contextmanager
def expect_raises(
    expected_exception,
    msg: str | None = None,
):
    """Fail test if expected exception is not raised.

    Parameters
    ----------
    expected_exception : Exception or tuple of Exceptions
        The exception type(s) expected to be raised.
    msg : str, optional
        Custom error message if wrong exception is raised.

    Yields
    ------
    contextmanager
        Context manager that captures the exception.

    Raises
    ------
    AssertionError
        If expected exception is not raised.
    """
    try:
        yield
        raise AssertionError(msg or f"Expected {expected_exception} to be raised")
    except expected_exception:
        pass
    except Exception as e:
        raise AssertionError(msg or f"Expected {expected_exception}, got {type(e).__name__}: {e}") from e


def expect_called_once(
    mock_obj,
    msg: str = "Expected mock to be called exactly once",
) -> None:
    """Fail test if mock object was not called exactly once.

    Parameters
    ----------
    mock_obj : unittest.mock.MagicMock
        The mock object to check.
    msg : str, optional
        Custom error message if call count is not one.

    Raises
    ------
    AssertionError
        If mock was not called exactly once.

    Returns
    -------
    None

    """
    if mock_obj.call_count != 1:
        raise AssertionError(msg)


def expect_not_called(
    mock_obj,
    msg: str = "Expected mock to not be called",
) -> None:
    """Fail test if mock object was called.

    Parameters
    ----------
    mock_obj : unittest.mock.MagicMock
        The mock object to check.
    msg : str, optional
        Custom error message if call count is not zero.

    Raises
    ------
    AssertionError
        If mock was called.

    Returns
    -------
    None

    """
    if mock_obj.call_count != 0:
        raise AssertionError(msg)


def write_tmp_tif(
    path,
    arr,
    *,
    crs: str = "EPSG:4326",
    transform=None,
    nodata=None,
    dtype: str = "float32",
) -> None:
    """Create a small single-band GeoTIFF for tests.

    Parameters
    ----------
    path : str
        File path where to write the GeoTIFF.
    arr : numpy.ndarray
        2D array with raster data.
    crs : str, optional
        Coordinate reference system (default: EPSG_4326).
    transform : rasterio.transform.Affine, optional
        Geotransform for the raster.
    nodata : any, optional
        Nodata value for the raster.
    dtype : str, optional
        Data type for the raster (default: "float32").

    Returns
    -------
    None

    """
    height, width = arr.shape
    if transform is None:
        # top-left y=10, pixel size 1x1
        transform = from_origin(0, 10, 1, 1)  # pixel size 1x1
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "transform": transform,
        "crs": crs,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
    del arr


def _write_xml(path: str, content: str) -> None:
    """Write XML text to file.

    Parameters
    ----------
    path : str
        File path to write XML content.
    content : str
        XML content to write.

    Returns
    -------
    None

    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, payload: dict) -> None:
    """Write JSON payload to file.

    Parameters
    ----------
    path : str
        File path to write JSON content.
    payload : dict
        JSON content to write.

    Returns
    -------
    None

    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
