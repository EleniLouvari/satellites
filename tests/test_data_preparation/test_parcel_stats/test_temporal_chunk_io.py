"""Verify chunk-aligned temporal I/O without changing fill semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from data_preparation.parcel_stats.core import streamed_raster
from tests.test_data_preparation.test_parcel_stats.test_temporal_pixel_pooling import _extractor


@pytest.mark.parametrize("rows", [None, 7, 256, 300, 512])
def test_temporal_blocks_cover_raster_once_with_chunk_alignment(tmp_path, rows):
    """Align large blocks and retain small strips inside one chunk before advancing."""
    extractor = _extractor(tmp_path, temporal_fill_block_rows=rows)
    variable = SimpleNamespace(shape=(135, 12, 1173, 2036), chunking=lambda: [1, 1, 256, 256])
    block_rows, block_columns = extractor._temporal_block_shape(variable)
    if rows is None:
        assert (block_rows, block_columns) == (256, 256)
    if rows is not None:
        assert block_rows <= rows
    if block_rows >= 256:
        assert block_rows % 256 == 0
    assert block_columns % 256 == 0
    coverage = np.zeros(variable.shape[-2:], dtype="uint8")
    for y, x in extractor._temporal_spatial_windows(variable, block_rows, block_columns):
        coverage[y, x] += 1
        assert x.start % 256 == 0
        if block_rows < 256:
            assert y.start // 256 == (y.stop - 1) // 256
        else:
            assert y.start % 256 == 0
    np.testing.assert_array_equal(coverage, 1)


@pytest.mark.parametrize("rows", [None, 7, 256, 300])
@pytest.mark.parametrize("mode", ["past_only", "bidirectional"])
def test_chunked_fill_matches_whole_cube_and_restores_caches(tmp_path, monkeypatch, rows, mode):
    """Fill across spatial chunk edges without losing coverage guards or provenance."""
    extractor = _extractor(tmp_path, temporal_fill_block_rows=rows, temporal_fill_mode=mode, temporal_fill_window_sizes=(3,))
    rng = np.random.default_rng(42)
    values = rng.normal(size=(5, 270, 270)).astype("float32")
    values[rng.random(values.shape) < 0.3] = np.nan
    allowed = np.array([True, False, True, True, True])
    source = values.copy()
    source[~allowed] = np.nan
    expected = extractor._merge_filled_values(
        values, extractor._fill_temporal_neighbors(source), np.ones((270, 270), dtype=bool), allowed
    )
    added = ~np.isfinite(values) & np.isfinite(expected)
    # Force both horizontal and vertical chunk edges without a large fixture.
    monkeypatch.setattr(streamed_raster, "TEMPORAL_BLOCK_BYTES", 5 * 256 * 256 * 4)
    with extractor._new_streamed_checkpoint(tmp_path / "fill.nc", list("abcde"), ["B04"], (270, 270)) as checkpoint:
        stored = checkpoint.variables["cleaned"]
        temporal = checkpoint.variables["temporal_filled_mask"]
        stored[:, 0] = values
        before = [var.get_var_chunk_cache() for var in (stored, temporal)]
        observed_caches = []
        fill = extractor._fill_temporal_neighbors

        def observe_cache(block):
            """Inspect cache settings during the real fill operation."""
            observed_caches.append(stored.get_var_chunk_cache()[0])
            return fill(block)

        monkeypatch.setattr(extractor, "_fill_temporal_neighbors", observe_cache)
        counts = (~np.isfinite(values)).sum(axis=(1, 2))[:, None]
        extractor._stream_temporal_fill(checkpoint, allowed[:, None], counts)
        np.testing.assert_allclose(stored[:, 0], expected, equal_nan=True)
        np.testing.assert_array_equal(temporal[:, 0], added)
        np.testing.assert_array_equal(counts[:, 0], (~np.isfinite(expected)).sum(axis=(1, 2)))
        assert [var.get_var_chunk_cache() for var in (stored, temporal)] == before
        assert min(observed_caches) > before[0][0]
        assert max(observed_caches) <= 64 * 1024**2


def test_temporal_cache_restored_on_fill_failure(tmp_path, monkeypatch):
    """Restore per-variable caches when a partial temporal operation fails."""
    extractor = _extractor(tmp_path)
    with extractor._new_streamed_checkpoint(tmp_path / "failure.nc", list("abcde"), ["B04"], (270, 270)) as checkpoint:
        variables = [checkpoint.variables[name] for name in ("cleaned", "temporal_filled_mask")]
        before = [var.get_var_chunk_cache() for var in variables]
        variables[0][:] = 1

        def fail(_):
            """Fail after cache tuning and the first read."""
            assert variables[0].get_var_chunk_cache()[0] > before[0][0]
            raise RuntimeError("fill interrupted")

        monkeypatch.setattr(extractor, "_fill_temporal_neighbors", fail)
        with pytest.raises(RuntimeError, match="fill interrupted"):
            extractor._stream_temporal_fill(checkpoint, np.ones((5, 1), dtype=bool), np.zeros((5, 1), dtype="int64"))
        assert [var.get_var_chunk_cache() for var in variables] == before


def test_automatic_block_and_cache_budgets_on_long_time_axis(tmp_path):
    """Subdivide oversized chunks in memory and cap caches for deep time series."""
    extractor = _extractor(tmp_path, temporal_fill_block_rows=None)
    variable = SimpleNamespace(shape=(2000, 1, 1024, 2048), chunking=lambda: [1, 1, 256, 256])
    rows, columns = extractor._temporal_block_shape(variable)
    assert rows < 256
    assert 2000 * rows * columns * 4 <= streamed_raster.TEMPORAL_BLOCK_BYTES
    with extractor._new_streamed_checkpoint(tmp_path / "metadata.nc", [str(i) for i in range(2000)], ["B04"], (256, 512)) as ds:
        variables = [ds.variables[name] for name in ("cleaned", "temporal_filled_mask")]
        with extractor._temporal_chunk_caches(variables, 256, 512):
            assert all(var.get_var_chunk_cache()[0] == 64 * 1024**2 for var in variables)
