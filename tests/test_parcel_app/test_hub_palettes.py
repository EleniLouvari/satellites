"""Fixed colour scales of HUB inspection previews."""

import numpy as np
import pytest

from data_preparation.parcel_stats.core.base import ParcelStatsBase
from parcel_app.sources.hub import palettes


def _rgb(colour):
    """Return the channel tuple of a ``#rrggbb`` colour."""
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def test_every_supported_index_has_an_increasing_scale():
    """Catch a configurable index that the map could not colour, or unsorted stops."""
    supported = {*ParcelStatsBase.SUPPORTED_SENTINEL2_INDICES, *ParcelStatsBase.SUPPORTED_SENTINEL1_INDICES}
    assert supported == set(palettes.INDEX_SCALES)
    for stops in palettes.INDEX_SCALES.values():
        positions = [value for value, _ in stops]
        assert positions == sorted(positions)


def test_ndvi_follows_the_requested_ramp():
    """Catch drift from dark blue/white below zero and light yellow to dark green above it."""
    values = np.array([[-1.0, -0.000001, 0.5, 1.0]], dtype="float32")
    image = palettes.colour_index(values, "NDVI")
    assert tuple(image[0, 0, :3]) == _rgb("#08306b")
    assert tuple(image[0, 1, :3]) == (255, 255, 255)
    assert tuple(image[0, 2, :3]) == _rgb("#78c679")
    assert tuple(image[0, 3, :3]) == _rgb("#006837")
    assert (image[..., 3] == 255).all()


def test_positive_ndvi_starts_at_light_yellow():
    """Catch a blend of white and yellow just above zero instead of the requested step."""
    image = palettes.colour_index(np.array([[0.0, 0.001]], dtype="float32"), "NDVI")
    assert abs(int(image[0, 1, 0]) - 0xFF) <= 1 and abs(int(image[0, 1, 2]) - 0xCC) <= 2


def test_out_of_range_values_are_clipped_and_nan_is_transparent():
    """Catch wrapped colours outside the display range or opaque missing pixels."""
    image = palettes.colour_index(np.array([[5.0, 1.0, np.nan]], dtype="float32"), "NDVI")
    assert tuple(image[0, 0]) == tuple(image[0, 1])
    assert image[0, 2, 3] == 0


def test_legend_describes_the_scale():
    """Catch a legend that disagrees with the colours drawn."""
    legend = palettes.legend("NDWI")
    assert legend == {"min": -0.8, "max": 0.8, "stops": [[-0.8, "#008000"], [0.0, "#ffffff"], [0.8, "#0000cc"]]}
    with pytest.raises(ValueError, match="Error: No preview colour scale"):
        palettes.legend("UNKNOWN")


def test_true_colour_stretches_reflectance_and_hides_missing_pixels():
    """Catch a different reflectance stretch or a pixel shown with a missing band."""
    red = np.array([[0.15, 0.6, 0.1]], dtype="float32")
    green = np.array([[0.0, 0.3, 0.1]], dtype="float32")
    blue = np.array([[0.3, 0.0, np.nan]], dtype="float32")
    image = palettes.true_colour(red, green, blue)
    assert tuple(image[0, 0]) == (128, 0, 255, 255)
    assert tuple(image[0, 1, :3]) == (255, 255, 0)
    assert image[0, 2, 3] == 0


def test_sentinel1_false_colour_uses_documented_db_stretches():
    """Catch a changed dB stretch, a wrong ratio channel or shown non-positive backscatter."""
    vv = np.array([[0.1, 0.0]], dtype="float32")  # -10 dB, then an invalid zero
    vh = np.array([[0.01, 0.01]], dtype="float32")  # -20 dB
    image = palettes.sentinel1_false_colour(vv, vh)
    # VV -10 dB in [-20, 0]; VH -20 dB in [-28, -8]; VV-VH = 10 dB in [0, 15].
    assert tuple(image[0, 0]) == (128, 102, 170, 255)
    assert image[0, 1, 3] == 0
