"""Fixed colour scales for HUB inspection previews.

Every scale is fixed across parcels and dates so colours stay comparable. A scale is a list
of ``(value, "#rrggbb")`` stops; values between stops are interpolated linearly and values
outside the first/last stop are clipped for display only (index values are never changed).
Pixels without a valid value are fully transparent.

Sources of the scales:

* NDVI-like vegetation indices: the requested ramp, dark blue to white over ``[-1, 0]`` and
  light yellow to dark green (ColorBrewer YlGn) over ``[0, 1]``.
* NDWI and MNDWI: Sentinel Hub NDWI script, green ``-0.8`` / white ``0`` / blue ``0.8``.
* NDMI: Sentinel Hub NDMI script, dry dark red to wet dark blue between ``-0.8`` and ``0.8``.
* PSRI: Sentinel Hub PSRI display range ``[-0.2, 0.4]``, green (low senescence) to brown.
* NBR: ColorBrewer BrBG diverging ramp over ``[-1, 1]`` (low/burnt or bare brown, high green).
* CI (red-edge chlorophyll index): ColorBrewer YlGn over ``[0, 6]``.
* Sentinel-1 R (VV/VH, linear) over ``[1, 15]`` and RVI over ``[0, 1]``: viridis and YlGn.
* Planet basemap NDVI uses the NDVI scale; its true colour uses the Sentinel-2 stretch.
"""

from __future__ import annotations

import numpy as np

_VEGETATION = [
    (-1.0, "#08306b"), (-0.5, "#6baed6"), (0.0, "#ffffff"),
    (0.0, "#ffffcc"), (0.25, "#c2e699"), (0.5, "#78c679"), (0.75, "#31a354"), (1.0, "#006837"),
]  # fmt: skip
_WATER = [(-0.8, "#008000"), (0.0, "#ffffff"), (0.8, "#0000cc")]
_MOISTURE = [
    (-0.8, "#800000"), (-0.24, "#ff0000"), (-0.032, "#ffff00"),
    (0.032, "#00ffff"), (0.24, "#0000ff"), (0.8, "#000080"),
]  # fmt: skip
_BURN = [(-1.0, "#8c510a"), (-0.5, "#d8b365"), (0.0, "#f5f5f5"), (0.5, "#5ab4ac"), (1.0, "#01665e")]
_SENESCENCE = [(-0.2, "#1a9850"), (0.1, "#ffffbf"), (0.4, "#8c510a")]
_CHLOROPHYLL = [(0.0, "#ffffcc"), (1.5, "#c2e699"), (3.0, "#78c679"), (4.5, "#31a354"), (6.0, "#006837")]
_VIRIDIS = [(1.0, "#440154"), (4.5, "#3b528b"), (8.0, "#21918c"), (11.5, "#5ec962"), (15.0, "#fde725")]
_RADAR_VEGETATION = [(0.0, "#ffffcc"), (0.25, "#c2e699"), (0.5, "#78c679"), (0.75, "#31a354"), (1.0, "#006837")]

INDEX_SCALES: dict[str, list[tuple[float, str]]] = {
    "NDVI": _VEGETATION, "GNDVI": _VEGETATION, "SAVI": _VEGETATION, "MSAVI": _VEGETATION,
    "EVI": _VEGETATION, "NDRE": _VEGETATION,
    "NDWI": _WATER, "MNDWI": _WATER, "NDMI": _MOISTURE, "NBR": _BURN, "PSRI": _SENESCENCE,
    "CI": _CHLOROPHYLL, "R": _VIRIDIS, "RVI": _RADAR_VEGETATION,
}  # fmt: skip

# True-colour reflectance shown linearly between 0 and 0.3 (brighter surfaces saturate).
TRUE_COLOUR_REFLECTANCE_MAX = 0.3
# Planet basemap surface reflectance is stored as reflectance x 10000.
PLANET_REFLECTANCE_SCALE_FACTOR = 1e-4
# Sentinel-1 false colour: (red, green, blue) = (VV dB, VH dB, VV dB - VH dB) with fixed stretches.
FALSE_COLOUR_DB_RANGES = ((-20.0, 0.0), (-28.0, -8.0), (0.0, 15.0))


def legend(name: str) -> dict:
    """Return the browser legend of an index scale: its display range and colour stops."""
    stops = _scale(name)
    return {"min": stops[0][0], "max": stops[-1][0], "stops": [[value, colour] for value, colour in stops]}


def _scale(name: str) -> list[tuple[float, str]]:
    """Return the colour stops of one index, failing clearly for an index without a scale."""
    if name not in INDEX_SCALES:
        raise ValueError(f"Error: No preview colour scale is defined for index {name}.")
    return INDEX_SCALES[name]


def _hex_channels(colour: str) -> tuple[int, int, int]:
    """Return the red, green and blue values of a ``#rrggbb`` colour."""
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def colour_index(values: np.ndarray, name: str) -> np.ndarray:
    """Colour one index array as an RGBA ``uint8`` image; non-finite pixels are transparent."""
    stops = _scale(name)
    positions = np.array([value for value, _ in stops], dtype="float64")
    channels = np.array([_hex_channels(colour) for _, colour in stops], dtype="float64")
    finite = np.isfinite(values)
    # Repeated stop positions (NDVI at 0) give a sharp step; np.interp takes the upper colour there.
    clipped = np.clip(np.where(finite, values, positions[0]), positions[0], positions[-1])
    image = np.zeros((*values.shape, 4), dtype="uint8")
    for channel in range(3):
        image[..., channel] = np.rint(np.interp(clipped, positions, channels[:, channel])).astype("uint8")
    image[..., 3] = np.where(finite, 255, 0)
    return image


def _stretch(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Scale values linearly from ``[low, high]`` to ``0..255``, clipping outside the range."""
    scaled = (np.nan_to_num(values, nan=low) - low) / (high - low)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255).astype("uint8")


def _compose(channels: list[np.ndarray], ranges, valid: np.ndarray) -> np.ndarray:
    """Stack three stretched channels into RGBA; pixels outside ``valid`` are transparent."""
    image = np.zeros((*valid.shape, 4), dtype="uint8")
    for position, (values, (low, high)) in enumerate(zip(channels, ranges, strict=True)):
        image[..., position] = _stretch(values, low, high)
    image[..., 3] = np.where(valid, 255, 0)
    return image


def true_colour(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Return RGBA true colour from physical reflectance; pixels missing any band are transparent."""
    valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    span = (0.0, TRUE_COLOUR_REFLECTANCE_MAX)
    return _compose([red, green, blue], (span, span, span), valid)


def basemap_true_colour(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Return RGBA true colour from scaled Planet basemap reflectance, on the Sentinel-2 true-colour stretch."""
    scale = PLANET_REFLECTANCE_SCALE_FACTOR
    return true_colour(red * scale, green * scale, blue * scale)


def sentinel1_false_colour(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Return RGBA Sentinel-1 false colour from linear backscatter: VV dB, VH dB and their dB difference.

    Only positive, finite backscatter is shown; the blue channel equals ``10 * log10(VV / VH)``.
    """
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_db = np.where(valid, 10.0 * np.log10(vv), np.nan)
        vh_db = np.where(valid, 10.0 * np.log10(vh), np.nan)
    return _compose([vv_db, vh_db, vv_db - vh_db], FALSE_COLOUR_DB_RANGES, valid)
