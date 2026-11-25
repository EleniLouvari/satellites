"""Sentinel-2 true-colour map tiles and acquisition dates from the Copernicus Data Space Sentinel Hub.

Tiles are fetched by the API so the Sentinel Hub configuration instance ID never reaches the browser.
"""

from fastapi import HTTPException

WMS_URL = "https://sh.dataspace.copernicus.eu/ogc/wms/{instance}"
WFS_URL = "https://sh.dataspace.copernicus.eu/ogc/wfs/{instance}"
WFS_TYPE = "DSS2"  # Sentinel-2 L2A acquisitions.
WFS_PAGE, WFS_MAX_PAGES = 100, 20  # Sentinel Hub returns at most 100 acquisitions per request.
LAYER = "TRUE_COLOR"
# Below zoom 10 a tile covers tens of kilometres, which costs many processing units for no inspection value.
MIN_ZOOM, MAX_ZOOM = 10, 18
HALF_WORLD = 20037508.342789244  # Half the Web Mercator (EPSG:3857) world width in metres.


def tile_bounds(z, x, y):
    """Return the EPSG:3857 bounds (min x, min y, max x, max y) of one XYZ tile."""
    size = 2 * HALF_WORLD / 2**z
    left, top = -HALF_WORLD + x * size, HALF_WORLD - y * size
    return left, top - size, left + size, top


def day_range(start_date, end_date):
    """Return a Sentinel Hub TIME value covering whole calendar days, both ends included."""
    return f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"


def get(url, params):
    """Call Sentinel Hub, turning network failures into a browser-safe error that never names the instance."""
    import requests

    try:
        return requests.get(url, params=params, timeout=30)
    except requests.RequestException as error:
        raise HTTPException(502, "Error: Sentinel Hub is unreachable.") from error


def fetch_tile(instance, z, x, y, start_date, end_date, max_cloud_cover):
    """Return one 256 px PNG: the least cloudy Sentinel-2 L2A image between the dates, within the cloud limit.

    Pass the same day as ``start_date`` and ``end_date`` to show a single acquisition date.
    """
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": LAYER,
        "CRS": "EPSG:3857",
        "BBOX": ",".join(f"{value:.3f}" for value in tile_bounds(z, x, y)),
        "WIDTH": 256,
        "HEIGHT": 256,
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
        # Same inclusive calendar-day range as the parcel statistics run (its end_date day is included).
        "TIME": day_range(start_date, end_date),
        "MAXCC": max_cloud_cover,
        "PRIORITY": "leastCC",
        "SHOWLOGO": "false",
    }
    response = get(WMS_URL.format(instance=instance), params)
    # Sentinel Hub reports errors as XML, sometimes with status 200.
    if response.status_code != 200 or not response.headers.get("content-type", "").startswith("image/"):
        raise HTTPException(502, "Error: Sentinel Hub did not return an image.")
    return response.content


def fetch_dates(instance, bounds, start_date, end_date, max_cloud_cover):
    """List the Sentinel-2 L2A acquisition dates over WGS84 ``bounds``, oldest first, with their lowest cloud cover.

    A parcel on the edge of several granules has several acquisitions per day; they become one date.
    """
    west, south, east, north = bounds
    clouds = {}
    for page in range(WFS_MAX_PAGES):
        params = {
            "SERVICE": "WFS",
            "REQUEST": "GetFeature",
            "VERSION": "2.0.0",
            "TYPENAMES": WFS_TYPE,
            "SRSNAME": "EPSG:4326",
            "BBOX": f"{south},{west},{north},{east}",  # WFS 2.0 uses latitude, longitude order for EPSG:4326.
            "TIME": day_range(start_date, end_date),
            "MAXCC": max_cloud_cover,
            "OUTPUTFORMAT": "application/json",
            "MAXFEATURES": WFS_PAGE,
            "FEATURE_OFFSET": page * WFS_PAGE,
        }
        response = get(WFS_URL.format(instance=instance), params)
        try:
            features = response.json()["features"] if response.status_code == 200 else None
        except ValueError:
            features = None
        if features is None:
            raise HTTPException(502, "Error: Sentinel Hub did not return the image dates.")
        for feature in features:
            date, cloud = feature["properties"]["date"], feature["properties"].get("cloudCoverPercentage")
            known = clouds.get(date)
            clouds[date] = cloud if known is None else min(known, cloud if cloud is not None else known)
        if len(features) < WFS_PAGE:
            break
    return [{"date": date, "cloud_cover": clouds[date]} for date in sorted(clouds)]
