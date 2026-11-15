"""Bounded, attribute-free upload inspection."""

import json
from pathlib import Path


def layer_metadata(frame, name):
    """Describe a layer without exposing parcel attributes in its map preview."""
    import geopandas as gpd
    from shapely.geometry import mapping

    empty = frame.geometry.isna() | frame.geometry.is_empty
    features = []
    if frame.crs:
        geometries = gpd.GeoSeries(frame.geometry[~empty].head(500), crs=frame.crs).to_crs(4326)
        for geometry in geometries:
            if not geometry.is_valid:
                continue
            feature = {"type": "Feature", "properties": {}, "geometry": mapping(geometry.simplify(0.0001))}
            candidate = {"type": "FeatureCollection", "features": features + [feature]}
            if len(json.dumps(candidate, allow_nan=False).encode()) > 1024 * 1024:
                break
            features.append(feature)
    bounds = frame.total_bounds.tolist() if len(frame) and not empty.all() else None
    return {
        "name": name,
        "columns": [str(c) for c in frame.columns if c != frame.geometry.name],
        "parcel_count": len(frame),
        "crs": str(frame.crs) if frame.crs else None,
        "bounds": bounds,
        "preview": {"type": "FeatureCollection", "features": features} if frame.crs else None,
        "geometry_invalid_count": int((~frame.geometry.is_valid & ~empty).sum()),
        "geometry_empty_count": int(empty.sum()),
    }


def inspect_upload(path):
    """Read supported actual vector content and describe every GeoPackage layer."""
    import geopandas as gpd

    path = Path(path)
    with path.open("rb") as source:
        header = source.read(16)
    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        if not header.startswith(b"PAR1"):
            raise ValueError("Error: File is not Parquet content.")
        return {"layers": [layer_metadata(gpd.read_parquet(path), None)]}
    if path.suffix.lower() not in {".geojson", ".gpkg"}:
        raise ValueError("Error: Unsupported vector format.")
    if path.suffix.lower() == ".gpkg":
        if header != b"SQLite format 3\x00":
            raise ValueError("Error: File is not GeoPackage content.")
        names = gpd.list_layers(path)["name"].tolist()
        if len(names) > 50:
            raise ValueError("Error: GeoPackage has too many layers; upload a smaller selection.")
    else:
        # Reject disguised virtual datasets before GDAL can follow external paths.
        with path.open(encoding="utf-8-sig") as source:
            document = json.load(source)
        if not isinstance(document, dict) or document.get("type") not in {"FeatureCollection", "Feature"}:
            raise ValueError("Error: File is not GeoJSON feature content.")
        names = [None]
    return {"layers": [layer_metadata(gpd.read_file(path, **({"layer": name} if name else {})), name) for name in names]}
