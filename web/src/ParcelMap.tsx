import { useMemo } from "react";
/** Render only server-simplified, attribute-free GeoJSON with bounded coordinates. */
export default function ParcelMap({ geojson }: { geojson: unknown }) {
  const paths = useMemo(() => {
    const rings: number[][][] = [];
    function geometry(g: any) {
      if (!g) return;
      if (g.type === "Polygon") rings.push(...g.coordinates);
      else if (g.type === "MultiPolygon")
        g.coordinates.forEach((p: number[][][]) => rings.push(...p));
      else if (g.type === "GeometryCollection") g.geometries.forEach(geometry);
    }
    const data = geojson as any;
    if (data?.type === "FeatureCollection")
      data.features?.forEach((f: any) => geometry(f.geometry));
    else if (data?.type === "Feature") geometry(data.geometry);
    else geometry(data);
    const points = rings
      .flat()
      .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    if (!points.length) return [];
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    points.forEach(([x, y]) => {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    });
    const scale = Math.min(560 / (maxX - minX || 1), 280 / (maxY - minY || 1));
    return rings.map(
      (r) =>
        r
          .map(
            ([x, y], i) =>
              `${i ? "L" : "M"}${(300 + (x - (maxX + minX) / 2) * scale).toFixed(2)},${(160 - (y - (maxY + minY) / 2) * scale).toFixed(2)}`,
          )
          .join(" ") + " Z",
    );
  }, [geojson]);
  return (
    <div className="parcel-map">
      {paths.length ? (
        <svg
          viewBox="0 0 600 320"
          role="img"
          aria-label="Simplified parcel footprint preview"
        >
          <defs>
            <pattern
              id="grid"
              width="30"
              height="30"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M30 0H0V30"
                fill="none"
                stroke="#ccd8bc"
                strokeWidth=".5"
              />
            </pattern>
          </defs>
          <rect width="600" height="320" fill="url(#grid)" />
          {paths.map((d, i) => (
            <path
              key={i}
              d={d}
              fill="#86ab62"
              fillOpacity=".55"
              stroke="#356b42"
              strokeWidth="1"
              fillRule="evenodd"
            />
          ))}
        </svg>
      ) : (
        <div className="map-empty">
          ◇<small>Parcel footprint appears after inspection</small>
        </div>
      )}
      <span className="map-label">
        {paths.length
          ? "SIMPLIFIED FOOTPRINT · NOT A BASEMAP"
          : "PARCEL PREVIEW"}
      </span>
    </div>
  );
}
