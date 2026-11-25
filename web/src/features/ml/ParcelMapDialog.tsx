import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { api } from "../../api";
import { Button, Notice } from "../../components/ui";
import AcquisitionTimeline from "./AcquisitionTimeline";
import HubPreviewControls from "./HubPreviewControls";

type Sentinel2 = {
  source?: "openeo" | "hub";
  available: boolean;
  start_date: string;
  end_date: string;
  max_cloud_cover: number;
  min_zoom: number;
  max_zoom: number;
};
type Acquisition = { date: string; cloud_cover: number | null };

function tileUrl(mlId: string, date: string): string {
  return `/api/ml/${mlId}/inspection/sentinel2/{z}/{x}/{y}.png?date=${date}`;
}

/** Pop-up map with one inspected parcel's footprint drawn over a background map. */
export default function ParcelMapDialog({
  mlId,
  parcelId,
  onClose,
}: {
  mlId: string;
  parcelId: string;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const container = useRef<HTMLDivElement>(null);
  const [mapInstance, setMapInstance] = useState<L.Map>();
  const [hub, setHub] = useState(false);
  const [error, setError] = useState("");
  const [criteria, setCriteria] = useState<Sentinel2>();
  const [dates, setDates] = useState<Acquisition[]>();
  const [datesError, setDatesError] = useState("");
  const [index, setIndex] = useState(0);
  const sentinel2Layer = useRef<L.TileLayer | undefined>(undefined);

  useEffect(() => {
    dialog.current?.showModal();
  }, []);

  useEffect(() => {
    if (!container.current) return;
    const map = L.map(container.current);
    setMapInstance(map);
    map.setView([38.5, 23.5], 6); // Greece, until the parcel loads.
    const satellite = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        maxZoom: 19,
        attribution:
          "Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
      },
    ).addTo(map);
    const streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
      // OpenStreetMap blocks tile requests without a Referer; the app's same-origin policy sends none,
      // so the tiles alone send the site origin (never the page path).
      referrerPolicy: "strict-origin",
    });
    const backgrounds = L.control
      .layers({ "Esri satellite": satellite, OpenStreetMap: streets }, undefined, {
        collapsed: false,
      })
      .addTo(map);
    let live = true;
    setError("");
    setHub(false);
    setCriteria(undefined);
    setDates(undefined);
    setDatesError("");
    // Sentinel-2 dates follow the parcel statistics run's dates and cloud limit, oldest first;
    // the API fetches the images so the Sentinel Hub instance ID stays on the server.
    api<Sentinel2>(`/ml/${mlId}/inspection/sentinel2`)
      .then(async (info) => {
        if (!live) return;
        if (info.source === "hub") { setHub(true); return; }
        if (!info.available) return;
        setCriteria(info);
        const found = await api<Acquisition[]>(
          `/ml/${mlId}/inspection/sentinel2/dates?parcel_id=${encodeURIComponent(parcelId)}`,
        );
        if (!live) return;
        setDates(found);
        setIndex(0);
        if (!found.length) return;
        const sentinel2 = L.tileLayer(tileUrl(mlId, found[0].date), {
          minZoom: info.min_zoom,
          maxNativeZoom: info.max_zoom,
          maxZoom: 19,
          attribution: "Contains modified Copernicus Sentinel data",
        });
        sentinel2Layer.current = sentinel2;
        backgrounds.addBaseLayer(sentinel2, "Sentinel-2");
        // Sentinel-2 is the default background whenever the parcel has images.
        map.removeLayer(satellite);
        sentinel2.addTo(map);
      })
      .catch((e) => {
        // Without Sentinel-2 the other backgrounds still work.
        if (live) setDatesError(e.message);
      });
    api<GeoJSON.Feature>(
      `/ml/${mlId}/inspection/geometry?parcel_id=${encodeURIComponent(parcelId)}`,
    )
      .then((feature) => {
        if (!live) return;
        const layer = L.geoJSON(feature, {
          style: { color: "#ffff00", weight: 3, fill: false },
        }).addTo(map);
        map.fitBounds(layer.getBounds(), { padding: [40, 40], maxZoom: 18 });
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
      sentinel2Layer.current = undefined;
      map.remove();
    };
  }, [mlId, parcelId]);

  // Leaflet measures its container only on window resizes, so redraw when the side panel
  // appears and narrows the map.
  useEffect(() => {
    if (!container.current || !mapInstance) return;
    const observer = new ResizeObserver(() => mapInstance.invalidateSize());
    observer.observe(container.current);
    return () => observer.disconnect();
  }, [mapInstance]);

  // Show the chosen acquisition date; the short delay avoids fetching every date passed while
  // dragging the timeline.
  useEffect(() => {
    const date = dates?.[index]?.date;
    if (!date) return;
    const timer = window.setTimeout(
      () => sentinel2Layer.current?.setUrl(tileUrl(mlId, date)),
      150,
    );
    return () => window.clearTimeout(timer);
  }, [mlId, dates, index]);

  return (
    <dialog
      ref={dialog}
      className="confirm-dialog parcel-map-dialog"
      aria-labelledby="parcel-map-title"
      onCancel={onClose}
    >
      <div className="help-head">
        <h2 id="parcel-map-title">Parcel {parcelId}</h2>
        <Button onClick={onClose}>Close</Button>
      </div>
      <Notice message={error} />
      {/* Left: imagery controls, timeline and information; right: only the map. */}
      <div className={`parcel-map-layout${criteria || hub ? " with-side" : ""}${hub ? " hub" : ""}`}>
        {(criteria || hub) && (
          <div className="parcel-map-side">
            {criteria && (
              <div className="parcel-map-dates">
                {datesError ? (
                  <span className="muted">Sentinel-2 dates are unavailable: {datesError}</span>
                ) : !dates ? (
                  <span className="muted">Finding Sentinel-2 dates…</span>
                ) : !dates.length ? (
                  <span className="muted">
                    No Sentinel-2 image of this parcel between {criteria.start_date} and{" "}
                    {criteria.end_date} with at most {criteria.max_cloud_cover}% cloud.
                  </span>
                ) : (
                  <>
                    <select
                      aria-label="Sentinel-2 image date"
                      value={index}
                      onChange={(e) => setIndex(Number(e.target.value))}
                    >
                      {dates.map((item, i) => (
                        <option key={item.date} value={i}>
                          {item.date}
                          {item.cloud_cover == null ? "" : ` · ${Math.round(item.cloud_cover)}% cloud`}
                        </option>
                      ))}
                    </select>
                    <span className="muted">
                      {index + 1} of {dates.length} · {criteria.start_date} – {criteria.end_date} · ≤
                      {criteria.max_cloud_cover}% cloud
                    </span>
                    <div className="parcel-map-steps">
                      <Button disabled={index === 0} onClick={() => setIndex(index - 1)}>
                        ← Older
                      </Button>
                      <Button
                        disabled={index === dates.length - 1}
                        onClick={() => setIndex(index + 1)}
                      >
                        Newer →
                      </Button>
                    </div>
                    <AcquisitionTimeline
                      dates={dates}
                      index={index}
                      startDate={criteria.start_date}
                      endDate={criteria.end_date}
                      onChange={setIndex}
                    />
                  </>
                )}
              </div>
            )}
            {hub && mapInstance && <HubPreviewControls key={`${mlId}:${parcelId}`} mlId={mlId} parcelId={parcelId} map={mapInstance} />}
          </div>
        )}
        <div ref={container} className="parcel-map-canvas" />
      </div>
    </dialog>
  );
}
