import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { api } from "../../api";
import { acquisitionsForLayer, beginPreviewLoad, previewAttribution, previewImageUrl, statisticsUrl } from "./hubPreview";
import type { ParcelAcquisitions, ParcelStatistics, PreviewMetadata } from "./hubPreview";

/** Load HUB metadata independently, then display just the selected image on its exact grid. */
export function useHubPreview(mlId: string, parcelId: string, map: L.Map) {
  const [metadata, setMetadata] = useState<PreviewMetadata>();
  const [coverage, setCoverage] = useState<ParcelAcquisitions>();
  const [layerName, setLayerName] = useState("");
  const [index, setIndex] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [opacity, setOpacityState] = useState(1);
  const [statistics, setStatistics] = useState<ParcelStatistics>();
  const [statisticsError, setStatisticsError] = useState("");
  // Refs let the slider restyle the displayed overlay without reloading its image,
  // and let a newly loaded image start at the opacity the user last chose.
  const opacityRef = useRef(1);
  const overlayRef = useRef<L.ImageOverlay | undefined>(undefined);
  const layer = metadata?.layers.find((item) => item.name === layerName);
  const acquisitions = acquisitionsForLayer(coverage?.acquisitions ?? [], layer);
  const acquisition = acquisitions[index];

  useEffect(() => {
    let live = true;
    const base = `/ml/${encodeURIComponent(mlId)}/inspection/hub`;
    // The two independent endpoints can overlap their metadata and geometry work.
    Promise.all([
      api<PreviewMetadata>(base),
      api<ParcelAcquisitions>(`${base}/acquisitions?parcel_id=${encodeURIComponent(parcelId)}`),
    ]).then(([info, found]) => {
      if (!live) return;
      setMetadata(info);
      setCoverage(found);
      setLayerName(info.layers[0]?.name ?? "");
    }).catch((e: Error) => { if (live) setError(e.message); });
    return () => { live = false; };
  }, [mlId, parcelId]);

  // Saved parcel statistics of the selected index; a failure here leaves the imagery usable.
  // Planet basemaps are reference imagery, not analysis inputs, so they have none.
  const statisticsLayer = layer?.kind === "index" && layer.sensor !== "PL" ? layer.name : "";
  useEffect(() => {
    setStatistics(undefined);
    setStatisticsError("");
    if (!statisticsLayer) return;
    let live = true;
    api<ParcelStatistics>(statisticsUrl(mlId, parcelId, statisticsLayer))
      .then((found) => { if (live) setStatistics(found); })
      .catch((e: Error) => { if (live) setStatisticsError(e.message); });
    return () => { live = false; };
  }, [mlId, parcelId, statisticsLayer]);

  useEffect(() => {
    if (!acquisition || !layer || !coverage) return;
    let live = true;
    let overlay: L.ImageOverlay | undefined;
    let objectUrl: string | undefined;
    let cancel: (() => void) | undefined;
    setLoading(true);
    setError("");
    const timer = window.setTimeout(() => {
      cancel = beginPreviewLoad(previewImageUrl(mlId, parcelId, acquisition, layer.name), (blob) => {
        if (!live) return;
        objectUrl = URL.createObjectURL(blob);
        // A dedicated pane keeps imagery below the parcel outline regardless of
        // which asynchronous request completed first.
        if (!map.getPane("hubPreview")) map.createPane("hubPreview").style.zIndex = "350";
        overlay = L.imageOverlay(objectUrl, coverage.bounds, {
          pane: "hubPreview",
          opacity: 0,
          interactive: false,
          attribution: previewAttribution(layer.sensor),
        });
        overlay.once("load", () => {
          if (!live) return;
          overlay?.setOpacity(opacityRef.current);
          overlayRef.current = overlay;
          setLoading(false);
        });
        overlay.once("error", () => {
          if (!live) return;
          overlay?.remove();
          setError("Error: HUB preview could not be displayed.");
          setLoading(false);
        });
        overlay.addTo(map);
      }, (message) => {
        if (!live) return;
        setError(message);
        setLoading(false);
      });
    }, 150);
    return () => {
      live = false;
      window.clearTimeout(timer);
      cancel?.();
      if (overlayRef.current === overlay) overlayRef.current = undefined;
      overlay?.remove();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [mlId, parcelId, map, acquisition, layer, coverage]);

  function selectLayer(name: string) {
    setLayerName(name);
    setIndex(0);
    setError("");
    setLoading(false);
  }

  /** Set the preview overlay opacity (0 transparent, 1 opaque); it persists across dates and layers. */
  function setOpacity(value: number) {
    opacityRef.current = value;
    setOpacityState(value);
    overlayRef.current?.setOpacity(value);
  }

  return {metadata, coverage, layer, acquisitions, index, setIndex, selectLayer, error, loading, opacity, setOpacity,
    statistics, statisticsError};
}
