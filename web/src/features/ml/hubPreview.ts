/** Public HUB preview metadata; asset locations and credentials stay on the server. */
export type PreviewLegend = { min: number; max: number; stops: [number, string][] };
/** S2/S1: Sentinel ARD acquisitions; PL: Planet monthly basemaps. */
export type PreviewSensor = "S2" | "S1" | "PL";
export type PreviewLayer = {
  name: string;
  label?: string;
  sensor: PreviewSensor;
  kind: "rgb" | "index";
  legend: PreviewLegend | null;
};
export type PreviewAcquisition = {
  id: string;
  datetime: string;
  sensor: PreviewSensor;
  cloud_cover: number | null;
  revision: string;
};
export type PreviewMetadata = {
  available: boolean;
  layers: PreviewLayer[];
  start_date: string;
  end_date: string;
};
export type ParcelAcquisitions = {
  bounds: [[number, number], [number, number]];
  acquisitions: PreviewAcquisition[];
};

/** UTC days, retaining sub-day positions for distinct passes on the same date. */
export function acquisitionTime(date: string): number {
  return Date.parse(date) / 86_400_000;
}

/** Snap a time-axis position to the closest acquisition rather than a calendar day. */
export function nearestAcquisition(times: number[], value: number): number {
  let best = 0;
  times.forEach((time, i) => {
    if (Math.abs(time - value) < Math.abs(times[best] - value)) best = i;
  });
  return best;
}

const SENSOR_NAMES: Record<PreviewSensor, string> = {S2: "Sentinel-2", S1: "Sentinel-1", PL: "Planet basemap"};

/** Readable source name of a layer's sensor. */
export function sensorName(sensor: PreviewSensor): string {
  return SENSOR_NAMES[sensor];
}

/** Image credit shown on the map for the selected source. */
export function previewAttribution(sensor: PreviewSensor): string {
  return sensor === "PL" ? "HUB · Imagery © Planet Labs PBC" : "HUB · Contains modified Copernicus Sentinel data";
}

/** Acquisition date for display: a month for monthly basemaps, otherwise the UTC timestamp. */
export function acquisitionLabel(item: Pick<PreviewAcquisition, "datetime" | "sensor">): string {
  if (item.sensor === "PL") {
    const month = new Date(item.datetime).toLocaleString("en-GB", {month: "long", year: "numeric", timeZone: "UTC"});
    return `${month} (monthly)`;
  }
  return item.datetime.replace("T", " ").replace(/Z$/, " UTC");
}

/** Select only acquisitions of the chosen layer's sensor. */
export function acquisitionsForLayer(items: PreviewAcquisition[], layer?: PreviewLayer) {
  return items.filter((item) => item.sensor === layer?.sensor);
}

/** Stable, encoded image URL so the browser can cache each rendering privately. */
export function previewImageUrl(mlId: string, parcelId: string, item: PreviewAcquisition, layer: string) {
  const params = new URLSearchParams({parcel_id: parcelId, acquisition: item.id, layer, rev: item.revision});
  return `/api/ml/${encodeURIComponent(mlId)}/inspection/hub/image.png?${params}`;
}

/** Preserve the numeric meaning of palette stops, including asymmetric ramps. */
export function legendGradient(legend: PreviewLegend) {
  const span = legend.max - legend.min || 1;
  const stops = legend.stops.map(([value, colour]) => `${colour} ${((value - legend.min) / span) * 100}%`);
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

/** Fetch one image; cancellation also blocks stale responses from non-abortable caches. */
export function beginPreviewLoad(url: string, loaded: (blob: Blob) => void, failed: (message: string) => void) {
  const controller = new AbortController();
  const fallback = "Error: HUB preview could not be loaded.";
  fetch(url, {credentials: "same-origin", signal: controller.signal})
    .then(async (response) => {
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : fallback);
      }
      const blob = await response.blob();
      if (!controller.signal.aborted) loaded(blob);
    })
    .catch((error: unknown) => {
      if (!controller.signal.aborted) failed(error instanceof Error ? error.message : fallback);
    });
  return () => controller.abort();
}

/** What each analysed index shows, with the formula the parcel statistics use (Sentinel-2 bands, Sentinel-1 linear power). */
const INDEX_DESCRIPTIONS: Record<string, string> = {
  NDVI: "Normalised Difference Vegetation Index, (B08 − B04) / (B08 + B04): amount and vigour of green vegetation. "
    + "Dense healthy crops are about 0.6–0.9, bare soil about 0.1–0.2 and water below 0.",
  GNDVI: "Green NDVI, (B08 − B03) / (B08 + B03): like NDVI but uses green light, so it follows chlorophyll content "
    + "and saturates less over dense canopies.",
  SAVI: "Soil-Adjusted Vegetation Index, 1.5 (B08 − B04) / (B08 + B04 + 0.5): NDVI corrected for soil brightness, "
    + "useful where vegetation cover is sparse.",
  MSAVI: "Modified Soil-Adjusted Vegetation Index: adjusts the soil correction to the vegetation present, limiting the "
    + "bare-soil effect in early growth stages.",
  EVI: "Enhanced Vegetation Index, 2.5 (B08 − B04) / (B08 + 6 B04 − 7.5 B02 + 1): vegetation greenness that saturates "
    + "less over dense canopies and is less affected by soil and haze than NDVI.",
  NDRE: "Normalised Difference Red Edge, (B08 − B05) / (B08 + B05): chlorophyll and nitrogen status, sensitive in "
    + "dense, mid- to late-season canopies where NDVI saturates.",
  CI: "Red-edge Chlorophyll Index, B08 / B05 − 1: canopy chlorophyll content; higher values mean more chlorophyll.",
  NDWI: "Normalised Difference Water Index (McFeeters), (B03 − B08) / (B03 + B08): open water is above 0, "
    + "vegetation and dry soil below 0. Highlights water bodies and flooded fields.",
  MNDWI: "Modified NDWI, (B03 − B11) / (B03 + B11): uses shortwave infrared to separate open water from soil and "
    + "built-up surfaces better than NDWI.",
  NDMI: "Normalised Difference Moisture Index, (B08 − B11) / (B08 + B11): water content of vegetation. Higher is "
    + "wetter; low or negative values suggest water stress or dry bare soil.",
  NBR: "Normalised Burn Ratio, (B08 − B12) / (B08 + B12): high for healthy vegetation, low for burnt, harvested or "
    + "bare surfaces.",
  PSRI: "Plant Senescence Reflectance Index, (B04 − B02) / B06: rises as leaves senesce and lose chlorophyll; near or "
    + "below 0 for green vegetation.",
  R: "Sentinel-1 cross-polarisation ratio, VV / VH: lower ratios usually mean more vegetation volume; high ratios "
    + "point to bare or smooth surfaces. Radar sees through clouds.",
  RVI: "Sentinel-1 Radar Vegetation Index, 4 VH / (VV + VH): about 0 for bare soil rising towards 1 for dense "
    + "vegetation. Radar sees through clouds.",
  PL_NDVI: "NDVI of the Planet monthly basemap, (NIR − red) / (NIR + red): amount and vigour of green vegetation, as "
    + "reference imagery.",
};

/** Short description of what an index layer shows; colour composites have none. */
export function indexDescription(layer?: Pick<PreviewLayer, "name" | "kind">): string | undefined {
  return layer?.kind === "index" ? INDEX_DESCRIPTIONS[layer.name] : undefined;
}

/** Spatial statistics of one parcel and index saved by the analysis; only calculated statistics are present. */
export type StatisticName = "mean" | "median" | "sd" | "range";
export type StatisticsPeriod = { start: string; values: Partial<Record<StatisticName, number | null>> };
export type ParcelStatistics = { variable: string; temporal_period: string | null; periods: StatisticsPeriod[] };

export const STATISTIC_LABELS: [StatisticName, string][] = [
  ["mean", "Mean"], ["median", "Median"], ["sd", "SD"], ["range", "Range"],
];

/** Statistics URL of one parcel's index layer. */
export function statisticsUrl(mlId: string, parcelId: string, layer: string) {
  const params = new URLSearchParams({parcel_id: parcelId, layer});
  return `/ml/${encodeURIComponent(mlId)}/inspection/hub/statistics?${params}`;
}

/** The statistics period containing an acquisition: the latest period starting on or before its UTC date. */
export function periodForAcquisition(periods: StatisticsPeriod[], datetime: string): StatisticsPeriod | undefined {
  const day = datetime.slice(0, 10);
  return periods.filter((period) => period.start <= day).at(-1);
}

/** One period of the parcel's index chart; null marks a period without a saved value. */
export type ChartPoint = { start: string; mean: number | null; sd: number | null };

/** Mean and SD of every statistics period, in the saved (chronological) order. */
export function chartPoints(periods: StatisticsPeriod[]): ChartPoint[] {
  return periods.map(({start, values}) => ({start, mean: values.mean ?? null, sd: values.sd ?? null}));
}

/** Runs of consecutive point indices with a mean, so missing periods break the line and band. */
export function chartSegments(points: ChartPoint[]): number[][] {
  const segments: number[][] = [];
  points.forEach((point, i) => {
    if (point.mean == null) return;
    if (i > 0 && points[i - 1].mean != null) segments.at(-1)!.push(i);
    else segments.push([i]);
  });
  return segments;
}

/** Value range covering mean ± SD of every period, padded by 8%; undefined when no period has a mean. */
export function chartDomain(points: ChartPoint[]): [number, number] | undefined {
  const lows: number[] = [];
  const highs: number[] = [];
  for (const {mean, sd} of points) {
    if (mean == null) continue;
    lows.push(mean - (sd ?? 0));
    highs.push(mean + (sd ?? 0));
  }
  if (!lows.length) return undefined;
  const low = Math.min(...lows);
  const high = Math.max(...highs);
  // A flat series still needs a visible span around its value.
  const pad = (high - low) * 0.08 || Math.abs(high) * 0.1 || 0.1;
  return [low - pad, high + pad];
}

/** About `count` evenly spaced round values (steps of 1, 2, 2.5 or 5 × 10ⁿ) inside the domain. */
export function chartTicks([low, high]: [number, number], count = 4): number[] {
  const raw = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((multiple) => multiple * power).find((value) => value >= raw)!;
  const ticks: number[] = [];
  // Rounding removes floating-point residue such as 0.30000000000000004.
  for (let value = Math.ceil(low / step) * step; value <= high + step * 1e-9; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

/** A statistic for display; null means the period had no valid pixels. */
export function formatStatistic(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(3);
}
