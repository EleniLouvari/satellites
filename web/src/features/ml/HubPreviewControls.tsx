import type L from "leaflet";
import { Button, Notice } from "../../components/ui";
import AcquisitionTimeline from "./AcquisitionTimeline";
import StatisticsChart from "./StatisticsChart";
import {
  STATISTIC_LABELS, acquisitionLabel, formatStatistic, indexDescription, legendGradient, periodForAcquisition, sensorName,
} from "./hubPreview";
import type { ParcelStatistics, PreviewAcquisition, PreviewLayer } from "./hubPreview";
import { useHubPreview } from "./useHubPreview";

/** HUB imagery choices and per-sensor acquisition timeline in the existing map. */
export default function HubPreviewControls({mlId, parcelId, map}: {mlId: string; parcelId: string; map: L.Map}) {
  const view = useHubPreview(mlId, parcelId, map);
  const {metadata, coverage, layer, acquisitions, index, setIndex, error, loading} = view;
  const selected = acquisitions[index];
  const dates = acquisitions.map((item) => ({id: item.id, date: item.datetime, cloud_cover: item.cloud_cover}));
  return (
    <div className="parcel-map-dates hub-preview-controls">
      <Notice message={error} />
      {!metadata && !error && <span role="status">Finding HUB imagery…</span>}
      {metadata && !metadata.available && <span role="status">HUB imagery is not configured on this server.</span>}
      {!!metadata?.layers.length && <>
        <label>Imagery layer{" "}
          <select aria-label="HUB imagery layer" value={layer?.name ?? ""} onChange={(e) => view.selectLayer(e.target.value)}>
            {metadata.layers.map((item) => <option key={item.name} value={item.name}>{item.label ?? item.name} · {sensorName(item.sensor)}</option>)}
          </select>
        </label>
        {!!acquisitions.length && <label className="hub-preview-opacity">Opacity
          <input type="range" min={0} max={100} step={5} value={Math.round(view.opacity * 100)}
            aria-label="HUB preview opacity" aria-valuetext={`${Math.round(view.opacity * 100)}%`}
            onChange={(e) => view.setOpacity(Number(e.target.value) / 100)} />
          <span className="muted">{Math.round(view.opacity * 100)}%</span>
        </label>}
        {!!acquisitions.length && <>
          <select aria-label="HUB acquisition" value={index} onChange={(e) => setIndex(Number(e.target.value))}>
            {acquisitions.map((item, i) => <option key={item.id} value={i}>
              {acquisitionLabel(item)}
              {item.cloud_cover == null ? "" : ` · ${Math.round(item.cloud_cover)}% scene cloud`}
            </option>)}
          </select>
          <span className="muted">{index + 1} of {acquisitions.length}</span>
          <div className="parcel-map-steps">
            <Button disabled={index === 0} onClick={() => setIndex(index - 1)}>← Older</Button>
            <Button disabled={index === acquisitions.length - 1} onClick={() => setIndex(index + 1)}>Newer →</Button>
            <span role="status" aria-live="polite">{loading ? "Loading HUB preview…" : selected && acquisitionLabel(selected)}</span>
          </div>
          <AcquisitionTimeline dates={dates} index={index} startDate={metadata.start_date} endDate={metadata.end_date}
            onChange={setIndex} ariaLabel="HUB acquisition timeline" />
        </>}
        {coverage && !acquisitions.length && <span role="status">No HUB acquisitions cover this parcel for the selected layer.</span>}
        {layer?.legend && <div className="hub-preview-legend" aria-label={`${layer.label ?? layer.name} colour scale`}>
          <strong>{layer.label ?? layer.name}</strong>
          <div className="hub-preview-ramp" style={{background: legendGradient(layer.legend)}} />
          <div className="hub-preview-stops">{layer.legend.stops
            .filter(([value], i, stops) => i === 0 || value !== stops[i - 1][0])
            .map(([value, colour]) => <span key={value} title={colour}
              style={{left: `${((value - layer.legend!.min) / (layer.legend!.max - layer.legend!.min || 1)) * 100}%`}}>{value}</span>)}</div>
          <span className="muted">Fixed display scale; values outside the range use the end colours.</span>
        </div>}
        {layer?.kind === "index" && <IndexInfo layer={layer} statistics={view.statistics} error={view.statisticsError}
          acquisition={selected} />}
        {layer?.sensor === "PL" ? <p className="hub-preview-note muted">
          Planet monthly basemap (PL_BSM) at its native resolution of about 4.8 m, as reference imagery: it is not
          an input of this analysis. True colour uses bands 6, 4, 2 (red, green, blue); NDVI is the basemap’s band 9.
        </p> : <p className="hub-preview-note muted">
          Observed HUB imagery at native source resolution. RGB shows clouds; indices mask cloud and shadow.
          Previews do not include the analysis’s local cleaning, gap filling or temporal aggregation.
          {layer?.sensor === "S1" && layer.kind === "rgb" && " Radar false colour: red = VV, green = VH, blue = VV/VH (dB stretches)."}
        </p>}
        {/* Static per-index chart below the info; it does not follow the selected acquisition. */}
        {layer?.kind === "index" && !!view.statistics?.periods.length && <StatisticsChart statistics={view.statistics} />}
      </>}
    </div>
  );
}

/** What the selected index shows and the parcel's saved statistics for the selected acquisition's period. */
function IndexInfo({layer, statistics, error, acquisition}: {
  layer: PreviewLayer; statistics?: ParcelStatistics; error: string; acquisition?: PreviewAcquisition;
}) {
  const description = indexDescription(layer);
  const period = statistics && acquisition ? periodForAcquisition(statistics.periods, acquisition.datetime) : undefined;
  const shown = period ? STATISTIC_LABELS.filter(([name]) => name in period.values) : [];
  let summary;
  if (layer.sensor === "PL") summary = null;
  else if (error) summary = <span className="muted">Parcel statistics are unavailable: {error}</span>;
  else if (!statistics) summary = <span className="muted">Loading parcel statistics…</span>;
  else if (!statistics.periods.length) summary = <span className="muted">
    The analysis saved no mean, median, SD or range of {layer.name} for this parcel.</span>;
  else if (!acquisition) summary = null;
  else if (!period) summary = <span className="muted">No statistics period of the analysis covers this acquisition.</span>;
  else summary = <>
    <strong>Parcel {layer.name} statistics · period from {period.start}
      {statistics.temporal_period ? ` (${statistics.temporal_period})` : ""}</strong>
    <dl className="hub-preview-stats">{shown.map(([name, label]) => <div key={name}>
      <dt>{label}</dt><dd>{formatStatistic(period.values[name])}</dd>
    </div>)}</dl>
    <span className="muted">From the analysis’s cleaned and filled pixels, so they can differ from this preview.</span>
  </>;
  if (!description && !summary) return null;
  return <div className="hub-preview-index" aria-live="polite">
    {description && <p>{description}</p>}
    {summary}
  </div>;
}
