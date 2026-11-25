import { useState } from "react";
import { chartDomain, chartPoints, chartSegments, chartTicks, formatStatistic } from "./hubPreview";
import type { ParcelStatistics } from "./hubPreview";

const WIDTH = 340;
const HEIGHT = 190;
const MARGIN = {top: 10, right: 12, bottom: 24, left: 42};
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * The parcel's saved mean (line) ± 1 SD (band) of one index for every statistics period.
 * It depends only on the index layer, not on the acquisition shown on the map.
 */
export default function StatisticsChart({statistics}: {statistics: ParcelStatistics}) {
  const [hover, setHover] = useState<number>();
  const points = chartPoints(statistics.periods);
  const domain = chartDomain(points);
  if (!domain) return null;

  const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
  // Periods are evenly spaced (one per reporting interval), so the x axis is the period index.
  const step = points.length > 1 ? plotWidth / (points.length - 1) : plotWidth;
  const x = (i: number) => MARGIN.left + (points.length > 1 ? i * step : plotWidth / 2);
  const y = (value: number) => MARGIN.top + (1 - (value - domain[0]) / (domain[1] - domain[0])) * plotHeight;
  const segments = chartSegments(points);
  const hasSd = points.some((point) => point.sd != null);

  // One label at each month change, thinned so that at most about six are shown.
  const monthStarts = points
    .map((point, i) => ({i, month: point.start.slice(0, 7)}))
    .filter(({month}, k, all) => k === 0 || month !== all[k - 1].month);
  const every = Math.max(1, Math.ceil(monthStarts.length / 6));
  const labels = monthStarts.filter((_, k) => k % every === 0).map(({i, month}, k) => {
    const [year, m] = month.split("-").map(Number);
    return {i, text: k === 0 || m === 1 ? `${MONTHS[m - 1]} ${year}` : MONTHS[m - 1]};
  });

  const hovered = hover == null ? undefined : points[hover];
  const caption = `Parcel ${statistics.variable} mean ± 1 SD per period`;
  return (
    <figure className="hub-preview-chart">
      <figcaption>
        <strong>{caption}</strong>
        {statistics.temporal_period && <span className="muted"> · {statistics.temporal_period}</span>}
        <span className="hub-preview-chart-key muted">
          <i className="line" /> mean{hasSd && <><i className="band" /> ± 1 SD</>}
        </span>
      </figcaption>
      <div className="hub-preview-chart-plot">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={caption} onMouseLeave={() => setHover(undefined)}>
          {chartTicks(domain).map((tick) => (
            <g key={tick} className="grid">
              <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(tick)} y2={y(tick)} />
              <text x={MARGIN.left - 6} y={y(tick)} dy="0.32em" textAnchor="end">{tick}</text>
            </g>
          ))}
          {labels.map(({i, text}) => (
            <text key={i} className="axis" x={x(i)} y={HEIGHT - 6} textAnchor={i === 0 ? "start" : "middle"}>{text}</text>
          ))}
          {segments.map((segment) => {
            const upper = segment.map((i) => `${x(i)},${y(points[i].mean! + (points[i].sd ?? 0))}`);
            const lower = segment.map((i) => `${x(i)},${y(points[i].mean! - (points[i].sd ?? 0))}`).reverse();
            const line = segment.map((i, k) => `${k ? "L" : "M"}${x(i)},${y(points[i].mean!)}`).join("");
            return (
              <g key={segment[0]}>
                {hasSd && segment.length > 1 && <polygon className="band" points={[...upper, ...lower].join(" ")} />}
                {/* A lone period has no band area, so its SD is drawn as a whisker. */}
                {hasSd && segment.length === 1 && <line className="whisker" x1={x(segment[0])} x2={x(segment[0])}
                  y1={y(points[segment[0]].mean! + (points[segment[0]].sd ?? 0))}
                  y2={y(points[segment[0]].mean! - (points[segment[0]].sd ?? 0))} />}
                <path className="mean" d={line} />
              </g>
            );
          })}
          {hovered && <line className="crosshair" x1={x(hover!)} x2={x(hover!)} y1={MARGIN.top} y2={HEIGHT - MARGIN.bottom} />}
          {points.map((point, i) => point.mean != null && (
            <circle key={point.start} className={i === hover ? "point active" : "point"} cx={x(i)} cy={y(point.mean)} r={i === hover ? 4.5 : 3} />
          ))}
          {/* Hit targets span the full period width, wider than the marks themselves. */}
          {points.map((point, i) => (
            <rect key={point.start} className="hit" x={x(i) - step / 2} y={MARGIN.top} width={step} height={plotHeight}
              onMouseEnter={() => setHover(i)} />
          ))}
        </svg>
        {hovered && (
          <div className="hub-preview-chart-tip" role="status"
            style={{left: `${(x(hover!) / WIDTH) * 100}%`, transform: `translateX(${hover! > points.length / 2 ? "-105%" : "5%"})`}}>
            <strong>{hovered.start}</strong>
            <span>Mean {formatStatistic(hovered.mean)}</span>
            {hasSd && <span>SD {formatStatistic(hovered.sd)}</span>}
          </div>
        )}
      </div>
      <details className="hub-preview-chart-table">
        <summary>Show values</summary>
        <table>
          <thead><tr><th>Period from</th><th>Mean</th>{hasSd && <th>SD</th>}</tr></thead>
          <tbody>{points.map((point) => (
            <tr key={point.start}>
              <td>{point.start}</td><td>{formatStatistic(point.mean)}</td>{hasSd && <td>{formatStatistic(point.sd)}</td>}
            </tr>
          ))}</tbody>
        </table>
      </details>
    </figure>
  );
}
