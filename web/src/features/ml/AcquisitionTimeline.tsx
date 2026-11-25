import { acquisitionTime as dayOf, nearestAcquisition } from "./hubPreview";

const DAY_MS = 86_400_000;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Horizontal time axis: dragging the handle snaps to the nearest acquisition date. */
export default function AcquisitionTimeline({
  dates,
  index,
  startDate,
  endDate,
  onChange,
  ariaLabel = "Sentinel-2 image date timeline",
}: {
  dates: {id?: string; date: string; cloud_cover: number | null}[];
  index: number;
  startDate: string;
  endDate: string;
  onChange: (index: number) => void;
  ariaLabel?: string;
}) {
  const days = dates.map((item) => dayOf(item.date));
  const first = Math.min(dayOf(startDate), ...days);
  const last = Math.max(dayOf(endDate), ...days, first + 1);
  const position = (day: number) => `${((day - first) / (last - first)) * 100}%`;

  // Month labels, thinned so that at most about twelve are shown.
  const labels: { day: number; text: string }[] = [];
  const start = new Date(first * DAY_MS);
  const cursor = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + 1, 1));
  const months: Date[] = [];
  while (cursor.getTime() / DAY_MS <= last) {
    months.push(new Date(cursor));
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  const every = Math.max(1, Math.ceil(months.length / 12));
  months.forEach((month, i) => {
    if (i % every) return;
    const m = month.getUTCMonth();
    labels.push({
      day: month.getTime() / DAY_MS,
      text: m === 0 || i === 0 ? `${MONTHS[m]} ${month.getUTCFullYear()}` : MONTHS[m],
    });
  });

  return (
    <div className="parcel-timeline">
      <div className="parcel-timeline-track">
        {dates.map((item, i) => (
          <button
            key={item.id ?? item.date}
            type="button"
            tabIndex={-1}
            className={`parcel-timeline-tick${i === index ? " active" : ""}`}
            style={{ left: position(days[i]) }}
            title={`${item.date}${
              item.cloud_cover == null ? "" : ` · ${Math.round(item.cloud_cover)}% cloud`
            }`}
            onClick={() => onChange(i)}
          />
        ))}
        <input
          type="range"
          aria-label={ariaLabel}
          aria-valuetext={dates[index]?.date}
          min={first}
          max={last}
          step={dates.some((item) => item.date.includes("T")) ? 1 / 86_400 : 1}
          value={days[index] ?? first}
          onChange={(e) => onChange(nearestAcquisition(days, Number(e.target.value)))}
          onKeyDown={(e) => {
            // Arrow keys move one acquisition at a time rather than one day.
            const step =
              e.key === "ArrowLeft" || e.key === "ArrowDown"
                ? -1
                : e.key === "ArrowRight" || e.key === "ArrowUp"
                  ? 1
                  : 0;
            if (!step) return;
            e.preventDefault();
            onChange(Math.min(dates.length - 1, Math.max(0, index + step)));
          }}
        />
      </div>
      <div className="parcel-timeline-labels" aria-hidden="true">
        {labels.map((label) => (
          <span key={label.day} style={{ left: position(label.day) }}>
            {label.text}
          </span>
        ))}
      </div>
    </div>
  );
}
