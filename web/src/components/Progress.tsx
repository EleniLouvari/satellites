import type { ProgressState } from "../api";

/** Percentages describe counted work in the named stage, never remaining time. */
export default function Progress({
  value,
  compact = false,
}: {
  value?: ProgressState;
  compact?: boolean;
}) {
  if (!value) return null;
  // openEO reports remote, local and partition counts; HUB reports batches and partitions.
  const details = [value.remote, value.local, value.partitions].filter(
    (metric): metric is ProgressState => metric != null,
  );
  const metrics = !compact && value.local ? details : [value];
  return (
    <div className={`completion ${compact ? "compact" : ""}`}>
      {metrics.map((metric) => (
        <div key={metric.stage}>
          <div className="completion-label">
            <span>{metric.stage}</span>
            <strong>
              {metric.percent == null ? "—" : `${metric.percent}%`}
            </strong>
          </div>
          {metric.percent != null && (
            <progress
              aria-label={metric.stage}
              value={metric.percent}
              max={100}
            />
          )}
          {!compact && metric.total != null && metric.total > 0 && (
            <small>
              {metric.completed} of {metric.total} completed
            </small>
          )}
        </div>
      ))}
      {!compact && value.local && (
        <small>
          {value.remote ? "Local batches" : "HUB batches"} count saved
          statistics and reports. Partition assembly follows batch processing.
          These counts do not estimate remaining time; results are ready after
          final publication.
        </small>
      )}
    </div>
  );
}
