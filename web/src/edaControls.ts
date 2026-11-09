/** Imported target/ID roles must remain selected and editable in the source table. */
export function includeRoleColumns(
  selected: string[],
  config: Record<string, unknown>,
  available: string[],
): string[] {
  const roles = [config.target_column, config.id_column].filter(
    (value): value is string =>
      typeof value === "string" && available.includes(value),
  );
  return [...new Set([...selected, ...roles])];
}

type ReportState = { eda_id: string; status: string; artifacts: {name: string}[] };

/** Open reports only for observed pending-to-complete transitions, once per run. */
export function newlyCompletedReport(previous: ReportState[], current: ReportState[]): string | undefined {
  const pending = new Set(previous.filter(r => ["queued", "running"].includes(r.status)).map(r => r.eda_id));
  return current.find(r => pending.has(r.eda_id) && r.status === "complete" &&
    r.artifacts.some(f => f.name === "report.html"))?.eda_id;
}
