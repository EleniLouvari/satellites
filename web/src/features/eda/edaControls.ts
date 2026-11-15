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

/** Split a column name into lower-case keyword tokens ("NDVI_median_2023" → ndvi, median, 2023). */
export function columnTokens(name: string): string[] {
  return name.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

/** Keywords shared by at least two columns, most common first; pure numbers (dates) are skipped. */
export function keywordOptions(names: string[]): { keyword: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const name of names)
    for (const token of new Set(columnTokens(name)))
      if (!/^\d+$/.test(token)) counts.set(token, (counts.get(token) ?? 0) + 1);
  return [...counts]
    .filter(([, count]) => count > 1)
    .map(([keyword, count]) => ({ keyword, count }))
    .sort((a, b) => b.count - a.count || a.keyword.localeCompare(b.keyword));
}

/** Keywords match whole name tokens, so "sd" does not match "ndvi" columns by accident. */
export function matchesKeywords(name: string, keywords: string[], mode: "any" | "all"): boolean {
  if (!keywords.length) return true;
  const tokens = new Set(columnTokens(name));
  const hit = (keyword: string) => tokens.has(keyword.toLowerCase());
  return mode === "all" ? keywords.every(hit) : keywords.some(hit);
}
