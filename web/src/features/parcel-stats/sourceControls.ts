import type { Schema } from "../../api";

/** Imagery providers a parcel-statistics analysis can read from. */
export type Source = "openeo" | "hub";

export const SOURCES: { id: Source; label: string; description: string }[] = [
  {
    id: "openeo",
    label: "Copernicus openEO",
    description: "Remote jobs on Copernicus Data Space with your own accounts.",
  },
  {
    id: "hub",
    label: "HUB Sentinel ARD",
    description: "Analysis-ready Sentinel-2/Sentinel-1 from the HUB catalog; no accounts needed.",
  },
];

/** Return a known source, defaulting older records to openEO. */
export function normalizeSource(value: unknown): Source {
  return value === "hub" ? "hub" : "openeo";
}

/** Return the display label of a source. */
export function sourceLabel(value: unknown): string {
  const source = normalizeSource(value);
  return SOURCES.find((item) => item.id === source)!.label;
}

/** HUB runs never carry Copernicus accounts; openEO keeps the selection order. */
export function accountsForSource(source: Source, selected: number[]): number[] {
  return source === "hub" ? [] : selected;
}

/** HUB runs always use batch scheduling; openEO keeps the user's choice. */
export function schedulingForSource(source: Source, scheduling: string): string {
  return source === "hub" ? "batches" : scheduling;
}

/** Whether the imagery step is complete: accounts for openEO, a configured server for HUB. */
export function sourceReady(source: Source, selected: number[], hubConfigured: boolean): boolean {
  return source === "hub" ? hubConfigured : selected.length > 0;
}

/** A processing schema is usable only once it describes the selected source; older servers imply openEO. */
export function schemaMatches(schema: Schema | undefined, source: Source): boolean {
  return !!schema && normalizeSource(schema.source) === source;
}

/**
 * Keep the user's values for settings both sources share, use the new schema's
 * defaults for the rest, and drop settings the new source does not accept.
 */
export function processingForSource(
  current: Record<string, unknown>,
  schema: Schema,
): Record<string, unknown> {
  const names = schema.fields.map((field) => field.name);
  return Object.fromEntries(
    names.map((name) => [name, name in current ? current[name] : schema.defaults[name]]),
  );
}
