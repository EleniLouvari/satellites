import type { Schema } from "../api";

type Field = Schema["fields"][number];

/** Describe the first inactive parent without changing the user's draft values. */
export function fieldDisabledReason(field: Field, values: Record<string, unknown>): string {
  for (const [parent, allowed] of Object.entries(field.enabled_when ?? {})) {
    let current = values[parent];
    if (typeof current === "string") {
      current = current.trim().toLowerCase();
      if (current === "inverse_distance_weighting") current = "idw";
      if (current === "nearest_neighbor") current = "nearest";
    }
    if (!allowed.includes(current)) {
      if (parent === "remove_outliers") return "Enable outlier removal to use this setting.";
      if (parent === "fill_nulls") return "Enable gap filling to use this setting.";
      if (parent === "interpolation_method") return `Available with ${allowed.map((value) => String(value).toUpperCase()).join(" or ")} interpolation.`;
      return `Available when ${parent.replaceAll("_", " ")} is ${allowed.join(" or ")}.`;
    }
  }
  return "";
}

/** Read one comma-separated list item: numbers, booleans and quoted text keep their JSON type. */
function listItem(token: string): unknown {
  try {
    return JSON.parse(token);
  } catch {
    return token.replace(/^'(.*)'$/, "$1");
  }
}

/**
 * Parse the text of a JSON setting. Lists also accept plain comma-separated values,
 * so "9999" and "9999, 8888" mean [9999] and [9999, 8888]. Returns undefined when invalid.
 */
export function parseDraft(type: string, raw: string): unknown {
  const text = raw.trim();
  if (!text) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    // Unbalanced JSON is a typing mistake, not a list of text values.
    if (type === "object" || /^[[{]/.test(text)) return undefined;
    parsed = text;
  }
  if (type === "object") return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : undefined;
  if (parsed === null || Array.isArray(parsed)) return parsed;
  if (parsed !== null && typeof parsed === "object") return undefined;
  const tokens = typeof parsed === "string" && parsed !== text ? [JSON.stringify(parsed)] : text.split(",").map((t) => t.trim());
  return tokens.every(Boolean) ? tokens.map(listItem) : undefined;
}

/** Unfinished inputs in disabled branches must not block saving the active settings. */
export function activeArrayErrors(fields: Field[], drafts: Record<string, string>, values: Record<string, unknown>): string[] {
  return fields
    .filter((field) => field.name in drafts && !fieldDisabledReason(field, values) && parseDraft(field.type, drafts[field.name]) === undefined)
    .map((field) => field.name);
}
