import type { Schema } from "./api";

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
      if (parent === "interpolation_method") return `Available with ${String(allowed[0]).toUpperCase()} interpolation.`;
      return `Available when ${parent.replaceAll("_", " ")} is ${allowed.join(" or ")}.`;
    }
  }
  return "";
}

/** Unfinished inputs in disabled branches must not block saving the active settings. */
export function activeArrayErrors(fields: Field[], drafts: Record<string, string>, values: Record<string, unknown>): string[] {
  return fields.filter((field) => {
    if (!(field.name in drafts) || fieldDisabledReason(field, values)) return false;
    try {
      const text = drafts[field.name].trim();
      const parsed = text ? JSON.parse(text) : null;
      return parsed !== null && (field.type === "object"
        ? typeof parsed !== "object" || Array.isArray(parsed)
        : !Array.isArray(parsed));
    } catch {
      return true;
    }
  }).map((field) => field.name);
}
