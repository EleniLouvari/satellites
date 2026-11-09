export type User = { username: string; csrf_token: string };
export type ProgressState = {
  stage: string;
  percent: number | null;
  completed?: number;
  total?: number;
  remote?: ProgressState;
  local?: ProgressState;
  partitions?: ProgressState;
};
export type Account = {
  eo_user_id: number;
  username: string;
  label: string;
  test_status: string;
  tested_at?: string;
};
export type Layer = {
  name: string | null;
  columns: string[];
  parcel_count: number;
  crs: string;
  bounds: number[];
  preview: unknown;
  geometry_invalid_count: number;
  geometry_empty_count: number;
};
export type Upload = {
  upload_id: string;
  filename: string;
  status: string;
  size_bytes: number;
  metadata: { layers?: Layer[] };
  error?: string;
};
export type Run = {
  run_id: string;
  scheduling: string;
  name: string;
  upload_id: string;
  layer: string | null;
  parcel_id_field: string;
  partition_field: string | null;
  processing: Record<string, unknown>;
  account_ids: number[];
  status: string;
  revision: number;
  validated_revision: number | null;
  summary: Record<string, unknown>;
  config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  duration_seconds?: number | null;
  progress?: ProgressState;
  error?: string;
};
export type Schema = {
  defaults: Record<string, unknown>;
  groups?: { id: string; label: string; description: string }[];
  fields: {
    name: string;
    type: string;
    default: unknown;
    choices?: unknown[];
    description?: string;
    group?: string;
    enabled_when?: Record<string, unknown[]>;
  }[];
  limits: { upload_limit_bytes: number; max_batch_workers: number };
};
export type Event = {
  event_id: number;
  stage: string;
  level: string;
  message: string;
  created_at: string;
};
export type Artifact = {
  artifact_id: string;
  name: string;
  kind: string;
  size_bytes: number;
  created_at: string;
};
let csrf = "";

export function duration(seconds?: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60) % 60;
  const hours = Math.floor(total / 3600);
  return hours
    ? `${hours}h ${minutes}m ${total % 60}s`
    : `${minutes}m ${total % 60}s`;
}
export function setCsrf(value: string) {
  csrf = value;
}
export async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (method !== "GET") {
    if (csrf) headers["X-CSRF-Token"] = csrf;
    headers["Content-Type"] =
      body instanceof File ? "application/octet-stream" : "application/json";
  }
  const response = await fetch(`/api${path}`, {
    method,
    credentials: "same-origin",
    headers,
    body:
      body === undefined
        ? undefined
        : body instanceof File
          ? body
          : JSON.stringify(body),
  });
  if (!response.ok) {
    if (response.status === 401)
      window.dispatchEvent(new CustomEvent("session-expired"));
    let detail = "Error: Request could not be completed.";
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* Generic safe fallback for proxy failures. */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}
export const date = (value?: string) =>
  value ? new Date(value).toLocaleString() : "—";
export const bytes = (value: number) =>
  value >= 1048576
    ? `${(value / 1048576).toFixed(1)} MiB`
    : `${(value / 1024).toFixed(1)} KiB`;
export function sanitizeConfig(
  value: unknown,
  schema: Schema,
): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Error: Import a JSON object.");
  const allowed = new Set(schema.fields.map((field) => field.name));
  return Object.fromEntries(
    Object.entries(value).filter(([key]) => allowed.has(key)),
  );
}
export function downloadJson(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
