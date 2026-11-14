import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

// Test the same pure import/export boundary used by the configuration screen.
const source = fs.readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
const code = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {sanitizeConfig, duration} = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);
const schema = {fields: [{name: "start_date"}, {name: "batch_workers"}, {name: "temporal_fill_window_sizes"}]};

const controlsSource = fs.readFileSync(new URL("../src/configControls.ts", import.meta.url), "utf8");
const controlsCode = ts.transpileModule(controlsSource, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {fieldDisabledReason, activeArrayErrors} = await import(`data:text/javascript;base64,${Buffer.from(controlsCode).toString("base64")}`);
const edaSource = fs.readFileSync(new URL("../src/edaControls.ts", import.meta.url), "utf8");
const edaCode = ts.transpileModule(edaSource, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {includeRoleColumns, newlyCompletedReport} = await import(`data:text/javascript;base64,${Buffer.from(edaCode).toString("base64")}`);

test("report auto-opens once on completion, never for old, failed or report-free runs", () => {
  const running = {eda_id: "a", status: "running", artifacts: []};
  const complete = {...running, status: "complete", artifacts: [{name: "report.html"}]};
  assert.equal(newlyCompletedReport([running], [complete]), "a");
  assert.equal(newlyCompletedReport([], [complete]), undefined);
  assert.equal(newlyCompletedReport([complete], [complete]), undefined);
  assert.equal(newlyCompletedReport([running], [{...complete, artifacts: []}]), undefined);
  assert.equal(newlyCompletedReport([running], [{...complete, status: "failed"}]), undefined);
});

test("imported EDA target and ID columns are included without unknown fields or duplicates", () => {
  assert.deepEqual(includeRoleColumns(["NDVI"], {target_column: "crop", id_column: "id"}, ["NDVI", "crop", "id"]), ["NDVI", "crop", "id"]);
  assert.deepEqual(includeRoleColumns(["id"], {id_column: "id", target_column: "foreign"}, ["id", "NDVI"]), ["id"]);
});

test("parent switches and interpolation methods control dependent inputs", () => {
  const distance = {name: "distance", enabled_when: {fill_nulls: [true], interpolation_method: ["idw"]}};
  assert.ok(fieldDisabledReason(distance, {fill_nulls: false, interpolation_method: "idw"}));
  assert.ok(fieldDisabledReason(distance, {fill_nulls: true, interpolation_method: "kriging"}));
  assert.equal(fieldDisabledReason(distance, {fill_nulls: true, interpolation_method: "IDW"}), "");
  assert.equal(fieldDisabledReason({name: "temporal_period"}, {temporal_reducer: "none"}), "");
});

test("invalid arrays block saving only while their controls are active", () => {
  const fields = [{name: "iqr_quantiles", enabled_when: {remove_outliers: [true]}}];
  const drafts = {iqr_quantiles: "[not JSON"};
  assert.deepEqual(activeArrayErrors(fields, drafts, {remove_outliers: false}), []);
  assert.deepEqual(activeArrayErrors(fields, drafts, {remove_outliers: true}), ["iqr_quantiles"]);
  assert.deepEqual(activeArrayErrors(fields, {iqr_quantiles: "[0.25, 0.75]"}, {remove_outliers: true}), []);
});

test("remote resource settings require a JSON object while arrays retain their contract", () => {
  const fields = [{name: "openeo_job_options", type: "object"}];
  assert.deepEqual(activeArrayErrors(fields, {openeo_job_options: '{"executor-memory":"4G"}'}, {}), []);
  assert.deepEqual(activeArrayErrors(fields, {openeo_job_options: "[]"}, {}), ["openeo_job_options"]);
  assert.deepEqual(activeArrayErrors(fields, {openeo_job_options: "42"}, {}), ["openeo_job_options"]);
  assert.deepEqual(activeArrayErrors(fields, {openeo_job_options: "{"}, {}), ["openeo_job_options"]);
  assert.deepEqual(activeArrayErrors(fields, {openeo_job_options: ""}, {}), []);
});

test("import retains supported fields but excludes paths, ownership and secrets", () => {
  const result = sanitizeConfig({start_date: "2024-01-01", batch_workers: 2,
    temporal_fill_window_sizes: [3, 5], output_dir: "/etc", app_user_id: 2,
    password: "never-keep", POSTGRES_HOST: "other-host"}, schema);
  assert.deepEqual(result, {start_date: "2024-01-01", batch_workers: 2, temporal_fill_window_sizes: [3, 5]});
});

test("import rejects non-object request envelopes", () => {
  for (const invalid of [null, [], 42, "settings"]) assert.throws(() => sanitizeConfig(invalid, schema));
});

test("durations show seconds, minutes and hours without inventing missing timestamps", () => {
  assert.equal(duration(null), "—");
  assert.equal(duration(0), "0s");
  assert.equal(duration(125), "2m 5s");
  assert.equal(duration(3661), "1h 1m 1s");
});
