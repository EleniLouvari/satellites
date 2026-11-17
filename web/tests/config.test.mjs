import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

// Test the same pure import/export boundary used by the configuration screen.
const source = fs.readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
const code = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {sanitizeConfig, duration} = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);
const schema = {fields: [{name: "start_date"}, {name: "batch_workers"}, {name: "temporal_fill_window_sizes"}]};

const controlsSource = fs.readFileSync(new URL("../src/components/configControls.ts", import.meta.url), "utf8");
const controlsCode = ts.transpileModule(controlsSource, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {fieldDisabledReason, activeArrayErrors, parseDraft} = await import(`data:text/javascript;base64,${Buffer.from(controlsCode).toString("base64")}`);
const edaSource = fs.readFileSync(new URL("../src/features/eda/edaControls.ts", import.meta.url), "utf8");
const edaCode = ts.transpileModule(edaSource, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {includeRoleColumns, newlyCompletedReport} = await import(`data:text/javascript;base64,${Buffer.from(edaCode).toString("base64")}`);

const mlSource = fs.readFileSync(new URL("../src/features/ml/mlControls.ts", import.meta.url), "utf8");
const mlCode = ts.transpileModule(mlSource, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {stepStates, canContinue, withoutRoles, trainingCounts, canRerun, requiredRetrain} = await import(`data:text/javascript;base64,${Buffer.from(mlCode).toString("base64")}`);

test("ML steps show what is done, where a run stopped, and when it can continue", () => {
  assert.deepEqual(stepStates("paused", 2), ["done", "done", "pending", "pending", "pending"]);
  assert.deepEqual(stepStates("running", 2), ["done", "done", "running", "pending", "pending"]);
  assert.deepEqual(stepStates("queued", 0), ["queued", "pending", "pending", "pending", "pending"]);
  assert.deepEqual(stepStates("interrupted", 3), ["done", "done", "done", "failed", "pending"]);
  assert.deepEqual(stepStates("complete", 5), ["done", "done", "done", "done", "done"]);
  assert.ok(canContinue("paused", 1) && canContinue("failed", 0) && canContinue("interrupted", 4));
  assert.ok(!canContinue("running", 1) && !canContinue("queued", 0) && !canContinue("complete", 5));
});

test("the ML target and ID columns are never kept as features", () => {
  assert.deepEqual(withoutRoles(["NDVI", "crop", "id"], {target_column: "crop", id_column: "id"}), ["NDVI"]);
  assert.deepEqual(withoutRoles(["NDVI"], {target_column: null}), ["NDVI"]);
});

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
  const capped = {name: "distance", enabled_when: {interpolation_method: ["idw", "nearest"]}};
  assert.equal(fieldDisabledReason(capped, {interpolation_method: "kriging"}), "Available with IDW or NEAREST interpolation.");
  assert.equal(fieldDisabledReason(capped, {interpolation_method: "nearest"}), "");
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

test("EDA keyword filter matches whole name parts and lists shared keywords", async () => {
  const {columnTokens, keywordOptions, matchesKeywords} = await import(`data:text/javascript;base64,${Buffer.from(edaCode).toString("base64")}`);
  assert.deepEqual(columnTokens("NDVI_median_2023-05"), ["ndvi", "median", "2023", "05"]);
  const names = ["NDVI_median_2023", "NDVI_sd_2023", "EVI_median_2023", "parcel_id"];
  assert.deepEqual(keywordOptions(names), [{keyword: "median", count: 2}, {keyword: "ndvi", count: 2}]);
  assert.ok(matchesKeywords("NDVI_sd", ["sd"], "any"));
  assert.ok(!matchesKeywords("NDVI_sdx", ["sd"], "any"));
  assert.ok(matchesKeywords("EVI_median", ["ndvi", "evi"], "any"));
  assert.ok(!matchesKeywords("EVI_median", ["ndvi", "median"], "all"));
  assert.ok(matchesKeywords("anything", [], "all"));
});

test("list settings accept plain comma-separated values as well as JSON", () => {
  assert.deepEqual(parseDraft("array", "9999"), [9999]);
  assert.deepEqual(parseDraft("array", "9999, 8888"), [9999, 8888]);
  assert.deepEqual(parseDraft("array", "[9999, 8888]"), [9999, 8888]);
  assert.deepEqual(parseDraft("array", "wheat, 'barley', \"oats\""), ["wheat", "barley", "oats"]);
  assert.deepEqual(parseDraft("array", '"wheat"'), ["wheat"]);
  assert.equal(parseDraft("array", ""), null);
  assert.equal(parseDraft("array", "[9999"), undefined);
  assert.equal(parseDraft("array", "1,,2"), undefined);
  assert.equal(parseDraft("array", '{"a": 1}'), undefined);
  const fields = [{name: "exclude_target_values", type: "array"}];
  assert.deepEqual(activeArrayErrors(fields, {exclude_target_values: "9999, 8888"}, {}), []);
});

test("Train step counts finished, failed and reused models as done, skipping missing libraries", () => {
  const models = [
    {name: "random_forest", status: "finished"},
    {name: "xgboost", status: "running"},
    {name: "lightgbm", status: "failed"},
    {name: "svm", status: "reused"},
    {name: "keras_lstm", status: "unavailable"},
    {name: "knn", status: "pending"},
  ];
  assert.deepEqual(trainingCounts(models), {total: 5, done: 3, failed: 1, unavailable: 1});
  assert.deepEqual(trainingCounts([]), {total: 0, done: 0, failed: 0, unavailable: 0});
});

// Render the real form to check nested grouping without a browser or API server.
async function formModule(path, dependencies = {}) {
  let code = ts.transpileModule(fs.readFileSync(new URL(path, import.meta.url), "utf8"), {
    compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX},
  }).outputText;
  for (const [name, url] of Object.entries({
    react: import.meta.resolve("react"),
    "react/jsx-runtime": import.meta.resolve("react/jsx-runtime"),
    ...dependencies,
  })) code = code.replaceAll(`from "${name}"`, `from "${url}"`);
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}

test("threshold subsections render inside their parent once and retain disabled controls", async () => {
  const {createElement} = await import("react");
  const {renderToStaticMarkup} = await import("react-dom/server");
  const formUrl = await formModule("../src/components/ConfigForm.tsx", {
    "../api": await formModule("../src/api.ts"),
    "./ui": await formModule("../src/components/ui.tsx"),
    "./configControls": await formModule("../src/components/configControls.ts"),
  });
  const {default: ConfigForm} = await import(formUrl);
  const groups = [{id: "inspection", label: "Inspection", description: "", subgroups: [
    {id: "thresholds", label: "Thresholds", description: ["Choose thresholds."]},
    {id: "data_reliability", parent: "thresholds", label: "Data reliability", description: ["Reliability modes."]},
    {id: "geometry", parent: "thresholds", label: "Geometry", description: ["Geometry modes."]},
  ]}];
  const fields = [
    {name: "score", subgroup: "thresholds", default: 25},
    {name: "reliability", subgroup: "data_reliability", default: 0.8},
    {name: "rank", subgroup: "geometry", default: 0.5, enabled_when: {mode: ["percentile"]}},
  ].map((f) => ({...f, group: "inspection", type: "number"}));
  const html = renderToStaticMarkup(createElement(ConfigForm, {
    schema: {groups, fields, defaults: {}}, value: {mode: "absolute"}, onChange() {},
  }));
  assert.match(html, /<h4>Thresholds<\/h4>/);
  assert.equal((html.match(/<h5>Data reliability<\/h5>/g) ?? []).length, 1);
  assert.equal((html.match(/<h5>Geometry<\/h5>/g) ?? []).length, 1);
  assert.match(html, /<h5>Geometry<\/h5>[\s\S]*?disabled=""/);
  assert.ok(html.indexOf("<h4>Thresholds") < html.indexOf("<h5>Data reliability"));
  assert.ok(html.indexOf("<h5>Geometry") < html.indexOf('value="25"'));
  for (const value of ["25", "0.8", "0.5"]) {
    assert.equal((html.match(new RegExp(`value="${value.replace(".", "\\.")}"`, "g")) ?? []).length, 1);
  }
});

test("ML re-runs need the earlier steps and mark the models whose training is stale", () => {
  assert.ok(canRerun("complete", 5, "train") && canRerun("complete", 5, "predict") && canRerun("paused", 2, "train"));
  assert.ok(!canRerun("paused", 2, "predict") && !canRerun("running", 5, "train") && !canRerun("queued", 5, "predict"));
  const rules = {train: [], predict: [], retrain_all_when: ["scoring_primary"], retrain_model_when: {keras_lstm: ["lstm_min_timesteps"]}};
  const previous = {selected_models: ["knn", "bayesian", "keras_lstm"], scoring_primary: "f1_macro", lstm_min_timesteps: 6,
    "tune_params.knn": {model__n_neighbors: [5]}};
  assert.deepEqual([...requiredRetrain(previous, {...previous}, rules)], []);
  assert.deepEqual([...requiredRetrain(previous, {...previous, "tune_params.knn": {model__n_neighbors: [3]}}, rules)], ["knn"]);
  assert.deepEqual([...requiredRetrain(previous, {...previous, lstm_min_timesteps: 3}, rules)], ["keras_lstm"]);
  assert.deepEqual([...requiredRetrain(previous, {...previous, scoring_primary: "accuracy"}, rules)], previous.selected_models);
});
