import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

// Test the pure source-selection rules used by the analysis wizard.
const source = fs.readFileSync(new URL("../src/features/parcel-stats/sourceControls.ts", import.meta.url), "utf8");
const code = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022}}).outputText;
const {
  normalizeSource, sourceLabel, accountsForSource, schedulingForSource, sourceReady, processingForSource,
  schemaMatches,
} = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

test("records created before HUB support are openEO", () => {
  assert.equal(normalizeSource(undefined), "openeo");
  assert.equal(normalizeSource("unknown"), "openeo");
  assert.equal(normalizeSource("hub"), "hub");
  assert.equal(sourceLabel("hub"), "HUB Sentinel ARD");
  assert.equal(sourceLabel(null), "Copernicus openEO");
});

test("HUB analyses never send Copernicus accounts and always use batch scheduling", () => {
  assert.deepEqual(accountsForSource("hub", [3, 1]), []);
  assert.deepEqual(accountsForSource("openeo", [3, 1]), [3, 1]);
  assert.equal(schedulingForSource("hub", "partitions"), "batches");
  assert.equal(schedulingForSource("openeo", "partitions"), "partitions");
});

test("the imagery step needs accounts for openEO and a configured server for HUB", () => {
  assert.ok(sourceReady("openeo", [1], false));
  assert.ok(!sourceReady("openeo", [], true));
  assert.ok(sourceReady("hub", [], true));
  assert.ok(!sourceReady("hub", [1, 2], false));
});

test("switching source keeps shared settings, adds new defaults and drops unsupported ones", () => {
  const hubSchema = {
    defaults: {start_date: "2025-01-01", working_epsg: 2100, hub_read_workers: 8},
    fields: [{name: "start_date"}, {name: "working_epsg"}, {name: "hub_read_workers"}],
  };
  const current = {start_date: "2024-06-01", working_epsg: 2100, openeo_parallel_jobs: 2, tile_size_metres: 5000};
  assert.deepEqual(processingForSource(current, hubSchema), {
    start_date: "2024-06-01", working_epsg: 2100, hub_read_workers: 8,
  });
});

test("saving waits for the schema of the selected source", () => {
  assert.ok(!schemaMatches(undefined, "hub"));
  assert.ok(!schemaMatches({source: "openeo", defaults: {}, fields: []}, "hub"));
  assert.ok(schemaMatches({source: "hub", defaults: {}, fields: []}, "hub"));
  // Schemas without a source field come from servers that only offer openEO.
  assert.ok(schemaMatches({defaults: {}, fields: []}, "openeo"));
  assert.ok(!schemaMatches({defaults: {}, fields: []}, "hub"));
});
