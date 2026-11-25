import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import ts from "typescript";

const path = new URL("../src/features/ml/hubPreview.ts", import.meta.url);
const source = fs.existsSync(path) ? fs.readFileSync(path, "utf8") : "export {};";
const code = ts.transpileModule(source, {
  compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022},
}).outputText;
const helpers = await import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);

test("timeline keeps same-day passes distinct while preserving date-only positions", () => {
  assert.equal(typeof helpers.acquisitionTime, "function");
  assert.equal(typeof helpers.nearestAcquisition, "function");
  const times = ["2025-05-01T05:00:00Z", "2025-05-01T17:00:00Z"].map(helpers.acquisitionTime);
  assert.equal(times[1] - times[0], 0.5);
  assert.equal(helpers.nearestAcquisition(times, times[1]), 1);
  assert.equal(helpers.acquisitionTime("2025-05-01"), Date.parse("2025-05-01") / 86_400_000);
});

test("layer selection keeps sensor acquisitions separate and handles no coverage", () => {
  assert.equal(typeof helpers.acquisitionsForLayer, "function");
  const acquisitions = [{id: "optical", sensor: "S2"}, {id: "radar", sensor: "S1"}];
  assert.deepEqual(helpers.acquisitionsForLayer(acquisitions, {sensor: "S1"}), [acquisitions[1]]);
  assert.deepEqual(helpers.acquisitionsForLayer(acquisitions, undefined), []);
  assert.deepEqual(helpers.acquisitionsForLayer([], {sensor: "S2"}), []);
});

test("preview URLs encode identifiers and include public cache revisions", () => {
  assert.equal(typeof helpers.previewImageUrl, "function");
  const url = new URL(helpers.previewImageUrl("ml", "parcel +&", {id: "s2|pass&x", revision: "abc123"}, "S1 false colour"), "https://local");
  assert.equal(url.pathname, "/api/ml/ml/inspection/hub/image.png");
  assert.equal(url.searchParams.get("parcel_id"), "parcel +&");
  assert.equal(url.searchParams.get("acquisition"), "s2|pass&x");
  assert.equal(url.searchParams.get("layer"), "S1 false colour");
  assert.equal(url.searchParams.get("rev"), "abc123");
});

test("legend uses numeric stop positions rather than evenly spacing unequal values", () => {
  assert.equal(typeof helpers.legendGradient, "function");
  assert.equal(helpers.legendGradient({min: -1, max: 1, stops: [[-1, "#000080"], [0, "#ffffff"], [0.1, "#ffff80"], [1, "#006000"]]}),
    "linear-gradient(to right, #000080 0%, #ffffff 50%, #ffff80 55.00000000000001%, #006000 100%)");
});

test("a cancelled image request cannot replace a newer selection even when fetch ignores abort", async () => {
  assert.equal(typeof helpers.beginPreviewLoad, "function");
  const original = globalThis.fetch;
  const requests = [];
  globalThis.fetch = (_url, options) => new Promise((resolve) => requests.push({resolve, signal: options.signal}));
  try {
    const displayed = [];
    const errors = [];
    const cancelOld = helpers.beginPreviewLoad("old", (blob) => displayed.push(blob), (message) => errors.push(message));
    cancelOld();
    helpers.beginPreviewLoad("new", (blob) => displayed.push(blob), (message) => errors.push(message));
    requests[1].resolve({ok: true, blob: async () => "new image"});
    requests[0].resolve({ok: true, blob: async () => "old image"});
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(displayed, ["new image"]);
    assert.deepEqual(errors, []);
    assert.ok(requests[0].signal.aborted);
  } finally {
    globalThis.fetch = original;
  }
});

test("image failures expose the server's safe message and handle non-JSON proxies", async () => {
  assert.equal(typeof helpers.beginPreviewLoad, "function");
  const original = globalThis.fetch;
  try {
    const errors = [];
    globalThis.fetch = async () => ({ok: false, json: async () => ({detail: "Error: Original imagery is unavailable."})});
    helpers.beginPreviewLoad("image", () => assert.fail("No image expected"), (message) => errors.push(message));
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(errors, ["Error: Original imagery is unavailable."]);
    globalThis.fetch = async () => ({ok: false, json: async () => {throw new Error("proxy HTML");}});
    helpers.beginPreviewLoad("image", () => assert.fail("No image expected"), (message) => errors.push(message));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(errors[1], "Error: HUB preview could not be loaded.");
  } finally {
    globalThis.fetch = original;
  }
});

test("Planet basemaps are named, credited and dated by month", () => {
  assert.equal(helpers.sensorName("PL"), "Planet basemap");
  assert.equal(helpers.sensorName("S1"), "Sentinel-1");
  assert.equal(helpers.previewAttribution("PL"), "HUB · Imagery © Planet Labs PBC");
  assert.match(helpers.previewAttribution("S2"), /Copernicus Sentinel/);
  assert.equal(helpers.acquisitionLabel({sensor: "PL", datetime: "2024-04-01T00:00:00Z"}), "April 2024 (monthly)");
  assert.equal(helpers.acquisitionLabel({sensor: "S2", datetime: "2024-04-03T09:15:59Z"}), "2024-04-03 09:15:59 UTC");
  const acquisitions = [{id: "pl|2024-04", sensor: "PL"}, {id: "optical", sensor: "S2"}];
  assert.deepEqual(helpers.acquisitionsForLayer(acquisitions, {sensor: "PL"}), [acquisitions[0]]);
});

test("index descriptions cover analysed indices but not colour composites", () => {
  assert.match(helpers.indexDescription({name: "NDVI", kind: "index"}), /vegetation/);
  for (const name of ["GNDVI", "SAVI", "MSAVI", "EVI", "NDRE", "CI", "NDWI", "MNDWI", "NDMI", "NBR", "PSRI", "R", "RVI", "PL_NDVI"]) {
    assert.equal(typeof helpers.indexDescription({name, kind: "index"}), "string", name);
  }
  assert.equal(helpers.indexDescription({name: "RGB", kind: "rgb"}), undefined);
  assert.equal(helpers.indexDescription(undefined), undefined);
});

test("an acquisition uses the latest statistics period starting on or before its UTC date", () => {
  const periods = [{start: "2025-04-01", values: {}}, {start: "2025-05-01", values: {}}];
  assert.equal(helpers.periodForAcquisition(periods, "2025-04-30T23:59:00Z"), periods[0]);
  assert.equal(helpers.periodForAcquisition(periods, "2025-05-01T10:00:00Z"), periods[1]);
  assert.equal(helpers.periodForAcquisition(periods, "2025-12-31"), periods[1]);
  assert.equal(helpers.periodForAcquisition(periods, "2025-03-31T10:00:00Z"), undefined);
  assert.equal(helpers.periodForAcquisition([], "2025-05-01"), undefined);
});

test("statistics are fetched per encoded layer and formatted with missing values marked", () => {
  const url = new URL(helpers.statisticsUrl("ml", "parcel +&", "NDVI"), "https://local");
  assert.equal(url.pathname, "/ml/ml/inspection/hub/statistics");
  assert.equal(url.searchParams.get("parcel_id"), "parcel +&");
  assert.equal(url.searchParams.get("layer"), "NDVI");
  assert.equal(helpers.formatStatistic(0.61234), "0.612");
  assert.equal(helpers.formatStatistic(null), "—");
  assert.equal(helpers.formatStatistic(undefined), "—");
});

test("chart points keep every period in order and mark missing mean or SD as null", () => {
  const points = helpers.chartPoints([
    {start: "2025-01-01", values: {mean: 0.2, sd: 0.05, median: 0.21}},
    {start: "2025-02-01", values: {mean: null}},
    {start: "2025-03-01", values: {median: 0.4}},
  ]);
  assert.deepEqual(points, [
    {start: "2025-01-01", mean: 0.2, sd: 0.05},
    {start: "2025-02-01", mean: null, sd: null},
    {start: "2025-03-01", mean: null, sd: null},
  ]);
});

test("chart segments break the line at periods without a mean", () => {
  const points = [0.1, 0.2, null, 0.4, null, null, 0.7, 0.8].map((mean, i) => ({start: String(i), mean, sd: null}));
  assert.deepEqual(helpers.chartSegments(points), [[0, 1], [3], [6, 7]]);
  assert.deepEqual(helpers.chartSegments([]), []);
});

test("chart domain covers mean ± SD with padding, also for flat or empty series", () => {
  const [low, high] = helpers.chartDomain([
    {start: "a", mean: 0.5, sd: 0.1}, {start: "b", mean: 0.2, sd: null}, {start: "c", mean: null, sd: 0.3},
  ]);
  assert.ok(low < 0.2 && low > 0.15);
  assert.ok(high > 0.6 && high < 0.65);
  const [flatLow, flatHigh] = helpers.chartDomain([{start: "a", mean: 0, sd: 0}]);
  assert.ok(flatLow < 0 && flatHigh > 0);
  assert.equal(helpers.chartDomain([{start: "a", mean: null, sd: null}]), undefined);
});

test("chart ticks are round values inside the domain", () => {
  assert.deepEqual(helpers.chartTicks([0.13, 0.92]), [0.2, 0.4, 0.6, 0.8]);
  assert.deepEqual(helpers.chartTicks([-1.1, 1.1]), [-1, 0, 1]);
  for (const tick of helpers.chartTicks([-18.4, -6.2])) assert.ok(tick >= -18.4 && tick <= -6.2);
});
