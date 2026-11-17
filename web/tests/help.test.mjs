import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

// Preserve previously published workflow URLs, generating their content from docs/.
// [guide overview, file served from web/public/help/, original under docs/]
const PAGES = [
  ["openeo", "openeo_parcel_statistics_workflow.html", "data_preparation/openeo/WORKFLOW_DIAGRAM.html"],
  ["eda", "eda_workflow.html", "eda/WORKFLOW_DIAGRAM.html"],
  ["ml", "ml_classification_workflow.html", "ml_classification/WORKFLOW_DIAGRAM.html"],
  ["inspection", "inspection_priority_workflow.html", "ml_classification/INSPECTION_WORKFLOW_DIAGRAM.html"],
];
const served = (name) => fs.readFileSync(new URL(`../public/help/${name}`, import.meta.url), "utf8");

for (const [stage, file, original] of PAGES) {
  test(`${file} is an exact copy of docs/${original}`, () => {
    assert.equal(served(file), fs.readFileSync(new URL(`../../docs/${original}`, import.meta.url), "utf8"));
  });

  test(`${file} loads nothing the Parcel Studio CSP would block`, () => {
    const page = served(file);
    // default-src 'self' and script-src 'self': no scripts, no remote fonts, stylesheets or images.
    assert.doesNotMatch(page, /<script\b/i);
    assert.doesNotMatch(page, /<link\b/i);
    assert.doesNotMatch(page, /(src|href)\s*=\s*["']?(https?:)?\/\//i);
    assert.doesNotMatch(page, /url\(\s*["']?(https?:)?\/\//i);
  });

  test(`${stage} overview links to its workflow in a separate tab`, () => {
    const guide = fs.readFileSync(new URL(`../public/guide/${stage}.html`, import.meta.url), "utf8");
    assert.ok(guide.includes(`href="/guide/assets/${original}" target="_blank" rel="noopener"`));
  });
}
