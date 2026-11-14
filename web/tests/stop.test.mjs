import test, { after } from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
after(async () => server.close());
const { default: StopAction } = await server.ssrLoadModule("/src/StopAction.tsx");
const render = (status, stop_requested = false) => renderToStaticMarkup(
  React.createElement(StopAction, { run: { run_id: "test", name: "Test", status, stop_requested }, onRequested() {} }),
);

test("running analysis exposes Stop analysis", () => {
  const html = render("running");
  assert.match(html, /Stop analysis/);
  assert.doesNotMatch(html, /disabled/);
});

test("requested stop disables duplicate requests while waiting", () => {
  const html = render("running", true);
  assert.match(html, /Stopping/);
  assert.match(html, /disabled/);
});

test("queued and terminal analyses have no active stop action", () => {
  for (const status of ["draft", "queued", "complete", "failed", "interrupted", "cancelled"]) {
    assert.equal(render(status), "");
  }
});
