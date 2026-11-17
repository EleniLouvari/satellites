import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("../", import.meta.url));
const output = path.join(web, "public", "guide");

test("generated guide follows the pipeline and every internal link resolves", async () => {
  const { buildGuide } = await import("../scripts/build-guide.mjs");
  buildGuide();
  const home = fs.readFileSync(path.join(output, "index.html"), "utf8");
  const stages = ["openEO acquisition", "Parcel statistics", "Feature enrichment", "EDA", "ML classification", "Inspection"];
  let previous = -1;
  for (const stage of stages) {
    const position = home.indexOf(`>${stage}</span>`);
    assert.ok(position > previous, `Expected ${stage} in execution order`);
    previous = position;
  }
  const pages = fs.readdirSync(output, { recursive: true }).filter((file) => file.endsWith(".html"));
  assert.ok(pages.length > 25, "The guide must include the existing Markdown articles");
  for (const file of pages) {
    const html = fs.readFileSync(path.join(output, file), "utf8");
    assert.doesNotMatch(html, /<script\b|<[^>]+\son\w+\s*=/i, file);
    for (const [, encoded] of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
      const href = encoded.replaceAll("&amp;", "&");
      if (href === "/" || /^(https?:|mailto:)/.test(href)) continue;
      assert.ok(href.startsWith("/guide/") || href.startsWith("#") || href.startsWith("/help/"), `${file}: ${href}`);
      const url = new URL(href, `http://localhost/guide/${file.replaceAll("\\", "/")}`);
      const target = path.join(web, "public", decodeURIComponent(url.pathname));
      assert.ok(fs.existsSync(target), `${file}: missing ${href}`);
      if (url.hash) {
        const content = fs.readFileSync(target, "utf8");
        assert.ok(content.includes(`id="${decodeURIComponent(url.hash.slice(1))}"`), `${file}: missing anchor ${href}`);
      }
    }
  }
});

test("Markdown rendering escapes HTML and rejects executable URLs", async () => {
  const { renderMarkdown } = await import("../scripts/build-guide.mjs");
  const { html } = renderMarkdown('# Hello\n\n## Same\n\n## Same\n\n<script>alert(1)</script>\n\n[x](javascript:alert)\n\n![x](data:text/html,hi)', "test.md", new Map());
  assert.match(html, /id="same"/);
  assert.match(html, /id="same-1"/);
  assert.doesNotMatch(html, /<script|href="javascript:|src="data:/);
  assert.match(html, /&lt;script&gt;/);
});

test("reference tables retain safe line breaks without permitting HTML attributes", async () => {
  const { renderMarkdown } = await import("../scripts/build-guide.mjs");
  const { html } = renderMarkdown("| Steps |\n| --- |\n| One<br>Two<br />Three<br onclick=alert(1)> |", "test.md", new Map());
  assert.match(html, /One<br>Two<br>Three/);
  assert.doesNotMatch(html, /<br onclick/);
});

test("rebuilding removes retired generated pages but preserves unowned files", async () => {
  const { buildGuide } = await import("../scripts/build-guide.mjs");
  buildGuide();
  const manifest = path.join(output, ".generated.json");
  assert.ok(fs.existsSync(manifest), "Generator must track the files it owns");
  fs.writeFileSync(path.join(output, "retired.html"), "retired generated fixture");
  fs.writeFileSync(path.join(output, "unowned-fixture.txt"), "preserve");
  const owned = JSON.parse(fs.readFileSync(manifest, "utf8"));
  fs.writeFileSync(manifest, JSON.stringify([...owned, "guide/retired.html", "../package.json"]));
  try {
    buildGuide();
    assert.equal(fs.existsSync(path.join(output, "retired.html")), false);
    assert.equal(fs.readFileSync(path.join(output, "unowned-fixture.txt"), "utf8"), "preserve");
    assert.ok(fs.existsSync(path.join(web, "package.json")));
  } finally {
    fs.unlinkSync(path.join(output, "unowned-fixture.txt"));
  }
});
