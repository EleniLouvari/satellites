/** Build the public pipeline guide from explicitly cataloged repository documents. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Marked } from "marked";
import GithubSlugger from "github-slugger";

const root = fileURLToPath(new URL("../../", import.meta.url));
const publicRoot = path.join(root, "web/public");
const guideRoot = path.join(publicRoot, "guide");

/** Read a cataloged source on Windows checkouts that retained an older filename case. */
function readSource(source) {
  let current = root;
  for (const segment of source.split("/")) {
    if (!segment || segment === "." || segment === "..") throw new Error(`Error: Invalid guide source ${source}`);
    const names = fs.readdirSync(current);
    const exact = names.find((name) => name === segment);
    const matches = exact ? [exact] : names.filter((name) => name.toLowerCase() === segment.toLowerCase());
    if (matches.length !== 1) throw new Error(`Error: Missing or ambiguous guide source ${source}`);
    current = path.join(current, matches[0]);
  }
  return fs.readFileSync(current);
}

/** Escape text and attribute values before inserting them into generated markup. */
function escape(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

/** Return headings using the same slug rules for the outline and rendered article. */
function headings(markdown) {
  const slugger = new GithubSlugger();
  const result = [];
  const parser = new Marked();
  parser.walkTokens(parser.lexer(markdown), (token) => {
    if (token.type === "heading") result.push({ id: slugger.slug(token.text), title: token.text, depth: token.depth });
  });
  return result;
}

/** Resolve only cataloged local references, safe external links and known heading anchors. */
function resolveLink(href, source, entries, outline) {
  if (/^(https?:\/\/|mailto:)/i.test(href)) return href;
  if (href === "/" || href.startsWith("/guide/")) return href;
  if (/^[a-z][a-z\d+.-]*:|^\/\//i.test(href)) return null;
  let decoded;
  try { decoded = decodeURIComponent(href); } catch { return null; }
  const [file, fragment] = decoded.split("#");
  const entry = file
    ? entries.get(path.posix.normalize(path.posix.join(path.posix.dirname(source), file)).toLowerCase())
    : { href: "", outline };
  if (!entry) return null;
  // Old documents sometimes name headings that have since changed. Keep the article reachable.
  const anchor = fragment && entry.outline?.some((heading) => heading.id === fragment) ? `#${fragment}` : "";
  return entry.href + anchor || "#article";
}

/** Render trusted repository Markdown while excluding raw HTML and executable URL schemes. */
export function renderMarkdown(markdown, source, entries) {
  const outline = headings(markdown);
  let headingIndex = 0;
  const parser = new Marked({ gfm: true, renderer: {
    heading(token) {
      const heading = outline[headingIndex++];
      return `<h${token.depth} id="${escape(heading.id)}">${this.parser.parseInline(token.tokens)}</h${token.depth}>\n`;
    },
    html(token) { return /^<br\s*\/?>$/i.test(token.text) ? "<br>" : escape(token.text); },
    link(token) {
      const label = this.parser.parseInline(token.tokens);
      const href = resolveLink(token.href, source, entries, outline);
      return href ? `<a href="${escape(href)}">${label}</a>`
        : `<span class="repository-reference" title="${escape(token.href)}">${label} <small>(repository reference)</small></span>`;
    },
    image(token) {
      const href = resolveLink(token.href, source, entries, outline);
      if (!href?.startsWith("/guide/assets/")) return `<span>${escape(token.text)} (see repository diagram)</span>`;
      return `<a href="${escape(href)}" target="_blank" rel="noopener"><img src="${escape(href)}"
        alt="${escape(token.text)}" loading="lazy"></a>`;
    },
    table(token) {
      const cell = (item, tag) => `<${tag}>${this.parser.parseInline(item.tokens)}</${tag}>`;
      return `<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable reference table"><table>
        <thead><tr>${token.header.map((item) => cell(item, "th")).join("")}</tr></thead>
        <tbody>${token.rows.map((row) => `<tr>${row.map((item) => cell(item, "td")).join("")}</tr>`).join("")}</tbody>
        </table></div>`;
    },
  } });
  return { html: parser.parse(markdown), outline };
}

/** Write a generated public file, creating its parent directory if necessary. */
function write(relative, content) {
  const destination = path.join(publicRoot, relative);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, content);
}

/** Generate the contents sidebar, expanding only the section being read. */
function sidebar(sections, current) {
  return sections.map((section, index) => {
    const active = current.section === section.id;
    const number = index === 0 ? "00" : index === sections.length - 1 ? "+" : `0${index}`;
    return `<details${active ? " open" : ""}><summary><b>${number}</b><span>${escape(section.title)}</span></summary>
      <a href="/guide/${section.id}.html"${current.id === section.id ? ' aria-current="page"' : ""}>Stage overview</a>
      ${section.articles.map(([id, title]) => `<a href="/guide/${id}.html"${current.id === id ? ' aria-current="page"' : ""}>
      ${escape(title)}</a>`).join("")}</details>`;
  }).join("");
}

/** Render the common document shell and sequential reading navigation. */
function page(sections, sequence, current, body, outline = []) {
  const position = sequence.findIndex((item) => item.id === current.id);
  const previous = sequence[position - 1];
  const next = sequence[position + 1];
  const navLink = (item, label) => item ? `<a href="/guide/${item.id}.html"><small>${label}</small>${escape(item.title)}</a>` : "<span></span>";
  return `<!doctype html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(current.title)} · Parcel Studio guide</title><link rel="stylesheet" href="/guide/guide.css"></head>
<body><a class="skip-link" href="#article">Skip to article</a>
<header class="guide-header"><a class="brand" href="/guide/index.html"><b aria-hidden="true">P</b><span>Parcel Studio<small>PIPELINE GUIDE</small></span></a>
<a class="workspace-link" href="/">Open workspace ↗</a></header>
<div class="guide-layout"><aside class="contents"><nav aria-label="Pipeline contents"><p class="eyebrow">CONTENTS · EXECUTION ORDER</p>
${sidebar(sections, current)}</nav></aside>
<main id="article" tabindex="-1"><div class="breadcrumb"><a href="/guide/index.html">Pipeline guide</a> / ${escape(current.title)}</div>
${body}<nav class="reading-nav" aria-label="Reading order">${navLink(previous, "← Previous")}${navLink(next, "Next →")}</nav>
<footer>Parcel Studio · Satellite observations. Grounded insight.</footer></main>
${outline.length ? `<aside class="on-page"><nav aria-label="On this page"><p class="eyebrow">ON THIS PAGE</p>${outline.filter((h) => h.depth === 2)
    .map((h) => `<a href="#${escape(h.id)}">${escape(h.title)}</a>`).join("")}</nav></aside>` : ""}
</div></body></html>`;
}

/** Render a pipeline overview with numbered steps and its ordered resource list. */
function overview(section, sections, entries) {
  const stepList = section.steps.map(([title, description], index) => `<li><b>${index + 1}</b><div><h3>${escape(title)}</h3>
    <p>${escape(description)}</p></div></li>`).join("");
  const workflow = section.id === "index" ? `<div class="journey">${sections.slice(1, -1).map((item, index) =>
    `<a href="/guide/${item.id}.html"><b>0${index + 1}</b><span>${escape(item.title)}</span><small>${escape(item.description)}</small></a>`).join("")}</div>` : "";
  const assets = (section.assets ?? []).map(([title, source]) => {
    const href = entries.get(source.toLowerCase()).href;
    return source.endsWith(".html")
      ? `<a class="workflow-link" href="${escape(href)}" target="_blank" rel="noopener">${escape(title)} ↗<small>HTML · opens in a new tab</small></a>`
      : `<figure><a href="${escape(href)}" target="_blank" rel="noopener"><img src="${escape(href)}" alt="${escape(title)}" loading="lazy"></a>
        <figcaption>${escape(title)} · select to open full size</figcaption></figure>`;
  }).join("");
  return `<p class="eyebrow">${section.id === "reference" ? "OPTIONAL TECHNICAL READING" : "UNDERSTAND THE PIPELINE"}</p>
    <h1>${escape(section.title)}</h1><p class="lead">${escape(section.description)}</p>${workflow}
    <dl class="handoff"><div><dt>Inputs</dt><dd>${escape(section.input)}</dd></div><div><dt>Outputs</dt><dd>${escape(section.output)}</dd></div></dl>
    ${stepList ? `<h2 id="execution">Processing order</h2><ol class="execution">${stepList}</ol>` : ""}
    <h2 id="reading">Read in order</h2><ol class="reading-list">${section.articles.map(([id, title], index) =>
      `<li><a href="/guide/${id}.html"><span>${String(index + 1).padStart(2, "0")}</span>${escape(title)}<b aria-hidden="true">→</b></a></li>`).join("")}</ol>
    ${assets ? `<h2 id="diagrams">Workflow diagrams</h2>${assets}` : ""}`;
}

/** Generate all cataloged articles, assets and backward-compatible HTML workflow URLs. */
export function buildGuide() {
  const sections = JSON.parse(fs.readFileSync(path.join(root, "docs/guide/contents.json"), "utf8"));
  const manifest = path.join(guideRoot, ".generated.json");
  const previousFiles = fs.existsSync(manifest) ? JSON.parse(fs.readFileSync(manifest, "utf8")) : [];
  const generated = new Set();
  const emit = (relative, content) => { write(relative, content); generated.add(relative); };
  const entries = new Map();
  const sequence = [];
  for (const section of sections) {
    sequence.push({ id: section.id, section: section.id, title: section.title });
    for (const [id, title, source] of section.articles) {
      const markdown = readSource(source).toString("utf8");
      const entry = { id, section: section.id, title, source, markdown, href: `/guide/${id}.html`, outline: headings(markdown) };
      if (sequence.some((item) => item.id === id)) throw new Error(`Error: Duplicate guide page ${id}`);
      entries.set(source.toLowerCase(), entry);
      sequence.push(entry);
    }
    for (const [, source, legacy] of section.assets ?? []) {
      const href = `/guide/assets/${source.replace(/^docs\//, "")}`;
      const content = readSource(source);
      if (source.endsWith(".html") && /<script\b|<link\b|\son\w+\s*=|(?:src|href)\s*=\s*["']?(?:https?:)?\/\//i.test(content.toString())) {
        throw new Error(`Error: Workflow asset requires scripts or external resources: ${source}`);
      }
      entries.set(source.toLowerCase(), { href });
      emit(href.slice(1), content);
      if (legacy) emit(`help/${legacy}`, content);
    }
  }
  for (const section of sections) {
    const current = sequence.find((item) => item.id === section.id);
    const outline = [{ id: "reading", title: "Read in order", depth: 2 }];
    if (section.steps.length) outline.unshift({ id: "execution", title: "Processing order", depth: 2 });
    if (section.assets?.length) outline.push({ id: "diagrams", title: "Workflow diagrams", depth: 2 });
    emit(`guide/${section.id}.html`, page(sections, sequence, current, overview(section, sections, entries), outline));
    for (const [, , source] of section.articles) {
      const entry = entries.get(source.toLowerCase());
      const rendered = renderMarkdown(entry.markdown, source, entries);
      emit(`guide/${entry.id}.html`, page(sections, sequence, entry,
        `<a class="stage-return" href="/guide/${section.id}.html">← ${escape(section.title)} overview</a>
        <article class="markdown">${rendered.html}</article>`, rendered.outline));
    }
  }
  emit("guide/guide.css", fs.readFileSync(new URL("./guide.css", import.meta.url)));
  // Retire only files recorded by this generator, after the replacement build succeeds.
  // Legacy /help URLs remain available; source documents and unowned files are never removed.
  for (const relative of previousFiles) {
    if (typeof relative !== "string" || !relative.startsWith("guide/") || generated.has(relative)) continue;
    const target = path.resolve(publicRoot, relative);
    if (!target.startsWith(`${guideRoot}${path.sep}`)) continue;
    if (fs.existsSync(target) && fs.lstatSync(target).isFile()) fs.unlinkSync(target);
  }
  fs.writeFileSync(manifest, JSON.stringify([...generated], null, 2));
  return sequence.length;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(`Built ${buildGuide()} pipeline guide pages from repository documentation.`);
}
