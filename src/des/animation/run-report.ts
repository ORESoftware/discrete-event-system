'use strict';

// RUST MIGRATION:
// - Target: src/des/animation/run_report.rs
// - Convert MetricRow, ReportSection, IndexEntry, IndexGroup, CatalogEntry, and CatalogSection to serde-friendly structs.
// - RunReportPage and SimulationIndexPage become builder structs with inherent add_* methods and Result<String, ReportRenderError> render methods.
// - HTML/string builders should use template/writer helpers; keep escape as a private helper or small trait-free utility.
// - Preserve relative-link behavior and avoid global state so the module ports cleanly to Rust ownership.

// =============================================================================
// RUST MIGRATION  —  target: src/des/animation/run-report.rs   (module des::animation::run_report)
// 1:1 file move. Builder classes that emit styled HTML report / index pages.
//
// Declarations → Rust:
//   interface MetricRow / ReportSection / IndexEntry / IndexGroup /
//             CatalogEntry / CatalogSection   -> struct (Option<T> for `?` fields)
//   class RunReportPage                       -> struct + impl (builder)
//   class SimulationIndexPage                 -> struct + impl (builder)
//
// Conversion notes (file-specific):
//   - `static escape(input)` -> an associated fn `fn escape(input: &str) -> String`
//     (chained `.replace(..)` map straight to `str::replace`).
//   - Builder methods returning `this` (`addSection`/`addGroup`/`addCatalog`)
//     -> return `&mut self` (or `Self`) for chaining.
//   - HTML produced via backtick template literals -> `format!` / `write!` into a
//     `String` (or a templating crate); keep markup identical.
//   - `private readonly` fields -> private struct fields set in a constructor fn.
// =============================================================================

// =============================================================================
// run-report.ts — class-only HTML builders for non-animation simulation runs.
//
//   • `RunReportPage`     — a styled single-run report (header, metric tables,
//                           captured console output) written into `out/`.
//   • `SimulationIndexPage` — the curated landing page that lists every
//                           simulation/run and links to its HTML.
//
// Pages use RELATIVE links and a dark theme that matches the animation player,
// so they render identically when served locally from `out/` or through the
// cluster gateway at `/des/out/`.
// =============================================================================

export interface MetricRow {
  label: string;
  value: string;
}

export interface ReportSection {
  heading: string;
  description?: string;
  metrics?: MetricRow[];
  /** Monospaced block (e.g. captured stdout). Rendered verbatim. */
  log?: string;
}

export class RunReportPage {
  private readonly sections: ReportSection[] = [];

  constructor(
    private readonly title: string,
    private readonly subtitle: string,
    private readonly backHref = '../index.html',
  ) {}

  addSection(section: ReportSection): this {
    this.sections.push(section);
    return this;
  }

  static escape(input: string): string {
    return input
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  private renderSection(s: ReportSection): string {
    const parts: string[] = [`<section><h2>${RunReportPage.escape(s.heading)}</h2>`];
    if (s.description) parts.push(`<p class="desc">${RunReportPage.escape(s.description)}</p>`);
    if (s.metrics && s.metrics.length) {
      parts.push('<table><tbody>');
      for (const m of s.metrics) {
        parts.push(`<tr><th>${RunReportPage.escape(m.label)}</th><td>${RunReportPage.escape(m.value)}</td></tr>`);
      }
      parts.push('</tbody></table>');
    }
    if (s.log) parts.push(`<pre class="log">${RunReportPage.escape(s.log)}</pre>`);
    parts.push('</section>');
    return parts.join('');
  }

  toHtml(): string {
    const body = this.sections.map(s => this.renderSection(s)).join('\n');
    return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${RunReportPage.escape(this.title)}</title>
<style>
:root{color-scheme:dark;}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:#0b1021;color:#e6edf3;}
main{max-width:960px;margin:0 auto;padding:28px 20px 72px;}
a.back{color:#58a6ff;text-decoration:none;font-size:.9rem;}
a.back:hover{text-decoration:underline;}
h1{font-size:1.7rem;margin:14px 0 4px;}
p.sub{color:#8b949e;margin:0 0 26px;font-size:.95rem;}
section{background:#161d33;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin:0 0 20px;}
h2{font-size:1.15rem;margin:0 0 8px;color:#f0f6fc;}
p.desc{color:#9aa5b1;margin:0 0 14px;font-size:.92rem;}
table{border-collapse:collapse;width:100%;margin:0 0 4px;}
th{text-align:left;color:#8b949e;font-weight:600;font-size:.82rem;padding:6px 12px 6px 0;white-space:nowrap;vertical-align:top;}
td{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86rem;color:#e6edf3;padding:6px 0;}
pre.log{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 16px;overflow:auto;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;line-height:1.5;color:#c9d1d9;}
</style></head><body><main>
<a class="back" href="${RunReportPage.escape(this.backHref)}">&larr; all simulations</a>
<h1>${RunReportPage.escape(this.title)}</h1>
<p class="sub">${RunReportPage.escape(this.subtitle)}</p>
${body}
</main></body></html>`;
  }
}

export interface IndexEntry {
  title: string;
  description: string;
  href: string;
  /** Short tag shown on the card, e.g. "animation" or "run report". */
  kind: string;
}

export interface IndexGroup {
  heading: string;
  blurb: string;
  entries: IndexEntry[];
}

/** A compact link in the full directory catalog. */
export interface CatalogEntry {
  href: string;
  label: string;
  size?: string;
}

export interface CatalogSection {
  heading: string;
  blurb: string;
  entries: CatalogEntry[];
}

export class SimulationIndexPage {
  private readonly groups: IndexGroup[] = [];
  private readonly catalogs: CatalogSection[] = [];

  constructor(private readonly title: string, private readonly subtitle: string) {}

  addGroup(group: IndexGroup): this {
    this.groups.push(group);
    return this;
  }

  addCatalog(section: CatalogSection): this {
    this.catalogs.push(section);
    return this;
  }

  private renderEntry(e: IndexEntry): string {
    const esc = RunReportPage.escape;
    return `<a class="card" href="${esc(e.href)}">
<span class="tag">${esc(e.kind)}</span>
<span class="card-title">${esc(e.title)}</span>
<span class="card-desc">${esc(e.description)}</span>
</a>`;
  }

  private renderGroup(g: IndexGroup): string {
    const esc = RunReportPage.escape;
    return `<section><h2>${esc(g.heading)}</h2><p class="blurb">${esc(g.blurb)}</p>
<div class="grid">${g.entries.map(e => this.renderEntry(e)).join('')}</div></section>`;
  }

  private renderCatalog(c: CatalogSection): string {
    const esc = RunReportPage.escape;
    const rows = c.entries.map(e =>
      `<li><a href="${esc(e.href)}"><span class="path">${esc(e.label)}</span>` +
      `${e.size ? `<span class="size">${esc(e.size)}</span>` : ''}</a></li>`).join('');
    return `<section><h2>${esc(c.heading)} <span class="count">${c.entries.length}</span></h2>
<p class="blurb">${esc(c.blurb)}</p><ul class="catalog">${rows}</ul></section>`;
  }

  toHtml(generatedAt: string): string {
    const esc = RunReportPage.escape;
    const body = this.groups.map(g => this.renderGroup(g)).join('\n')
      + '\n' + this.catalogs.map(c => this.renderCatalog(c)).join('\n');
    return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(this.title)}</title>
<style>
:root{color-scheme:dark;}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:#0b1021;color:#e6edf3;}
main{max-width:1040px;margin:0 auto;padding:36px 22px 80px;}
h1{font-size:2rem;margin:0 0 6px;}
p.sub{color:#8b949e;margin:0 0 32px;font-size:1rem;}
section{margin:0 0 34px;}
h2{font-size:1.25rem;margin:0 0 4px;color:#f0f6fc;}
p.blurb{color:#9aa5b1;margin:0 0 16px;font-size:.92rem;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;}
a.card{display:flex;flex-direction:column;gap:6px;background:#161d33;border:1px solid #21262d;
border-radius:12px;padding:16px 18px;text-decoration:none;transition:border-color .15s,transform .15s;}
a.card:hover{border-color:#58a6ff;transform:translateY(-2px);}
.tag{align-self:flex-start;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:#0b1021;
background:#58a6ff;border-radius:999px;padding:2px 9px;font-weight:700;}
.card-title{color:#f0f6fc;font-size:1.05rem;font-weight:600;}
.card-desc{color:#9aa5b1;font-size:.86rem;line-height:1.4;}
.count{display:inline-block;font-size:.72rem;color:#0b1021;background:#7d8590;border-radius:999px;
padding:1px 8px;vertical-align:middle;font-weight:700;}
ul.catalog{list-style:none;padding:0;margin:0;column-width:330px;column-gap:14px;}
ul.catalog li{break-inside:avoid;margin:0 0 4px;}
ul.catalog a{display:flex;justify-content:space-between;gap:10px;align-items:baseline;
padding:6px 10px;border:1px solid #21262d;border-radius:7px;text-decoration:none;background:#11172b;}
ul.catalog a:hover{border-color:#58a6ff;background:#161d33;}
ul.catalog .path{color:#58a6ff;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
ul.catalog .size{color:#586069;font-size:.72rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;}
footer{color:#586069;font-size:.8rem;margin-top:20px;}
footer code{color:#8b949e;}
</style></head><body><main>
<h1>${esc(this.title)}</h1>
<p class="sub">${esc(this.subtitle)}</p>
${body}
<footer>Generated ${esc(generatedAt)} · served from the discrete-event-system <code>out/</code> directory.</footer>
</main></body></html>`;
  }
}
