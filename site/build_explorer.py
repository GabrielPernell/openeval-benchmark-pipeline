#!/usr/bin/env python3
"""
build_explorer.py
=================
Put the JSON-driven coverage explorer into the site's index.html.

Stage 1 wrote all 28 benchmark tables straight into the page (51 KB -> 327 KB).
This replaces them with a shell plus a renderer that fetches `coverage.json`,
so the page goes back to being small and refreshing the numbers means
regenerating one data file. It also adds per-benchmark download links, pointing
at each benchmark's parquet shards on HuggingFace.

The data file is named by the `data-src` attribute on the explorer, so pointing
the page at a copy hosted elsewhere (the dataset's own summary JSON, say) is a
one-attribute change as long as that file has the same shape.

Two sentences higher up in the section are also corrected: they still said 24
benchmarks and ~70 models per benchmark. They now carry the real figures as
static text and are refreshed from the JSON when it loads.

Re-runnable: each inserted block sits between coverage-explorer markers and is
replaced rather than duplicated. Line endings are preserved (the site file is
LF-only).

Usage
-----
    python build_explorer.py                  # edits the site index.html in place
    python build_explorer.py --site PATH --copy-json
"""
import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
SITE = r"C:\Users\gabri\OneDrive\Desktop\open-eval.github.io-main\index.html"
JSON_SRC = os.path.join(HERE, "coverage.json")
DEFAULT_BENCH = "moralbench"   # the benchmark the section opens on
DATA_SRC = "coverage.json"     # fetched relative to the page

HTML_START, HTML_END = "<!-- coverage-explorer:start -->", "<!-- coverage-explorer:end -->"
CSS_START, CSS_END = "/* coverage-explorer:start */", "/* coverage-explorer:end */"
JS_START, JS_END = "// coverage-explorer:start", "// coverage-explorer:end"

CSS = r"""    /* coverage-explorer:start */
    /* ── Coverage explorer ────────────────────────────────────── */
    .cov-explorer [hidden] { display: none !important; }

    .cov-picker {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }

    .cov-label {
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .cov-combo { position: relative; }

    .cov-search {
      font: inherit;
      font-size: 0.9rem;
      color: var(--ink);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 100px;
      padding: 0.45rem 2.4rem 0.45rem 1rem;
      min-width: 18rem;
    }

    .cov-search:focus { outline: none; border-color: var(--accent); }
    .cov-search:disabled { color: var(--faint); background: var(--bg); }

    .cov-toggle {
      position: absolute;
      top: 50%;
      right: 0.4rem;
      transform: translateY(-50%);
      border: 0;
      background: none;
      color: var(--muted);
      font-size: 0.8rem;
      line-height: 1;
      padding: 0.35rem 0.5rem;
      cursor: pointer;
    }

    .cov-toggle:hover { color: var(--ink); }
    .cov-toggle:disabled { color: var(--faint); cursor: default; }

    .cov-list {
      position: absolute;
      z-index: 20;
      top: calc(100% + 0.35rem);
      left: 0;
      min-width: 100%;
      max-height: 18rem;
      overflow-y: auto;
      margin: 0;
      padding: 0.3rem 0;
      list-style: none;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 8px 24px rgba(26, 38, 26, 0.08);
    }

    .cov-list li {
      padding: 0.4rem 1rem;
      font-family: var(--mono);
      font-size: 0.8rem;
      color: var(--ink);
      white-space: nowrap;
      cursor: pointer;
    }

    .cov-list li.is-active,
    .cov-list li:hover { background: var(--bg); }

    .cov-list li[aria-selected="true"] { font-weight: 600; }

    .cov-list li.cov-nomatch {
      font-family: var(--sans);
      color: var(--muted);
      cursor: default;
      background: none;
    }

    .cov-status { font-size: 0.9rem; color: var(--muted); }
    .cov-status.is-error { color: var(--ink); }

    .cov-summary {
      font-size: 0.95rem;
      color: var(--muted);
      margin-bottom: 0.75rem;
    }

    .cov-summary strong { color: var(--ink); font-weight: 600; }

    /* ── Per-benchmark downloads ──────────────────────────────── */
    .cov-dl {
      display: flex;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 0.5rem 0.6rem;
      margin-bottom: 1rem;
    }

    .cov-dl-label {
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      padding-top: 0.4rem;
    }

    .cov-chip-group { position: relative; }

    /* A one-shard table links straight to the file; a sharded one opens a
       list, so both have to look like the same control. */
    a.cov-chip,
    .cov-chip-group > summary.cov-chip {
      display: inline-flex;
      align-items: baseline;
      gap: 0.45rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 100px;
      padding: 0.3rem 0.85rem;
      font-size: 0.8rem;
      color: var(--ink);
      cursor: pointer;
      list-style: none;
      white-space: nowrap;
    }

    .cov-chip-group > summary.cov-chip::-webkit-details-marker { display: none; }

    a.cov-chip:hover,
    .cov-chip-group > summary.cov-chip:hover {
      border-color: var(--accent);
      text-decoration: none;
    }

    .cov-chip-group[open] > summary.cov-chip { border-color: var(--accent); }

    .cov-chip-name { font-weight: 600; }
    .cov-chip-meta { color: var(--muted); font-variant-numeric: tabular-nums; }

    /* A closed <details> hides its content with content-visibility, which an
       absolutely positioned child escapes when its containing block is the
       <details> itself -- so the shard list has to be hidden outright. */
    .cov-chip-group:not([open]) > .cov-shards { display: none; }

    .cov-shards {
      position: absolute;
      z-index: 20;
      top: calc(100% + 0.35rem);
      left: 0;
      min-width: 100%;
      max-height: 16rem;
      overflow-y: auto;
      margin: 0;
      padding: 0.3rem 0;
      list-style: none;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 8px 24px rgba(26, 38, 26, 0.08);
    }

    .cov-shards a {
      display: flex;
      gap: 1.25rem;
      justify-content: space-between;
      padding: 0.35rem 1rem;
      font-family: var(--mono);
      font-size: 0.75rem;
      color: var(--ink);
      white-space: nowrap;
    }

    .cov-shards a:hover { background: var(--bg); text-decoration: none; }

    .cov-dl-browse { font-size: 0.8rem; padding-top: 0.35rem; }

    /* ── Per-model table ──────────────────────────────────────── */
    .cov-table-wrap {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      max-height: 26rem;
      overflow: auto;
    }

    .cov-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }

    .cov-table th,
    .cov-table td {
      padding: 0.55rem 1rem;
      text-align: right;
      white-space: nowrap;
    }

    .cov-table th:first-child,
    .cov-table td:first-child { text-align: left; }

    .cov-table th {
      position: sticky;
      top: 0;
      background: var(--surface);
      box-shadow: inset 0 -1px 0 var(--border);
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .cov-table td {
      border-top: 1px solid var(--border);
      color: var(--ink);
      font-variant-numeric: tabular-nums;
    }

    .cov-table tbody tr:first-child td { border-top: none; }

    .cov-table td:first-child { font-family: var(--mono); font-size: 0.8rem; }

    .cov-note {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 0.75rem;
    }

    @media (max-width: 480px) {
      .cov-combo { width: 100%; }
      .cov-search { min-width: 0; width: 100%; }
      .cov-table th,
      .cov-table td { padding: 0.45rem 0.7rem; }
      .cov-shards { max-width: 78vw; }
      .cov-shards a { gap: 0.75rem; font-size: 0.7rem; }
    }
    /* coverage-explorer:end */
"""

JS = r"""  // coverage-explorer:start
  // ── Coverage explorer ────────────────────────────────────────
  // Everything shown here is rendered from the file named by data-src, so the
  // numbers are refreshed by regenerating that file, not by editing the page.
  (function () {
    const root = document.getElementById('cov-explorer');
    if (!root) return;

    const input = document.getElementById('cov-search');
    const list = document.getElementById('cov-list');
    const toggle = root.querySelector('.cov-toggle');
    const status = document.getElementById('cov-status');
    const summary = document.getElementById('cov-summary');
    const downloads = document.getElementById('cov-downloads');
    const tableWrap = document.getElementById('cov-table-wrap');
    const tbody = document.getElementById('cov-rows');
    const note = document.getElementById('cov-note');

    let cov = null, names = [], options = [], noMatch = null;
    let selected = null, active = -1, clickedIn = false;

    const int = n => n.toLocaleString('en-US');
    const plural = (n, w) => int(n) + ' ' + w + (n === 1 ? '' : 's');
    // "MMLU Pro" and "mmlu-pro" should both find mmlu-pro.
    const normalize = v => v.trim().toLowerCase().replace(/[\s_]+/g, '-');
    const shown = () => options.filter(o => !o.hidden);

    function bytes(n) {
      const units = ['B', 'KB', 'MB', 'GB', 'TB'];
      let i = 0;
      while (n >= 1000 && i < units.length - 1) { n /= 1000; i++; }
      return (i > 0 && n < 10 ? n.toFixed(1) : Math.round(n)) + ' ' + units[i];
    }

    function el(tag, cls, text) {
      const e = document.createElement(tag);
      if (cls) e.className = cls;
      if (text !== undefined) e.textContent = text;
      return e;
    }

    // ---- the benchmark picker -------------------------------------------
    function buildOptions() {
      list.textContent = '';
      names.forEach(name => {
        const li = el('li', null, name);
        li.id = 'cov-opt-' + name;
        li.setAttribute('role', 'option');
        li.setAttribute('aria-selected', 'false');
        li.dataset.bench = name;
        list.appendChild(li);
      });
      noMatch = el('li', 'cov-nomatch', 'No benchmark matches');
      noMatch.hidden = true;
      list.appendChild(noMatch);
      options = Array.from(list.querySelectorAll('[role="option"]'));
    }

    function setActive(i) {
      const vis = shown();
      options.forEach(o => o.classList.remove('is-active'));
      active = vis.length ? (i + vis.length) % vis.length : -1;
      if (active < 0) { input.removeAttribute('aria-activedescendant'); return; }
      vis[active].classList.add('is-active');
      input.setAttribute('aria-activedescendant', vis[active].id);
      vis[active].scrollIntoView({ block: 'nearest' });
    }

    function open(query) {
      let n = 0;
      options.forEach(o => {
        const hit = !query || o.dataset.bench.includes(query);
        o.hidden = !hit;
        n += hit;
      });
      noMatch.hidden = n > 0;
      list.hidden = false;
      input.setAttribute('aria-expanded', 'true');
      // Start the highlight on the benchmark already showing, so Enter on an
      // untouched list doesn't jump to the first one alphabetically.
      const at = shown().findIndex(o => o.dataset.bench === selected);
      setActive(at >= 0 ? at : 0);
    }

    function close() {
      list.hidden = true;
      input.setAttribute('aria-expanded', 'false');
      input.removeAttribute('aria-activedescendant');
    }

    function choose(name) {
      options.forEach(o => o.setAttribute('aria-selected',
                                          String(o.dataset.bench === name)));
      selected = name;
      input.value = name;
      close();
      render(name);
    }

    // ---- downloads ------------------------------------------------------
    function fileUrl(path) {
      return cov.dataset.resolve + path + '?download=true';
    }

    function chip(label, shards) {
      if (!shards || !shards.length) return null;
      const total = shards.reduce((sum, s) => sum + s[1], 0);
      const meta = shards.length === 1 ? bytes(total)
                                       : shards.length + ' files · ' + bytes(total);
      if (shards.length === 1) {
        const a = el('a', 'cov-chip');
        a.href = fileUrl(shards[0][0]);
        a.appendChild(el('span', 'cov-chip-name', label));
        a.appendChild(el('span', 'cov-chip-meta', meta));
        return a;
      }
      // Several shards can't be a single link, so the chip opens a list of
      // them instead. The size is on every row: some run past a gigabyte.
      const group = el('details', 'cov-chip-group');
      const head = el('summary', 'cov-chip');
      head.appendChild(el('span', 'cov-chip-name', label));
      head.appendChild(el('span', 'cov-chip-meta', meta + ' ▾'));
      group.appendChild(head);
      const ul = el('ul', 'cov-shards');
      shards.forEach(([path, size]) => {
        const a = el('a');
        a.href = fileUrl(path);
        a.appendChild(el('span', null, path.split('/').pop()));
        a.appendChild(el('span', 'cov-chip-meta', bytes(size)));
        const li = el('li');
        li.appendChild(a);
        ul.appendChild(li);
      });
      group.appendChild(ul);
      return group;
    }

    // A data file without file listings (the dataset's own summary JSON has
    // none) still gives a working table; it just has nothing to download.
    function renderDownloads(bench) {
      downloads.textContent = '';
      const files = bench.files || {};
      const chips = [chip('Items', files.item), chip('Responses', files.response)]
        .filter(Boolean);
      if (!chips.length || !cov.dataset) { downloads.hidden = true; return; }
      downloads.appendChild(el('span', 'cov-dl-label', 'Download'));
      chips.forEach(c => downloads.appendChild(c));
      const browse = el('a', 'cov-dl-browse', 'Browse all files ↗');
      browse.href = cov.dataset.url + '/tree/' + (cov.dataset.revision || 'main');
      browse.target = '_blank';
      browse.rel = 'noopener';
      downloads.appendChild(browse);
      downloads.hidden = false;
    }

    // ---- the per-model table --------------------------------------------
    function render(name) {
      const bench = cov.benchmarks[name];
      if (!bench) return;

      summary.textContent = '';
      summary.appendChild(el('strong', null, name));
      summary.appendChild(document.createTextNode(
        ' · ' + int(bench.items) + ' archived items · ' +
        plural(bench.models, 'model') + ' · ' +
        plural(bench.responses, 'response')));
      summary.hidden = false;

      renderDownloads(bench);

      // Rows arrive sorted by items covered, which for a fixed denominator is
      // the same order as coverage.
      const frag = document.createDocumentFragment();
      bench.rows.forEach(([model, covered, responses]) => {
        const tr = el('tr');
        tr.appendChild(el('td', null, model));
        tr.appendChild(el('td', null, int(covered)));
        tr.appendChild(el('td', null, bench.items
          ? (100 * covered / bench.items).toFixed(1) + '%' : '—'));
        tr.appendChild(el('td', null, int(responses)));
        frag.appendChild(tr);
      });
      tbody.textContent = '';
      tbody.appendChild(frag);
      tableWrap.hidden = false;
      tableWrap.scrollTop = 0;
    }

    // ---- the section's headline figures ---------------------------------
    function refreshTotals() {
      const counts = Object.keys(cov.benchmarks)
        .map(k => cov.benchmarks[k].models).sort((a, b) => a - b);
      const mid = counts.length
        ? (counts.length % 2 ? counts[(counts.length - 1) / 2]
          : Math.round((counts[counts.length / 2 - 1] + counts[counts.length / 2]) / 2))
        : null;
      document.querySelectorAll('[data-cov-total]').forEach(node => {
        const v = cov.totals[node.dataset.covTotal];
        if (v != null) node.textContent = int(v);
      });
      document.querySelectorAll('[data-cov-median="models"]').forEach(node => {
        if (mid != null) node.textContent = int(mid);
      });
    }

    // ---- wiring ---------------------------------------------------------
    // Opening always lists every benchmark, not just the one already in the
    // box, and selects the text so typing replaces it.
    input.addEventListener('focus', () => { input.select(); open(''); });
    input.addEventListener('mousedown', () => {
      clickedIn = document.activeElement !== input;
    });
    input.addEventListener('mouseup', e => {
      // A click that focuses the box would otherwise drop the caret and undo
      // the selection made on focus.
      if (clickedIn) { e.preventDefault(); clickedIn = false; }
    });
    input.addEventListener('click', () => { if (list.hidden) open(''); });

    // Typing only filters the list. Nothing is filled in while editing.
    input.addEventListener('input', () => open(normalize(input.value)));

    input.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        // With the box still showing the chosen name, arrowing in should list
        // every benchmark rather than filter down to that one name.
        if (list.hidden) open(input.value === selected ? '' : normalize(input.value));
        else setActive(active + (e.key === 'ArrowDown' ? 1 : -1));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const vis = shown();
        if (active >= 0 && vis[active]) choose(vis[active].dataset.bench);
        else if (cov.benchmarks[normalize(input.value)]) choose(normalize(input.value));
      } else if (e.key === 'Escape') {
        input.value = selected;
        close();
      }
    });

    // Leaving without choosing puts back the name of the table on screen, so
    // the box never disagrees with what's shown.
    input.addEventListener('blur', () => { input.value = selected; close(); });

    // mousedown rather than click, so a choice lands before the box blurs.
    list.addEventListener('mousedown', e => {
      e.preventDefault();
      const o = e.target.closest('[role="option"]');
      if (o) choose(o.dataset.bench);
    });
    toggle.addEventListener('mousedown', e => {
      e.preventDefault();
      if (!list.hidden) { close(); return; }
      if (document.activeElement === input) open('');
      else input.focus();
    });

    // Only one shard list open at a time, and a click elsewhere closes it.
    downloads.addEventListener('toggle', e => {
      if (!e.target.open) return;
      downloads.querySelectorAll('details[open]').forEach(d => {
        if (d !== e.target) d.open = false;
      });
    }, true);
    document.addEventListener('mousedown', e => {
      if (!downloads.contains(e.target)) {
        downloads.querySelectorAll('details[open]').forEach(d => { d.open = false; });
      }
    });

    // ---- load -----------------------------------------------------------
    function failed(err) {
      console.error('coverage explorer:', err);
      status.className = 'cov-status is-error';
      // Opened by double-clicking the file, the page is on a file:// origin and
      // the browser blocks it from fetching the JSON sitting right beside it.
      // That looks like missing data, so say what it actually is.
      status.textContent = location.protocol === 'file:'
        ? 'This page is open straight from disk, and browsers block a file:// ' +
          'page from reading the data file next to it. Serve the folder over ' +
          'http instead — "python -m http.server" in this folder, then open ' +
          'localhost:8000. Hosted, it loads normally. '
        : 'Coverage data could not be loaded. ';
      const a = el('a', null, 'Browse the dataset on HuggingFace ↗');
      a.href = 'https://huggingface.co/datasets/Open-Eval-Commons/OpenEval';
      a.target = '_blank';
      a.rel = 'noopener';
      status.appendChild(a);
      status.hidden = false;
    }

    fetch(root.dataset.src)
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + root.dataset.src);
        return r.json();
      })
      .then(data => {
        cov = data;
        names = Object.keys(cov.benchmarks).sort();
        if (!names.length) throw new Error('no benchmarks in ' + root.dataset.src);
        buildOptions();
        const wanted = root.dataset.default;
        selected = cov.benchmarks[wanted] ? wanted : names[0];
        input.disabled = false;
        toggle.disabled = false;
        input.placeholder = 'Search ' + names.length + ' benchmarks…';
        choose(selected);
        note.textContent = (cov.as_of ? 'Counts as of ' + cov.as_of + '. ' : '') +
          'Downloads are the parquet files published in the OpenEval dataset.';
        refreshTotals();
        status.hidden = true;
      })
      .catch(failed);
  })();
  // coverage-explorer:end
"""

HTML = f"""    {HTML_START}
    <h3 class="section-title" style="margin-top: 4rem;">Explore coverage</h3>
    <p class="section-sub">Pick a benchmark to see which models have responses in OpenEval, how much of the benchmark each one covers, and to download that benchmark's data.</p>
    <div class="cov-explorer" id="cov-explorer" data-src="{DATA_SRC}" data-default="{DEFAULT_BENCH}">
      <div class="cov-picker">
        <label for="cov-search" class="cov-label" id="cov-label">Benchmark</label>
        <div class="cov-combo">
          <input id="cov-search" class="cov-search" type="text" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="cov-list" autocomplete="off" spellcheck="false" placeholder="Loading benchmarks…" disabled>
          <button type="button" class="cov-toggle" tabindex="-1" aria-label="Show all benchmarks" disabled>▾</button>
          <ul id="cov-list" class="cov-list" role="listbox" aria-labelledby="cov-label" hidden></ul>
        </div>
      </div>
      <p class="cov-status" id="cov-status" aria-live="polite">Loading coverage data…</p>
      <p class="cov-summary" id="cov-summary" hidden></p>
      <div class="cov-dl" id="cov-downloads" hidden></div>
      <div class="cov-table-wrap" id="cov-table-wrap" hidden>
        <table class="cov-table">
          <thead><tr><th scope="col">Model</th><th scope="col">Items Covered</th><th scope="col">Coverage</th><th scope="col">Responses</th></tr></thead>
          <tbody id="cov-rows"></tbody>
        </table>
      </div>
      <p class="cov-note" id="cov-note"></p>
    </div>
    {HTML_END}
"""

# The two stale sentences. Each is matched loosely enough to also match the
# version this script already wrote, so a second run is a no-op.
SENTENCES = [
    (re.compile(r"OpenEval currently indexes results from .*? benchmarks, spanning"),
     'OpenEval currently indexes results from '
     '<span data-cov-total="benchmarks">{benchmarks}</span> benchmarks, spanning'),
    (re.compile(r'<p class="section-sub">OpenEval covers a growing range of '
                r'open and proprietary models.*?</p>'),
     '<p class="section-sub">OpenEval covers a growing range of open and proprietary '
     'models &mdash; <span data-cov-total="models">{models}</span> distinct models in '
     'all, and a median of <span data-cov-median="models">{median_models}</span> per '
     'benchmark.</p>'),
]


def replace_or_insert(s, start, end, block, anchor_index):
    """Swap an existing marked block, or insert the new one at anchor_index."""
    if start in s:
        a = s.rindex("\n", 0, s.index(start)) + 1          # start of the marker line
        b = s.index("\n", s.index(end)) + 1                 # end of the marker line
        return s[:a] + block + s[b:]
    return s[:anchor_index] + block + s[anchor_index:]


def fill_sentences(s, stats):
    for pattern, replacement in SENTENCES:
        hits = pattern.findall(s)
        if len(hits) != 1:
            raise RuntimeError(f"expected 1 match for {pattern.pattern!r}, got {len(hits)}")
        s = pattern.sub(lambda _m: replacement.format(**stats), s, count=1)
    return s


def apply(s, stats):
    """Return the page with the explorer blocks refreshed and the copy fixed."""
    # HTML: end of the coverage section's container, after the model legend.
    if HTML_START in s:
        s = replace_or_insert(s, HTML_START, HTML_END, HTML, 0)
    else:
        sec = s.index('<section class="section" id="coverage">')
        sec_end = s.index("</section>", sec)
        s = replace_or_insert(s, HTML_START, HTML_END, HTML,
                              s.rindex("  </div>", sec, sec_end))

    # CSS: just before the single </style>.
    assert s.count("  </style>") == 1
    s = replace_or_insert(s, CSS_START, CSS_END, CSS, s.index("  </style>"))

    # JS: just before the single </script>.
    assert s.count("</script>") == 1
    s = replace_or_insert(s, JS_START, JS_END, JS, s.index("</script>"))

    return fill_sentences(s, stats)


def summary_stats(path):
    with open(path, encoding="utf-8") as f:
        cov = json.load(f)
    counts = sorted(b["models"] for b in cov["benchmarks"].values())
    mid = len(counts) // 2
    median = counts[mid] if len(counts) % 2 else round((counts[mid - 1] + counts[mid]) / 2)
    if cov["benchmarks"].get(DEFAULT_BENCH) is None:
        raise RuntimeError(f"default benchmark {DEFAULT_BENCH} is not in {path}")
    return {"benchmarks": f"{cov['totals']['benchmarks']:,}",
            "models": f"{cov['totals']['models']:,}",
            "median_models": f"{median:,}",
            "as_of": cov["as_of"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=SITE, help="the site's index.html")
    ap.add_argument("--json", default=JSON_SRC, help="the coverage.json to read stats from")
    ap.add_argument("--copy-json", action="store_true",
                    help="also copy coverage.json next to index.html")
    args = ap.parse_args()

    stats = summary_stats(args.json)

    with open(args.site, "rb") as f:
        raw = f.read()

    # A git clone on Windows checks the page out with CRLF (core.autocrlf) while
    # the repo stores LF; the ZIP download is LF throughout. Either is fine as
    # long as the file goes back the way it came, so nothing shows up in the
    # diff but the actual edit.
    crlf = b"\r\n" in raw
    if crlf and raw.count(b"\r\n") != raw.count(b"\n"):
        sys.exit("site index.html mixes CRLF and bare LF; normalize it first")
    newline = "\r\n" if crlf else "\n"

    out = apply(raw.replace(b"\r\n", b"\n").decode("utf-8"), stats)
    with open(args.site, "w", encoding="utf-8", newline=newline) as f:
        f.write(out)

    print(f"wrote {args.site}")
    print(f"  {len(raw)/1024:.0f} KB -> {os.path.getsize(args.site)/1024:.0f} KB "
          f"({'CRLF' if crlf else 'LF'} preserved)")
    print(f"  reads {DATA_SRC}, opens on '{DEFAULT_BENCH}', "
          f"{stats['benchmarks']} benchmarks as of {stats['as_of']}")

    if args.copy_json:
        dest = os.path.join(os.path.dirname(os.path.abspath(args.site)), DATA_SRC)
        shutil.copyfile(args.json, dest)
        print(f"  copied {DATA_SRC} -> {dest} ({os.path.getsize(dest)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
