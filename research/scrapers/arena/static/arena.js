"use strict";

// Blind A/B judging. Entrant identities are never rendered before a vote is
// committed; the server sends them but the UI only reveals them in the toast
// after /api/vote returns.

let current = null;
let busy = false;

const $ = (id) => document.getElementById(id);

function esc(text) {
  return text.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------- markdown
// Small deliberate subset. The default view is the raw string in a <pre>,
// because that is what actually lands in a corpus; this is only a convenience
// toggle, so it does not need to be a full CommonMark implementation.
function renderMarkdown(src) {
  const lines = src.split("\n");
  const out = [];
  let inCode = false, listType = null;

  const closeList = () => { if (listType) { out.push(`</${listType}>`); listType = null; } };
  // Scraped content decides these URLs, so only http(s) (and inline images) may reach
  // an href/src. `esc` already blocks attribute breakout; this blocks the rest --
  // a `[x](javascript:...)` in an extractor's output would otherwise render as a live
  // clickable link in the reviewer's browser.
  const safeUrl = (url, allowData) =>
    /^https?:\/\//i.test(url) || (allowData && /^data:image\//i.test(url)) ? url : "";

  const inline = (text) => {
    let s = esc(text);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)[^)]*\)/g, (m, alt, url) => {
      const safe = safeUrl(url, true);
      return safe ? `<img alt="${alt}" src="${safe}" />` : `<span>${alt}</span>`;
    });
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g, (m, label, url) => {
      const safe = safeUrl(url, false);
      return safe
        ? `<a href="${safe}" rel="noopener nofollow">${label}</a>`
        : `<span>${label}</span>`;
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|\W)\*([^*\n]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/(^|\W)_([^_\n]+)_/g, "$1<em>$2</em>");
    return s;
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (/^```/.test(line)) {
      closeList();
      out.push(inCode ? "</code></pre>" : '<pre><code>');
      inCode = !inCode;
      continue;
    }
    if (inCode) { out.push(esc(raw)); continue; }
    if (!line.trim()) { closeList(); continue; }

    let match;
    if ((match = line.match(/^(#{1,6})\s+(.*)$/))) {
      closeList();
      const level = Math.min(match[1].length, 6);
      out.push(`<h${level}>${inline(match[2])}</h${level}>`);
    } else if (/^\s*([-*+])\s+/.test(line)) {
      if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
      out.push(`<li>${inline(line.replace(/^\s*[-*+]\s+/, ""))}</li>`);
    } else if (/^\s*\d+[.)]\s+/.test(line)) {
      if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
      out.push(`<li>${inline(line.replace(/^\s*\d+[.)]\s+/, ""))}</li>`);
    } else if (/^\s*>\s?/.test(line)) {
      closeList();
      out.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ""))}</blockquote>`);
    } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      closeList(); out.push("<hr />");
    } else if (/^\s*\|.*\|\s*$/.test(line)) {
      closeList();
      if (/^\s*\|[\s:|-]+\|\s*$/.test(line)) continue;   // alignment row
      const cells = line.trim().replace(/^\||\|$/g, "").split("|");
      out.push("<table><tr>" + cells.map((c) => `<td>${inline(c.trim())}</td>`).join("") + "</tr></table>");
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  if (inCode) out.push("</code></pre>");
  return out.join("\n");
}

// Unchanged text adjacent to a difference is context worth keeping; the rest is what
// makes a 90%-identical pair unreadable. These two numbers are the whole trade-off:
// keep this much on each side of a difference, collapse anything longer.
const CONTEXT_CHARS = 200;
const COLLAPSE_MIN_CHARS = 2 * CONTEXT_CHARS + 80;

// Jump the reference pane to a section of the source page.
//
// The pane is cross-origin for live pages, so its scroll position cannot be set
// directly -- but a URL fragment can, and MediaWiki heading ids are the heading text
// with spaces as underscores. The saved snapshots carry the same ids, so one anchor
// serves the live iframe and the saved-DOM fallback alike.
function jumpSource(anchor) {
  if (!anchor || !current) return;
  const frame = $("ref");
  const base = (frame.src || "").split("#")[0];
  if (!base) return;
  const target = `${base}#${anchor}`;
  // Re-assigning an identical src is a no-op, so a second click on the same chip
  // would do nothing after the reviewer had scrolled away. Clearing the fragment
  // first makes the navigation happen again.
  if (frame.src === target) frame.src = base;
  frame.src = target;
}

function regionChip(region, marks) {
  const trail = (marks.sections && marks.sections[String(region)]) || [];
  const anchor = (marks.anchors && marks.anchors[String(region)]) || "";
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.dataset.chipRegion = region;
  const where = trail.length ? trail.join(" › ") : "(before any heading)";
  chip.textContent = anchor ? `diff ${region} · ${where} ↗` : `diff ${region} · ${where}`;
  if (anchor) {
    chip.classList.add("linked");
    chip.title = `show "${trail[trail.length - 1]}" in the source pane`;
    chip.addEventListener("click", () => jumpSource(anchor));
  } else {
    chip.title = "this difference sits above the first heading";
  }
  return chip;
}

function collapseStub(text) {
  const stub = document.createElement("span");
  stub.className = "collapsed";
  const lines = text.split("\n").length;
  stub.textContent = `[…] ${lines} identical lines`;
  stub.title = "click to expand";
  // Expanding in place rather than repainting keeps the scroll position, which is the
  // only reason the reviewer collapsed the document in the first place.
  stub.addEventListener("click", () => stub.replaceWith(document.createTextNode(text)));
  return stub;
}

function paint(containerId, text, spans, marks) {
  const container = $(containerId);
  const asMarkdown = $("render-md").checked;
  if (asMarkdown) {
    // Marks are offsets into the raw text; mapping them onto rendered HTML would mean
    // re-parsing and re-anchoring. Markdown mode shows plain output.
    container.innerHTML = `<div class="md">${renderMarkdown(text)}</div>`;
    container.scrollTop = 0;
    return;
  }
  container.innerHTML = "<pre></pre>";
  const pre = container.querySelector("pre");

  if (!spans || !$("mark-diffs").checked) {
    pre.textContent = text;
    container.scrollTop = 0;
    return;
  }

  const collapse = $("diffs-only").checked;
  let lastChipRegion = 0;

  spans.forEach((span, index) => {
    if (span.kind === "none") {
      // This side has nothing for the region. Show the chip anyway plus an explicit
      // "nothing here", so the two columns stay aligned and a one-sided difference
      // reads as one -- rather than as a column that simply stops.
      if (span.region !== lastChipRegion) {
        pre.appendChild(regionChip(span.region, marks || {}));
        lastChipRegion = span.region;
      }
      const gap = document.createElement("span");
      gap.className = "nothing";
      gap.dataset.region = span.region;
      gap.textContent = "(nothing on this side)";
      pre.appendChild(gap);
      return;
    }
    if (span.kind === "diff") {
      if (span.region !== lastChipRegion) {
        pre.appendChild(regionChip(span.region, marks || {}));
        lastChipRegion = span.region;
      }
      const el = document.createElement("mark");
      // A purely-whitespace difference is the fused-citation defect, where the entire
      // change is one absent space. Unstyled it is invisible, and it is the most
      // common real defect in this corpus, so it gets a glyph.
      el.className = span.text.trim() ? "d" : "d ws";
      el.dataset.region = span.region;
      el.textContent = span.text;
      pre.appendChild(el);
      return;
    }

    if (!collapse || span.text.length <= COLLAPSE_MIN_CHARS) {
      pre.appendChild(document.createTextNode(span.text));
      return;
    }
    // Context is only useful next to a difference. A shared run at the very start of
    // the document has nothing before it to contextualise, and one at the very end has
    // nothing after -- so each side of the stub is kept only if a difference adjoins it.
    const diffBefore = index > 0;
    const diffAfter = index < spans.length - 1;
    if (diffBefore) pre.appendChild(document.createTextNode(span.text.slice(0, CONTEXT_CHARS)));
    pre.appendChild(collapseStub(span.text));
    if (diffAfter) pre.appendChild(document.createTextNode(span.text.slice(-CONTEXT_CHARS)));
  });
  container.scrollTop = 0;
}

// ------------------------------------------------------------ diff navigation
//
// Highlighting alone is not enough on a 700k-character output: the reviewer still
// has to find the marks. These step both columns to the same region at once, so the
// two sides stay comparable.

let diffCursor = 0;

function diffRegions() {
  return current && current.marks ? current.marks.regions : 0;
}

function setDiffCounter() {
  const badge = $("diffcount");
  const total = diffRegions();
  const marking = $("mark-diffs").checked && !$("render-md").checked;
  badge.hidden = !marking;
  if (!marking) return;
  if (!current || !current.marks) { badge.textContent = ""; return; }
  const m = current.marks;
  // `regions === 0` means identical *after* blank-line and trailing-space
  // normalisation, which is not the same as byte-identical -- a pair differing only
  // in how many newlines separate its blocks lands here. Saying "identical" of two
  // outputs the reviewer can see are different lengths reads as a broken diff.
  const sameBytes = current.output_a === current.output_b;
  badge.textContent = total === 0
    ? (sameBytes ? "identical" : "identical apart from blank lines")
    : `${total} diff${total === 1 ? "" : "s"} · ` +
      `${Math.round(m.similarity * 100)}% shared · ` +
      `left-only ${m.a_only_lines}, right-only ${m.b_only_lines}` +
      (diffCursor ? ` · at ${diffCursor}/${total}` : "");
}

function gotoDiff(step) {
  const total = diffRegions();
  if (!total || $("render-md").checked || !$("mark-diffs").checked) return;
  diffCursor += step;
  if (diffCursor < 1) diffCursor = total;
  if (diffCursor > total) diffCursor = 1;
  let scrolled = false;
  for (const id of ["out-a", "out-b"]) {
    // A one-sided region has a `.nothing` marker instead of a `<mark>`, and stepping
    // to it must still scroll both columns -- otherwise the side that dropped the
    // content is the one the reviewer never gets shown.
    const el = $(id).querySelector(
      `mark[data-region="${diffCursor}"], .nothing[data-region="${diffCursor}"]`
    );
    $(id).querySelectorAll("mark.at, .nothing.at").forEach((m) => m.classList.remove("at"));
    if (el) {
      el.classList.add("at");
      el.scrollIntoView({ block: "center" });
      scrolled = true;
    }
  }
  if (!scrolled) return;
  // Stepping to a difference also points the reference pane at its section, so
  // "which side is right" is one keystroke away rather than a manual hunt.
  const marks = current.marks || {};
  const anchor = (marks.anchors && marks.anchors[String(diffCursor)]) || "";
  if (anchor && $("follow-source").checked) jumpSource(anchor);
  // Mark the chip too, so the reviewer can see which region they are on in a
  // collapsed document where the mark itself may be a single space.
  for (const id of ["out-a", "out-b"]) {
    $(id).querySelectorAll(".chip.at").forEach((c) => c.classList.remove("at"));
    const chip = $(id).querySelector(`.chip[data-chip-region="${diffCursor}"]`);
    if (chip) chip.classList.add("at");
  }
  setDiffCounter();
}

function repaint() {
  if (!current) return;
  const marks = current.marks || {};
  paint("out-a", current.output_a, marks.a, marks);
  paint("out-b", current.output_b, marks.b, marks);
  diffCursor = 0;
  setDiffCounter();
}

// ---------------------------------------------------------------- matchup

function setCounter(human, auto) {
  $("counter").textContent = `${human} votes · ${auto} auto-draws`;
}

// The mode survives a refresh. A <select> resets to its first option on every page
// load, so telling the reviewer to "refresh and pick duel" silently put them back in
// explore mode -- nine votes went to the wrong sampler before this was noticed.
// localStorage keeps it offline-safe; no network, no server state.
const MODE_KEY = "arena-mode";
const MODES = ["explore", "calibration", "duel", "wiki3", "challengers",
               "challengers-ours", "content3", "floorcheck", "committed", "crossplay",
               "converters", "bottomhalf"];

function currentMode() {
  return $("mode").value;
}

function restoreMode() {
  let saved = null;
  try {
    saved = localStorage.getItem(MODE_KEY);
  } catch (error) {
    saved = null;                      // private mode / storage disabled
  }
  if (saved && MODES.includes(saved)) $("mode").value = saved;
}

function rememberMode(mode) {
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch (error) {
    /* not worth failing a vote over */
  }
}

// Both non-default modes report progress here, but they count different things:
// calibration counts the two pools' overlap (what agreement is computed over), duel
// counts votes on the one pair, split by page shape because that is the axis the
// difference under test lives on.
function setCalibProgress(info) {
  const box = $("calib-progress");
  if (!info) { box.hidden = true; return; }
  box.hidden = false;
  if (info.entrants) {
    // Focus mode: show which pair is on screen and how the set is filling up. The
    // per-pair split is the useful number -- a closed set is only settled when every
    // pair in it has votes, not when the total looks big.
    const perPair = Object.entries(info.per_pair || {})
      .map(([pair, n]) => `${pair.replace(/ vs /, "/")} ${n.judged}`)
      .join(" · ");
    box.textContent =
      `${info.judged} judged over ${info.pages} pages · ${info.remaining} left` +
      (perPair ? ` · ${perPair}` : "") +
      (info.axis && info.axis !== "none" ? ` · split by ${info.axis}` : "");
    return;
  }
  box.textContent = `${info.overlap} shared · ${info.remaining} left`;
}

// Two loads can be in flight at once: the module-level `loadMatchup()` fires with the
// remembered mode, and a `/focus/<mode>` URL immediately switches the mode, which
// fires a second one. Without sequencing whichever response lands last wins, so the
// slower first request could paint an out-of-mode page while the dropdown said
// otherwise -- and in a wiki-only mode that means being served an unpinned page whose
// reference pane no longer matches what the extractors read. Caught by opening
// /focus/content3 and getting a gov.uk page.
let loadSeq = 0;

async function loadMatchup() {
  const seq = ++loadSeq;
  busy = true;
  setButtons(false);
  const mode = currentMode();
  const response = await fetch(`/api/matchup?mode=${mode}`);
  const data = await response.json();
  if (seq !== loadSeq) return;         // a newer load started; this answer is stale
  setCalibProgress(mode === "explore" ? null : data.calibration);

  if (data.exhausted) {
    current = null;
    const exhaustedNote = {
      calibration:
        '<strong>No panel-judged comparisons left to re-check.</strong> ' +
        'Switch back to explore to keep judging new pages, and see the ' +
        'agreement table in the Report.',
      duel:
        '<strong>This pair has been judged on every page where the two differ.</strong> ' +
        'Run <code>just arena-duel</code> for the split by page shape.',
      wiki3:
        '<strong>Every pair in this set has been judged on all 47 shared pages.</strong> ' +
        'Run <code>just arena-wiki3</code> for the standings.',
      content3:
        '<strong>All three pairs judged on every pinned wiki page where they differ.</strong> ' +
        'Run <code>just arena-focus ours,trafilatura,resiliparse</code> for the standings ' +
        '&mdash; note it pools the unpinned pages\' older votes, which this mode does not collect.',
      floorcheck:
        '<strong>trafilatura and the floor judged on every pinned wiki page.</strong> ' +
        'Run <code>just arena-focus trafilatura,scrapling-md</code> for the standings ' +
        '&mdash; note it pools the unpinned pages, which are where the floor won.',
      committed:
        '<strong>The branch, the committed version and stock trafilatura ' +
        'judged on every pinned wiki page where they differ.</strong> ' +
        'Run <code>just arena-focus ours,ours@committed,trafilatura</code> for the standings.',
      crossplay:
        '<strong>All three of the missing pairs judged on every pinned wiki page ' +
        'where they differ.</strong> The board can now be ordered without going ' +
        'through trafilatura. Run <code>just arena-deck</code> after refreshing the numbers.',
      converters:
        '<strong>All three converter pairs judged on every pinned wiki page ' +
        'where they differ.</strong> They can now be ordered against each other, ' +
        'not just against trafilatura.',
      bottomhalf:
        '<strong>What we run now has been judged against all three converters ' +
        'on every pinned wiki page where they differ.</strong> ' +
        'The bottom of the board no longer rests on a shared opponent.',
      explore:
        '<strong>Every pair on every page has been judged.</strong> ' +
        'Open the Report for final results.',
    };
    $("meta").innerHTML = exhaustedNote[mode] || exhaustedNote.explore;
    paint("out-a", ""); paint("out-b", "");
    $("diffcount").hidden = true;
    setCounter(data.human_votes, data.auto_draws);
    return;
  }

  current = data;
  const page = data.page;
  $("meta").innerHTML =
    (data.mode !== "explore"
      ? `<span class="tag calib">${esc(data.mode)}${
          data.calibration && data.calibration.serving
            ? " · " + esc(data.calibration.serving.pair)
            : ""
        }</span>`
      : "") +
    `<span class="tag">${esc(page.bucket)}</span>` +
    `<span class="tag">${esc(page.lang)}</span>` +
    `<span class="tag">${esc(page.shape)}</span>` +
    `<span>${esc(page.title || "(untitled)")}</span>` +
    `<span class="url">${esc(page.url)}</span>`;
  $("original").href = page.url;

  // Reference pane. The variant matters: the extractors read `raw` on 97 of 100 pages,
  // and this link used to omit it, so /snapshot defaulted to `rendered` -- a document
  // neither column was produced from.
  const variant = data.reference_variant || "raw";
  const saved = `/snapshot/${page.id}?variant=${variant}`;
  $("saved-dom").href = saved;

  // Live page where the site allows framing, saved bytes where it does not. Safe for
  // the wiki slice because those URLs are pinned to ?oldid= permalinks, so the live
  // render is the snapshot and cannot drift from it.
  const live = data.frameable !== false;
  $("ref").src = live ? page.url : `${saved}&reader=1`;
  $("refkind").textContent = live
    ? "Source page — live"
    : `Source page — saved ${variant} (site refuses framing)`;

  // The saved-DOM pane serves the same variant the extractor read, so it cannot drift
  // from the text under review. What can drift is an unpinned live page.
  const badge = $("stale");
  if (!live && !page.url.includes("oldid=")) {
    badge.hidden = false;
    badge.textContent = "unpinned page — live content may have moved on";
  } else {
    badge.hidden = true;
  }
  $("refscroll").scrollTop = 0;
  paint("out-a", data.output_a, data.marks && data.marks.a, data.marks);
  paint("out-b", data.output_b, data.marks && data.marks.b, data.marks);
  diffCursor = 0;
  setDiffCounter();
  setCounter(data.human_votes, data.auto_draws);

  busy = false;
  setButtons(true);
}

function setButtons(enabled) {
  document.querySelectorAll("[data-vote]").forEach((b) => { b.disabled = !enabled; });
}

function showToast(picked, reveal) {
  const label = (side, info) => {
    const ms = info.extract_ms == null ? "?" : info.extract_ms.toFixed(1);
    return `<span class="${side}">${esc(info.name)}</span> ` +
           `<span class="pill">${esc(info.variant || "")}</span> ` +
           `${ms} ms · ${info.chars ?? "?"} chars`;
  };
  const verdict = picked === "a" ? "you picked A"
    : picked === "b" ? "you picked B"
    : picked === "draw" ? "draw" : "both bad";
  $("toast").innerHTML =
    `A = ${label("a", reveal.a)} &nbsp;|&nbsp; B = ${label("b", reveal.b)} ` +
    `&nbsp;→&nbsp; <span class="picked">${verdict}</span>`;
  $("toast").hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { $("toast").hidden = true; }, 4500);
}

async function submit(winner) {
  if (busy || !current) return;
  if (winner === "skip") { await loadMatchup(); return; }

  busy = true;
  setButtons(false);
  const payload = {
    page_id: current.page.id,
    entrant_a: current.entrant_a,
    entrant_b: current.entrant_b,
    winner,
    // Tagged so a semi-blind duel vote is never counted as a fully blind one.
    mode: current.mode || "explore",
  };
  const response = await fetch("/api/vote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  showToast(winner, data.reveal);
  setCounter(data.human_votes, data.auto_draws);
  await loadMatchup();
}

// ------------------------------------------------------------ leaderboard

function boardRow(row) {
  const record = `${row.wins}/${row.losses}/${row.draws}`;
  const flag = row.wiki_only ? ' <span class="pill">wiki only</span>' : "";
  const layer = row.layer === "floor" ? ' <span class="pill">floor</span>' : "";
  return `<tr>
    <td>${esc(row.name)}${layer}${flag}</td>
    <td>${row.bt}</td><td>${row.elo}</td><td>${record}</td><td>${row.games}</td>
    <td>${row.ok ?? 0}</td><td>${row.empty ?? 0}</td><td>${row.error ?? 0}</td>
    <td>${row.median_extract_ms ?? "-"}</td>
    <td>${row.median_total_ms ?? "-"}</td>
  </tr>`;
}

function fetchRow(row) {
  const failures = Object.entries(row.failures || {})
    .map(([kind, count]) => `${count} ${kind}`)
    .join(", ");
  return `<tr><td>${esc(row.name)}</td><td>${row.success_rate}%</td>
    <td>${row.ok}</td><td>${row.median_ms}</td><td>${row.total_s}</td>
    <td>${failures || "&mdash;"}</td></tr>`;
}

function fetchLayerHtml(fetch) {
  if (!fetch || !fetch.measured) {
    return `<p class="note">Fetch layer not measured yet &mdash; run
      <code>just arena-fetchers</code>.</p>`;
  }
  // Two figures per pair: byte-identical markdown, and token cosine. A fetcher can
  // agree on all the content and still not match byte for byte, so one number alone
  // has read as the other being wrong.
  const pairs = (fetch.pairs || [])
    .filter((p) => !p.self)
    .map(
      (p) =>
        `${esc(p.a)} vs ${esc(p.b)}: ${p.pct}% identical` +
        (p.near_pct === null ? "" : `, ${p.near_pct}% near-identical`) +
        ` (${p.total} pages)`,
    )
    .join("<br />");
  return `
    <h3 class="section">Fetch layer <span class="pill">measured, not rated</span></h3>
    <p class="note">HTTP clients and browser fetchers. These have no main-content
      model, so they are measured rather than voted on: pairing a fetcher with an
      extractor emits exactly what that extractor emits alone.</p>
    <table class="board">
      <thead><tr><th>fetcher</th><th>success</th><th>ok</th><th>median ms</th>
        <th>total s</th><th>failures</th></tr></thead>
      <tbody>${fetch.rows.map(fetchRow).join("")}</tbody>
    </table>
    <p class="note">${fetch.urls} URLs · ${fetch.dynamic_pages} pages excluded as
      dynamic (they differ when the same fetcher runs twice, so they cannot tell
      fetchers apart).<br />${pairs}</p>`;
}

function boardTable(rows) {
  return `<table class="board">
    <thead><tr>
      <th>entrant</th><th>BT</th><th>ELO</th><th>W/L/D</th><th>games</th>
      <th>ok</th><th>empty</th><th>err</th><th>extract ms</th><th>total ms</th>
    </tr></thead>
    <tbody>${rows.map(boardRow).join("")}</tbody>
  </table>`;
}

async function openBoard(pool) {
  $("board").hidden = false;
  $("board-body").innerHTML = "loading…";
  // The two pools are separate ratings, never a blend. With no pool requested the
  // server picks one that has votes, so the board never opens flat at 1500.
  const query = pool ? `?judge=${pool}` : "";
  const data = await (await fetch(`/api/leaderboard${query}`)).json();
  $("board-pool").value = data.judge;
  const bias = data.position_bias;
  const controlled = data.position_bias_controlled;
  const poolLabel = data.judge === "llm" ? "automated" : "your";
  const other = data.judge === "llm" ? "human" : "llm";
  const empty = !data.human_votes && !data.auto_draws
    ? `<p class="note"><strong>No ${poolLabel} votes yet</strong> — every rating below
       is the 1500 starting value, not a measurement.
       ${data.pools[other] ? `The ${other === "llm" ? "automated" : "human"} pool has
       ${data.pools[other]} votes; switch pool above to see it.` : ""}</p>`
    : "";
  const wiki = data.wiki_rows && data.wiki_rows.length
    ? `<h3 class="section">Wiki slice <span class="pill">${data.wiki_votes} votes</span></h3>
       <p class="note">Fitted on wiki-page votes only. This is the only place a
       wiki-only entrant appears: a rating earned over the 50 wiki pages is not on
       the same scale as one earned over all 100, so ranking them together would
       overstate it.</p>
       ${boardTable(data.wiki_rows)}`
    : "";
  $("board-body").innerHTML = `
    ${empty}
    ${boardTable(data.rows)}
    <p class="note">
      ${data.human_votes} ${poolLabel} votes · ${data.auto_draws} auto-draws ·
      ${data.pages} pages · per-bucket votes:
      ${Object.entries(data.bucket_votes).map(([k, v]) => `${k} ${v}`).join(" · ")}<br />
      Left-column win rate ${(bias.left_win_rate * 100).toFixed(1)}%
      over ${bias.decided} decided votes — near 50% means no position bias.
      ${controlled && controlled.advantage !== null
        ? `Controlled for which entrant sat where (${controlled.pairs} pairs judged
           in both orders): first slot worth
           ${(controlled.advantage * 100 >= 0 ? "+" : "")}${(controlled.advantage * 100).toFixed(1)}
           points.`
        : ""}
      BT confidence intervals are in the Report.
    </p>
    ${wiki}
    ${fetchLayerHtml(data.fetch_layer)}`;
  renderAgreement();
}

function agreementHtml(a) {
  if (!a.cells) {
    return `<h3 class="section">Panel agreement</h3>
      <p class="note">The two pools have not judged a single comparison in common,
        so there is no evidence either way about whether the automated panel matches
        your judgement. ${a.candidates} panel-judged comparisons are available to
        re-check &mdash; tick <strong>calibration mode</strong> to start.</p>`;
  }
  const kappa = a.kappa === null ? "n/a" : a.kappa.toFixed(2);
  const verdict = a.kappa === null ? ""
    : a.kappa >= 0.6 ? " Substantial &mdash; the panel is a usable stand-in."
    : a.kappa >= 0.4 ? " Moderate &mdash; treat the panel as a rough pre-screen only."
    : " Weak &mdash; the panel's ranking is not evidence about your preferences.";
  return `<h3 class="section">Panel agreement
      <span class="pill">${a.cells} shared comparison${a.cells === 1 ? "" : "s"}</span></h3>
    <table class="board">
      <thead><tr><th>measure</th><th>rate</th><th>n</th><th>what it means</th></tr></thead>
      <tbody>
      <tr><td>exact</td><td>${a.exact ?? "-"}%</td><td>${a.cells}</td>
        <td class="why">same winner, distinguishing draw from both-bad</td></tr>
      <tr><td>equivalent</td><td>${a.equivalent ?? "-"}%</td><td>${a.cells}</td>
        <td class="why">same winner; draw and both-bad treated alike, as the ratings do</td></tr>
      <tr><td>decisive only</td><td>${a.decisive ?? "-"}%</td><td>${a.decisive_n}</td>
        <td class="why">cells where both picked a winner</td></tr>
      <tr><td>Cohen&rsquo;s &kappa;</td><td>${kappa}</td><td>${a.cells}</td>
        <td class="why">agreement above chance</td></tr>
      </tbody>
    </table>
    <p class="note">${verdict}
      Same left/right order: ${a.same_order ?? "-"}% of ${a.same_order_n} ·
      flipped: ${a.flipped_order ?? "-"}% of ${a.flipped_order_n}. A large gap between
      those two means the agreement is partly a shared position bias rather than
      shared judgement. ${a.candidates} panel cells remain un-rechecked.</p>`;
}

async function renderAgreement() {
  const box = $("agreement-body");
  if (!box) return;
  try {
    box.innerHTML = agreementHtml(await (await fetch("/api/agreement")).json());
  } catch (error) {
    box.innerHTML = `<p class="note">Agreement unavailable: ${esc(String(error))}</p>`;
  }
}

// ------------------------------------------------------------------- wiring

document.querySelectorAll("[data-vote]").forEach((button) => {
  button.addEventListener("click", () => submit(button.dataset.vote));
});
$("show-board").addEventListener("click", () => openBoard());
$("board-pool").addEventListener("change", () => openBoard($("board-pool").value));
$("close-board").addEventListener("click", () => { $("board").hidden = true; });
$("board").addEventListener("click", (event) => {
  if (event.target === $("board")) $("board").hidden = true;
});
$("mode").addEventListener("change", () => {
  rememberMode(currentMode());
  loadMatchup();
});
["render-md", "mark-diffs", "diffs-only"].forEach((id) => {
  $(id).addEventListener("change", repaint);
});
$("next-diff").addEventListener("click", () => gotoDiff(1));
$("prev-diff").addEventListener("click", () => gotoDiff(-1));

// Scrollable panes are focusable so they can be scrolled with the keyboard.
// While one has focus, arrow keys must scroll rather than vote — otherwise
// reading a long output with the keyboard silently casts votes.
function inScrollPane() {
  const active = document.activeElement;
  return !!active && active.classList &&
    (active.classList.contains("out") || active.classList.contains("refscroll"));
}

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (!$("board").hidden) {
    if (event.key === "Escape") $("board").hidden = true;
    return;
  }
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (key === "l") { event.preventDefault(); openBoard(); return; }
  if (key === "u") { event.preventDefault(); undoLast(); return; }
  // j/k step between differences. Deliberately not the arrow keys: those vote, and
  // a reviewer hunting for the next diff must not cast one by accident.
  if (key === "j") { event.preventDefault(); gotoDiff(1); return; }
  if (key === "k") { event.preventDefault(); gotoDiff(-1); return; }

  const arrows = { ArrowLeft: "a", ArrowRight: "b", ArrowDown: "draw" };
  if (arrows[key]) {
    if (inScrollPane()) return;          // let the pane scroll
    event.preventDefault();
    submit(arrows[key]);
    return;
  }
  const letters = { a: "a", b: "b", d: "draw", x: "both_bad", s: "skip" };
  if (letters[key]) { event.preventDefault(); submit(letters[key]); }
});

async function undoLast() {
  if (busy) return;
  const data = await (await fetch("/api/undo", { method: "POST" })).json();
  setCounter(data.human_votes, data.auto_draws);
  if (data.undone) {
    const r = data.removed;
    $("toast").innerHTML =
      `Undid vote #${r.id}: <span class="a">${esc(r.entrant_a)}</span> vs ` +
      `<span class="b">${esc(r.entrant_b)}</span> (was &ldquo;${esc(r.winner)}&rdquo;). ` +
      `That page/pair can be drawn again.`;
  } else {
    $("toast").innerHTML = "Nothing to undo.";
  }
  $("toast").hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { $("toast").hidden = true; }, 4500);
  await loadMatchup();
}

$("undo").addEventListener("click", undoLast);

// Restore before the first fetch, so a reload resumes the mode it was left in rather
// than quietly reverting to explore.
restoreMode();
loadMatchup();
