"""Report rendering, served live at /report and dumpable to a standalone file.

Everything is computed on demand from the append-only vote log, so the report is
usable from vote #1 and the leaderboard always reflects current ratings. The
computation itself lives in stats.py; this module turns it into a page.

Charts are hand-rolled inline SVG: no chart library, no CDN, so the page works
with networking disabled.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from research.scrapers import rating, stats, store
from research.scrapers.extractors import BY_NAME


# ------------------------------------------------------------------ svg bits


def _bar_chart_svg(rows: list[tuple[str, float, float, float]]) -> str:
    """Rating with 95% CI whiskers. Overlapping bars = indistinguishable."""
    if not rows:
        return "<p class='note'>No votes yet.</p>"
    values = [value for _, value, low, high in rows]
    lows = [low or value for _, value, low, high in rows]
    highs = [high or value for _, value, low, high in rows]
    minimum = min(min(values), min(lows)) - 30
    maximum = max(max(values), max(highs)) + 30
    span = max(maximum - minimum, 1.0)

    row_height, label_width, chart_width = 26, 190, 520
    height = row_height * len(rows) + 30
    parts = [
        f'<svg viewBox="0 0 {label_width + chart_width + 70} {height}" '
        f'width="100%" role="img" aria-label="Ratings with confidence intervals">'
    ]
    for index, (name, value, low, high) in enumerate(rows):
        y = index * row_height + 18
        x = label_width + (value - minimum) / span * chart_width
        parts.append(
            f'<text x="{label_width - 8}" y="{y + 4}" text-anchor="end" '
            f'font-size="11" fill="currentColor">{html.escape(name)}</text>'
        )
        parts.append(
            f'<line x1="{label_width}" y1="{y}" x2="{x:.1f}" y2="{y}" '
            f'stroke="var(--accent)" stroke-width="6" opacity="0.55" />'
        )
        if low and high:
            x1 = label_width + (low - minimum) / span * chart_width
            x2 = label_width + (high - minimum) / span * chart_width
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" '
                f'stroke="currentColor" stroke-width="1.5" opacity="0.75" />'
                f'<line x1="{x1:.1f}" y1="{y - 4}" x2="{x1:.1f}" y2="{y + 4}" stroke="currentColor" />'
                f'<line x1="{x2:.1f}" y1="{y - 4}" x2="{x2:.1f}" y2="{y + 4}" stroke="currentColor" />'
            )
        parts.append(
            f'<text x="{label_width + chart_width + 8}" y="{y + 4}" font-size="11" '
            f'fill="currentColor">{value:.0f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _scatter_svg(points: list[tuple[str, float, float]]) -> str:
    """Quality against cost, log-x. Deliberately not collapsed to one score."""
    import math

    points = [(n, q, c) for n, q, c in points if c and c > 0]
    if not points:
        return "<p class='note'>No timing data yet.</p>"
    width, height, pad = 640, 320, 46
    xs = [math.log10(c) for _, _, c in points]
    ys = [q for _, q, _ in points]
    x_min, x_max = min(xs) - 0.15, max(xs) + 0.15
    y_min, y_max = min(ys) - 25, max(ys) + 25

    def sx(value: float) -> float:
        return pad + (value - x_min) / max(x_max - x_min, 1e-9) * (width - 2 * pad)

    def sy(value: float) -> float:
        return (
            height
            - pad
            - (value - y_min) / max(y_max - y_min, 1e-9) * (height - 2 * pad)
        )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Quality versus cost">'
    ]
    parts.append(
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="currentColor" opacity="0.35" />'
    )
    parts.append(
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" '
        f'stroke="currentColor" opacity="0.35" />'
    )
    parts.append(
        f'<text x="{width / 2}" y="{height - 8}" font-size="11" text-anchor="middle" '
        f'fill="currentColor" opacity="0.8">median extract ms/page, 1 CPU (log scale)</text>'
    )
    parts.append(
        f'<text x="12" y="{height / 2}" font-size="11" fill="currentColor" '
        f'opacity="0.8" transform="rotate(-90 12 {height / 2})">BT rating</text>'
    )
    for exponent in range(-1, 5):
        if not (x_min <= exponent <= x_max):
            continue
        x = sx(exponent)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" '
            f'stroke="currentColor" opacity="0.10" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad + 14}" font-size="10" '
            f'text-anchor="middle" fill="currentColor" opacity="0.7">'
            f"{10**exponent:g}</text>"
        )
    for name, quality, cost in points:
        x, y = sx(math.log10(cost)), sy(quality)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--accent)" '
            f'opacity="0.85" />'
        )
        parts.append(
            f'<text x="{x + 7:.1f}" y="{y + 3.5:.1f}" font-size="10" '
            f'fill="currentColor">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# -------------------------------------------------------------------- tables


def _table(headers: list[str], rows: list[list[str]], classes: str = "board") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f'<table class="{classes}"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _win_matrix_html(matrix: dict, names: list[str]) -> str:
    head = "".join(f"<th>{html.escape(n.split('@')[0][:9])}</th>" for n in names)
    rows = []
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append('<td style="opacity:.25">–</td>')
                continue
            cell = matrix.get(a, {}).get(b, {})
            games = cell.get("games", 0)
            if not games:
                cells.append('<td style="opacity:.35">·</td>')
                continue
            rate = (cell["wins"] + 0.5 * cell["draws"]) / games
            shade = f"rgba(59,91,219,{0.10 + 0.55 * rate:.2f})"
            cells.append(
                f'<td style="background:{shade}" title="{cell["wins"]:g}W '
                f'{cell["losses"]:g}L {cell["draws"]}D of {games}">'
                f"{rate * 100:.0f}%</td>"
            )
        rows.append(f"<tr><td>{html.escape(a)}</td>" + "".join(cells) + "</tr>")
    return (
        f'<table class="board"><thead><tr><th>wins as row vs column</th>{head}</tr>'
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )


# ---------------------------------------------------------------------- main

# The methodology section is prose, not code: it changes when the protocol changes,
# never when the aggregates do. Kept as a sibling HTML partial so it can be edited as
# markup, and so this module stays about computation.
METHODS_PATH = Path(__file__).resolve().parent / "methods.html"


TOKENS_PATH = Path(__file__).resolve().parent / "arena" / "static" / "tokens.css"


def _tokens_css() -> str:
    return TOKENS_PATH.read_text(encoding="utf-8")


def _methods_html() -> str:
    return METHODS_PATH.read_text(encoding="utf-8")


def _pair(pairs: list[dict], a: str, b: str) -> dict:
    """One pairwise row, or an empty dict if the similarity pass has not run."""
    return next((row for row in pairs if {row["a"], row["b"]} == {a, b}), {})


def _fetch_layer_html() -> str:
    """The measured fetch-layer comparison, if fetch_layer.py has been run."""
    path = store.fetch_layer_json()
    if not path.exists():
        return (
            "<h2 id='fetchers'>Fetch layer</h2><p class='note'>Not measured yet "
            "&mdash; run <code>just arena-fetchers</code>.</p>"
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for name, row in data["fetchers"].items():
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(row["outcomes"].items()))
        rows.append(
            [
                html.escape(name),
                f"<strong>{row['success_rate']}%</strong>",
                f"{row['ok']}/{data['urls']}",
                f"{row['median_ms'] or '&mdash;'}",
                html.escape(outcomes),
            ]
        )

    divergent = data.get("divergent_pages", [])
    examples = "".join(
        f"<li><code>{html.escape(row['url'][:90])}</code> &mdash; "
        + ", ".join(f"{html.escape(k)} {v:,}" for k, v in row["chars"].items())
        + "</li>"
        for row in divergent[:8]
    )

    agreement = {"pairs": stats.fetch_layer_pairs()}
    _scrapling = _pair(agreement["pairs"], "ours-httpx", "scrapling")
    _stealthy = _pair(agreement["pairs"], "ours-httpx", "scrapling-stealthy")
    pair_rows = [
        [
            f"{html.escape(row['a'])} vs {html.escape(row['b'])}"
            + (" <em>(control)</em>" if row["self"] else ""),
            f"<strong>{row['pct']}%</strong>",
            f"{row['near_pct']}%" if row["near_pct"] is not None else "&mdash;",
            f"{row['same']}/{row['total']}",
        ]
        for row in agreement.get("pairs", [])
    ]
    pair_table = (
        _table(
            ["fetcher pair", "identical", "near-identical", "pages"],
            pair_rows,
        )
        if pair_rows
        else "<p class='note'>No pairwise data &mdash; run <code>just "
        "arena-similarity</code>.</p>"
    )

    return f"""
<h2 id="fetchers">Fetch layer</h2>
<p class="note">Scrapy and Scrapling sit at a different layer from the extractors:
they retrieve HTML and have no main-content model. Voting on them would be
meaningless &mdash; <code>scrapy + html-text</code> emits output byte-identical to
<code>html-text</code> &mdash; so they are measured here instead of rated.</p>
<div class="scroll">{
        _table(["fetcher", "success rate", "ok", "median ms", "outcomes"], rows)
    }</div>

<h3>Do the fetchers actually disagree?</h3>
<p class="note">Two controls make this falsifiable. First, the same client fetches
every URL <em>twice</em>: pages whose content changes between two identical fetches
are dynamic (timestamps, nonces, vote counts, rotating ad slots) and are excluded,
since page dynamism would otherwise masquerade as a fetcher difference. Second, both
sides are converted to markdown before comparison, so that a re-serialized DOM is
not counted as a different document. Getting this wrong is not hypothetical &mdash;
two earlier versions of this measurement reported a divergence count that was
entirely an artifact of the comparison (see Bugs found).</p>
<div class="panel">
<strong>{data.get("dynamic_pages", 0)}</strong> of {data["urls"]} pages excluded as
dynamic.<br />
Of the {data["content_agreed"] + data["content_divergent"]} stable pages:
<strong>{data["content_agreed"]}</strong> identical across all four fetchers,
<strong>{data["content_divergent"]}</strong> divergent somewhere. That headline count
is dominated by the browser: read the pairwise table below instead, which says which
fetchers differ from which.
</div>
<h3>Pairwise agreement</h3>
<p class="note">Two columns because there are two questions. <em>Identical</em> is
byte-identical markdown; <em>near-identical</em> is token cosine, i.e. whether an
extractor would see the same document. They diverge sharply &mdash; a fetcher can
agree on essentially all the content and still not produce the same bytes &mdash; and
quoting one without the other has read as the other being wrong. The self-pair rows
are the control: they must read {
        next(
            (
                f"{row['pct']}%"
                for row in agreement.get("pairs", [])
                if row["self"] and row["a"] == "ours-httpx"
            ),
            "n/a",
        )
    }, which is what licenses reading the rest. Distinct causes are
visible, and lumping them into one "do the fetchers agree" number hides all of
them.</p>
<div class="scroll">{pair_table}</div>
<div class="panel">
<p><strong>1. Two HTTP clients are interchangeable.</strong> httpx and Scrapy return
identical content on {
        next(
            (
                f"{row['pct']}% of pages ({row['same']}/{row['total']})"
                for row in agreement.get("pairs", [])
                if {row["a"], row["b"]} == {"ours-httpx", "scrapy"}
            ),
            "nearly every page",
        )
    }. This is the empirical justification for not making Scrapy a
voting entrant: it cannot change what an extractor sees, so votes spent on it would
have been wasted. Caveat on Scrapy's median latency above &mdash; it runs a scheduler
with concurrency and per-domain politeness, so per-request wall time includes queue
wait and is <em>not</em> a like-for-like comparison against a serial client.</p>
<p><strong>2. Scrapling's HTTP fetcher changes the bytes, not the content.</strong>
It is byte-identical to raw HTTP on {
        f"{_scrapling.get('pct', 0)}% ({_scrapling.get('same', 0)}/"
        f"{_scrapling.get('total', 0)})"
        if _scrapling
        else "an unmeasured share"
    } of pages, and
near-identical on {
        f"{_scrapling['near_pct']}%"
        if _scrapling.get("near_pct") is not None
        else "an unmeasured share"
    }. Those two numbers together are the
finding: the content an extractor would see is the same, and the difference is
representational. This is the empirical reason Scrapling's HTTP fetcher is not a
voting entrant either &mdash; it cannot change what an extractor reads.</p>
<p><strong>3. A browser fetch genuinely differs, and dominates the count.</strong>
StealthyFetcher is byte-identical to raw HTTP on only {
        f"{_stealthy.get('pct', 0)}% ({_stealthy.get('same', 0)}/"
        f"{_stealthy.get('total', 0)})"
        if _stealthy
        else "a small minority"
    } of pages, and
unlike Scrapling it also moves the content: {
        f"{_stealthy['near_pct']}%"
        if _stealthy.get("near_pct") is not None
        else "under 100%"
    } near-identical, sometimes
returning 2&ndash;3&times; the text (JavaScript-rendered). A 100% cosine with a low
byte match means formatting; anything under it means content, and this is the only
pair that shows it. That is the same effect the
<a href="#rendering">rendering payoff</a> section measures from the other direction,
and it is a real difference rather than a representational one.</p>
<p><strong>Methodological caveat.</strong> The dynamism control fetches twice
back-to-back, while the cross-fetcher passes run minutes apart. A page that changes
on a scale of minutes but not seconds therefore passes the control and then shows up
as a fetcher disagreement. The control bounds the artifact; it does not eliminate it,
so treat small pairwise gaps as an upper bound.</p>
</div>
{
        f"<p class='note'>Sample pages, character counts per fetcher:</p><ul class='note'>{examples}</ul>"
        if examples
        else ""
    }
"""


def build_html() -> str:
    # Lead with the pool that has evidence. The human pool is the primary metric by
    # design, but when it is empty every rating collapses to the 1500 start value and
    # the headline leaderboard reads as "all extractors are equal" -- the single most
    # misleading thing this report could say, especially while a populated automated
    # pool sits further down the page. Which pool is in view is stated everywhere it
    # matters; the two are never merged.
    primary_judge = store.HUMAN
    votes = store.load_votes(judge=store.HUMAN)
    if len(votes) < store.MIN_POOL_VOTES:
        automated = store.load_votes(judge=store.LLM)
        if len(automated) > len(votes):
            primary_judge = store.LLM
            votes = automated
    pool_label = "human" if primary_judge == store.HUMAN else "automated (LLM panel)"
    judged_votes = [vote for vote in votes if not vote.get("auto")]
    runs = store.load_run_index()
    pages = {page["id"]: page for page in store.load_pages()}
    config = store.config()

    names, general, _ = stats.entrant_split()
    wiki_names = names

    # Intervals need enough votes to mean anything; below that the bootstrap just
    # resamples the 1500 prior.
    computed = stats.fit(
        votes,
        general,
        bootstrap_rounds=int(config["rating"]["bootstrap_rounds"])
        if len(votes) >= 10
        else 0,
    )
    coverage = stats.coverage_table(runs)
    speed = stats.speed_table(runs)
    shape_coverage = stats.coverage_by_shape(runs, pages)
    judge_pool_html = _judge_pool_html(general)

    ordered = sorted(general, key=lambda n: computed.bt.get(n, 0.0), reverse=True)
    # Measured over the entrants the precise pass actually covers. wikiextractor-v2
    # runs in its own Python 3.10 interpreter and is skipped by that pass, so
    # including it here would report every timing at its coarser repeat count and
    # blame an abort that never happened.
    timed_here = {
        name: row for name, row in speed.items() if not BY_NAME[name].wiki_only
    }
    speed_repeats = min((row["repeats"] for row in timed_here.values()), default=0)
    speed_repeats_max = max(
        (row["max_repeats"] for row in timed_here.values()), default=0
    )
    mixed_timing = (
        ""
        if speed_repeats == speed_repeats_max
        else f"""<p class="note"><strong>Mixed timing protocol.</strong> Repeat counts
range from {speed_repeats} to {speed_repeats_max} across cells, which means the
precise timing pass did not finish &mdash; it aborts when the machine becomes busy,
by design. Compare timings within a column with care until it is re-run to
completion.</p>"""
    )

    # ---- leaderboard
    leaderboard_rows = []
    for name in ordered:
        record = computed.records.get(name, rating.Record())
        cover = coverage.get(name, {})
        low, high = computed.ci_low.get(name), computed.ci_high.get(name)
        interval = f"{low:.0f}&ndash;{high:.0f}" if low and high else "&mdash;"
        layer = BY_NAME[name].layer
        tag = ' <span class="pill">floor</span>' if layer == "floor" else ""
        leaderboard_rows.append(
            [
                f"{html.escape(name)}{tag}",
                f"<strong>{computed.bt.get(name, 0):.0f}</strong>",
                interval,
                f"{computed.elo.get(name, 0):.0f}",
                f"{record.wins:g}/{record.losses:g}/{record.draws}",
                str(record.games),
                f"{cover.get('success_rate', 0)}%",
            ]
        )

    chart_rows = [
        (
            name,
            computed.bt.get(name, 0.0),
            computed.ci_low.get(name, 0.0),
            computed.ci_high.get(name, 0.0),
        )
        for name in ordered
    ]

    # ---- speed
    speed_rows = []
    for name in sorted(speed, key=lambda n: speed[n]["median_extract_ms"] or 0):
        row = speed[name]
        speed_rows.append(
            [
                html.escape(name),
                f"<strong>{row['median_extract_ms']}</strong>",
                f"{row['mean_extract_ms']}",
                f"{row['p95_extract_ms']}",
                f"±{row['repeat_cv_pct']}%"
                if row["repeat_cv_pct"] is not None
                else "&mdash;",
                "yes" if row["needs_browser"] else "no",
                f"{row['chars_per_ms']}",
                f"{row['cpu_hours_per_1m']} h",
            ]
        )
    infrastructure = stats.infrastructure_cost()

    scatter = _scatter_svg(
        [
            (
                name,
                computed.bt.get(name, 0.0),
                speed.get(name, {}).get("median_extract_ms") or 0.0,
            )
            for name in general
        ]
    )

    # ---- coverage
    coverage_rows = []
    for name in sorted(coverage, key=lambda n: -coverage[n]["success_rate"]):
        row = coverage[name]
        coverage_rows.append(
            [
                html.escape(name),
                f"<strong>{row['success_rate']}%</strong>",
                f"{row['ok']}/{row['applicable']}",
                str(row["empty"]),
                str(row["error"]),
                f"{row['median_chars']:,}",
                f"{row['mean_chars']:,}",
            ]
        )

    # ---- per-bucket
    slices = stats.bucket_ratings(votes, pages, general)
    bucket_headers = ["entrant"] + [f"{k} (n={v['votes']})" for k, v in slices.items()]
    bucket_rows = []
    for name in ordered:
        cells = [html.escape(name)]
        for values in slices.values():
            score = values["ratings"].get(name)
            cells.append(f"{score:.0f}" if score is not None else "&mdash;")
        bucket_rows.append(cells)

    # ---- rendering payoff
    payoff_rows = [
        [
            html.escape(f"{row['raw']} → {row['rendered']}"),
            f"<strong>{row['differed_pct']}%</strong>",
            str(row["differed"]),
            str(row["same"]),
            f"<span class='note'>{html.escape(row['note'])}</span>"
            if row["note"]
            else "",
        ]
        for row in stats.rendering_payoff(runs)
    ]

    # ---- wiki slice
    wiki_votes = [
        vote
        for vote in votes
        if vote["page_id"] in pages and pages[vote["page_id"]]["is_wiki"]
    ]
    wiki_bt = rating.bradley_terry(wiki_votes, wiki_names) if wiki_votes else {}
    wiki_rows = [
        [
            html.escape(name),
            f"{wiki_bt.get(name, 0):.0f}",
            f"{coverage.get(name, {}).get('success_rate', 0)}%",
        ]
        for name in sorted(wiki_names, key=lambda n: wiki_bt.get(n, 0.0), reverse=True)
    ]

    matrix_html = _win_matrix_html(rating.win_matrix(votes, ordered), ordered)
    bias = rating.position_bias(votes)
    bias_controlled = rating.position_bias_controlled(votes)
    bucket_counts = stats.bucket_vote_counts(votes, pages)

    language_counts: dict[str, int] = defaultdict(int)
    shape_counts: dict[str, int] = defaultdict(int)
    for page in pages.values():
        language_counts[page["lang"]] += 1
        shape_counts[page["shape"]] += 1

    # Inlined, not linked: the report is dumped to a single standalone file that has
    # to open with no server and no network. Read from the arena's tokens.css so the
    # two surfaces cannot drift -- these values used to be retyped here.
    style = _tokens_css() + """
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
      font-size:14.5px;line-height:1.6}
    .wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
    h1{font-size:1.7em;margin:0 0 4px}
    h2{margin-top:2em;padding-bottom:6px;border-bottom:1px solid var(--line);font-size:1.25em}
    h3{margin-top:1.5em;font-size:1.05em}
    code{font-family:var(--mono);font-size:.9em;background:var(--panel);padding:1px 4px;
      border-radius:3px}
    .lede{color:var(--muted);margin:0 0 8px}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:14px 16px;margin:14px 0}
    .scroll{overflow-x:auto}
    table.board{border-collapse:collapse;width:100%;font-size:12.5px}
    table.board th,table.board td{border-bottom:1px solid var(--line);padding:6px 9px;
      text-align:right;white-space:nowrap}
    table.board th:first-child,table.board td:first-child{text-align:left}
    table.board th{color:var(--muted);font-weight:600}
    .pill{font-family:var(--mono);font-size:10.5px;border:1px solid var(--line);
      border-radius:999px;padding:0 6px;color:var(--muted)}
    .note{color:var(--muted);font-size:12.5px}
    ul{padding-left:20px}
    .ex{border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:12px 0}
    .ex header{background:var(--panel);padding:6px 10px;font-size:12px;color:var(--muted);
      border-bottom:1px solid var(--line)}
    .ex .pair{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
    .ex pre{margin:0;padding:10px;background:var(--panel);font-family:var(--mono);
      font-size:11.5px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;
      overflow:auto}
    @media (max-width:800px){.ex .pair{grid-template-columns:1fr}}
    """

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Scraper Arena — Results</title><style>{style}</style></head>
<body><div class="wrap">
<h1>Scraper Arena &mdash; results</h1>
<p class="lede">Blind pairwise comparison of HTML&rarr;text extractors for the
llms4eu corpus. Ratings below are from the <strong>{pool_label}</strong> vote pool:
{len(judged_votes)} judged votes, {len(votes) - len(judged_votes)} auto-draws, over
{len(pages)} pages.{
        ""
        if primary_judge == store.HUMAN
        else " The human pool is empty, so the automated panel is what is shown; "
        "the two pools are compared against each other further down."
    }</p>

<div class="panel">
<strong>Corpus.</strong> {len(pages)} pages in four buckets
({
        ", ".join(
            f"{k} {v}"
            for k, v in sorted(
                (b, sum(1 for p in pages.values() if p["bucket"] == b))
                for b in {p["bucket"] for p in pages.values()}
            )
        )
    }).
Languages: {", ".join(f"{k} {v}" for k, v in sorted(language_counts.items()))}.
{len(shape_counts)} distinct page shapes.<br />
<strong>Votes per bucket.</strong>
{", ".join(f"{k} {v}" for k, v in bucket_counts.items()) or "none yet"}.<br />
<strong>Position bias.</strong> left column won
{bias["left_win_rate"] * 100:.1f}% of {bias["decided"]} decided votes
(near 50% means none). {
        (
            "Controlled for slot composition &mdash; restricted to the "
            f"{bias_controlled['pairs']} pairs judged in <em>both</em> orders, where "
            "the same entrant is compared against itself &mdash; the first slot is "
            f"worth {bias_controlled['advantage'] * 100:+.1f} points "
            f"({bias_controlled['first_rate'] * 100:.1f}% winning when shown first "
            f"vs {bias_controlled['second_rate'] * 100:.1f}% when shown second, over "
            f"{bias_controlled['first_games']}/{bias_controlled['second_games']} "
            "games). The raw rate above is larger than that because the first slot "
            "also happened to hold stronger entrants; comparing it against fitted "
            "ratings would be circular, since a positional advantage inflates "
            "whichever entrants sat there."
        )
        if bias_controlled["advantage"] is not None
        else "Too few pairs judged in both orders to separate position from strength."
    }
</div>

<h2 id="leaderboard">Leaderboard <span class="pill">{pool_label}</span></h2>
<p class="note">Bradley-Terry rating with 95% bootstrap intervals. Where intervals
overlap, the entrants are not distinguishable at this vote count. Ordered by rating,
always.</p>
<div class="panel">{_bar_chart_svg(chart_rows)}</div>
<div class="scroll">{
        _table(
            ["entrant", "BT", "95% CI", "ELO", "W/L/D", "games", "pages handled"],
            leaderboard_rows,
        )
    }</div>

<h2 id="speed">Speed</h2>
<p class="note"><strong>How this was measured.</strong> Pinned to a single CPU on
an otherwise idle machine, {speed_repeats} timed repeats per page after warmups,
{mixed_timing}
median across the {len(pages)} pages. The pass refuses to run above a load average
of {
        config["timing"][
            "max_cpu_busy_percent"
        ]:.0f}% CPU, because contention inflates timings
by an amount that varies per entrant and would silently reorder this table. The
&plusmn; column is the median within-page coefficient of variation across repeats
&mdash; a measure of how stable each number is.</p>
<p class="note"><code>extract ms</code> is the only column that is a property of
the tool, so it is the headline. Fetching is deliberately <em>not</em> folded in:
it is network-bound, dominated by the remote server, and near-identical across
entrants, so including it would drown two orders of magnitude of genuine
difference. It is stated once below instead.</p>
<div class="scroll">{
        _table(
            [
                "entrant",
                "extract ms (median)",
                "mean",
                "p95",
                "repeat spread",
                "needs browser",
                "chars/ms",
                "1M pages CPU",
            ],
            speed_rows,
        )
    }</div>
<div class="panel">
<strong>Shared infrastructure cost per page</strong> (median, not attributed to any
entrant): raw HTTP fetch <strong>{infrastructure["raw_fetch_ms"]} ms</strong>,
Playwright render <strong>{infrastructure["render_ms"]} ms</strong>.
Entrants marked &ldquo;needs browser&rdquo; pay the render cost instead of the
fetch cost &mdash; roughly {
        (infrastructure["render_ms"] or 1)
        / (infrastructure["raw_fetch_ms"] or 1):.1f}&times;
more &mdash; which is why the <a href="#rendering">rendering payoff</a> section
matters: that cost is only worth paying where the output actually changes.
</div>
<p class="note"><code>chars/ms</code> measures throughput, not quality: an entrant
that emits boilerplate scores well on it by definition, which is why the floors
lead that column.</p>
<h3>Quality against cost</h3>
<p class="note">Deliberately not collapsed into a single quality-per-millisecond
score &mdash; the exchange rate between the two depends on which pipeline is asking.
Entrants toward the upper left are on the Pareto frontier.</p>
<div class="panel">{scatter}</div>

<h2 id="coverage">Coverage &amp; failures</h2>
<p class="note">How many pages each entrant actually handled. An entrant that wins
on the pages it manages but returns nothing on a third of them is not a winner, so
this table belongs next to the leaderboard rather than in an appendix.</p>
<div class="scroll">{
        _table(
            [
                "entrant",
                "pages handled",
                "ok / applicable",
                "empty",
                "error",
                "median chars",
                "mean chars",
            ],
            coverage_rows,
        )
    }</div>
<p class="note">Denominators differ: <code>wikiextractor-v2</code> is only
applicable to the 50 wiki pages, so its percentage is not comparable with the
others' &mdash; hence the explicit <code>ok / applicable</code> column.
<code>median chars</code> is descriptive, not a quality measure: the floors emit
the most text precisely because they remove no boilerplate.</p>

<h3>Empty output by page shape, not by language</h3>
<p class="note">This table is here to block a conclusion the raw counts invite.
jusText returns nothing on 18 of 100 pages and 12 of those sit in
<code>eu-other</code>, which reads as a multilingual weakness &mdash; and would line
up suspiciously well with the SIGIR 2025 result cited below. Auditing every one of
those pages showed the cause is page <em>shape</em>. Stoplists load correctly for
all seven languages (Slovenian 2160 words, Hungarian 3253, against English's 453),
every page's declared language is correct, and on each failure <em>100% of
extracted paragraphs are classified boilerplate</em> &mdash; honest behaviour for a
paragraph-level classifier handed a link index. English pages of the same shape
&mdash; Hacker News, BBC News, the Guardian, gov.uk &mdash; fail identically. The
buckets differ because <code>eu-other</code> is 56% index-like against
<code>en-other</code>'s 32%: government portals, news front pages and tourism
SPAs.</p>
<div class="scroll">{
        _table(
            [
                "entrant",
                "empty on index-like pages",
                "empty on article pages",
            ],
            [
                [
                    html.escape(row["entrant"]),
                    f"{row['index_empty']}/{row['index_n']}"
                    + (
                        f" ({row['index_rate']}%)"
                        if row["index_rate"] is not None
                        else ""
                    ),
                    f"{row['article_empty']}/{row['article_n']}"
                    + (
                        f" ({row['article_rate']}%)"
                        if row["article_rate"] is not None
                        else ""
                    ),
                ]
                for row in shape_coverage
            ],
        )
    }</div>
<p class="note"><strong>goose3 is the exception, and for an interesting reason.</strong>
It is the only entrant that fails more on <em>article</em> pages than the shape
split alone would predict, and auditing all 17 of its empties found every one to be
genuine goose3 behaviour rather than an adapter artifact. The mechanism is that
goose3's <code>post_cleanup()</code> keeps only a hardcoded whitelist of tags
(<code>p, ul, ol, h1&ndash;h6</code>) and deletes the chosen content node's other
children. So it locates the right node and then destroys it: content nested in
<code>&lt;section&gt;</code> (Wikipedia via Parsoid), <code>&lt;pre&gt;</code> (RFC
2616, 351 blocks), <code>&lt;tr&gt;</code> (table-structured articles) or
<code>&lt;blockquote&gt;</code> (the arXiv abstract) all vanish. Widening that
whitelist was tested: it recovers 6 of the 17 with no new empties, but changes output
length on 47 of the 83 working pages, often by large multiples (Berlin
782&nbsp;&rarr;&nbsp;160,429 chars). That is rewriting the tool under test rather
than fixing our adapter, so it was <em>not</em> applied &mdash; it is reported here
as an upstream limitation.</p>
<p class="note">Index-like means the shape label mentions a listing, an index, a
front page or a JavaScript-shell SPA &mdash; 22 of the 100 pages. Two of those SPAs
(<code>gradrajhenburg.si</code>) ship a ~2.5&nbsp;KB script-only shell with no text
nodes at all in the raw snapshot, so <em>every</em> raw-input entrant is empty there
by necessity; that is a fact about the page, not about the tool.</p>

{judge_pool_html}

<h2 id="ours">Where <code>ours</code> loses, and why</h2>
<p class="note">The automated pool ranks <code>ours</code> and
<code>ours@raw-only</code> near the bottom, below stock <code>trafilatura</code>.
Read the judges' stated reasons rather than the rank alone: the deficit is not one
weakness but two unrelated ones, and only one of them is a quality problem.</p>
<p class="note"><strong>1. Output size on prose pages &mdash; a configuration
choice, not a defect.</strong> On wiki pages <code>ours</code> emits a median
129,772 characters against stock trafilatura's 79,364, i.e. <strong>1.6&times;</strong>
as many for the same article. The cause is our own call: <code>extract_markdown</code>
passes <code>include_links=True</code>, <code>include_images=True</code> and
<code>include_formatting=True</code>, so every wikilink becomes
<code>[text](absolute-url "title")</code>. Judges penalised this repeatedly under
token economy &mdash; "far more token-economical without stripped-link bloat",
"fragments sentences around links". Whether that is wrong depends on the consumer:
links are wasted tokens for a pretraining corpus and useful signal for a RAG index
that follows them. It is a knob, and this benchmark says the default setting costs
us.</p>
<p class="note"><strong>2. The listing branch drops prose &mdash; a real
weakness.</strong> On index-like pages <code>ours</code> emits a median 1,760
characters where resiliparse gives 2,696 and the no-op floor gives 7,930. Our
ladder classifies these pages as listings and emits a link list, discarding the
surrounding prose. Four of the twelve <code>ours</code> matchups lost precisely
this way, with judges noting dropped headlines, agenda entries and section text.
This is not a formatting preference; content is being lost.</p>
<p class="note">Both effects are invisible in the coverage table, because
<code>ours</code> is one of only three entrants that returned non-empty output on
all 100 pages. It never fails; it is judged worse when it succeeds. That is the
case for reading the leaderboard and the coverage table together.</p>

<h2 id="buckets">Per-slice ratings</h2>
<p class="note">The English vs EU-language split is the point of this benchmark:
the one genuinely multilingual study in the literature
(<a href="https://dl.acm.org/doi/10.1145/3726302.3730234">SIGIR 2025</a>) finds
trafilatura's macro-F1 falling to ~0.77 on non-English text, while every other
published comparison is English-dominant.</p>
<div class="scroll">{_table(bucket_headers, bucket_rows)}</div>

<h2 id="rendering">Rendering payoff</h2>
<p class="note">Share of pages where running the same extractor over the
Playwright-rendered DOM changed the output at all. Where it changed nothing, the
browser bought nothing &mdash; and its cost is in the <code>input ms</code> column
above.</p>
<div class="scroll">{
        _table(
            [
                "pairing",
                "output differed",
                "pages differed",
                "pages identical",
                "caveat",
            ],
            payoff_rows,
        )
    }</div>

{_fetch_layer_html()}

<h2 id="wiki">Wiki slice</h2>
<p class="note">The only table where <code>wikiextractor-v2</code> appears: it
consumes wikitext rather than HTML, so a rating computed over a different page
population does not belong in the overall leaderboard. See the Wikiextractor
caveats under Limitations &mdash; especially the absent template expansion.</p>
<div class="scroll">{
        _table(["entrant", "BT (wiki pages)", "pages handled"], wiki_rows)
    }</div>

<h2 id="matrix">Head-to-head</h2>
<p class="note">Row's win rate against column, draws counted as half. Hover a cell
for the raw tally.</p>
<div class="scroll">{matrix_html}</div>

{_examples_html(votes, pages, runs)}

{_methods_html()}

<h2>Reproducing</h2>
<pre class="panel" style="white-space:pre-wrap"><code>just arena-snapshot   # fetch the corpus once (network)
just arena-extract    # run every entrant, timed
just arena-wiki       # the Python 3.10 wikiextractor pass
just arena            # serve the arena at 127.0.0.1:{config["arena_port"]}
just arena-report     # dump this page standalone</code></pre>
<p class="note">Generated from {len(votes)} vote records and {len(runs)} extractor
runs. Ratings recompute from the append-only log on every load.</p>
</div></body></html>"""


def _judge_pool_html(names: list[str]) -> str:
    """Human ratings beside the automated panel's, never merged into one number."""
    comparison = stats.judge_pool_comparison(names)
    if not comparison["llm_votes"]:
        return ""

    # The automated pool stands on its own. Before the human has voted at all, the
    # main leaderboard above is flat at 1500, so this is the only place any rating
    # is visible -- gating it behind "both pools populated" would hide the one
    # result that exists.
    llm_votes = store.load_votes(judge=store.LLM)
    llm_only = rating.compute(llm_votes, names, bootstrap_rounds=0)
    llm_ordered = sorted(
        (name for name in names if llm_only.records.get(name, rating.Record()).games),
        key=lambda n: llm_only.bt.get(n, 0.0),
        reverse=True,
    )
    own_table = ""
    if llm_ordered:
        own_table = (
            "<h3>Automated panel leaderboard</h3>"
            "<div class='scroll'>"
            + _table(
                ["entrant", "BT", "ELO", "W/L/D", "games"],
                [
                    [
                        html.escape(name),
                        f"<strong>{llm_only.bt.get(name, 0):.0f}</strong>",
                        f"{llm_only.elo.get(name, 0):.0f}",
                        f"{llm_only.records[name].wins:g}/"
                        f"{llm_only.records[name].losses:g}/"
                        f"{llm_only.records[name].draws}",
                        str(llm_only.records[name].games),
                    ]
                    for name in llm_ordered
                ],
            )
            + "</div>"
        )

    bias = stats.length_bias(llm_votes, names, store.load_run_index())
    bias_html = ""
    if bias:
        bias_html = (
            "<h3>Length-bias check on the automated pool</h3>"
            "<p class='note'>Rank correlation between automated rating and median "
            f"output length is <strong>{bias['spearman']:+.2f}</strong> across "
            f"{bias['entrants']} entrants &mdash; {html.escape(bias['verdict'])}. "
            "This is the check that decides whether the automated leaderboard is "
            'worth reading at all: "longer" is the easiest signal for a judge to '
            "latch onto and the least meaningful one, since an extractor that "
            "removes nothing has perfect coverage by construction. A strong "
            "positive correlation would mean the pool had measured verbosity "
            "rather than quality.</p>"
        )

    correlation = comparison["spearman"]
    if comparison["human_votes"] < 10 or correlation is None:
        verdict = (
            "<p class='note'>No comparison yet: a rank correlation needs a few dozen "
            f"votes in each pool, and the human pool holds "
            f"{comparison['human_votes']} against the automated pool's "
            f"{comparison['llm_votes']}. The automated ratings below stand on their "
            "own until then.</p>"
        )
        table = ""
    else:
        strength = (
            "strong"
            if correlation >= 0.8
            else "moderate"
            if correlation >= 0.5
            else "weak"
        )
        verdict = (
            f"<p class='note'>Rank correlation between the two pools is "
            f"<strong>{correlation:+.2f}</strong> ({strength}) over "
            f"{len(comparison['rated'])} entrants rated by both. "
            + (
                "That is high enough that the automated panel is a usable proxy for "
                "bulk comparisons, though the human pool remains the reference."
                if correlation >= 0.8
                else "That is not high enough to treat the automated panel as a "
                "substitute for human judgement; read it as a second opinion only."
            )
            + "</p>"
        )
        rows = [
            [
                html.escape(name),
                f"{comparison['human_bt'][name]:.0f}",
                f"{comparison['llm_bt'][name]:.0f}",
                f"{comparison['llm_bt'][name] - comparison['human_bt'][name]:+.0f}",
            ]
            for name in sorted(
                comparison["rated"],
                key=lambda n: comparison["human_bt"][n],
                reverse=True,
            )
        ]
        table = (
            "<div class='scroll'>"
            + _table(["entrant", "human BT", "automated BT", "delta"], rows)
            + "</div>"
        )

    agreement = ""
    if comparison["pairs"]:
        agreement = (
            f"<p class='note'>On the {len(comparison['pairs'])} pairs both pools "
            f"judged, they lean the same way "
            f"<strong>{comparison['agree']}%</strong> of the time (a lean between "
            "0.4 and 0.6 counts as no lean).</p>"
        )
    calibration = _calibration_html()

    return f"""
<h2 id="judges">Human votes vs the automated panel</h2>
<p class="note">Two independent rating pools over the same corpus: this reviewer's
votes, and a panel of LLM judges each shown one blinded matchup. They are
<strong>never merged</strong>. An LLM panel and a human reviewer measure different
things, and a blended rating would describe neither &mdash; worse, a few hundred
cheap automated votes would swamp a few dozen careful human ones. Each pool also
samples against only its own history, so automated judging never consumes matchups
the human has not seen.</p>
<p class="note">The automated judges see a tighter brief than the human does: a
neutral tag-strip of the page instead of a rendered one, and long outputs cut to
three windows (head, middle, tail) with the true character count stated. That makes
them weaker on coverage for very long outputs, which is the main reason to keep the
human pool as the reference rather than the tiebreaker.</p>
{verdict}
{own_table}
{bias_html}
{table}
{agreement}
{calibration}
<p class="note">Pool sizes: {comparison["human_votes"]} human,
{comparison["llm_votes"]} automated.</p>"""


def _calibration_html() -> str:
    """Cell-level agreement: the check that decides if the panel means anything.

    Kept separate from the pair-lean number above because it answers a stricter
    question. Two pools can agree on which of two extractors is better overall while
    disagreeing on most individual documents, and it is the per-document agreement
    that determines whether the panel's votes can stand in for the reviewer's.
    """
    a = stats.judge_cell_agreement()
    if not a["llm_judged"]:
        return ""

    if not a["cells"]:
        return (
            "<h3>Calibration: not yet measurable</h3>"
            "<p class='note'>The two pools have judged <strong>no comparison in "
            "common</strong>, so there is no evidence either way on whether the "
            "automated panel matches this reviewer's judgement. That is a direct "
            "consequence of the sampling design: cells are never repeated within a "
            "pool and each pool samples against only its own history, so the two "
            f"drifted apart. {a['candidates']} panel-judged comparisons are available "
            "to re-check; the arena's <strong>calibration mode</strong> serves them "
            "blind, and this section becomes a measurement once it has been run.</p>"
        )

    kappa = a["kappa"]
    if kappa is None:
        strength = (
            "Cohen's &kappa; is undefined here (every shared comparison drew the same "
            "verdict), so the rate above is not yet distinguishable from chance."
        )
    elif kappa >= 0.6:
        strength = (
            "By Landis&ndash;Koch that is <strong>substantial</strong> agreement: the "
            "automated pool is a defensible stand-in for bulk comparisons, with the "
            "human pool still the reference for the headline result."
        )
    elif kappa >= 0.4:
        strength = (
            "By Landis&ndash;Koch that is <strong>moderate</strong> agreement: usable "
            "as a pre-screen to decide where the reviewer should look, not as "
            "evidence about the reviewer's preferences."
        )
    else:
        strength = (
            "By Landis&ndash;Koch that is <strong>slight to fair</strong> agreement: "
            "the automated leaderboard should not be read as an estimate of this "
            "reviewer's ranking, and the automated pool's conclusions need human "
            "confirmation before they are reported as findings."
        )

    order_note = ""
    if a["same_order_n"] and a["flipped_order_n"]:
        gap = (a["same_order"] or 0) - (a["flipped_order"] or 0)
        order_note = (
            f"Agreement is {a['same_order']}% on the {a['same_order_n']} cells where "
            f"both pools happened to see the same left/right order and "
            f"{a['flipped_order']}% on the {a['flipped_order_n']} where the order was "
            f"flipped, a gap of {gap:+.1f} points. "
            + (
                "A large positive gap would mean part of the agreement is a shared "
                "position bias rather than shared judgement; this one is small "
                "enough not to explain the result."
                if abs(gap) < 15
                else "That gap is large enough to matter: some of the apparent "
                "agreement is the two instruments sharing a first-slot preference, "
                "not agreeing about the documents."
            )
        )

    rows = [
        [
            "exact",
            f"{a['exact']}%",
            str(a["cells"]),
            "same winning entrant, and the same no-winner category when neither "
            "named one",
        ],
        [
            "equivalent",
            f"<strong>{a['equivalent']}%</strong>",
            str(a["cells"]),
            "same winner, with <code>draw</code> and <code>both_bad</code> collapsed "
            "&mdash; the comparison that matches how the ratings score them",
        ],
        [
            "decisive only",
            f"{a['decisive']}%" if a["decisive"] is not None else "&mdash;",
            str(a["decisive_n"]),
            "restricted to cells where both pools named a winner, so differing "
            "draw habits do not enter",
        ],
        [
            "Cohen's &kappa;",
            f"{kappa:+.2f}" if kappa is not None else "&mdash;",
            str(a["cells"]),
            "agreement above what the two pools' marginal habits give by chance, "
            "over the three-option choice set each cell actually offered "
            "(left / right / no winner)",
        ],
    ]

    disagreements = ""
    if a["disagreements"]:
        disagreements = (
            "<h4>Where they disagreed</h4><div class='scroll'>"
            + _table(
                ["page", "bucket", "pair", "human picked", "panel picked"],
                [
                    [
                        html.escape(row["page"][:60]),
                        html.escape(row["bucket"]),
                        html.escape(row["pair"]),
                        html.escape(row["human"]),
                        html.escape(row["llm"]),
                    ]
                    for row in a["disagreements"]
                ],
            )
            + "</div>"
        )

    return f"""
<h3>Calibration: agreement on the same comparison</h3>
<p class="note">Measured over the {a["cells"]} (page, pair)
cell{"" if a["cells"] == 1 else "s"} both pools have
judged &mdash; the same document, the same two outputs, scored independently. This is
the strict test. The pair-lean figure above tolerates disagreeing on most individual
pages so long as the aggregate verdict matches; this does not.</p>
<div class="scroll">{_table(["measure", "rate", "n", "meaning"], rows)}</div>
<p class="note">{strength} {order_note}</p>
<p class="note">{a["candidates"]} panel-judged comparisons remain un-rechecked. Note
that the panel has used <code>both_bad</code> on
{sum(count for key, count in a["confusion"].items() if key.endswith("|both_bad"))} of
its shared cells: an instrument that always names a winner will manufacture a signal
on pages where both outputs are unusable, which is exactly where the floor entrants
should be exposed.</p>
{disagreements}"""


def _examples_html(votes: list[dict], pages: dict, runs: list[dict]) -> str:
    """Side-by-side excerpts from the pages where entrants disagreed most."""
    by_page: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        if row["status"] == "ok":
            by_page[row["page_id"]].append(row)

    spread = []
    for page_id, rows in by_page.items():
        if page_id not in pages or len(rows) < 3:
            continue
        lengths = [row["output_chars"] or 0 for row in rows]
        if not max(lengths):
            continue
        spread.append((max(lengths) - min(lengths), page_id, rows))
    spread.sort(reverse=True, key=lambda item: item[0])

    blocks = []
    for _, page_id, rows in spread[:3]:
        page = pages[page_id]
        rows.sort(key=lambda row: row["output_chars"] or 0)
        shortest, longest = rows[0], rows[-1]
        left = store.run_output(page_id, shortest["entrant"]) or ""
        right = store.run_output(page_id, longest["entrant"]) or ""
        blocks.append(f"""
<div class="ex">
  <header>{html.escape(page["bucket"])} &middot; {html.escape(page["lang"])} &middot;
  {html.escape(page["shape"])} &middot; {html.escape(page["url"])}</header>
  <div class="pair">
    <div><pre>{html.escape(shortest["entrant"])} — {shortest["output_chars"]:,} chars
────────────────
{html.escape(left[:1400])}</pre></div>
    <div><pre>{html.escape(longest["entrant"])} — {longest["output_chars"]:,} chars
────────────────
{html.escape(right[:1400])}</pre></div>
  </div>
</div>""")

    return (
        "<h2 id='examples'>Examples</h2>"
        "<p class='note'>Pages with the widest spread in output length across "
        "entrants — where the shortest and longest extraction disagree most "
        "about what the page contains. Excerpts truncated.</p>" + "".join(blocks)
    )


def main() -> None:
    target = store.data_dir() / "report.html"
    target.write_text(build_html(), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

