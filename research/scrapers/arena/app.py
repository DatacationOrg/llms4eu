"""Local FastAPI app for blind pairwise judging.

Runs entirely offline: every asset is served from this process, the reference
panes come off local disk, and no request ever leaves the machine. That is
deliberate -- this is meant to be usable on a plane.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research.scrapers import rating, report, stats, store
from research.scrapers.arena import marks
from research.scrapers.extractors import BY_NAME, REGISTRY
from research.scrapers.sampler import Sampler


CONFIG = store.config()
RATING = CONFIG["rating"]
SAMPLING = CONFIG["sampling"]
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Scraper Arena")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Read-only companion instance: serves the report and leaderboard but refuses to hand
# out matchups. Without this, opening the stats instance would load the judging UI,
# which fetches a matchup on page load -- consuming one of the reviewer's unseen pairs
# and recording any auto-draws it passed on the way. Read from the environment too, so
# it still holds when the app is started through uvicorn directly.
STATS_ONLY = os.environ.get("ARENA_STATS_ONLY", "").lower() in ("1", "true", "yes")

_served_at: dict[str, float] = {}


# Named entrant sets for focus mode. `duel` settles the one pair with a code change
# behind it; `wiki3` is the largest set that can legitimately include
# wikiextractor-v2, since the 47 wiki pages are where all three produce output.
FOCUS_SETS = {
    "duel": ("ours", "trafilatura"),
    "wiki3": ("ours", "trafilatura", "wikiextractor-v2"),
    # Challenger round: three libraries never previously benchmarked, each played
    # against trafilatura only. See CHAMPION_OF.
    "challengers": ("trafilatura", "markitdown", "crawl4ai", "docling"),
    # Same three newcomers against what we actually ship. Stock trafilatura is the
    # weaker of our two configs on precisely the page shape the newcomers keep most of
    # (listings and homepages, median 2.7k chars vs their 9-23k), so a challenger round
    # against stock alone would flatter them on the half of the corpus that matters
    # least and never test them against the pipeline.
    "challengers-ours": ("ours", "markitdown", "crawl4ai", "docling"),
    # The only comparison the similarity matrix leaves open. It split the eleven
    # entrants into two families sharing 44-71% of their words, so cross-family pairs
    # are not a ranking question at all -- a different document wins, and any judge
    # separates them for free. Inside the content-model family these three sit at
    # 88-97% word-cosine of each other, which is close enough that a vote is the only
    # instrument that can tell them apart, and far enough that there is something to
    # tell. It is also the last matchup votes can still move; everything else the
    # similarity matrix already separates. Full round-robin at three pairs,
    # deliberately not champion-pinned: there is
    # no incumbent here to defend, `ours` is one of the three under test, and
    # resiliparse-vs-trafilatura is a question the study has an actual stake in
    # (DCLM/Dolma chose one, FineWeb/HPLT the other).
    "content3": ("ours", "trafilatura", "resiliparse"),
    # The floor, checked properly. `scrapling-md` (markdownify over the whole `<body>`,
    # no content model) tops the Bradley-Terry board, and every win behind that standing
    # is on the unpinned half of the corpus where the reference pane no longer shows the
    # reviewer what the extractors read. On the pinned wiki pages its whole record is
    # one loss to trafilatura, one to resiliparse@rendered and one both-bad -- three
    # votes, which is not enough to answer "is the board wrong, or are we?". This duel
    # answers it on evidence we trust. Two entrants, so no champion pin is needed.
    "floorcheck": ("trafilatura", "scrapling-md"),
    # The comparison the study never actually ran. `ours` calls the working tree, so
    # every vote we hold describes the five uncommitted fixes; the version on the
    # branch (images, links, dedup, favor_precision, bs4 pre-clean) has never faced a
    # judge. Measured, it keeps 38% of the table rows on the wiki slice and collapses
    # 11 of 47 index pages -- but "measurably different" is not "worse to read", and
    # the fixes are only worth shipping if a reviewer agrees. Three entrants, full
    # round-robin: committed-vs-branch is the question, committed-vs-trafilatura is
    # the same question asked without our wrapper in the way, and branch-vs-trafilatura
    # is the 15-draw result this round re-tests as a control. Wiki-only because the
    # loss it looks for is table rows inside articles; the index-page collapses
    # (tagesschau.de 45 chars against 7,474) are on the unpinned half and can be read
    # without a vote.
    "committed": ("ours@committed", "ours", "trafilatura"),
    # Closing the star. With `ours` taken off the board (it is 100% word-identical to
    # trafilatura on wiki, so listing it as a rival entrant only invites the reading
    # that our pipeline lost to the library it *is*), the remaining four entrants are
    # joined only through trafilatura: resiliparse, scrapling-md and ours@committed
    # have played it 15 / 14 / 9 times and each other zero. A Bradley-Terry fit on a
    # star graph cannot order the leaves -- their ratings are whatever their record
    # against the single hub implies, and the two at the bottom land on the same number
    # by construction rather than by evidence. These three pairs are the missing edges.
    "crossplay": ("ours@committed", "resiliparse", "scrapling-md"),
    # The document converters, ordered against each other rather than only against the
    # incumbent. Each has 2 wiki votes, all of them against trafilatura, so on the board
    # they are three more leaves of the same star -- rateable, but not orderable. These
    # are the three cross-edges. They are the tools an audience is most likely to have
    # heard of, which is reason enough to be able to say where they land. Expect
    # confirmation rather than surprise: markitdown and scrapling-md are 98.6%
    # word-identical.
    "converters": ("crawl4ai", "markitdown", "docling"),
    # The one claim on the board with no direct evidence behind it. `ours@committed`
    # finishes below all three document converters, but it has never played any of them:
    # its rating comes from trafilatura, resiliparse and scrapling-md, and theirs comes
    # from trafilatura and each other. Since the settings on the branch put our output
    # *in* the converter family -- 82% word overlap with crawl4ai, 63% with trafilatura --
    # this is the like-for-like fight, and the one an audience is right to ask for.
    # Champion-pinned: the converters' own three edges were judged in `converters`, and
    # re-judging them here would spend votes on an answer we already have.
    "bottomhalf": ("ours@committed", "crawl4ai", "markitdown", "docling"),
}
DUEL_DEFAULT = FOCUS_SETS["duel"]

# Modes where one entrant is pinned to every pair. Without this a four-entrant focus
# set would also pair the newcomers against each other -- six pairs instead of three,
# half of them answering a question nobody asked.
CHAMPION_OF = {
    "challengers": "trafilatura",
    "challengers-ours": "ours",
    "bottomhalf": "ours@committed",
}

# Modes whose population is restricted to the pinned wiki pages. The reference pane is
# a live iframe and only the wiki slice carries `?oldid=` permalinks, so it is the only
# half of the corpus where the reviewer is looking at the bytes the extractors read.
# Session 6 measured the cost of ignoring this: 7-0 on the pinned pages, 2-2-3 on the
# unpinned ones. Once the reference pane serves the saved snapshot instead of the live
# URL, this set should shrink to nothing rather than grow.
WIKI_ONLY = {
    "content3",
    "floorcheck",
    "committed",
    "crossplay",
    "converters",
    "bottomhalf",
}


class VoteRequest(BaseModel):
    page_id: str
    entrant_a: str
    entrant_b: str
    winner: str  # a | b | draw | both_bad
    # Recorded on the vote so a semi-blind duel vote is never mistaken for a fully
    # blind one. Duel mode names the pair up front, which is weaker evidence.
    mode: str = "explore"


def _entrant_names() -> list[str]:
    return [entrant.name for entrant in REGISTRY]


def _sampler() -> Sampler:
    return Sampler(
        pages=store.load_pages(),
        run_index=store.load_run_index(),
        epsilon=float(SAMPLING["epsilon"]),
        candidate_pairs=int(SAMPLING["candidate_pairs"]),
    )


def _elo() -> dict[str, float]:
    elo, _, _ = rating.online_elo(
        store.load_votes(),
        _entrant_names(),
        float(RATING["initial"]),
        float(RATING["k_factor"]),
    )
    return elo


@app.get("/")
def index():
    """Whole-output review is the entry point.

    The comparison is between different libraries over a freshly pinned corpus, where
    the outputs differ everywhere and the whole output is the thing being judged.
    """
    if STATS_ONLY:
        return RedirectResponse("/report")
    return HTMLResponse((STATIC_DIR / "arena.html").read_text(encoding="utf-8"))


# One template rather than one page per mode: every focus round shares the comparison
# UI and differs only in the sampler's entrant set, so a second copy of arena.html --
# or a second copy of this handler -- would drift. The per-round reasoning lives with
# FOCUS_SETS above, next to the entrants it is about.
PRESELECT_SCRIPT = """
<script>
  // Runs after arena.js has restored any remembered mode, so this wins for this URL.
  addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("mode");
    if (select) { select.value = "%s"; select.dispatchEvent(new Event("change")); }
  });
</script>
"""


@app.get("/focus/{mode}", response_class=HTMLResponse)
def focus_page(mode: str) -> HTMLResponse:
    """The page arena with one focus round preselected, as a bookmarkable URL.

    Validated against FOCUS_SETS so a typo 404s instead of silently serving the
    default mode -- which would look like the round being reviewed and record votes
    against the wrong entrant set.
    """
    if mode not in FOCUS_SETS:
        raise HTTPException(status_code=404, detail=f"unknown focus mode: {mode}")
    if STATS_ONLY:
        return RedirectResponse("/report")
    html = (STATIC_DIR / "arena.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("</body>", (PRESELECT_SCRIPT % mode) + "</body>"))


@app.get("/pages", response_class=HTMLResponse)
def page_arena():
    """Alias for `/`, kept so existing links and bookmarks keep working."""
    if STATS_ONLY:
        return RedirectResponse("/report")
    return HTMLResponse((STATIC_DIR / "arena.html").read_text(encoding="utf-8"))


@app.get("/api/matchup")
def matchup(
    mode: str = Query(
        "explore",
        description=(
            "'explore' for new cells, 'calibration' to re-judge panel cells, "
            "'duel'/'wiki3'/'focus' for a closed arena over a named entrant set"
        ),
    ),
    pair: str = Query(
        "",
        description=(
            "focus mode: two or more entrant names, comma-separated. Defaults to the "
            "set named by `mode`, or to the duel pair."
        ),
    ),
) -> JSONResponse:
    if STATS_ONLY:
        raise HTTPException(
            status_code=409,
            detail="this instance is read-only; judge on the labelling instance",
        )
    focus_modes = ("focus", *FOCUS_SETS)
    if mode not in ("explore", "calibration", *focus_modes):
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode}")

    focus_set: tuple[str, ...] = ()
    if mode in focus_modes:
        named = tuple(name.strip() for name in pair.split(",") if name.strip())
        focus_set = named or FOCUS_SETS.get(mode, DUEL_DEFAULT)
        if len(set(focus_set)) < 2:
            raise HTTPException(
                status_code=400, detail="focus mode needs at least two entrant names"
            )
        for name in focus_set:
            if name not in BY_NAME:
                raise HTTPException(status_code=400, detail=f"unknown entrant: {name}")

    votes = store.load_votes()
    sampler = _sampler()
    calibration: dict | None = None
    if focus_set:
        selected, auto_draws, calibration = sampler.focus_matchup(
            votes,
            focus_set,
            champion=CHAMPION_OF.get(mode),
            wiki_only=mode in WIKI_ONLY,
        )
    elif mode == "calibration":
        # Blind on both sides: the panel's verdict is never sent to the client, so the
        # reviewer cannot anchor on it. Agreement is computed afterwards from the two
        # stored votes.
        selected, auto_draws, calibration = sampler.calibration_matchup(
            votes, store.load_votes(judge=store.LLM), _elo()
        )
    else:
        selected, auto_draws = sampler.next_matchup(votes, _elo())

    for draw in auto_draws:
        store.append_vote(
            draw.page_id, draw.entrant_a, draw.entrant_b, "draw", auto=True
        )

    if selected is None:
        human, auto = store.vote_count()
        return JSONResponse(
            {
                "exhausted": True,
                "mode": mode,
                "calibration": calibration,
                "human_votes": human,
                "auto_draws": auto,
                "auto_recorded": len(auto_draws),
            }
        )

    page = selected.page
    left = store.run_output(page["id"], selected.entrant_a) or ""
    right = store.run_output(page["id"], selected.entrant_b) or ""
    _variants = _run_variants(page["id"], (selected.entrant_a, selected.entrant_b))
    human, auto = store.vote_count()
    _served_at[page["id"]] = time.perf_counter()

    return JSONResponse(
        {
            "exhausted": False,
            "mode": mode,
            "calibration": calibration,
            "page": {
                "id": page["id"],
                "url": page["url"],
                "title": page["title"],
                "bucket": page["bucket"],
                "lang": page["lang"],
                "shape": page["shape"],
                "is_wiki": bool(page["is_wiki"]),
            },
            # Deliberately not labelled with entrant names: the client shows A/B only
            # and reveals identity after the vote is recorded.
            "entrant_a": selected.entrant_a,
            "entrant_b": selected.entrant_b,
            "output_a": left,
            "output_b": right,
            "chars_a": len(left),
            "chars_b": len(right),
            # Whole-document diff marks. Blind-safe: it says *where* the two outputs
            # differ, never which library produced which side.
            "marks": marks.to_marked_json(marks.mark(left, right)),
            "frameable": _frameable(page["id"]),
            "reference_variant": _variants[selected.entrant_a],
            "variant_a": _variants[selected.entrant_a],
            "variant_b": _variants[selected.entrant_b],
            "human_votes": human,
            "auto_draws": auto,
            "auto_recorded": len(auto_draws),
        }
    )


def same_origin(request: Request) -> None:
    """Reject cross-site writes.

    The server binds 127.0.0.1, which keeps other machines out but not other *pages*:
    any site open in the reviewer's browser can POST to localhost, and these endpoints
    append to and delete from the vote log. `Sec-Fetch-Site` is set by the browser and
    cannot be spoofed by page script, so it is the whole check. A request with no such
    header is not from a browser (curl, a test client) and is allowed.
    """
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        raise HTTPException(status_code=403, detail="cross-site request refused")


@app.post("/api/vote")
def vote(request: VoteRequest, _: None = Depends(same_origin)) -> JSONResponse:
    if request.winner not in ("a", "b", "draw", "both_bad"):
        raise HTTPException(status_code=400, detail=f"bad winner: {request.winner}")
    for name in (request.entrant_a, request.entrant_b):
        if name not in BY_NAME:
            raise HTTPException(status_code=400, detail=f"unknown entrant: {name}")

    started = _served_at.pop(request.page_id, None)
    latency_ms = (time.perf_counter() - started) * 1000 if started else None
    # Only non-default modes are journalled. Tagging every ordinary vote "explore"
    # would bloat the mirror without distinguishing anything.
    # Every focus variant journals as "focus": the distinguishing fact is that the
    # reviewer knew the entrant set, not which set it was -- and the set is recoverable
    # from the entrant names on the vote itself.
    reason = None if request.mode == "explore" else request.mode
    vote_id = store.append_vote(
        request.page_id,
        request.entrant_a,
        request.entrant_b,
        request.winner,
        auto=False,
        latency_ms=latency_ms,
        reason=reason,
    )
    human, auto = store.vote_count()

    runs = {
        row["entrant"]: row
        for row in store.load_run_index()
        if row["page_id"] == request.page_id
    }

    def reveal(name: str) -> dict:
        row = runs.get(name, {})
        entrant = BY_NAME[name]
        return {
            "name": name,
            "label": entrant.label,
            "layer": entrant.layer,
            "variant": row.get("variant"),
            "extract_ms": row.get("extract_ms"),
            "chars": row.get("output_chars"),
        }

    return JSONResponse(
        {
            "vote_id": vote_id,
            "human_votes": human,
            "auto_draws": auto,
            # Revealed only now that the vote is committed.
            "reveal": {"a": reveal(request.entrant_a), "b": reveal(request.entrant_b)},
        }
    )


@app.get("/api/agreement")
def agreement() -> JSONResponse:
    """Cell-level agreement between the two pools -- the calibration result.

    Served on the stats instance too: it is a read-only measurement, and watching it
    move is the point of running calibration mode in the first place.
    """
    result = stats.judge_cell_agreement()
    # The disagreement list is for the report, where there is room to read it; the
    # modal only needs the headline rates.
    result.pop("disagreements", None)
    return JSONResponse(result)


@app.post("/api/undo")
def undo(_: None = Depends(same_origin)) -> JSONResponse:
    removed = store.undo_last_vote()
    human, auto = store.vote_count()
    return JSONResponse(
        {
            "undone": removed is not None,
            "removed": removed,
            "human_votes": human,
            "auto_draws": auto,
        }
    )


def default_pool(pools: dict[str, int]) -> str:
    """Which rating pool to show when the caller did not ask for one.

    The human pool is the primary metric, but showing it while it holds one or two
    votes renders a board that is almost entirely the 1500 starting prior -- it reads
    "all extractors are equal" when it means "you have barely started". So the
    automated pool stays in view until the human pool has enough votes to say
    something, and the selector switches between them either way.
    """
    if pools.get(store.HUMAN, 0) >= store.MIN_POOL_VOTES:
        return store.HUMAN
    if pools.get(store.LLM, 0) >= store.MIN_POOL_VOTES:
        return store.LLM
    # Nothing meaningful anywhere: show the primary pool rather than invent one.
    return store.HUMAN if pools.get(store.HUMAN) or not pools else store.LLM


def _fetch_layer() -> dict:
    """Fetch-layer measurements, if `just arena-fetchers` has been run.

    Scrapy and the Scrapling fetchers sit a layer below the extractors: they
    retrieve HTML and have no main-content model, so they are never voted on -- a
    fetcher paired with an extractor emits output byte-identical to that extractor
    alone. They are still part of the answer, so the leaderboard reports them as
    measurements alongside the rated entrants rather than leaving them invisible.
    """
    path = store.fetch_layer_json()
    if not path.exists():
        return {"measured": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "name": name,
            "success_rate": row["success_rate"],
            "ok": row["ok"],
            "median_ms": row["median_ms"],
            "total_s": row["total_s"],
            # Anything that is not "ok": blocks, anti-bot pages, transport errors.
            "failures": {
                key: count for key, count in row["outcomes"].items() if key != "ok"
            },
        }
        for name, row in sorted(
            data["fetchers"].items(), key=lambda item: -item[1]["success_rate"]
        )
    ]
    return {
        "measured": True,
        "urls": data["urls"],
        "rows": rows,
        "dynamic_pages": data.get("dynamic_pages", 0),
        "content_agreed": data.get("content_agreed", 0),
        "content_divergent": data.get("content_divergent", 0),
        "pairs": stats.fetch_layer_pairs(),
    }


@app.get("/api/leaderboard")
def leaderboard(
    judge: str | None = Query(
        None,
        description="rating pool: 'human', 'llm', or unset to pick a populated pool",
    ),
) -> JSONResponse:
    if judge is not None and judge not in (store.HUMAN, store.LLM):
        raise HTTPException(status_code=400, detail=f"unknown judge pool: {judge}")

    pools = store.judge_pools()
    if judge is None:
        judge = default_pool(pools)

    votes = store.load_votes(judge=judge)
    names = _entrant_names()
    runs = store.load_run_index()
    pages = {page["id"]: page for page in store.load_pages()}
    coverage = stats.coverage_table(runs)
    speed = stats.speed_table(runs)

    # The wiki-only slice is fitted separately, on wiki-page votes only -- see
    # stats.entrant_split for why the two populations cannot share a ranking.
    _, general, wiki_only = stats.entrant_split()

    computed = stats.fit(votes, general)  # no CIs live; the report computes them

    def describe(name: str, ratings: rating.Ratings) -> dict:
        record = ratings.records.get(name, rating.Record())
        return {
            "name": name,
            "layer": BY_NAME[name].layer,
            "wiki_only": name in wiki_only,
            "elo": round(ratings.elo.get(name, 0.0), 1),
            "bt": round(ratings.bt.get(name, 0.0), 1),
            "wins": record.wins,
            "losses": record.losses,
            "draws": record.draws,
            "games": record.games,
            **coverage.get(name, {}),
            **speed.get(name, {}),
        }

    rows = [describe(name, computed) for name in general]
    # Always ordered by current rating -- the leaderboard is dynamic by design.
    rows.sort(key=lambda row: row["bt"], reverse=True)

    wiki_votes = [
        vote
        for vote in votes
        if vote["page_id"] in pages and pages[vote["page_id"]]["is_wiki"]
    ]
    wiki_computed = stats.fit(wiki_votes, names)
    wiki_rows = [
        describe(name, wiki_computed)
        for name in names
        if wiki_computed.records.get(name, rating.Record()).games
    ]
    wiki_rows.sort(key=lambda row: row["bt"], reverse=True)

    judged, auto = store.vote_count(judge=judge)
    return JSONResponse(
        {
            "rows": rows,
            "wiki_rows": wiki_rows,
            "wiki_votes": len(wiki_votes),
            "fetch_layer": _fetch_layer(),
            "judge": judge,
            "pools": pools,
            "human_votes": judged,
            "auto_draws": auto,
            "pages": len(pages),
            "bucket_votes": stats.bucket_vote_counts(votes, pages),
            "position_bias": rating.position_bias(votes),
            "position_bias_controlled": rating.position_bias_controlled(votes),
        }
    )


@app.get("/api/pressure")
def pressure() -> JSONResponse:
    """Sampling weight per pair, so the active-sampling scheme is inspectable."""
    return JSONResponse({"rows": _sampler().pair_pressure(store.load_votes(), _elo())})


# references 404s. Without its CSS a single broken <img> can expand to fill the
# viewport -- the arXiv snapshot renders as one page-high magnifying glass -- which
# makes the ground-truth reference unusable precisely when it is needed. This caps
# replaced elements and restores a readable column. It is injected only for the
# in-app reference pane; "Open saved page" serves the document untouched.
READER_CSS = """<style id="arena-reader">
  img, svg, video, iframe, canvas { max-width: 240px !important; max-height: 160px !important; }
  body { max-width: 60rem; margin: 0 auto; padding: 1rem 1.25rem;
         font: 14px/1.5 system-ui, sans-serif; background: #fff; color: #111; }
  * { position: static !important; float: none !important; }
  [style*="display:none"], [hidden] { display: none !important; }
</style>
"""


@app.get("/snapshot/{page_id}")
def snapshot(
    page_id: str,
    variant: str = Query("rendered"),
    reader: bool = Query(False, description="cap broken assets so it stays readable"),
) -> HTMLResponse:
    """A saved DOM, served with no network.

    `variant` matters more than it looks. The extractors read `raw` on 97 of 100
    corpus pages, so `rendered` is not the document they were given -- and on a
    page whose content turns over (a news index, a journal listing) the two are
    genuinely different text. Reviewing against the wrong one produces draws that
    say nothing about the extractors.
    """
    if variant not in ("raw", "rendered"):
        raise HTTPException(status_code=400, detail=f"unknown variant: {variant}")
    # Checked against the corpus rather than pattern-matched. Page ids are uuid5
    # (store.page_id), so a real one can never contain a path separator, and this
    # keeps a request parameter from selecting a path at all.
    if page_id not in {row["id"] for row in store.load_pages(include_dropped=True)}:
        raise HTTPException(status_code=404, detail="unknown page")
    path = store.data_dir() / "snapshots" / page_id / f"{variant}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no {variant} snapshot")

    html = path.read_text(encoding="utf-8", errors="replace")
    if reader:
        html = READER_CSS + html
    return HTMLResponse(html)


def _run_variants(page_id: str, entrants: tuple[str, ...]) -> dict[str, str]:
    """Which saved DOM each entrant was actually given."""
    rows = {row["entrant"]: row for row in store.load_runs(page_id)}
    return {name: (rows.get(name, {}).get("variant") or "raw") for name in entrants}


def _frameable(page_id: str) -> bool:
    """Whether the reference pane may show the live URL in an iframe.

    Measured at fetch time from `X-Frame-Options` / CSP `frame-ancestors` and stored on
    the raw snapshot. Unmeasured (null) is treated as not frameable: falling back to the
    saved DOM always shows the right bytes, whereas guessing wrong shows the reviewer an
    empty pane with no explanation.
    """
    for row in store.load_snapshots_for(page_id):
        if row["variant"] == "raw":
            return bool(row.get("frameable"))
    return False



@app.get("/report", response_class=HTMLResponse)
def live_report() -> HTMLResponse:
    # Rebuilt per request, so a browser refresh always reflects votes cast since the
    # page was opened -- but only if the browser actually asks. Without no-store it
    # will happily serve the copy it already has.
    return HTMLResponse(
        report.build_html(),
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


def main() -> None:
    global STATS_ONLY

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the scraper arena.")
    parser.add_argument("--host", default=CONFIG["arena_host"])
    parser.add_argument("--port", type=int, default=int(CONFIG["arena_port"]))
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="serve the report and leaderboard only; refuse to hand out matchups",
    )
    arguments = parser.parse_args()
    if arguments.stats_only:
        STATS_ONLY = True

    print(
        f"{'stats' if STATS_ONLY else 'arena'} on "
        f"http://{arguments.host}:{arguments.port}"
        f"{'/report' if STATS_ONLY else '/'}",
        flush=True,
    )
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
