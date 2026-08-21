"""Aggregates over the arena's votes, runs and snapshots.

Pure computation: every function takes rows (or reads them from the store) and
returns numbers. No HTML, no SVG, no request handling -- which is what lets the
web app, the judge CLI and the sampler all share one definition of "the standings"
instead of each growing its own.

Split out of report.py, which now does nothing but render these.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from research.scrapers import rating, store
from research.scrapers.extractors import REGISTRY


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


# --------------------------------------------------------------- aggregates


def coverage_table(runs: list[dict]) -> dict[str, dict]:
    """How many pages each entrant actually handled.

    This is a first-class result, not a footnote: an entrant that wins on the
    pages it manages but returns nothing on a third of them is not a winner.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        grouped[row["entrant"]].append(row)

    table: dict[str, dict] = {}
    for entrant, rows in grouped.items():
        applicable = [row for row in rows if row["status"] != "n/a"]
        chars = [
            row["output_chars"] or 0 for row in applicable if row["status"] == "ok"
        ]
        table[entrant] = {
            "ok": sum(row["status"] == "ok" for row in rows),
            "empty": sum(row["status"] == "empty" for row in rows),
            "error": sum(row["status"] == "error" for row in rows),
            "na": sum(row["status"] == "n/a" for row in rows),
            "applicable": len(applicable),
            "success_rate": round(
                100.0 * sum(row["status"] == "ok" for row in rows) / len(applicable), 1
            )
            if applicable
            else 0.0,
            "mean_chars": int(statistics.mean(chars)) if chars else 0,
            "median_chars": int(statistics.median(chars)) if chars else 0,
        }
    return table


# A page whose main content is a list of links to elsewhere, or which ships an
# empty shell for JavaScript to fill. Both defeat any main-content model, and both
# are unevenly distributed across the buckets, which is why coverage has to be read
# against shape rather than against language.
INDEX_SHAPE_MARKERS = ("listing", "spa", "front", "index")


def is_index_like(shape: str | None) -> bool:
    lowered = (shape or "").lower()
    return any(marker in lowered for marker in INDEX_SHAPE_MARKERS)


def coverage_by_shape(runs: list[dict], pages: dict[str, dict]) -> list[dict]:
    """Empty-output rate split by page shape rather than by language.

    This exists to prevent a specific wrong conclusion. jusText returns nothing on
    18 of 100 pages, 12 of them in `eu-other`, which reads as "jusText is weak on
    European languages" -- a claim that would line up neatly with the published
    SIGIR 2025 multilingual result and be entirely unfounded here.

    Auditing every one of those pages found the cause is page shape, not language:
    jusText's stoplists load correctly for all seven languages (Slovenian 2160
    words, Hungarian 3253, against English's 453), every page's declared language
    is correct, and on the failures 100% of extracted paragraphs are classified
    boilerplate -- which is honest behaviour for a paragraph-level classifier fed a
    link index. English pages of the same shape (Hacker News, BBC, the Guardian,
    gov.uk) fail identically. The buckets differ because `eu-other` holds 56%
    index-like pages against `en-other`'s 32%: government portals, news front pages
    and tourism SPAs.

    So this table is the control. If an entrant's empty rate is similar on
    index-like pages across both language groups, the apparent language effect is
    corpus composition.
    """
    shapes = {
        page_id: is_index_like(page.get("shape")) for page_id, page in pages.items()
    }
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"index": [], "article": []}
    )
    for row in runs:
        if row["status"] == "n/a" or row["page_id"] not in shapes:
            continue
        key = "index" if shapes[row["page_id"]] else "article"
        grouped[row["entrant"]][key].append(row)

    def empty_rate(rows: list[dict]) -> tuple[int, int, float | None]:
        if not rows:
            return 0, 0, None
        empty = sum(row["status"] == "empty" for row in rows)
        return empty, len(rows), round(100.0 * empty / len(rows), 1)

    table = []
    for entrant, split in grouped.items():
        index_empty, index_n, index_rate = empty_rate(split["index"])
        article_empty, article_n, article_rate = empty_rate(split["article"])
        table.append(
            {
                "entrant": entrant,
                "index_empty": index_empty,
                "index_n": index_n,
                "index_rate": index_rate,
                "article_empty": article_empty,
                "article_n": article_n,
                "article_rate": article_rate,
            }
        )
    table.sort(key=lambda row: -(row["index_rate"] or 0))
    return table


def spearman(first: list[float], second: list[float]) -> float | None:
    """Rank correlation, ties averaged. Hand-rolled to avoid a scipy dependency."""
    if len(first) != len(second) or len(first) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while (
                end + 1 < len(order)
                and values[order[end + 1]] == values[order[position]]
            ):
                end += 1
            average = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = average
            position = end + 1
        return result

    rank_a, rank_b = ranks(first), ranks(second)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    covariance = sum(
        (a - mean_a) * (b - mean_b) for a, b in zip(rank_a, rank_b, strict=True)
    )
    spread_a = math.sqrt(sum((a - mean_a) ** 2 for a in rank_a))
    spread_b = math.sqrt(sum((b - mean_b) ** 2 for b in rank_b))
    if not spread_a or not spread_b:
        return None
    # Clamp: floating-point error puts a perfect correlation at 1.0000000000000002,
    # and this value is rendered into the report.
    return max(-1.0, min(1.0, covariance / (spread_a * spread_b)))


def length_bias(votes: list[dict], names: list[str], runs: list[dict]) -> dict | None:
    """Does this pool's rating just track output length?

    The load-bearing sanity check on any judging pool, automated or human. "Longer"
    is the easiest signal to latch onto and the least meaningful: an extractor that
    removes nothing has maximal content coverage by construction. If ratings
    correlate strongly and positively with median output length, the pool has
    measured verbosity and the leaderboard should not be believed.

    A near-zero or negative correlation is the pass condition -- it says the pool
    rewarded something other than volume.
    """
    if len(votes) < 10:
        return None
    computed = rating.compute(votes, names, bootstrap_rounds=0)
    coverage = coverage_table(runs)
    rated = [
        name
        for name in names
        if computed.records.get(name, rating.Record()).games
        and coverage.get(name, {}).get("median_chars")
    ]
    if len(rated) < 3:
        return None
    correlation = spearman(
        [computed.bt[name] for name in rated],
        [float(coverage[name]["median_chars"]) for name in rated],
    )
    if correlation is None:
        return None
    return {
        "spearman": correlation,
        "entrants": len(rated),
        "verdict": (
            "ratings track output length -- treat this leaderboard with suspicion"
            if correlation > 0.6
            else "no meaningful length bias"
            if correlation > -0.3
            else "ratings favour shorter output, as the rubric intends"
        ),
    }


def judge_pool_comparison(names: list[str]) -> dict:
    """Human ratings against the automated panel's, as two independent instruments.

    The pools are never merged. An LLM panel and a human reviewer measure different
    things, and a blended rating would describe neither; worse, it would let a few
    hundred cheap automated votes swamp a few dozen careful human ones. Reporting
    them side by side turns that into an actual result: how far an automated panel
    can stand in for a human on this task, measured by rank correlation and by
    per-pair agreement.
    """
    human = store.load_votes(judge=store.HUMAN)
    llm = store.load_votes(judge=store.LLM)
    if not human or not llm:
        return {
            "human_votes": len(human),
            "llm_votes": len(llm),
            "spearman": None,
            "pairs": [],
            "agree": None,
        }

    human_bt = rating.bradley_terry(human, names)
    llm_bt = rating.bradley_terry(llm, names)
    rated = [
        name
        for name in names
        if any(name in (v["entrant_a"], v["entrant_b"]) for v in human)
        and any(name in (v["entrant_a"], v["entrant_b"]) for v in llm)
    ]
    correlation = spearman(
        [human_bt[name] for name in rated], [llm_bt[name] for name in rated]
    )

    # Per-pair preference direction, for pairs both pools actually judged.
    def directions(votes: list[dict]) -> dict[tuple[str, str], float]:
        totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for vote in votes:
            if vote.get("auto"):
                continue
            a, b = vote["entrant_a"], vote["entrant_b"]
            score = (
                1.0 if vote["winner"] == "a" else 0.0 if vote["winner"] == "b" else 0.5
            )
            key = (a, b) if a <= b else (b, a)
            totals[key].append(score if (a, b) == key else 1.0 - score)
        return {
            key: sum(scores) / len(scores) for key, scores in totals.items() if scores
        }

    human_direction, llm_direction = directions(human), directions(llm)
    shared = sorted(set(human_direction) & set(llm_direction))
    pairs = []
    agreed = 0
    for key in shared:
        human_score, llm_score = human_direction[key], llm_direction[key]

        # "Agree" means they lean the same way, treating 0.4-0.6 as no lean.
        def lean(value: float) -> int:
            return 0 if 0.4 <= value <= 0.6 else (1 if value > 0.6 else -1)

        same = lean(human_score) == lean(llm_score)
        agreed += int(same)
        pairs.append(
            {
                "pair": f"{key[0]} vs {key[1]}",
                "human": round(human_score, 2),
                "llm": round(llm_score, 2),
                "agree": same,
            }
        )

    return {
        "human_votes": len(human),
        "llm_votes": len(llm),
        "spearman": correlation,
        "rated": rated,
        "human_bt": human_bt,
        "llm_bt": llm_bt,
        "pairs": pairs,
        "agree": round(100.0 * agreed / len(pairs), 1) if pairs else None,
    }


def judge_cell_agreement() -> dict:
    """Do the two pools agree on the *same comparison*? The calibration measure.

    `judge_pool_comparison` compares each pool's aggregate lean per entrant pair,
    which is informative but forgiving: two pools can reach the same overall verdict
    on `trafilatura vs goose3` while disagreeing on most individual pages. This is
    the strict version -- same page, same pair, same document, both pools' verdicts
    lined up -- and it is the only thing that licenses treating the automated pool as
    a stand-in for the reviewer.

    It reports three rates, because they answer different questions:

    * `exact` -- same winning entrant, and when neither pool named a winner, the same
      category (`draw` vs `both_bad`). Deliberately *not* also requiring the same
      left/right slot: the side is randomised independently per pool, so a
      side-sensitive measure is `equivalent` multiplied by a coin flip. An earlier
      version did require it and reported 35% where the real figure was 68%.
    * `equivalent` -- `draw` and `both_bad` collapsed, since `rating.py` scores both
      as half a win. This is the rate that matters for whether the ratings agree.
    * `decisive` -- restricted to cells where *both* pools picked a winner. Draw
      handling differs sharply between the instruments (the panel has never once used
      `both_bad`), so mixing draws in conflates two separate disagreements.

    Cohen's kappa accompanies the equivalent rate: with an unbalanced verdict
    distribution, raw agreement is inflated by chance alone, and kappa nets that out.
    Agreement is also split by whether the two pools happened to see the same
    left/right order, because a position bias shared by both instruments would show up
    as agreement that is really a shared artifact.

    **Kappa is computed over the per-cell choice set, not over entrant names.** An
    earlier version labelled each verdict with the winning entrant's name and reported
    kappa = 0.638. That is wrong, and wrong in the flattering direction: it computes
    chance agreement as if `goose3` were an available answer on a page where goose3 was
    never shown, so the marginals spread across 12 entrants and expected agreement
    collapses to ~11%. A rater actually chooses among three options -- left wins, right
    wins, no winner. Labelling by position in the alphabetically-sorted pair
    (`first`/`second`/`no-winner`) keeps the label space equal to the real choice set;
    on the same 37 cells that gives kappa = 0.272, and the bootstrap CI includes zero.
    The raw rates were never affected.
    """
    human = [v for v in store.load_votes(judge=store.HUMAN) if not v.get("auto")]
    llm = [v for v in store.load_votes(judge=store.LLM) if not v.get("auto")]

    def by_cell(votes: list[dict]) -> dict[tuple[str, str, str], dict]:
        index: dict[tuple[str, str, str], dict] = {}
        for vote in votes:
            a, b = sorted((vote["entrant_a"], vote["entrant_b"]))
            # First verdict wins if a cell was somehow judged twice in one pool: a
            # later re-judgement of the same document is not independent evidence.
            index.setdefault((vote["page_id"], a, b), vote)
        return index

    human_cells, llm_cells = by_cell(human), by_cell(llm)
    shared = sorted(set(human_cells) & set(llm_cells))

    empty = {
        "cells": 0,
        "human_judged": len(human),
        "llm_judged": len(llm),
        "candidates": len(set(llm_cells) - set(human_cells)),
        "exact": None,
        "equivalent": None,
        "decisive": None,
        "decisive_n": 0,
        "kappa": None,
        "same_order": None,
        "same_order_n": 0,
        "flipped_order": None,
        "flipped_order_n": 0,
        "confusion": {},
        "disagreements": [],
    }
    if not shared:
        return empty

    def winner_of(vote: dict) -> str:
        """Verdict as the winning entrant name, or the no-winner category."""
        if vote["winner"] == "a":
            return vote["entrant_a"]
        if vote["winner"] == "b":
            return vote["entrant_b"]
        return vote["winner"]

    pages = {page["id"]: page for page in store.load_pages()}
    labels: list[tuple[str, str]] = []
    rows: list[dict] = []
    exact = same_order_hits = same_order_n = flipped_hits = flipped_n = 0
    decisive_hits = decisive_n = 0
    confusion: dict[str, int] = defaultdict(int)

    for cell in shared:
        h, m = human_cells[cell], llm_cells[cell]
        h_win, m_win = winner_of(h), winner_of(m)
        h_label = "no-winner" if h_win in ("draw", "both_bad") else h_win
        m_label = "no-winner" if m_win in ("draw", "both_bad") else m_win
        agree = h_label == m_label

        exact += int(h_win == m_win)

        # Entrant names identify *which* tool won and drive the disagreement table;
        # kappa needs the choice set the rater actually faced, so it gets positions
        # within the sorted pair instead. See the docstring.
        def slot(label: str) -> str:
            if label == "no-winner":
                return "no-winner"
            return "first" if label == cell[1] else "second"

        labels.append((slot(h_label), slot(m_label)))
        confusion[f"{h['winner']}|{m['winner']}"] += 1

        if h["entrant_a"] == m["entrant_a"]:
            same_order_n += 1
            same_order_hits += int(agree)
        else:
            flipped_n += 1
            flipped_hits += int(agree)

        if h_label != "no-winner" and m_label != "no-winner":
            decisive_n += 1
            decisive_hits += int(agree)

        if not agree:
            page = pages.get(cell[0], {})
            rows.append(
                {
                    "page": page.get("title") or cell[0][:8],
                    "url": page.get("url", ""),
                    "bucket": page.get("bucket", ""),
                    "pair": f"{cell[1]} vs {cell[2]}",
                    "human": h_label,
                    "llm": m_label,
                }
            )

    equivalent = sum(1 for h, m in labels if h == m)
    total = len(labels)

    def rate(hits: int, count: int) -> float | None:
        return round(100.0 * hits / count, 1) if count else None

    return {
        "cells": total,
        "human_judged": len(human),
        "llm_judged": len(llm),
        "candidates": len(set(llm_cells) - set(human_cells)),
        "exact": rate(exact, total),
        "equivalent": rate(equivalent, total),
        "decisive": rate(decisive_hits, decisive_n),
        "decisive_n": decisive_n,
        "kappa": cohens_kappa(labels),
        "same_order": rate(same_order_hits, same_order_n),
        "same_order_n": same_order_n,
        "flipped_order": rate(flipped_hits, flipped_n),
        "flipped_order_n": flipped_n,
        "confusion": dict(sorted(confusion.items())),
        "disagreements": rows,
    }


def cohens_kappa(labels: list[tuple[str, str]]) -> float | None:
    """Agreement above what the two raters' marginal habits would produce by chance.

    Needed because raw agreement flatters an unbalanced task: if both instruments
    pick the same slot 80% of the time, they agree ~68% of the time knowing nothing
    about the document. Kappa of 0 means "no better than that", 1 means perfect.

    Callers must pass labels drawn from the choice set the rater faced. Passing a
    wider space (e.g. entrant names, when only two entrants were on screen) inflates
    the result; `judge_cell_agreement` documents the case where that happened.
    """
    total = len(labels)
    if total == 0:
        return None
    observed = sum(1 for a, b in labels if a == b) / total
    first: dict[str, int] = defaultdict(int)
    second: dict[str, int] = defaultdict(int)
    for a, b in labels:
        first[a] += 1
        second[b] += 1
    expected = sum(
        (first[name] / total) * (second[name] / total)
        for name in set(first) | set(second)
    )
    if expected >= 1.0:
        # Both raters always answered the same single label; agreement is total but
        # kappa is undefined rather than perfect, so say so instead of dividing by 0.
        return None
    return round((observed - expected) / (1.0 - expected), 3)


def head_to_head_by_shape(a: str, b: str, judge: str | None = None) -> dict:
    """One pair's record split into index-like and article pages.

    The overall record for `ours vs trafilatura` averages two populations that behave
    oppositely: `favor_precision=True` returns nothing on link-index pages and a little
    more than stock on prose. A single win rate hides that completely, so the split is
    the result, not a breakdown of it.
    """
    pages = {page["id"]: page for page in store.load_pages()}
    runs = {(row["page_id"], row["entrant"]): row for row in store.load_run_index()}
    pool = store.HUMAN if judge is None else judge
    canonical = tuple(sorted((a, b)))

    split: dict[str, dict] = {
        kind: {"wins_a": 0, "wins_b": 0, "draws": 0, "games": 0, "pages": []}
        for kind in ("index-like", "article")
    }
    for vote in store.load_votes(judge=pool):
        if vote.get("auto"):
            continue
        if tuple(sorted((vote["entrant_a"], vote["entrant_b"]))) != canonical:
            continue
        page = pages.get(vote["page_id"])
        if not page:
            continue
        kind = "index-like" if is_index_like(page.get("shape")) else "article"
        entry = split[kind]
        entry["games"] += 1
        if vote["winner"] in ("draw", "both_bad"):
            entry["draws"] += 1
            winner = "draw"
        else:
            won = vote["entrant_a"] if vote["winner"] == "a" else vote["entrant_b"]
            entry["wins_a" if won == a else "wins_b"] += 1
            winner = won
        entry["pages"].append(
            {
                "url": page["url"],
                "shape": page.get("shape"),
                "winner": winner,
                "chars_a": (runs.get((page["id"], a)) or {}).get("output_chars"),
                "chars_b": (runs.get((page["id"], b)) or {}).get("output_chars"),
                # Duel votes are only semi-blind: the reviewer knew which pair was
                # under test, though not which column. Kept visible rather than
                # silently pooled with the fully blind votes.
                "duel": vote.get("reason") == "duel",
            }
        )

    for entry in split.values():
        decided = entry["wins_a"] + entry["wins_b"]
        entry["win_rate_a"] = (
            round(100.0 * entry["wins_a"] / decided, 1) if decided else None
        )
    total_games = sum(entry["games"] for entry in split.values())
    duel_votes = sum(
        1 for entry in split.values() for row in entry["pages"] if row["duel"]
    )
    return {
        "a": a,
        "b": b,
        "judge": pool,
        "games": total_games,
        "duel_votes": duel_votes,
        "split": split,
    }


def subset_standings(
    entrants: tuple[str, ...], judge: str | None = None, bootstrap_rounds: int = 600
) -> dict:
    """Bradley-Terry standings for a closed set, over the pages all of them handled.

    Distinct from the main leaderboard on purpose. `wikiextractor-v2` consumes wikitext
    and only exists on wiki pages, so its rating in the full board comes from a
    different -- and easier -- page population than everyone else's; ranking it there
    would be a real statistical error, which is why the board keeps it in a wiki slice.
    A closed set fixes that by construction: the population is the pages where *every*
    named entrant produced output, so all of them are rated over the same documents.

    Votes are counted only when both sides are in the set and the page is in that
    shared population, so a stray explore-mode vote on an out-of-population page cannot
    leak in. Semi-blind focus votes are counted but reported separately.
    """
    names = sorted(set(entrants))
    pool = store.HUMAN if judge is None else judge
    pages = {page["id"]: page for page in store.load_pages()}

    ok: dict[str, set[str]] = defaultdict(set)
    for row in store.load_run_index():
        if row["status"] == "ok":
            ok[row["page_id"]].add(row["entrant"])
    # Restricted to live pages, not merely to pages with runs. Dropping a page leaves
    # its extractor runs behind -- repinning the wiki set to `?oldid=` permalinks
    # retired 50 pages whose runs are all still in the table -- so a population built
    # from the run index alone contains ids that `load_pages()` cannot resolve, and the
    # language split below raised KeyError on the first one.
    population = {
        page_id
        for page_id, handled in ok.items()
        if page_id in pages and set(names) <= handled
    }

    relevant = [
        vote
        for vote in store.load_votes(judge=pool)
        if vote["entrant_a"] in names
        and vote["entrant_b"] in names
        and vote["page_id"] in population
    ]
    judged = [vote for vote in relevant if not vote.get("auto")]
    # A vote's `reason` is the arena mode verbatim, and named focus sets tag it with
    # the set name (`wiki3`), not the literal "focus" -- so a hardcoded
    # ("focus", "duel") membership test silently reported every wiki3 vote as
    # blind-explore. Every tagged mode except calibration is semi-blind, so test for
    # that instead of enumerating set names that grow with FOCUS_SETS.
    focus_votes = sum(
        1 for vote in judged if vote.get("reason") and vote["reason"] != "calibration"
    )

    ratings = rating.compute(relevant, names, bootstrap_rounds=bootstrap_rounds)
    matrix = rating.win_matrix(relevant, names)

    languages = {
        ("en" if (pages[p].get("lang") or "") == "en" else "other") for p in population
    }
    per_language: dict[str, dict] = {}
    if len(languages) > 1:
        for label in ("en", "other"):
            subset = [
                vote
                for vote in relevant
                if (
                    (
                        "en"
                        if (pages[vote["page_id"]].get("lang") or "") == "en"
                        else "other"
                    )
                    == label
                )
            ]
            if not [v for v in subset if not v.get("auto")]:
                continue
            fitted = rating.bradley_terry(subset, names)
            per_language[label] = {
                "votes": len([v for v in subset if not v.get("auto")]),
                "bt": {name: round(fitted.get(name, 0.0)) for name in names},
            }

    rows = []
    for name in sorted(names, key=lambda n: -ratings.bt.get(n, 0.0)):
        record = ratings.records.get(name, rating.Record())
        rows.append(
            {
                "entrant": name,
                "bt": round(ratings.bt.get(name, 0.0)),
                "elo": round(ratings.elo.get(name, 0.0)),
                "ci_low": round(ratings.ci_low.get(name, 0.0)),
                "ci_high": round(ratings.ci_high.get(name, 0.0)),
                "wins": record.wins,
                "losses": record.losses,
                "draws": record.draws,
                "games": record.games,
            }
        )

    return {
        "entrants": names,
        "judge": pool,
        "pages": len(population),
        "votes": len(judged),
        "focus_votes": focus_votes,
        "auto_draws": len(relevant) - len(judged),
        "rows": rows,
        "matrix": matrix,
        "per_language": per_language,
    }


def speed_table(runs: list[dict]) -> dict[str, dict]:
    """Per-entrant timing.

    `extract_ms` is the headline: it is the only column that is a property of the
    tool. `input_ms` is shared infrastructure -- the ~0.8 s raw HTTP fetch is
    common to nearly every entrant and is dominated by the remote server, so
    folding it in would drown two orders of magnitude of real difference between
    extractors. It is reported separately by `infrastructure_cost()`, and here
    only as a browser-or-not distinction.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        if row["status"] in ("ok", "empty"):
            grouped[row["entrant"]].append(row)

    table: dict[str, dict] = {}
    for entrant, rows in grouped.items():
        extract = [row["extract_ms"] or 0.0 for row in rows]
        inputs = [row["input_ms"] or 0.0 for row in rows]
        totals = [(row["extract_ms"] or 0.0) + (row["input_ms"] or 0.0) for row in rows]
        chars = sum(row["output_chars"] or 0 for row in rows)
        elapsed = sum(extract) or 1.0

        # Within-page spread across repeats, i.e. how stable the measurement is.
        stability: list[float] = []
        for row in rows:
            raw = row.get("extract_ms_all")
            if not raw:
                continue
            try:
                samples = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                continue
            if len(samples) > 1 and statistics.mean(samples) > 0:
                stability.append(
                    statistics.pstdev(samples) / statistics.mean(samples) * 100
                )

        # Reported as a min, not a max. The precise timing pass aborts if the
        # machine gets busy, which would otherwise leave some cells re-timed with
        # 10 repeats and the rest on the coarse 3-repeat default, while the report
        # claimed 10 for everything. Taking the minimum understates rather than
        # overstates, and `max_repeats` makes a mixed run visible.
        repeat_counts = [
            len(json.loads(row["extract_ms_all"]))
            for row in rows
            if row.get("extract_ms_all")
        ]

        variants = {row["variant"] for row in rows}
        table[entrant] = {
            "median_extract_ms": round(statistics.median(extract), 2)
            if extract
            else None,
            "mean_extract_ms": round(statistics.mean(extract), 2) if extract else None,
            "p95_extract_ms": round(_percentile(extract, 0.95), 2) if extract else None,
            "repeat_cv_pct": round(statistics.median(stability), 1)
            if stability
            else None,
            "repeats": min(repeat_counts, default=0),
            "max_repeats": max(repeat_counts, default=0),
            "median_input_ms": round(statistics.median(inputs), 1) if inputs else None,
            "median_total_ms": round(statistics.median(totals), 1) if totals else None,
            "needs_browser": "rendered" in variants,
            "chars_per_ms": round(chars / elapsed, 1),
            # CPU-only wall clock for a million pages, single-threaded. Excludes
            # fetching, which is network-bound and identical across entrants.
            "cpu_hours_per_1m": round(
                (statistics.median(extract) if extract else 0.0)
                * 1_000_000
                / 1000
                / 3600,
                1,
            ),
        }
    return table


def infrastructure_cost() -> dict[str, float | None]:
    """Shared per-page cost of producing each input variant.

    Reported once rather than charged to each entrant: it is a property of the
    network and the remote server, not of any extractor.
    """
    live = {page["id"] for page in store.load_pages()}
    by_variant: dict[str, list[float]] = defaultdict(list)
    for row in store.load_snapshots():
        # Restrict to selected pages: snapshots also exist for the ~37 candidates
        # that were validated and dropped, which would skew the medians.
        if row["page_id"] not in live:
            continue
        if row["variant"] in ("raw", "rendered", "wikitext") and row["fetch_ms"]:
            by_variant[row["variant"]].append(float(row["fetch_ms"]))
    return {
        "raw_fetch_ms": round(statistics.median(by_variant["raw"]), 1)
        if by_variant["raw"]
        else None,
        "render_ms": round(statistics.median(by_variant["rendered"]), 1)
        if by_variant["rendered"]
        else None,
        "wikitext_ms": round(statistics.median(by_variant["wikitext"]), 1)
        if by_variant["wikitext"]
        else None,
    }


def bucket_vote_counts(votes: list[dict], pages: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for vote in votes:
        page = pages.get(vote["page_id"])
        if page:
            counts[page["bucket"]] += 1
    return dict(sorted(counts.items()))


def rendering_payoff(runs: list[dict]) -> list[dict]:
    """Does the headless browser change anything, and where?

    Pairs each `X` with `X@rendered` and counts pages whose output differs at
    all. Paired with input_ms this answers whether Playwright earns its cost.
    """
    by_key: dict[tuple[str, str], dict] = {
        (row["page_id"], row["entrant"]): row for row in runs
    }
    pairs = [
        ("trafilatura", "trafilatura@rendered"),
        ("resiliparse", "resiliparse@rendered"),
        ("ours@raw-only", "ours"),
    ]
    page_ids = {row["page_id"] for row in runs}

    # `ours` only consumes the rendered DOM where its own ladder escalated. On
    # every other page it read the same raw HTML as `ours@raw-only`, so counting
    # those as "rendering changed the output" would badly overstate the browser's
    # value -- it read 69% when the ladder had escalated on 3% of pages.
    escalated = sum(
        1 for row in runs if row["entrant"] == "ours" and row["variant"] == "rendered"
    )

    results = []
    for raw_name, rendered_name in pairs:
        differed = same = missing = 0
        for page_id in page_ids:
            left = by_key.get((page_id, raw_name))
            right = by_key.get((page_id, rendered_name))
            if not left or not right:
                missing += 1
                continue
            # A missing hash means n/a or error on one side, not a difference.
            # The old sentinel comparison silently scored those as "differed",
            # which would inflate the apparent value of rendering.
            if not left["output_sha256"] or not right["output_sha256"]:
                missing += 1
                continue
            if left["output_sha256"] == right["output_sha256"]:
                same += 1
            else:
                differed += 1
        total = differed + same
        row = {
            "raw": raw_name,
            "rendered": rendered_name,
            "differed": differed,
            "same": same,
            "missing": missing,
            "differed_pct": round(100.0 * differed / total, 1) if total else 0.0,
            "note": "",
        }
        if rendered_name == "ours":
            row["note"] = (
                f"our ladder escalated to the rendered DOM on only {escalated} "
                f"page(s); differences elsewhere are not caused by rendering"
            )
        results.append(row)
    return results


def bucket_ratings(votes: list[dict], pages: dict[str, dict], names: list[str]):
    """Bradley-Terry per slice. The English vs EU-language split is the point."""
    slices = {
        "wiki": lambda page: bool(page["is_wiki"]),
        "non-wiki": lambda page: not page["is_wiki"],
        "English": lambda page: page["lang"] == "en",
        "EU languages": lambda page: page["lang"] != "en",
    }
    output: dict[str, dict] = {}
    for label, predicate in slices.items():
        subset = [
            vote
            for vote in votes
            if vote["page_id"] in pages and predicate(pages[vote["page_id"]])
        ]
        output[label] = {
            "votes": len(subset),
            "ratings": rating.bradley_terry(subset, names) if subset else {},
        }
    return output


SIMILARITY_PATH = Path(__file__).resolve().parent / "deck_data.json"


def fetch_layer_pairs() -> list[dict]:
    """Pairwise agreement between fetchers, from the similarity pass.

    This used to hash the `fetchcmp-*` snapshot rows in the database instead. That
    was wrong in a way that was invisible in the output: the `sha256` column was
    only ever filled for 43-45 of the 100 pages, so the percentages described a
    thin and unstated slice, and they compared raw visible text rather than the
    markdown a converter would actually see. `similarity.py` reads the bodies off
    disk over the full page set and compares extracted markdown, so it is the
    instrument to trust -- see RESEARCH_LOG.md, "Is the fetch-layer agreement
    believable? (session 13)".

    Read from the file `just arena-similarity` writes rather than recomputed:
    the similarity pass re-extracts every body with trafilatura, which is far too
    slow to do inside a request. Returns [] when it has not been run.

    Pairwise rather than all-at-once: a single "do all four agree" count lumps
    together three very different causes (a genuinely equivalent HTTP client, a
    re-serialized DOM, and a browser that executes JavaScript) and so answers
    nothing. The self-pair is the control -- it must read 100%, otherwise the
    dynamic-page filter is not working and every other number is suspect.
    """
    if not SIMILARITY_PATH.exists():
        return []
    block = json.loads(SIMILARITY_PATH.read_text(encoding="utf-8")).get("fetchers", {})
    exact = block.get("exact") or {}
    cosine = block.get("cosine") or {}
    rows = []
    for key, value in exact.items():
        a, b = key.split("|")
        if value is None:
            continue
        near = cosine.get(key) or {}
        rows.append(
            {
                "a": a,
                "b": b,
                # Two questions, not one. `pct` is byte-identical markdown, which is
                # the strict reading and the one a caching or dedup argument needs.
                # `near_pct` is token cosine, which answers "would an extractor see
                # the same document" -- these differ by 40 points for Scrapling, and
                # quoting either alone has read as the other being wrong.
                "pct": round(value["pct"], 1),
                "near_pct": round(near["pct"], 1) if near.get("pct") is not None else None,
                "same": round(value["pct"] * value["n"] / 100.0),
                "total": value["n"],
                "self": a == b,
            }
        )
    return rows


# --------------------------------------------------------- entrants and ratings


def entrant_split() -> tuple[list[str], list[str], set[str]]:
    """All entrants, the full-corpus ones, and the wiki-only ones.

    `wiki_only` entrants are rated over a different page population (the pinned wiki
    slice), so their Bradley-Terry number is not on the same scale as a full-corpus
    one. Ranking them together would be a real statistical error, so every surface
    that shows a leaderboard needs this split -- which is why it lives here rather
    than being spelled out again in the app and in the report.
    """
    names = [entrant.name for entrant in REGISTRY]
    general = [entrant.name for entrant in REGISTRY if not entrant.wiki_only]
    return names, general, {name for name in names if name not in general}


def fit(votes: list[dict], names: list[str], bootstrap_rounds: int = 0) -> rating.Ratings:
    """Fit ratings with the configured priors.

    `bootstrap_rounds=0` skips the confidence intervals, which is what the live
    leaderboard wants: the intervals cost a few hundred refits and the report is the
    surface that shows them.
    """
    config = store.config()["rating"]
    return rating.compute(
        votes,
        names,
        float(config["initial"]),
        float(config["k_factor"]),
        bootstrap_rounds=bootstrap_rounds,
    )
