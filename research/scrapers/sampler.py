"""Matchup selection.

Uniform-random pairing wastes the reviewer's time: once a floor entrant has lost
20 straight to trafilatura, the 21st comparison carries no information. So pairs
are drawn with probability proportional to

    weight(i, j) = p(1 - p) / sqrt(1 + n_ij)

where p is the ELO-expected win probability and n_ij is how many times that pair
has already been judged.

* `p(1 - p)` peaks at 0.25 for an even matchup and decays toward 0 as one side
  dominates -- a 400-point gap gets ~1/3 the weight of an even one, 800 points
  ~1/10. That is the "stop re-running last place against first place" property.
* `1 / sqrt(1 + n_ij)` spreads coverage toward under-sampled pairs.

Draws are *proportional*, not argmax: a deterministic pick would lock onto a
single matchup and starve every other pair. An epsilon fraction is drawn
uniformly so early noise cannot permanently freeze the ordering.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from research.scrapers import rating, stats


@dataclass(frozen=True)
class Matchup:
    page: dict
    entrant_a: str
    entrant_b: str


@dataclass(frozen=True)
class AutoDraw:
    page_id: str
    entrant_a: str
    entrant_b: str


class Sampler:
    def __init__(
        self,
        pages: list[dict],
        run_index: list[dict],
        epsilon: float = 0.10,
        candidate_pairs: int = 24,
        seed: int | None = None,
    ) -> None:
        self.pages = {page["id"]: page for page in pages}
        self.epsilon = epsilon
        self.candidate_pairs = candidate_pairs
        self.random = random.Random(seed)

        # Only entrants that actually produced usable output can be compared.
        self.eligible: dict[str, list[str]] = defaultdict(list)
        self.hashes: dict[tuple[str, str], str] = {}
        for row in run_index:
            if row["status"] != "ok":
                continue
            if row["page_id"] not in self.pages:
                continue
            self.eligible[row["page_id"]].append(row["entrant"])
            self.hashes[(row["page_id"], row["entrant"])] = row["output_sha256"] or ""

        self.buckets: dict[str, list[str]] = defaultdict(list)
        for page_id, page in self.pages.items():
            if len(self.eligible.get(page_id, [])) >= 2:
                self.buckets[page["bucket"]].append(page_id)

    # ---------------------------------------------------------------- state

    def _vote_state(self, votes: list[dict]):
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        page_votes: dict[str, int] = defaultdict(int)
        bucket_votes: dict[str, int] = defaultdict(int)
        seen: set[tuple[str, str, str]] = set()
        for vote in votes:
            a, b = sorted((vote["entrant_a"], vote["entrant_b"]))
            pair_counts[(a, b)] += 1
            page_votes[vote["page_id"]] += 1
            seen.add((vote["page_id"], a, b))
            page = self.pages.get(vote["page_id"])
            if page:
                bucket_votes[page["bucket"]] += 1
        return pair_counts, page_votes, bucket_votes, seen

    def _pick_bucket(self, bucket_votes: dict[str, int]) -> str | None:
        """Round-robin across buckets so per-bucket ratings stay balanced.

        Stopping at any vote count then still leaves roughly equal data per
        bucket -- 120 en-wiki votes against 30 eu-other would sink the
        English-vs-EU comparison, which is a headline result.
        """
        available = [name for name, pages in self.buckets.items() if pages]
        if not available:
            return None
        fewest = min(bucket_votes.get(name, 0) for name in available)
        return self.random.choice(
            [name for name in available if bucket_votes.get(name, 0) == fewest]
        )

    def _pick_page(self, bucket: str, page_votes: dict[str, int]) -> str:
        """Within a bucket, prefer the least-judged pages."""
        candidates = self.buckets[bucket]
        fewest = min(page_votes.get(page_id, 0) for page_id in candidates)
        return self.random.choice(
            [page_id for page_id in candidates if page_votes.get(page_id, 0) == fewest]
        )

    # ------------------------------------------------------------- matchups

    def next_matchup(
        self, votes: list[dict], elo: dict[str, float], max_attempts: int = 400
    ) -> tuple[Matchup | None, list[AutoDraw]]:
        """Return the next matchup to show, plus any auto-draws to record.

        Byte-identical outputs are recorded as draws and skipped without ever
        being displayed. That rule is what makes the raw-vs-rendered axis
        affordable: on a static page `X` and `X@rendered` agree exactly, so the
        reviewer only sees the pairing when rendering actually changed something.
        """
        pair_counts, page_votes, bucket_votes, seen = self._vote_state(votes)
        auto_draws: list[AutoDraw] = []

        for _ in range(max_attempts):
            bucket = self._pick_bucket(bucket_votes)
            if bucket is None:
                return None, auto_draws
            page_id = self._pick_page(bucket, page_votes)
            entrants = sorted(self.eligible[page_id])
            if len(entrants) < 2:
                return None, auto_draws

            pairs = [
                (a, b)
                for index, a in enumerate(entrants)
                for b in entrants[index + 1 :]
                if (page_id, a, b) not in seen
            ]
            if not pairs:
                # This page is exhausted; stop offering it and try another.
                self.buckets[bucket] = [
                    other for other in self.buckets[bucket] if other != page_id
                ]
                continue

            self.random.shuffle(pairs)
            candidates = pairs[: self.candidate_pairs]

            if self.random.random() < self.epsilon:
                chosen = self.random.choice(candidates)
            else:
                weights = [self._weight(a, b, elo, pair_counts) for a, b in candidates]
                total = sum(weights)
                if total <= 0:
                    chosen = self.random.choice(candidates)
                else:
                    chosen = self.random.choices(candidates, weights=weights, k=1)[0]

            a, b = chosen
            if self.hashes.get((page_id, a)) == self.hashes.get((page_id, b)):
                auto_draws.append(AutoDraw(page_id, a, b))
                seen.add((page_id, a, b))
                pair_counts[(a, b)] += 1
                continue

            # Randomise which side is shown left; entrant_a is always the left
            # column, so position bias stays measurable afterwards.
            if self.random.random() < 0.5:
                a, b = b, a
            return Matchup(self.pages[page_id], a, b), auto_draws

        return None, auto_draws

    def _weight(
        self,
        a: str,
        b: str,
        elo: dict[str, float],
        pair_counts: dict[tuple[str, str], int],
    ) -> float:
        default = rating.DEFAULT_INITIAL
        probability = rating.expected_score(elo.get(a, default), elo.get(b, default))
        informativeness = probability * (1.0 - probability)
        played = pair_counts.get(tuple(sorted((a, b))), 0)
        return informativeness / ((1.0 + played) ** 0.5)

    # ----------------------------------------------------------- diagnostics

    def pair_pressure(self, votes: list[dict], elo: dict[str, float]) -> list[dict]:
        """Current sampling weight per pair, so the scheme is inspectable."""
        pair_counts, _, _, _ = self._vote_state(votes)
        names = sorted({name for names in self.eligible.values() for name in names})
        rows = []
        for index, a in enumerate(names):
            for b in names[index + 1 :]:
                rows.append(
                    {
                        "pair": f"{a} vs {b}",
                        "played": pair_counts.get((a, b), 0),
                        "weight": round(self._weight(a, b, elo, pair_counts), 5),
                    }
                )
        rows.sort(key=lambda row: row["weight"], reverse=True)
        return rows

    # ------------------------------------------------------------ calibration

    def calibration_matchup(
        self,
        own_votes: list[dict],
        other_votes: list[dict],
        elo: dict[str, float],
    ) -> tuple[Matchup | None, list[AutoDraw], dict]:
        """Re-serve a cell the *other* pool already judged, for agreement scoring.

        Normal sampling deliberately never repeats a (page, pair) cell, and each pool
        samples against only its own history. The consequence is that the human and
        the LLM panel drift into disjoint corners of the corpus -- after 17 and 195
        votes respectively they had exactly one cell in common, so there was no way to
        tell whether the panel's ranking reflected the reviewer's judgement or was
        confident noise.

        This mode restricts candidates to cells the other pool decided and this one
        has not, so the intersection grows on purpose. Selection is stratified by ELO
        gap rather than information-weighted: agreement measured only on close
        matchups would understate it, and only on lopsided ones would flatter it.
        Neither is the number we want, so the sample is spread across the range.

        The left/right order is randomised independently of the order the other pool
        saw. Reusing their order would let a shared position bias inflate agreement,
        and inverting it would deflate it; randomising keeps the two comparable and
        lets the report split agreement by whether the orders happened to match.
        """
        _, _, _, own_seen = self._vote_state(own_votes)
        other_cells = [
            (vote["page_id"], *sorted((vote["entrant_a"], vote["entrant_b"])))
            for vote in other_votes
            # Auto-draws record byte-identical output, not a judgement, so they are
            # not something agreement can be measured against.
            if not vote.get("auto")
        ]

        pool: list[tuple[str, str, str]] = []
        deduped: set[tuple[str, str, str]] = set()
        for cell in other_cells:
            page_id, a, b = cell
            if cell in own_seen or cell in deduped or page_id not in self.pages:
                continue
            if self.hashes.get((page_id, a)) == self.hashes.get((page_id, b)):
                continue
            deduped.add(cell)
            pool.append(cell)

        overlap = own_seen & set(other_cells)
        progress = {"overlap": len(overlap), "remaining": len(pool)}
        if not pool:
            return None, [], progress

        chosen = self._pick_by_gap(pool, overlap, elo)
        page_id, a, b = chosen
        if self.random.random() < 0.5:
            a, b = b, a
        return Matchup(self.pages[page_id], a, b), [], progress

    def _pick_by_gap(
        self,
        pool: list[tuple[str, str, str]],
        overlap: set[tuple[str, str, str]],
        elo: dict[str, float],
    ) -> tuple[str, str, str]:
        """Draw uniformly from whichever ELO-gap third is least represented so far.

        Three bands split at the pool's own gap terciles, so the bands stay populated
        whatever the rating spread happens to be. Preferring the least-sampled band
        keeps the calibration set spanning easy and hard comparisons alike even if the
        reviewer stops after a handful.
        """
        default = rating.DEFAULT_INITIAL

        def gap(cell: tuple[str, str, str]) -> float:
            _, a, b = cell
            return abs(elo.get(a, default) - elo.get(b, default))

        gaps = sorted(gap(cell) for cell in pool)
        low_cut = gaps[len(gaps) // 3]
        high_cut = gaps[(2 * len(gaps)) // 3]

        def band(cell: tuple[str, str, str]) -> int:
            value = gap(cell)
            return 0 if value <= low_cut else (1 if value <= high_cut else 2)

        bands: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
        for cell in pool:
            bands[band(cell)].append(cell)
        done: dict[int, int] = defaultdict(int)
        for cell in overlap:
            done[band(cell)] += 1

        fewest = min(done.get(index, 0) for index in bands)
        candidates = [index for index in bands if done.get(index, 0) == fewest]
        return self.random.choice(bands[self.random.choice(candidates)])

    # ----------------------------------------------------------------- focus

    def focus_matchup(
        self,
        votes: list[dict],
        entrants: tuple[str, ...],
        champion: str | None = None,
        *,
        wiki_only: bool = False,
    ) -> tuple[Matchup | None, list[AutoDraw], dict]:
        """Run a small closed arena over a named set of entrants.

        The information-weighted sampler spreads clicks over 55 pairs, which is right
        for building a leaderboard and wrong for settling one question. With two
        entrants this is a duel; with three or more it is a round-robin, and every
        vote on an entrant outside the set is a vote not spent on the question.

        Pages are restricted to those where **every** named entrant produced output.
        That restriction is what makes the resulting ratings comparable to each other:
        Bradley-Terry over a set where one entrant played an easier subset of pages is
        not a ranking, it is an artifact. It is also why `wikiextractor-v2` can be
        included at all -- against `ours` and `trafilatura` the shared population is
        the 47 wiki pages all three handle, and the standings describe exactly that.

        `wiki_only` narrows the population further, to pages carrying `is_wiki`. It is
        an *instrument* restriction, not a topical preference: the reference pane is a
        live iframe, and only the 50 wiki pages are pinned to `?oldid=` permalinks, so
        only there does the reviewer see the bytes the extractors actually read. On the
        50 unpinned pages the iframe shows today's document against a snapshot fetched
        days earlier, and session 6's non-wiki votes came back 2-2-3 -- what
        coin-flipping looks like -- against 7-0 on the pinned slice. A vote cast
        through a broken instrument is worse than no vote, because it is indexed and
        counted like a real one. Applied before the every-entrant-produced-output
        restriction so the two compose: the population is the wiki pages *and* the
        pages the whole set handled.

        Pairs are drawn round-robin (least-judged first) so no pair starves, and pages
        are stratified on an axis chosen by `_focus_strata` from what actually varies
        in the eligible set.

        This mode is only **semi-blind**: naming the entrants tells the reviewer who is
        involved, though not which column is which. Votes are tagged `focus` so the
        report never pools them silently with fully blind ones.
        """
        names = sorted(set(entrants))
        if len(names) < 2:
            return None, [], {"entrants": names, "error": "need at least two entrants"}
        if champion is not None and champion not in names:
            return None, [], {"entrants": names, "error": f"champion {champion!r} not in set"}

        _, page_votes, _, seen = self._vote_state(votes)
        pairs = [(a, b) for index, a in enumerate(names) for b in names[index + 1 :]]
        # Challenger format: every pair includes the champion, so N newcomers cost N
        # pairs instead of N(N+1)/2. Registering a newcomer is cheap; comparing the
        # newcomers to each other is a question nobody asked, and at ~12 votes/pair the
        # budget only stretches to the question that was asked -- "is this clearly worse
        # than what we ship". Which column the champion lands in is still randomised
        # downstream, so the reviewer cannot learn a side.
        if champion is not None:
            pairs = [pair for pair in pairs if champion in pair]

        # Pages every entrant handled. Restricting up front, rather than per pair,
        # keeps all pairs on one page population so their ratings stay comparable.
        # The wiki filter runs first, so `pages` in the progress payload reports the
        # labellable population rather than a count the reviewer cannot vote through.
        candidates = list(self.pages.items())
        if wiki_only:
            candidates = [(page_id, page) for page_id, page in candidates
                          if page.get("is_wiki")]
        population = [
            page_id
            for page_id, page in candidates
            if set(names) <= set(self.eligible.get(page_id, []))
        ]
        axis, strata_of = self._focus_strata(population)

        # Counted only inside the shared population, matching `subset_standings`. An
        # earlier version counted any vote on the pair, so an explore-mode vote on a
        # page outside the population reported "1 judged, 47 of 47 remaining" and
        # deprioritised that pair in the round-robin on the strength of a vote the
        # standings correctly ignore.
        in_population = set(population)
        pair_set = set(pairs)
        judged_pairs: dict[tuple[str, str], int] = defaultdict(int)
        judged_strata: dict[str, int] = defaultdict(int)
        for vote in votes:
            pair = tuple(sorted((vote["entrant_a"], vote["entrant_b"])))
            if pair not in pair_set or vote["page_id"] not in in_population:
                continue
            judged_pairs[pair] += 1
            judged_strata[strata_of(vote["page_id"])] += 1

        # Open cells: (pair, page) not yet judged and not byte-identical.
        open_cells: list[tuple[tuple[str, str], str]] = []
        for pair in pairs:
            for page_id in population:
                if (page_id, *pair) in seen:
                    continue
                if self.hashes.get((page_id, pair[0])) == self.hashes.get(
                    (page_id, pair[1])
                ):
                    continue
                open_cells.append((pair, page_id))

        progress = {
            "entrants": names,
            "axis": axis,
            "pages": len(population),
            "judged": sum(judged_pairs.values()),
            "remaining": len(open_cells),
            "per_pair": {
                f"{a} vs {b}": {
                    "judged": judged_pairs[(a, b)],
                    "remaining": sum(1 for pair, _ in open_cells if pair == (a, b)),
                }
                for a, b in pairs
            },
            "per_stratum": dict(judged_strata),
        }
        if not open_cells:
            return None, [], progress

        # Least-judged pair that still has an open cell, then least-judged stratum
        # within it, then the least-judged page. Each tie broken randomly so repeated
        # calls do not lock onto one cell.
        live_pairs = {pair for pair, _ in open_cells}
        fewest = min(judged_pairs[pair] for pair in live_pairs)
        pair = self.random.choice(
            sorted(p for p in live_pairs if judged_pairs[p] == fewest)
        )

        candidates = [page_id for p, page_id in open_cells if p == pair]
        by_stratum: dict[str, list[str]] = defaultdict(list)
        for page_id in candidates:
            by_stratum[strata_of(page_id)].append(page_id)
        thinnest = min(judged_strata.get(name, 0) for name in by_stratum)
        stratum = self.random.choice(
            sorted(
                name for name in by_stratum if judged_strata.get(name, 0) == thinnest
            )
        )

        pool = by_stratum[stratum]
        least = min(page_votes.get(page_id, 0) for page_id in pool)
        page_id = self.random.choice([p for p in pool if page_votes.get(p, 0) == least])

        a, b = pair
        if self.random.random() < 0.5:
            a, b = b, a
        progress["serving"] = {"pair": f"{pair[0]} vs {pair[1]}", "stratum": stratum}
        return Matchup(self.pages[page_id], a, b), [], progress

    def _focus_strata(self, population: list[str]):
        """Choose a stratification axis that actually varies in this page set.

        Fixed stratification silently degrades. Splitting `ours vs trafilatura` by page
        shape is exactly right -- `favor_precision=True` empties link-index pages and
        slightly enriches prose, so pooling the two averages the effect away. But the
        three-way set including `wikiextractor-v2` is 47 wiki articles with *zero*
        index-like pages, where that same split puts everything in one bucket and
        balances nothing.

        So the axis is picked from the population: page shape if both classes are
        present, else English vs other language (the SIGIR multilingual question, and
        what varies across the wiki set: 25 English against 22 in five other
        languages), else a single bucket.
        """

        shapes = {stats.is_index_like(self.pages[p].get("shape")) for p in population}
        if len(shapes) > 1:
            return (
                "page shape",
                lambda page_id: (
                    "index-like"
                    if stats.is_index_like(self.pages[page_id].get("shape"))
                    else "article"
                ),
            )

        languages = {(self.pages[p].get("lang") or "") == "en" for p in population}
        if len(languages) > 1:
            return (
                "language",
                lambda page_id: (
                    "english"
                    if (self.pages[page_id].get("lang") or "") == "en"
                    else "other-language"
                ),
            )

        return "none", lambda page_id: "all"
