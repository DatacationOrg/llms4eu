"""Tests for the automated judging harness and the report helpers it feeds."""

from research.scrapers import stats
from research.scrapers.judge import clip


def test_clip_returns_short_text_unchanged():
    assert clip("short output", 5000, "output 1") == "short output"


def test_clip_samples_head_middle_and_tail():
    """A judge must be able to see the middle, not just the ends.

    Head-and-tail only was a real defect: readability's output on one page is 178k
    characters, so a 10k budget showed 5.7% of it -- an infobox and a reference
    list -- leaving the judge unable to answer the coverage question it was asked.
    """
    text = "".join(f"[{index:06d}]" for index in range(4000))  # 32k, position-encoded
    excerpt = clip(text, 10000, "output 1")

    assert len(excerpt) < len(text)
    assert "[000000]" in excerpt, "head window missing"
    assert "[003999]" in excerpt, "tail window missing"
    # A marker from the middle third must survive.
    assert any(f"[{index:06d}]" in excerpt for index in range(1900, 2100)), (
        "middle window missing"
    )
    assert excerpt.count("elided") == 2, "expected exactly two elision markers"


def test_clip_states_how_much_was_dropped():
    text = "x" * 50_000
    excerpt = clip(text, 10_000, "output 2")
    reported = sum(
        int(part.replace(",", ""))
        for line in excerpt.splitlines()
        if line.startswith("[...")
        for part in [line.split()[1]]
    )
    assert reported == 40_000, "elision markers must account for every dropped char"


def test_spearman_detects_perfect_and_inverted_agreement():
    assert stats.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) > 0.999
    assert stats.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) < -0.999


def test_spearman_needs_enough_points():
    assert stats.spearman([1, 2], [1, 2]) is None
    assert stats.spearman([1, 2, 3], [1, 2]) is None


def test_spearman_handles_ties_without_dividing_by_zero():
    assert stats.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert stats.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == 1.0


def test_index_like_shapes():
    for shape in ("listing", "gov-listing", "tourism-spa", "spa-docs", "front-page"):
        assert stats.is_index_like(shape), shape
    for shape in ("biography", "prose", "science-math", "recipe", None):
        assert not stats.is_index_like(shape), shape


def test_coverage_by_shape_separates_the_two_populations():
    """The table that stops 'jusText is weak on European languages' being published.

    Same tool, same empty count, but concentrated entirely on index-like pages --
    which is a statement about page shape, not about language.
    """
    pages = {
        "p1": {"shape": "listing"},
        "p2": {"shape": "gov-listing"},
        "p3": {"shape": "biography"},
        "p4": {"shape": "prose"},
    }
    runs = [
        {"page_id": "p1", "entrant": "justext", "status": "empty"},
        {"page_id": "p2", "entrant": "justext", "status": "empty"},
        {"page_id": "p3", "entrant": "justext", "status": "ok"},
        {"page_id": "p4", "entrant": "justext", "status": "ok"},
    ]
    row = stats.coverage_by_shape(runs, pages)[0]
    assert row["index_empty"] == 2 and row["index_n"] == 2
    assert row["index_rate"] == 100.0
    assert row["article_empty"] == 0 and row["article_n"] == 2
    assert row["article_rate"] == 0.0


def test_coverage_by_shape_ignores_not_applicable_cells():
    pages = {"p1": {"shape": "listing"}}
    runs = [{"page_id": "p1", "entrant": "wikiextractor-v2", "status": "n/a"}]
    assert stats.coverage_by_shape(runs, pages) == []


def _sampler(hashes: dict[str, str]):
    """Minimal single-page sampler over entrants with the given output hashes."""
    from research.scrapers.sampler import Sampler

    pages = [
        {"id": "p1", "url": "u", "bucket": "en-wiki", "lang": "en", "shape": "prose"}
    ]
    run_index = [
        {
            "page_id": "p1",
            "entrant": name,
            "status": "ok",
            "output_sha256": digest,
            "variant": "raw",
        }
        for name, digest in hashes.items()
    ]
    return Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=7)


def test_identical_outputs_auto_draw_once_and_are_not_reoffered():
    """An auto-draw already in the vote log must not be drawn a second time.

    This was a live bug in batch preparation: auto-draws were written to the
    database but not fed back to the sampler within the same batch, so a pair
    auto-drawn on one iteration was invisible on the next and got recorded again,
    double-counting draws in the ratings.
    """
    sampler = _sampler({"a": "same", "b": "same"})
    selected, auto_draws = sampler.next_matchup([], {})
    assert selected is None, "byte-identical pair must never be shown to a judge"
    assert len(auto_draws) == 1

    recorded = [
        {
            "page_id": draw.page_id,
            "entrant_a": draw.entrant_a,
            "entrant_b": draw.entrant_b,
            "winner": "draw",
            "auto": 1,
        }
        for draw in auto_draws
    ]
    selected_again, auto_again = _sampler({"a": "same", "b": "same"}).next_matchup(
        recorded, {}
    )
    assert selected_again is None
    assert auto_again == [], "already-recorded auto-draw was drawn a second time"


def test_differing_outputs_are_shown_to_a_judge():
    sampler = _sampler({"a": "one", "b": "two"})
    selected, auto_draws = sampler.next_matchup([], {})
    assert auto_draws == []
    assert selected is not None
    assert {selected.entrant_a, selected.entrant_b} == {"a", "b"}


def test_normalised_text_ignores_attribute_and_comment_noise():
    """Regex tag-stripping produced a false finding; this pins the fix.

    `<[^>]+>` stops at the first `>` inside an attribute value or a comment, so
    Wikipedia's JSON-in-attribute payloads leaked fragments of markup into the
    "visible text". Scrapling returns an lxml-reserialized DOM where those
    attributes are re-quoted and comments are gone, so the leak occurred on one
    side of the comparison only and 79 of 93 pages were reported as fetcher
    disagreements when the content was identical.
    """
    from research.scrapers.fetch_layer import normalised_hash, normalised_text

    raw = (
        "<html><body><!-- nav > sidebar -->"
        '<div data-mw=\'{"parts":[{"i":0}],"x":"a>b"}\'>Real content here</div>'
        "<script>var x = 1 > 0;</script></body></html>"
    )
    reserialized = (
        "<html><body>"
        '<div data-mw="{&quot;parts&quot;:[{&quot;i&quot;:0}]}">Real content here</div>'
        "</body></html>"
    )

    assert normalised_text(raw) == "Real content here"
    assert normalised_hash(raw) == normalised_hash(reserialized)


def test_normalised_text_still_sees_genuine_content_differences():
    """The parser must not normalise away what the comparison exists to find."""
    from research.scrapers.fetch_layer import normalised_hash

    full = "<html><body><p>Article body</p><p>Second paragraph</p></body></html>"
    truncated = "<html><body><p>Article body</p></body></html>"
    assert normalised_hash(full) != normalised_hash(truncated)


# ------------------------------------------------------- calibration sampling


def _multi_page_sampler(pages_count: int = 6, seed: int = 3):
    """A sampler over several pages, each with three distinct-output entrants."""
    from research.scrapers.sampler import Sampler

    pages = [
        {
            "id": f"p{index}",
            "url": f"u{index}",
            "bucket": "en-wiki",
            "lang": "en",
            "shape": "prose",
        }
        for index in range(pages_count)
    ]
    run_index = [
        {
            "page_id": page["id"],
            "entrant": name,
            "status": "ok",
            "output_sha256": f"{page['id']}-{name}",
            "variant": "raw",
        }
        for page in pages
        for name in ("alpha", "beta", "gamma")
    ]
    return Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=seed)


def _judged(page_id: str, a: str, b: str, winner: str = "a") -> dict:
    return {
        "page_id": page_id,
        "entrant_a": a,
        "entrant_b": b,
        "winner": winner,
        "auto": 0,
    }


def test_calibration_serves_only_cells_the_other_pool_judged():
    other = [_judged("p2", "alpha", "beta")]
    selected, _, progress = _multi_page_sampler().calibration_matchup([], other, {})

    assert selected is not None
    assert selected.page["id"] == "p2"
    assert sorted((selected.entrant_a, selected.entrant_b)) == ["alpha", "beta"]
    assert progress == {"overlap": 0, "remaining": 1}


def test_calibration_skips_cells_this_pool_already_judged():
    """The point is to grow the overlap, so an already-shared cell is not re-served."""
    other = [_judged("p1", "alpha", "beta"), _judged("p3", "beta", "gamma")]
    own = [_judged("p1", "beta", "alpha", "b")]

    selected, _, progress = _multi_page_sampler().calibration_matchup(own, other, {})

    assert selected is not None
    assert selected.page["id"] == "p3"
    assert progress["overlap"] == 1
    assert progress["remaining"] == 1


def test_calibration_exhausts_cleanly_when_the_overlap_is_complete():
    other = [_judged("p1", "alpha", "beta")]
    own = [_judged("p1", "alpha", "beta", "b")]

    selected, auto, progress = _multi_page_sampler().calibration_matchup(own, other, {})

    assert selected is None
    assert auto == []
    assert progress == {"overlap": 1, "remaining": 0}


def test_calibration_ignores_auto_draws_in_the_other_pool():
    """An auto-draw is a byte-identical output, not a judgement to agree with."""
    other = [
        dict(_judged("p1", "alpha", "beta"), auto=1, winner="draw"),
        _judged("p4", "alpha", "gamma"),
    ]
    selected, _, progress = _multi_page_sampler().calibration_matchup([], other, {})

    assert selected is not None
    assert selected.page["id"] == "p4"
    assert progress["remaining"] == 1


def test_calibration_randomises_the_side_independently_of_the_other_pool():
    """Reusing the other pool's left/right order would let a shared position bias
    inflate agreement, so the order is drawn fresh each time."""
    other = [_judged("p0", "alpha", "beta")]
    orders = set()
    for seed in range(12):
        selected, _, _ = _multi_page_sampler(seed=seed).calibration_matchup(
            [], other, {}
        )
        orders.add((selected.entrant_a, selected.entrant_b))
    assert orders == {("alpha", "beta"), ("beta", "alpha")}


def test_calibration_spreads_across_the_elo_gap_range():
    """Sampling must not concentrate on close matchups.

    Agreement measured only on near-ties understates it and only on lopsided pairs
    flatters it, so the least-sampled gap band is preferred. Here the overlap already
    holds two close cells, so the next draw must come from a wider-gap band.
    """
    elo = {"alpha": 1900.0, "beta": 1500.0, "gamma": 1490.0}
    other = [
        _judged("p0", "beta", "gamma"),  # gap 10  -> narrow band
        _judged("p1", "beta", "gamma"),  # gap 10  -> narrow band
        _judged("p2", "alpha", "beta"),  # gap 400 -> wide band
        _judged("p3", "alpha", "gamma"),  # gap 410 -> wide band
    ]
    own = [_judged("p0", "beta", "gamma", "b"), _judged("p1", "gamma", "beta", "a")]

    picked = set()
    for seed in range(20):
        selected, _, _ = _multi_page_sampler(seed=seed).calibration_matchup(
            own, other, elo
        )
        picked.add(selected.page["id"])
    assert picked <= {"p2", "p3"}, "narrow band was already sampled twice"


# ------------------------------------------------------------ focus sampling


def _focus_sampler(pages_spec: dict[str, tuple[str, str]], seed: int = 3, *,
                   entrants=("ours", "trafilatura", "wikiextractor-v2")):
    """Pages given as {page_id: (shape, lang)}, each carrying `entrants`.

    Entrant membership is per page so a page can be made ineligible for the set by
    leaving one of them out -- which is the real situation with wikiextractor-v2.
    """
    from research.scrapers.sampler import Sampler

    pages = [
        {
            "id": page_id,
            "url": f"https://example.org/{page_id}",
            "bucket": "en-wiki",
            "lang": lang,
            "shape": shape,
        }
        for page_id, (shape, lang) in pages_spec.items()
    ]
    run_index = [
        {
            "page_id": page_id,
            "entrant": name,
            "status": "ok",
            "output_sha256": f"{page_id}-{name}",
            "variant": "raw",
        }
        for page_id in pages_spec
        for name in entrants
    ]
    return Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=seed)


def _judged(page_id: str, a: str, b: str, winner: str = "a") -> dict:
    return {
        "page_id": page_id,
        "entrant_a": a,
        "entrant_b": b,
        "winner": winner,
        "auto": 0,
    }


TRIO = ("ours", "trafilatura", "wikiextractor-v2")


def test_focus_only_offers_pairs_from_the_named_set():
    spec = {"p1": ("biography", "en"), "p2": ("geography", "nl")}
    for seed in range(10):
        selected, _, _ = _focus_sampler(spec, seed=seed).focus_matchup([], TRIO)
        assert selected.entrant_a in TRIO and selected.entrant_b in TRIO
        assert selected.entrant_a != selected.entrant_b


def test_focus_restricts_pages_to_those_every_entrant_handled():
    """The shared population is what makes a 3-way rating comparable.

    `wikiextractor-v2` only exists on wiki pages; rating it over pages the others also
    played but it did not would compare ratings earned on different documents.
    """
    from research.scrapers.sampler import Sampler

    pages = [
        {"id": "wiki", "url": "u1", "bucket": "en-wiki", "lang": "en",
         "shape": "biography"},
        {"id": "news", "url": "u2", "bucket": "en-other", "lang": "en",
         "shape": "listing"},
    ]
    run_index = [
        {"page_id": "wiki", "entrant": name, "status": "ok",
         "output_sha256": f"wiki-{name}", "variant": "raw"}
        for name in TRIO
    ] + [
        # The news page has no wikitext, so wikiextractor never ran on it.
        {"page_id": "news", "entrant": name, "status": "ok",
         "output_sha256": f"news-{name}", "variant": "raw"}
        for name in ("ours", "trafilatura")
    ]
    sampler = Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=1)

    for seed in range(10):
        s = Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=seed)
        selected, _, progress = s.focus_matchup([], TRIO)
        assert selected.page["id"] == "wiki"
        assert progress["pages"] == 1
    assert sampler is not None


def test_focus_rotates_pairs_so_none_starves():
    """A closed set is settled only when every pair has votes, not when the total is big."""
    spec = {f"p{i}": ("biography", "en") for i in range(6)}
    votes: list[dict] = []
    served = []
    for seed in range(3):
        selected, _, _ = _focus_sampler(spec, seed=seed).focus_matchup(votes, TRIO)
        pair = tuple(sorted((selected.entrant_a, selected.entrant_b)))
        served.append(pair)
        votes.append(_judged(selected.page["id"], *pair))
    assert len(set(served)) == 3, f"expected all three pairs, got {served}"


def test_focus_stratifies_by_language_when_shape_does_not_vary():
    """Adaptive axis: the wiki set has no index-like pages, so shape balances nothing.

    Splitting by English vs other language is the axis that actually varies there, and
    it happens to be the multilingual question the benchmark exists to answer.
    """
    spec = {"en1": ("biography", "en"), "nl1": ("biography", "nl")}
    _, _, progress = _focus_sampler(spec).focus_matchup([], TRIO)
    assert progress["axis"] == "language"

    # With the English side already judged, the next page must be the other language.
    own = [_judged("en1", "ours", "trafilatura")]
    selected, _, _ = _focus_sampler(spec).focus_matchup(own, TRIO)
    assert selected.page["id"] == "nl1"


def test_focus_prefers_page_shape_when_both_classes_are_present():
    spec = {"idx": ("listing", "en"), "art": ("biography", "en")}
    _, _, progress = _focus_sampler(spec).focus_matchup([], TRIO)
    assert progress["axis"] == "page shape"


def test_focus_reports_no_axis_when_nothing_varies():
    spec = {"a": ("biography", "en"), "b": ("geography", "en")}
    _, _, progress = _focus_sampler(spec).focus_matchup([], TRIO)
    assert progress["axis"] == "none"


def test_focus_does_not_repeat_a_judged_cell():
    spec = {"p1": ("biography", "en")}
    own = [_judged("p1", "ours", "trafilatura"),
           _judged("p1", "ours", "wikiextractor-v2")]
    selected, _, progress = _focus_sampler(spec).focus_matchup(own, TRIO)
    pair = tuple(sorted((selected.entrant_a, selected.entrant_b)))
    assert pair == ("trafilatura", "wikiextractor-v2")
    assert progress["remaining"] == 1


def test_focus_ignores_votes_involving_entrants_outside_the_set():
    spec = {"p1": ("biography", "en")}
    own = [_judged("p1", "ours", "goose3")]
    _, _, progress = _focus_sampler(spec).focus_matchup(own, TRIO)
    assert progress["judged"] == 0


def test_focus_skips_byte_identical_pairs():
    from research.scrapers.sampler import Sampler

    pages = [{"id": "p1", "url": "u", "bucket": "en-wiki", "lang": "en",
              "shape": "biography"}]
    run_index = [
        {"page_id": "p1", "entrant": name, "status": "ok",
         "output_sha256": "same", "variant": "raw"}
        for name in ("ours", "trafilatura")
    ]
    sampler = Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=1)
    selected, _, progress = sampler.focus_matchup([], ("ours", "trafilatura"))
    assert selected is None
    assert progress["remaining"] == 0


def test_focus_exhausts_cleanly():
    spec = {"p1": ("biography", "en")}
    own = [
        _judged("p1", "ours", "trafilatura"),
        _judged("p1", "ours", "wikiextractor-v2"),
        _judged("p1", "trafilatura", "wikiextractor-v2"),
    ]
    selected, auto, progress = _focus_sampler(spec).focus_matchup(own, TRIO)
    assert selected is None
    assert auto == []
    assert progress["remaining"] == 0
    assert progress["judged"] == 3


def test_focus_randomises_which_side_each_entrant_takes():
    spec = {"p1": ("biography", "en")}
    orders = set()
    for seed in range(20):
        selected, _, _ = _focus_sampler(spec, seed=seed).focus_matchup(
            [], ("ours", "trafilatura")
        )
        orders.add((selected.entrant_a, selected.entrant_b))
    assert orders == {("ours", "trafilatura"), ("trafilatura", "ours")}


def test_focus_reported_pair_does_not_leak_the_display_side():
    """The `serving.pair` label is shown in the UI, so it must be order-independent."""
    spec = {"p1": ("biography", "en")}
    labels = set()
    for seed in range(20):
        selected, _, progress = _focus_sampler(spec, seed=seed).focus_matchup(
            [], ("ours", "trafilatura")
        )
        labels.add(progress["serving"]["pair"])
        assert selected is not None
    assert labels == {"ours vs trafilatura"}


def test_focus_rejects_a_single_entrant():
    spec = {"p1": ("biography", "en")}
    selected, _, progress = _focus_sampler(spec).focus_matchup([], ("ours",))
    assert selected is None
    assert "error" in progress


def test_focus_progress_ignores_votes_outside_the_shared_population():
    """Progress must agree with `subset_standings`, which restricts to that population.

    A live explore-mode vote on a non-wiki page made the counter read "1 judged, 47 of
    47 remaining" for `ours vs trafilatura` and pushed the round-robin away from that
    pair, on the strength of a vote no standings table counts.
    """
    from research.scrapers.sampler import Sampler

    pages = [
        {"id": "wiki", "url": "u1", "bucket": "en-wiki", "lang": "en",
         "shape": "biography"},
        {"id": "spa", "url": "u2", "bucket": "eu-other", "lang": "en",
         "shape": "tourism-spa"},
    ]
    run_index = [
        {"page_id": "wiki", "entrant": name, "status": "ok",
         "output_sha256": f"wiki-{name}", "variant": "raw"}
        for name in TRIO
    ] + [
        {"page_id": "spa", "entrant": name, "status": "ok",
         "output_sha256": f"spa-{name}", "variant": "raw"}
        for name in ("ours", "trafilatura")
    ]
    sampler = Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=1)

    outside = [_judged("spa", "ours", "trafilatura", "draw")]
    _, _, progress = sampler.focus_matchup(outside, TRIO)
    assert progress["judged"] == 0
    assert progress["per_pair"]["ours vs trafilatura"]["judged"] == 0
    assert progress["per_stratum"] == {}


# ------------------------------------------------------- wiki-only focus population

CONTENT3 = ("ours", "resiliparse", "trafilatura")


def _mixed_corpus_sampler(seed: int = 1, entrants=CONTENT3, wiki_entrants=CONTENT3):
    """One pinned wiki page and one unpinned non-wiki page, both otherwise eligible.

    `is_wiki` is the only difference between them, so any page the sampler serves
    identifies which restriction fired.
    """
    from research.scrapers.sampler import Sampler

    pages = [
        {"id": "wiki", "url": "https://en.wikipedia.org/wiki/X?oldid=1",
         "bucket": "en-wiki", "lang": "en", "shape": "biography", "is_wiki": 1},
        {"id": "news", "url": "https://example.org/news", "bucket": "en-other",
         "lang": "en", "shape": "listing", "is_wiki": 0},
    ]
    run_index = [
        {"page_id": "wiki", "entrant": name, "status": "ok",
         "output_sha256": f"wiki-{name}", "variant": "raw"}
        for name in wiki_entrants
    ] + [
        {"page_id": "news", "entrant": name, "status": "ok",
         "output_sha256": f"news-{name}", "variant": "raw"}
        for name in entrants
    ]
    return Sampler(pages=pages, run_index=run_index, epsilon=0.0, seed=seed)


def test_wiki_only_excludes_pages_that_are_not_pinned():
    """The reference pane is a live iframe, and only wiki is pinned to `?oldid=`.

    Session 6 measured what a vote on an unpinned page is worth: 2-2-3 on the 50
    non-wiki pages against 7-0 on the pinned slice, because the reviewer was judging
    two extractions of a document they could not see. The flag makes that population
    unreachable rather than merely discouraged.
    """
    for seed in range(10):
        selected, _, progress = _mixed_corpus_sampler(seed).focus_matchup(
            [], CONTENT3, wiki_only=True
        )
        assert progress["pages"] == 1
        assert selected.page["id"] == "wiki"


def test_wiki_only_defaults_off_so_existing_modes_are_unchanged():
    """Every other focus mode must keep its whole shared population."""
    _, _, unrestricted = _mixed_corpus_sampler().focus_matchup([], CONTENT3)
    assert unrestricted["pages"] == 2

    served = {
        _mixed_corpus_sampler(seed).focus_matchup([], CONTENT3)[0].page["id"]
        for seed in range(20)
    }
    assert served == {"wiki", "news"}, "the non-wiki page must still be reachable"


def test_wiki_only_composes_with_the_every_entrant_restriction():
    """Both filters apply: wiki pages *and* pages the whole set handled.

    A wiki page one entrant returned nothing on is still not comparable, so narrowing
    to wiki must not smuggle it back in.
    """
    sampler = _mixed_corpus_sampler(wiki_entrants=("ours", "trafilatura"))
    selected, _, progress = sampler.focus_matchup([], CONTENT3, wiki_only=True)
    assert progress["pages"] == 0
    assert progress["remaining"] == 0
    assert selected is None


def test_wiki_only_still_serves_all_three_pairs():
    """A wiki-restricted round-robin is still a round-robin, not a champion format.

    Three entrants and no pinned champion means three pairs, and `resiliparse vs
    trafilatura` -- the one neither `ours` plays in -- must be among them: DCLM and
    Dolma chose one of those two, FineWeb and HPLT the other.
    """
    votes: list[dict] = []
    served = []
    for seed in range(3):
        selected, _, progress = _mixed_corpus_sampler(seed).focus_matchup(
            votes, CONTENT3, wiki_only=True
        )
        pair = tuple(sorted((selected.entrant_a, selected.entrant_b)))
        served.append(pair)
        votes.append(_judged(selected.page["id"], *pair))
        assert len(progress["per_pair"]) == 3
    assert len(set(served)) == 3, f"expected all three pairs, got {served}"
    assert ("resiliparse", "trafilatura") in set(served)
