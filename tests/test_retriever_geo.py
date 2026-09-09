"""GeoScopedRetriever: soft (over-fetch, fuse) and strict (filter, widen, boost)."""

from src.db.pages import PageLocation
from src.retrieval.base import RankedChunk
from src.retrieval.retrievers.geo import GeoScopedRetriever
from src.shared.geo_scope import GeoScope
from src.shared.geocode import Coordinates

CASTLE = Coordinates(45.989, 15.466)
FAR = Coordinates(46.5, 13.5)


def _located(page_id, coords, nuts3="SI036", nuts2="SI03", country="SI"):
    return PageLocation(
        page_id,
        "primary",
        page_id,
        page_id,
        None,
        coords.latitude,
        coords.longitude,
        "point",
        country,
        nuts2,
        nuts3,
    )


class FixedResolver:
    def __init__(self, scopes):
        self.scopes = scopes
        self.calls = 0

    def resolve(self, query):
        self.calls += 1
        return self.scopes.get(query)


class StageFactory:
    """Records which scope each stage was built for; returns canned chunks per level."""

    def __init__(self, by_level):
        self.by_level = by_level
        self.built = []
        self.limits = []
        self.scopes = []

    def __call__(self, scope):
        level = scope.level if scope else "none"
        self.built.append(level)
        if scope is not None:
            self.scopes.append(scope)
        chunks = self.by_level[level]
        factory = self

        class Stage:
            name = f"stage-{level}"

            def retrieve(self, query, limit):
                factory.limits.append(limit)
                return chunks[:limit]

            def retrieve_batch(self, queries, limit):
                factory.limits.append(limit)
                return {i: chunks[:limit] for i in range(len(queries))}

        return Stage()


def _chunks(*ids):
    return [RankedChunk(id=i, score=1.0 - n * 0.1, text=i) for n, i in enumerate(ids)]


def _retriever(factory, resolver, mode="strict", locations=None, shares=None, **kw):
    locations = locations or {}
    return GeoScopedRetriever(
        name="geo",
        resolver=resolver,
        build_stage=factory,
        chunk_locations=lambda ids: {i: locations[i] for i in ids if i in locations},
        mode=mode,
        scope_share=(lambda scope: shares.get(scope.level)) if shares else None,
        **kw,
    )


# --- strict -----------------------------------------------------------------------


def test_unresolved_query_runs_the_unfiltered_stage_once():
    factory = StageFactory({"none": _chunks("a", "b")})
    retriever = _retriever(factory, FixedResolver({}))

    assert [c.id for c in retriever.retrieve("anything", 10)] == ["a", "b"]
    assert factory.built == ["none"]


def test_thin_filtered_result_widens_until_enough_comes_back():
    scope = GeoScope(country_code="SI", nuts2="SI03", nuts3="SI036")
    factory = StageFactory(
        {"nuts3": _chunks("only"), "nuts2": _chunks("x", "y", "z"), "none": []}
    )
    retriever = _retriever(factory, FixedResolver({"q": scope}), min_candidates=2)

    chunks = retriever.retrieve("q", 10)

    assert [c.id for c in chunks] == ["x", "y", "z"]
    assert factory.built == ["nuts3", "nuts2"]
    assert retriever.stats.widenings == 1
    assert retriever.stats.levels == {"nuts2": 1}


def test_widening_ends_at_no_filter_so_a_bad_scope_cannot_empty_the_result():
    scope = GeoScope(country_code="XX")
    factory = StageFactory({"country": [], "none": _chunks("fallback")})
    retriever = _retriever(factory, FixedResolver({"q": scope}), min_candidates=1)

    assert [c.id for c in retriever.retrieve("q", 10)] == ["fallback"]
    assert factory.built == ["country", "none"]


def test_include_null_widens_when_the_scope_itself_selected_too_little():
    # With include_null the filtered stage fills up with unlocated pages, so the
    # count alone would never widen; only located in-scope chunks count.
    # The cached scope says include_null=False; the retriever's policy wins.
    scope = GeoScope(country_code="SI", nuts2="SI03", nuts3="SI036")
    factory = StageFactory(
        {
            "nuts3": _chunks("in", "bio1", "bio2"),
            "nuts2": _chunks("in", "in2", "bio1"),
            "none": [],
        }
    )
    locations = {
        "in": _located("in", CASTLE),
        "in2": _located("in2", CASTLE, nuts3="SI037"),
    }
    retriever = _retriever(
        factory,
        FixedResolver({"q": scope}),
        locations=locations,
        min_candidates=2,
        include_null=True,
    )

    chunks = retriever.retrieve("q", 10)

    assert [c.id for c in chunks] == ["in", "in2", "bio1"]
    assert factory.built == ["nuts3", "nuts2"]
    assert all(s.include_null for s in factory.scopes)


def test_strict_boost_reorders_by_distance_and_leaves_unlocated_chunks_alone():
    scope = GeoScope(latitude=CASTLE.latitude, longitude=CASTLE.longitude)  # boost only
    chunks = [
        RankedChunk(id="far", score=0.90, text=""),
        RankedChunk(id="near", score=0.85, text=""),
        RankedChunk(id="nowhere", score=0.80, text=""),
    ]
    factory = StageFactory({"none": chunks})
    retriever = _retriever(
        factory,
        FixedResolver({"q": scope}),
        locations={"far": _located("far", FAR), "near": _located("near", CASTLE)},
        boost_weight=0.25,
    )

    ranked = retriever.retrieve("q", 10)

    assert [c.id for c in ranked] == ["near", "nowhere", "far"]
    by_id = {c.id: c.score for c in ranked}
    assert by_id["near"] == 0.85  # multiplier 1.0 at the place
    assert by_id["nowhere"] == 0.80  # no coordinates: untouched
    assert 0.90 * 0.75 <= by_id["far"] < 0.90 * 0.76  # floor of the multiplier


def test_strict_batch_groups_queries_by_scope_and_widens_per_query():
    posavje = GeoScope(country_code="SI", nuts2="SI03", nuts3="SI036")
    factory = StageFactory(
        {"nuts3": _chunks("p"), "nuts2": _chunks("p", "q"), "none": _chunks("u", "v")}
    )
    resolver = FixedResolver({"in posavje": posavje, "also posavje": posavje})
    retriever = _retriever(factory, resolver, min_candidates=2)

    rankings = retriever.retrieve_batch(["in posavje", "plain", "also posavje"], 10)

    assert [c.id for c in rankings[0]] == ["p", "q"]
    assert [c.id for c in rankings[1]] == ["u", "v"]
    assert [c.id for c in rankings[2]] == ["p", "q"]
    # One stage per distinct (scope, level), built once and reused.
    assert sorted(factory.built) == ["none", "nuts2", "nuts3"]
    assert resolver.calls == 3
    retriever.retrieve("in posavje", 10)
    assert len(factory.built) == 3


def test_resolver_failure_falls_back_to_unscoped(capsys):
    class Broken:
        def resolve(self, query):
            raise ConnectionError("gazetteer down")

    factory = StageFactory({"none": _chunks("a")})
    retriever = _retriever(factory, Broken())

    assert [c.id for c in retriever.retrieve("q", 10)] == ["a"]
    assert "unscoped" in capsys.readouterr().out


# --- soft -------------------------------------------------------------------------


def test_soft_point_scope_overfetches_and_keeps_unlocated_pages_neutral():
    scope = GeoScope(
        latitude=CASTLE.latitude,
        longitude=CASTLE.longitude,
        radius_km=25,
        nuts3="SI036",
    )
    chunks = [
        RankedChunk(id="far", score=0.90, text=""),
        RankedChunk(id="bio", score=0.85, text=""),
        RankedChunk(id="near", score=0.80, text=""),
        RankedChunk(id="tail", score=0.10, text=""),
    ]
    factory = StageFactory({"none": chunks})
    retriever = _retriever(
        factory,
        FixedResolver({"q": scope}),
        mode="soft",
        locations={"far": _located("far", FAR), "near": _located("near", CASTLE)},
        boost_weight=0.3,
        overfetch=4,
    )

    ranked = retriever.retrieve("q", 2)

    # Never filtered: one unfiltered stage, asked for limit * overfetch.
    assert factory.built == ["none"] and factory.limits == [8]
    # The biography (no footprint) keeps s_geo = 1 and now leads; the far page
    # drops below the nearby one; the result is cut back to the limit.
    assert [c.id for c in ranked] == ["bio", "near"]
    assert retriever.stats.levels == {"radius": 1} and retriever.stats.widenings == 0


def test_soft_region_scope_scores_membership_and_skips_uninformative_levels():
    scope = GeoScope(country_code="SI", nuts2="SI03", nuts3="SI036")
    # Text scores close together, as a reranked top-k is; the tail sets the range.
    chunks = [
        RankedChunk(id="elsewhere", score=0.90, text=""),
        RankedChunk(id="inside", score=0.88, text=""),
        RankedChunk(id="bio", score=0.86, text=""),
        RankedChunk(id="tail", score=0.10, text=""),
    ]
    factory = StageFactory({"none": chunks})
    locations = {
        "inside": _located("inside", CASTLE),
        "elsewhere": _located("elsewhere", FAR, nuts3="SI037", nuts2="SI04"),
    }
    # NUTS-3 keeps 95% of located pages: no information. NUTS-2 keeps 60%.
    retriever = _retriever(
        factory,
        FixedResolver({"q": scope}),
        mode="soft",
        locations=locations,
        shares={"nuts3": 0.95, "nuts2": 0.6, "country": 1.0},
        max_scope_share=0.9,
    )

    ranked = retriever.retrieve("q", 3)

    assert [c.id for c in ranked] == ["inside", "bio", "elsewhere"]
    assert retriever.stats.levels == {"nuts2": 1} and retriever.stats.widenings == 1


def test_soft_scope_that_keeps_every_located_page_is_a_no_op():
    scope = GeoScope(country_code="SI")
    chunks = _chunks("a", "b", "c")
    factory = StageFactory({"none": chunks})
    retriever = _retriever(
        factory,
        FixedResolver({"q": scope}),
        mode="soft",
        shares={"country": 1.0},
    )

    assert [c.id for c in retriever.retrieve("q", 2)] == ["a", "b"]
    assert factory.limits == [2]  # plain retrieval, no over-fetch
    assert retriever.stats.levels == {"none": 1}


def test_soft_batch_splits_scoped_from_plain_queries():
    scope = GeoScope(latitude=CASTLE.latitude, longitude=CASTLE.longitude, radius_km=10)
    chunks = [
        RankedChunk(id="far", score=0.9, text=""),
        RankedChunk(id="near", score=0.8, text=""),
        RankedChunk(id="x", score=0.1, text=""),
    ]
    factory = StageFactory({"none": chunks})
    retriever = _retriever(
        factory,
        FixedResolver({"near castle": scope}),
        mode="soft",
        locations={"far": _located("far", FAR), "near": _located("near", CASTLE)},
        overfetch=2,
    )

    rankings = retriever.retrieve_batch(["plain", "near castle"], 2)

    assert [c.id for c in rankings[0]] == ["far", "near"]
    assert [c.id for c in rankings[1]] == ["near", "far"]
    assert sorted(factory.limits) == [2, 4]
    assert retriever.stats.resolved == 1 and retriever.stats.unresolved == 1
