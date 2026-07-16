import json
import sqlite3

import pytest
import httpx

from src.okf.answer import (
    EvidenceAssessment,
    NavigationAction,
    OKFAnswer,
    answer_question,
)
from src.okf import generate
from src.okf.bundle import regenerate_indexes, validate_bundle, write_concept
from src.okf.document import OKFDocument, OKFDocumentError
from src.okf import evidence
from src.okf.paths import concept_path, parse_concept_id
from src.okf.source import load_source_pages
from src.shared.llm import AzureFoundryStructuredLlm


def _document(title: str = "Castle") -> OKFDocument:
    return OKFDocument(
        frontmatter={
            "type": "Destination",
            "title": title,
            "description": "A documented cultural heritage destination.",
            "timestamp": "2026-07-14T00:00:00+00:00",
        },
        body="# Overview\n\nFact.\n",
    )


def test_document_roundtrip_and_validation():
    document = _document()
    parsed = OKFDocument.parse(document.serialize())

    assert parsed.frontmatter == document.frontmatter
    assert parsed.body.strip() == document.body.strip()

    with pytest.raises(OKFDocumentError, match="Missing frontmatter"):
        OKFDocument(frontmatter={"type": "Reference"}, body="Fact").serialize()


def test_azure_structured_error_includes_final_cause(monkeypatch):
    client = AzureFoundryStructuredLlm("https://example.test", "secret", "model")
    monkeypatch.setattr(
        AzureFoundryStructuredLlm,
        "_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            httpx.ReadTimeout("request timed out")
        ),
    )

    with pytest.raises(RuntimeError, match="ReadTimeout: request timed out"):
        client.structured_output("prompt", generate.ConceptProposal, retries=2)


def test_concept_paths_reject_unsafe_and_reserved_ids(tmp_path):
    assert parse_concept_id("destinations/rajhenburg") == (
        "destinations",
        "rajhenburg",
    )
    assert concept_path(tmp_path, "destinations/rajhenburg").name == "rajhenburg.md"

    for value in ("../secret", "destinations/../secret", "index", "places/index"):
        with pytest.raises(ValueError):
            parse_concept_id(value)


def test_bundle_indexes_and_reports_broken_links(tmp_path):
    root = tmp_path / "bundle"
    document = _document()
    document = OKFDocument(
        frontmatter=document.frontmatter,
        body=document.body + "\n[Missing](missing.md)\n",
    )
    write_concept(root, "destinations/rajhenburg", document)

    regenerate_indexes(root)
    report = validate_bundle(root)

    assert report.valid
    assert len(report.warnings) == 1
    assert "[Castle](rajhenburg.md)" in (root / "destinations" / "index.md").read_text()
    assert "destinations/index.md" in (root / "index.md").read_text()


def test_source_adapter_reads_only_eligible_complete_pages(tmp_path):
    db = tmp_path / "pages.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key, source text, url text, title text,
              error text, page_kind text
            );
            create table page_markdown_content (page_id text, markdown text);
            create table page_sources (source text primary key, language text);
            insert into page_sources values ('castle', 'sl');
            insert into page_metadata values ('ok', 'castle', 'https://a', 'A', null, 'prose');
            insert into page_metadata values ('bad', 'castle', 'https://b', 'B', 'failed', 'empty');
            insert into page_markdown_content values ('ok', '# Whole page');
            insert into page_markdown_content values ('bad', '');
            """
        )

    pages = load_source_pages(db)

    assert len(pages) == 1
    assert pages[0].id == "ok"
    assert pages[0].markdown == "# Whole page"
    assert pages[0].language == "sl"


class _StubAzure:
    def structured_output(self, _prompt, schema, **_kwargs):
        if schema is generate.ConceptProposal:
            return generate.ConceptProposal(
                concept_id="destinations/rajhenburg",
                type="Destination",
                title="Rajhenburg Castle",
                description="A castle above Brestanica.",
                tags=["castle"],
                aliases=["Grad Rajhenburg"],
                search_terms=["Brestanica fortress"],
            )
        if schema is generate.ResolutionBatch:
            return generate.ResolutionBatch(
                concepts=[
                    generate.CanonicalConcept(
                        concept_id="destinations/rajhenburg",
                        type="Destination",
                        title="Grad Rajhenburg",
                        description="A castle above Brestanica.",
                        tags=["castle"],
                        aliases=["Rajhenburg Castle"],
                    )
                ],
                decisions=[
                    generate.ResolutionDecision(
                        page_id="p1",
                        canonical_concept_id="destinations/rajhenburg",
                    )
                ],
            )
        return generate.EnrichedConcept(
            title="Rajhenburg Castle",
            description="A castle above Brestanica.",
            tags=["heritage"],
            aliases=["Reichenburg Castle"],
            search_terms=["Rajhenburg", "Brestanica", "medieval castle"],
            retrieval_queries=[
                "What stands above Brestanica?",
                "Where is Rajhenburg Castle?",
                "Kaj stoji nad Brestanico?",
            ],
            source_summary="Rajhenburg Castle is a heritage castle above Brestanica.",
            facts=["Rajhenburg Castle stands above Brestanica."],
            body="# Overview\n\nThe castle stands above Brestanica.\n\n# Citations\n\n- [Source](https://example.test/castle)",
        )


def test_generation_writes_provenance_and_resumes(monkeypatch, tmp_path):
    db = tmp_path / "pages.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            create table page_metadata (
              id text primary key, source text, url text, title text,
              error text, page_kind text
            );
            create table page_markdown_content (page_id text, markdown text);
            create table page_sources (source text primary key, language text);
            insert into page_sources values ('castle', 'en');
            insert into page_metadata values ('p1', 'castle', 'https://example.test/castle', 'Castle', null, 'prose');
            insert into page_markdown_content values ('p1', '# Complete source page');
            """
        )
    monkeypatch.setattr(generate, "ROOT", tmp_path)
    monkeypatch.setattr(
        generate,
        "CONFIG",
        {
            **generate.CONFIG,
            "source_db": "pages.db",
            "bundle_path": "bundle",
            "checkpoint_path": ".local/checkpoint.json",
            "retrieval_checkpoint_path": ".local/retrieval-refresh.json",
        },
    )
    monkeypatch.setattr(generate, "_azure_client", lambda: _StubAzure())

    first = generate.generate_bundle()
    second = generate.generate_bundle()
    refreshed = generate.generate_bundle(refresh_retrieval=True)
    resumed_refresh = generate.generate_bundle(refresh_retrieval=True)
    concept = OKFDocument.parse(
        (tmp_path / "bundle/destinations/rajhenburg.md").read_text()
    )
    checkpoint = json.loads((tmp_path / ".local/checkpoint.json").read_text())
    refresh_checkpoint = json.loads(
        (tmp_path / ".local/retrieval-refresh.json").read_text()
    )

    assert first["processed_pages"] == 1
    assert second["resumed_pages"] == 1
    assert first["discovered_pages"] == 1
    assert first["resolved_pages"] == 1
    assert second["discovered_pages"] == 0
    assert second["resolved_pages"] == 0
    assert refreshed["generation_mode"] == "refresh_retrieval"
    assert refreshed["discovered_pages"] == 0
    assert refreshed["resolved_pages"] == 0
    assert refreshed["processed_pages"] == 0
    assert refreshed["resumed_pages"] == 1
    assert resumed_refresh["processed_pages"] == 0
    assert resumed_refresh["resumed_pages"] == 1
    assert concept.frontmatter["source_page_ids"] == ["p1"]
    assert concept.frontmatter["source_urls"] == ["https://example.test/castle"]
    assert concept.frontmatter["aliases"] == [
        "Rajhenburg Castle",
        "Grad Rajhenburg",
        "Reichenburg Castle",
    ]
    assert "Brestanica fortress" in concept.frontmatter["search_terms"]
    assert concept.frontmatter["source_evidence"]["p1"]["facts"] == [
        "Rajhenburg Castle stands above Brestanica."
    ]
    assert "# Source evidence by page" in concept.body
    assert concept.body.count("https://example.test/castle") == 1
    assert first["retrieval_metadata"] == {
        "concepts_with_search_terms": 1,
        "concepts_with_retrieval_queries": 1,
        "source_evidence_pages": 1,
        "atomic_facts": 1,
    }
    assert checkpoint["enriched_pages"] == {"p1": "destinations/rajhenburg"}
    assert refresh_checkpoint["enriched_pages"] == {"p1": "destinations/rajhenburg"}


def test_oversized_page_splits_on_markdown_boundaries(monkeypatch):
    monkeypatch.setattr(
        generate,
        "CONFIG",
        {**generate.CONFIG, "max_page_chars": 100, "page_part_chars": 70},
    )
    markdown = "# Castle\n\n" + "A" * 55 + "\n\n## History\n\n" + "B" * 55

    parts = generate._page_parts(markdown)

    assert len(parts) >= 2
    assert [part.index for part in parts] == list(range(1, len(parts) + 1))
    assert all(part.total == len(parts) for part in parts)
    assert all(len(part.markdown) <= 70 for part in parts)
    assert parts[0].heading_path == "Castle"
    assert any("History" in part.heading_path for part in parts)
    assert "".join(part.markdown.replace("\n", "") for part in parts).count("A") == 55
    assert "".join(part.markdown.replace("\n", "") for part in parts).count("B") == 55


def test_enrichment_merges_multiple_parts_under_one_source_page():
    page = generate.SourcePage(
        id="p1",
        source="castle",
        url="https://example.test/castle",
        title="Castle",
        language="en",
        markdown="# Castle",
    )
    plan = generate.CanonicalConcept(
        concept_id="destinations/castle",
        type="Destination",
        title="Castle",
        description="A castle.",
    )

    def enrichment(label: str) -> generate.EnrichedConcept:
        return generate.EnrichedConcept(
            title="Castle",
            description="A castle.",
            search_terms=[label, "castle", "heritage"],
            retrieval_queries=[
                f"What is {label}?",
                f"Where is {label}?",
                f"When was {label}?",
            ],
            source_summary=f"Summary {label}.",
            facts=[f"Fact {label}."],
            body=f"# {label}\n\nFact {label}.",
        )

    first = generate._document(page, plan, enrichment("one"), None)
    second = generate._document(page, plan, enrichment("two"), first)
    source = second.frontmatter["source_evidence"]["p1"]

    assert second.frontmatter["source_page_ids"] == ["p1"]
    assert source["summary"] == "Summary one. Summary two."
    assert source["facts"] == ["Fact one.", "Fact two."]
    assert source["search_terms"] == [
        "one",
        "castle",
        "heritage",
        "two",
    ]
    assert second.body.count("https://example.test/castle") == 1


def test_enrichment_bounds_merged_retrieval_metadata():
    page = generate.SourcePage(
        "p1", "source", "https://example.test", "Title", "en", "# Title"
    )
    plan = generate.CanonicalConcept(
        concept_id="destinations/title",
        type="Destination",
        title="Title",
        description="Description.",
        search_terms=[f"plan-{index}" for index in range(30)],
    )
    enriched = generate.EnrichedConcept(
        title="Title",
        description="Description.",
        search_terms=[f"new-{index}" for index in range(30)],
        retrieval_queries=[f"Question {index}?" for index in range(30)],
        facts=[f"Fact {index}." for index in range(70)],
        body="# Title\n\nBody.",
    )

    document = generate._document(page, plan, enriched, None)
    evidence = document.frontmatter["source_evidence"]["p1"]

    assert len(document.frontmatter["search_terms"]) == 40
    assert len(document.frontmatter["retrieval_queries"]) == 24
    assert len(evidence["facts"]) == 60
    assert len(evidence["search_terms"]) == 30
    assert len(evidence["retrieval_queries"]) == 24


def test_multipart_enrichment_resumes_after_last_completed_part(monkeypatch, tmp_path):
    monkeypatch.setattr(
        generate,
        "CONFIG",
        {**generate.CONFIG, "max_page_chars": 100, "page_part_chars": 70},
    )
    page = generate.SourcePage(
        "p1",
        "source",
        "https://example.test",
        "Title",
        "en",
        "# Title\n\n" + "A" * 55 + "\n\n## History\n\n" + "B" * 55,
    )
    plan = generate.CanonicalConcept(
        concept_id="destinations/title",
        type="Destination",
        title="Title",
        description="Description.",
    )
    catalog = generate.CanonicalCatalog(
        concepts={plan.concept_id: plan}, assignments={page.id: plan.concept_id}
    )
    checkpoint = generate.GenerationCheckpoint()
    checkpoint_path = tmp_path / "checkpoint.json"

    class _FailSecondPart:
        calls = 0

        def structured_output(self, _prompt, _schema, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("part failed")
            return generate.EnrichedConcept(
                title="Title",
                description="Description.",
                facts=[f"Fact {self.calls}."],
                body="# Title\n\nBody.",
            )

    first_client = _FailSecondPart()
    processed, _skipped = generate._enrich_catalog(
        first_client,
        [page],
        catalog,
        checkpoint,
        checkpoint_path,
        tmp_path / "bundle",
        force=False,
    )

    assert processed == 0
    assert checkpoint.enriched_parts == {"p1": 1}

    class _Succeed:
        calls = 0

        def structured_output(self, _prompt, _schema, **_kwargs):
            self.calls += 1
            return generate.EnrichedConcept(
                title="Title",
                description="Description.",
                facts=["Recovered fact."],
                body="# Title\n\nRecovered body.",
            )

    second_client = _Succeed()
    processed, _skipped = generate._enrich_catalog(
        second_client,
        [page],
        catalog,
        checkpoint,
        checkpoint_path,
        tmp_path / "bundle",
        force=False,
    )

    assert processed == 1
    assert second_client.calls == len(generate._page_parts(page.markdown)) - 1
    assert checkpoint.enriched_parts == {}
    assert checkpoint.enriched_pages == {"p1": "destinations/title"}


def test_exact_alias_resolution_reuses_canonical_concept():
    catalog = generate.CanonicalCatalog(
        concepts={
            "destinations/grad-rajhenburg": generate.CanonicalConcept(
                concept_id="destinations/grad-rajhenburg",
                type="Destination",
                title="Grad Rajhenburg",
                description="A castle.",
                aliases=["Rajhenburg Castle"],
            )
        }
    )
    proposal = generate.PageProposal(
        page_id="english-page",
        concept_id="destinations/rajhenburg-castle",
        type="destination",
        title="Rajhenburg Castle",
        description="A castle.",
        tags=["heritage"],
        search_terms=["Brestanica fortress"],
    )

    generate._resolve_exact_matches([proposal], catalog)

    assert catalog.assignments == {"english-page": "destinations/grad-rajhenburg"}
    assert len(catalog.concepts) == 1
    concept = catalog.concepts["destinations/grad-rajhenburg"]
    assert concept.tags == ["heritage"]
    assert concept.search_terms == ["Brestanica fortress"]


def test_resolution_splits_failed_large_batch():
    proposals = [
        generate.PageProposal(
            page_id=f"p{index}",
            concept_id=f"people/person-{index}",
            type="Person",
            title=f"Person {index}",
            description="A person.",
        )
        for index in range(4)
    ]

    class _SplittingAzure:
        def structured_output(self, prompt, schema, **_kwargs):
            assert schema is generate.ResolutionBatch
            proposed = json.loads(prompt.split("PROPOSALS:\n", maxsplit=1)[1])
            page_ids = [item["page_id"] for item in proposed]
            if len(page_ids) > 2:
                raise RuntimeError("truncated response")
            return generate.ResolutionBatch(
                concepts=[
                    generate.CanonicalConcept(
                        concept_id=f"people/{page_id}",
                        type="Person",
                        title=page_id,
                        description="A person.",
                    )
                    for page_id in page_ids
                ],
                decisions=[
                    generate.ResolutionDecision(
                        page_id=page_id,
                        canonical_concept_id=f"people/{page_id}",
                    )
                    for page_id in page_ids
                ],
            )

    catalog = generate.CanonicalCatalog()
    generate._resolve_with_fallback(_SplittingAzure(), proposals, catalog)

    assert catalog.assignments == {
        f"p{index}": f"people/p{index}" for index in range(4)
    }


def test_existing_bundle_seeds_canonical_catalog(tmp_path):
    root = tmp_path / "bundle"
    write_concept(root, "destinations/grad-rajhenburg", _document("Grad Rajhenburg"))
    catalog = generate.CanonicalCatalog()

    generate._seed_catalog_from_bundle(root, catalog)

    assert list(catalog.concepts) == ["destinations/grad-rajhenburg"]
    assert catalog.concepts["destinations/grad-rajhenburg"].title == "Grad Rajhenburg"


class _StubNavigator:
    def __init__(self, actions):
        self.actions = iter(actions)

    def structured_output(self, _prompt, schema, **_kwargs):
        if schema is EvidenceAssessment:
            return EvidenceAssessment(sufficient=True, reason="Directly supported.")
        assert schema is NavigationAction
        return next(self.actions)


def test_navigation_opens_only_advertised_whole_document(tmp_path):
    root = tmp_path / "bundle"
    document = OKFDocument(
        frontmatter={
            **_document().frontmatter,
            "source_page_ids": ["p1"],
            "source_evidence": {
                "p1": {"title": "Rajhenburg", "url": "https://example.test/castle"}
            },
        },
        body=_document().body,
    )
    write_concept(root, "destinations/rajhenburg", document)
    regenerate_indexes(root)
    client = _StubNavigator(
        [
            NavigationAction(action="open", path="destinations/index.md"),
            NavigationAction(action="open", path="destinations/rajhenburg.md"),
            NavigationAction(action="open_source", page_id="p1"),
            NavigationAction(
                action="answer",
                answer="Rajhenburg is a cultural heritage destination.",
                citations=["p1"],
            ),
        ]
    )

    result = answer_question(
        "What is Rajhenburg?",
        bundle_root=root,
        client=client,
        source_reader=lambda page_id: (
            "# Grad Rajhenburg\n\nRajhenburg is a cultural heritage destination "
            "above Brestanica."
            if page_id == "p1"
            else None
        ),
    )

    assert result.sufficient
    assert result.citations == ["p1"]
    assert result.sources == ["p1"]
    assert result.query_count == 5
    assert result.visited == [
        "index.md",
        "destinations/index.md",
        "destinations/rajhenburg.md",
    ]


def test_navigation_rejects_unadvertised_path(tmp_path):
    root = tmp_path / "bundle"
    write_concept(root, "destinations/rajhenburg", _document())
    regenerate_indexes(root)
    client = _StubNavigator([NavigationAction(action="open", path="../outside.md")])

    result = answer_question(
        "Question",
        bundle_root=root,
        client=client,
        source_reader=lambda _page_id: None,
    )

    assert not result.sufficient
    assert result.query_count == 1
    assert "not advertised" in result.reason


def test_rejected_evidence_backtracks_to_unvisited_concept(tmp_path):
    root = tmp_path / "bundle"
    write_concept(
        root,
        "destinations/first",
        OKFDocument(
            frontmatter={**_document("First").frontmatter, "source_page_ids": ["pf"]},
            body=_document("First").body,
        ),
    )
    write_concept(
        root,
        "destinations/second",
        OKFDocument(
            frontmatter={**_document("Second").frontmatter, "source_page_ids": ["ps"]},
            body=_document("Second").body,
        ),
    )
    regenerate_indexes(root)

    class _BacktrackingNavigator:
        def __init__(self):
            self.actions = iter(
                [
                    NavigationAction(action="open", path="destinations/index.md"),
                    NavigationAction(action="open", path="destinations/first.md"),
                    NavigationAction(action="open_source", page_id="pf"),
                    NavigationAction(
                        action="answer",
                        answer="Unsupported",
                        citations=["pf"],
                    ),
                    NavigationAction(action="open", path="destinations/second.md"),
                    NavigationAction(action="open_source", page_id="ps"),
                    NavigationAction(
                        action="answer",
                        answer="Supported",
                        citations=["ps"],
                    ),
                ]
            )
            self.assessments = iter(
                [
                    EvidenceAssessment(sufficient=False, reason="Missing detail."),
                    EvidenceAssessment(sufficient=True, reason="Direct support."),
                ]
            )

        def structured_output(self, _prompt, schema, **_kwargs):
            if schema is EvidenceAssessment:
                return next(self.assessments)
            return next(self.actions)

    result = answer_question(
        "Question",
        bundle_root=root,
        client=_BacktrackingNavigator(),
        source_reader=lambda page_id: f"# Article {page_id}\n\nRaw article body text.",
    )

    assert result.sufficient
    assert result.answer == "Supported"
    assert result.citations == ["ps"]
    assert result.sources == ["pf", "ps"]
    assert result.query_count == 9
    assert [
        entry["sufficient"]
        for entry in result.trace
        if entry["kind"] == "evidence_assessment"
    ] == [False, True]


def test_okf_page_evidence_retriever_maps_visits_and_effort(monkeypatch, tmp_path):
    root = tmp_path / "bundle"
    document = _document()
    document = OKFDocument(
        frontmatter={**document.frontmatter, "source_page_ids": ["p1", "p2"]},
        body=document.body,
    )
    write_concept(root, "destinations/rajhenburg", document)
    regenerate_indexes(root)
    monkeypatch.setattr(
        evidence,
        "answer_question",
        lambda *_args, **_kwargs: OKFAnswer(
            answer="Answer",
            citations=["destinations/rajhenburg"],
            visited=["index.md", "destinations/rajhenburg.md"],
            query_count=3,
            sufficient=True,
            reason="",
        ),
    )

    retriever = evidence.OKFPageEvidenceRetriever(root)
    ranked = retriever.retrieve("Question", limit=10)

    assert [item.id for item in ranked] == ["p1", "p2"]
    assert retriever.covered_page_ids == {"p1", "p2"}
    assert retriever.total_queries() == 3
    assert retriever.average_queries_per_question() == 3


def test_retrieval_ready_filter_excludes_failed_source_pages(monkeypatch, tmp_path):
    root = tmp_path / "bundle"
    document = _document()
    document = OKFDocument(
        frontmatter={
            **document.frontmatter,
            "source_page_ids": ["ready", "failed"],
            "source_evidence": {
                "ready": {
                    "facts": ["Supported fact."],
                    "retrieval_queries": ["Which fact is supported?"],
                }
            },
        },
        body=document.body,
    )
    write_concept(root, "destinations/rajhenburg", document)
    regenerate_indexes(root)
    monkeypatch.setattr(
        evidence,
        "answer_question",
        lambda *_args, **_kwargs: OKFAnswer(
            answer="Answer",
            citations=["destinations/rajhenburg"],
            visited=["destinations/rajhenburg.md"],
            query_count=1,
            sufficient=True,
            reason="",
        ),
    )

    legacy_retriever = evidence.OKFPageEvidenceRetriever(root)
    retriever = evidence.OKFPageEvidenceRetriever(root, retrieval_ready_only=True)
    ranked = retriever.retrieve("Question", limit=10)

    assert legacy_retriever.covered_page_ids == {"ready", "failed"}
    assert retriever.covered_page_ids == {"ready"}
    assert [item.id for item in ranked] == ["ready"]


def test_metadata_search_supplies_ranked_candidate_and_records_trace(
    monkeypatch, tmp_path
):
    root = tmp_path / "bundle"
    castle = OKFDocument(
        frontmatter={
            **_document().frontmatter,
            "aliases": ["Grad Rajhenburg"],
            "tags": ["fortress"],
            "source_page_ids": ["castle-page"],
        },
        body=_document().body,
    )
    write_concept(root, "destinations/rajhenburg", castle)
    write_concept(root, "people/someone", _document("Someone"))
    regenerate_indexes(root)
    captured = {}

    def _answer(*_args, **kwargs):
        captured["candidates"] = list(kwargs["candidate_paths"])
        return OKFAnswer(
            answer="Answer",
            citations=["destinations/rajhenburg"],
            visited=["index.md", "destinations/rajhenburg.md"],
            query_count=2,
            sufficient=True,
            reason="supported",
            trace=[{"kind": "navigation", "action": "answer"}],
        )

    monkeypatch.setattr(evidence, "answer_question", _answer)
    retriever = evidence.OKFSearchPageEvidenceRetriever(root)

    ranked = retriever.retrieve("Grad Rajhenburg fortress", limit=10)

    assert captured["candidates"][0] == "destinations/rajhenburg.md"
    assert [item.id for item in ranked] == ["castle-page"]
    assert retriever.action_log[0]["trace"][0]["action"] == "answer"


def test_large_collection_is_split_into_bounded_semantic_indexes(tmp_path):
    root = tmp_path / "bundle"
    for index in range(21):
        document = OKFDocument(
            frontmatter={
                **_document(f"Person {index:02d}").frontmatter,
                "type": "Person",
                "aliases": [f"Alias {index:02d}"],
            },
            body=_document().body,
        )
        write_concept(root, f"people/person-{index:02d}", document)

    regenerate_indexes(root)

    people_index = (root / "people/index.md").read_text()
    first_browse = (root / "people/_browse-01/index.md").read_text()
    assert "_browse-01/index.md" in people_index
    assert "Contains 20 entries" in people_index
    assert first_browse.count("* [Person") == 20
    assert "Keywords: Alias 00." in first_browse
