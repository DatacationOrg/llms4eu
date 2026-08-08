import json
import sqlite3

import pytest
import httpx

from src.okf import answer
from src.okf.answer import (
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
from src.eval.metrics import score_rankings
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


def test_concept_proposal_normalizes_safe_punctuation():
    proposal = generate.ConceptProposal(
        concept_id="geography/utc+02:00",
        type="Time zone",
        title="UTC+02:00",
        description="A time zone.",
    )

    assert proposal.concept_id == "geography/utc-02-00"


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
            )
        if schema is generate.ResolutionBatch:
            return generate.ResolutionBatch(
                concepts=[
                    generate.CanonicalConcept(
                        concept_id="destinations/rajhenburg",
                        type="Destination",
                        title="Grad Rajhenburg",
                        description="A castle above Brestanica.",
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
        },
    )
    monkeypatch.setattr(generate, "_azure_client", lambda: _StubAzure())

    first = generate.generate_bundle()
    second = generate.generate_bundle()
    concept = OKFDocument.parse(
        (tmp_path / "bundle/destinations/rajhenburg.md").read_text()
    )
    checkpoint = json.loads((tmp_path / ".local/checkpoint.json").read_text())

    assert first["processed_pages"] == 1
    assert second["resumed_pages"] == 1
    assert first["discovered_pages"] == 1
    assert first["resolved_pages"] == 1
    assert second["discovered_pages"] == 0
    assert second["resolved_pages"] == 0
    assert concept.frontmatter["source_page_ids"] == ["p1"]
    assert list(concept.frontmatter) == [
        "type",
        "title",
        "description",
        "timestamp",
        "source_page_ids",
    ]
    assert concept.body.count("https://example.test/castle") == 1
    assert "retrieval_metadata" not in first
    assert checkpoint["enriched_pages"] == {"p1": "destinations/rajhenburg"}


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


def test_enrichment_merges_page_provenance_and_citations():
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
            body=f"# {label}\n\nFact {label}.\n\n# Citations\n\n- [Source](https://example.test/castle)",
        )

    first = generate._document(page, plan, enrichment("one"), None)
    second = generate._document(page, plan, enrichment("two"), first)
    assert second.frontmatter["source_page_ids"] == ["p1"]
    assert second.body.startswith("# two\n\nFact two.")
    assert second.body.count("https://example.test/castle") == 1


def test_clean_generation_removes_bundle_and_private_state(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "old.md").write_text("old")
    state_root = tmp_path / ".local/okf"
    state_root.mkdir(parents=True)
    (state_root / "obsolete.json").write_text("old")

    generate._reset_generation(bundle, state_root)

    assert not bundle.exists()
    assert not state_root.exists()


def test_clean_generation_rejects_partial_selection():
    with pytest.raises(ValueError, match="full corpus"):
        generate.generate_bundle(source="castle", clean=True)


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
        overwrite=False,
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
        overwrite=False,
    )

    assert processed == 1
    assert second_client.calls == len(generate._page_parts(page.markdown)) - 1


def test_repeated_enrichment_failure_uses_source_fallback(tmp_path):
    page = generate.SourcePage(
        "p1",
        "source",
        "https://example.test",
        "Title",
        "en",
        "# Title\n\nOriginal source fact.",
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
    checkpoint = generate.GenerationCheckpoint(failures={page.id: "prior failure"})

    class _Fail:
        def structured_output(self, _prompt, _schema, **_kwargs):
            raise RuntimeError("failed again")

    processed, _skipped = generate._enrich_catalog(
        _Fail(),
        [page],
        catalog,
        checkpoint,
        tmp_path / "checkpoint.json",
        tmp_path / "bundle",
        overwrite=False,
    )

    concept = generate.OKFDocument.parse(
        (tmp_path / "bundle" / "destinations" / "title.md").read_text()
    )
    assert processed == 1
    assert checkpoint.enriched_pages == {"p1": "destinations/title"}
    assert checkpoint.failures == {}
    assert "failed again" in checkpoint.fallback_pages["p1"]
    assert "Original source fact." in concept.body
    assert checkpoint.enriched_parts == {}
    assert checkpoint.enriched_pages == {"p1": "destinations/title"}


def test_exact_title_resolution_reuses_canonical_concept():
    catalog = generate.CanonicalCatalog(
        concepts={
            "destinations/grad-rajhenburg": generate.CanonicalConcept(
                concept_id="destinations/grad-rajhenburg",
                type="Destination",
                title="Grad Rajhenburg",
                description="A castle.",
            )
        }
    )
    proposal = generate.PageProposal(
        page_id="english-page",
        concept_id="destinations/grad-rajhenburg",
        type="destination",
        title="Grad Rajhenburg",
        description="A castle.",
    )

    generate._resolve_exact_matches([proposal], catalog)

    assert catalog.assignments == {"english-page": "destinations/grad-rajhenburg"}
    assert len(catalog.concepts) == 1


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


def test_resolution_splits_batch_with_unknown_concepts():
    proposals = [
        generate.PageProposal(
            page_id=f"p{index}",
            concept_id=f"places/place-{index}",
            type="Place",
            title=f"Place {index}",
            description="A place.",
        )
        for index in range(2)
    ]

    class _IncompleteAzure:
        def structured_output(self, prompt, schema, **_kwargs):
            assert schema is generate.ResolutionBatch
            proposed = json.loads(prompt.split("PROPOSALS:\n", maxsplit=1)[1])
            page_ids = [item["page_id"] for item in proposed]
            concepts = []
            if len(page_ids) == 1:
                concepts.append(
                    generate.CanonicalConcept(
                        concept_id=f"places/{page_ids[0]}",
                        type="Place",
                        title=page_ids[0],
                        description="A place.",
                    )
                )
            return generate.ResolutionBatch(
                concepts=concepts,
                decisions=[
                    generate.ResolutionDecision(
                        page_id=page_id,
                        canonical_concept_id=f"places/{page_id}",
                    )
                    for page_id in page_ids
                ],
            )

    catalog = generate.CanonicalCatalog()
    generate._resolve_with_fallback(_IncompleteAzure(), proposals, catalog)

    assert catalog.assignments == {"p0": "places/p0", "p1": "places/p1"}


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
        assert schema is NavigationAction
        return next(self.actions)


def test_navigation_opens_only_advertised_whole_document(tmp_path):
    root = tmp_path / "bundle"
    document = OKFDocument(
        frontmatter={
            **_document().frontmatter,
            "source_page_ids": ["p1"],
        },
        body=(
            "# Overview\n\nRajhenburg is a cultural heritage destination "
            "above Brestanica.\n"
        ),
    )
    write_concept(root, "destinations/rajhenburg", document)
    regenerate_indexes(root)
    client = _StubNavigator(
        [
            NavigationAction(action="open", path="destinations/index.md"),
            NavigationAction(action="open", path="destinations/rajhenburg.md"),
            NavigationAction(
                action="answer",
                answer="Rajhenburg is a cultural heritage destination.",
                citations=["destinations/rajhenburg.md"],
            ),
        ]
    )

    result = answer_question(
        "What is Rajhenburg?",
        bundle_root=root,
        client=client,
    )

    assert result.sufficient
    assert result.citations == ["destinations/rajhenburg.md"]
    assert result.query_count == 3
    assert result.visited == [
        "index.md",
        "destinations/index.md",
        "destinations/rajhenburg.md",
    ]


def test_navigation_uses_query_focused_excerpt_for_oversized_concept(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "bundle"
    document = OKFDocument(
        frontmatter={**_document().frontmatter, "source_page_ids": ["p1"]},
        body=(
            "# Roman Empire\n\n"
            + "unrelated material " * 400
            + "\n\nCourier travel from Mainz to Rome took nine days.\n"
        ),
    )
    write_concept(root, "history/roman-empire", document)
    regenerate_indexes(root)
    monkeypatch.setitem(answer.CONFIG["navigation"], "max_context_chars", 5_000)
    client = _StubNavigator(
        [
            NavigationAction(action="open", path="history/index.md"),
            NavigationAction(action="open", path="history/roman-empire.md"),
            NavigationAction(
                action="answer",
                answer="Nine days.",
                citations=["history/roman-empire.md"],
            ),
        ]
    )

    result = answer_question(
        "How long did courier travel from Mainz to Rome take?",
        bundle_root=root,
        client=client,
    )

    assert result.sufficient
    assert result.citations == ["history/roman-empire.md"]
    assert "history/roman-empire.md" in result.visited
    assert any(item["kind"] == "document_excerpt" for item in result.trace)
    excerpt = answer._document_context(
        document.serialize(),
        "How long did courier travel from Mainz to Rome take?",
        5_000,
    )
    assert "Courier travel from Mainz to Rome took nine days." in excerpt


def test_navigation_rejects_unadvertised_path(tmp_path):
    root = tmp_path / "bundle"
    write_concept(root, "destinations/rajhenburg", _document())
    regenerate_indexes(root)
    client = _StubNavigator([NavigationAction(action="open", path="../outside.md")])

    result = answer_question(
        "Question",
        bundle_root=root,
        client=client,
    )

    assert not result.sufficient
    assert result.query_count == 1
    assert "not advertised" in result.reason


def test_navigation_can_read_multiple_concepts_before_answering(tmp_path):
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

    class _MultiConceptNavigator:
        def __init__(self):
            self.actions = iter(
                [
                    NavigationAction(action="open", path="destinations/index.md"),
                    NavigationAction(action="open", path="destinations/first.md"),
                    NavigationAction(action="open", path="destinations/second.md"),
                    NavigationAction(
                        action="answer",
                        answer="Supported",
                        citations=["destinations/second.md"],
                    ),
                ]
            )

        def structured_output(self, _prompt, schema, **_kwargs):
            assert schema is NavigationAction
            return next(self.actions)

    result = answer_question(
        "Question",
        bundle_root=root,
        client=_MultiConceptNavigator(),
    )

    assert result.sufficient
    assert result.answer == "Supported"
    assert result.citations == ["destinations/second.md"]
    assert result.query_count == 4
    assert result.visited == [
        "index.md",
        "destinations/index.md",
        "destinations/first.md",
        "destinations/second.md",
    ]


def test_okf_concept_retriever_ranks_citations_before_visits(monkeypatch, tmp_path):
    root = tmp_path / "bundle"
    write_concept(root, "destinations/first", _document("First"))
    write_concept(root, "destinations/second", _document("Second"))
    regenerate_indexes(root)
    monkeypatch.setattr(
        evidence,
        "answer_question",
        lambda *_args, **_kwargs: OKFAnswer(
            answer="Answer",
            citations=["destinations/second.md"],
            visited=[
                "index.md",
                "destinations/first.md",
                "destinations/second.md",
            ],
            query_count=3,
            sufficient=True,
            reason="",
        ),
    )

    retriever = evidence.OKFConceptRetriever(root)
    ranked = retriever.retrieve("Question", limit=10)

    assert [item.id for item in ranked] == [
        "destinations/second.md",
        "destinations/first.md",
    ]
    assert retriever.total_queries() == 3
    assert retriever.average_queries_per_question() == 3


def test_page_projection_gives_sibling_pages_the_same_concept():
    page_concepts = evidence.invert_page_map(
        {"destinations/castle.md": ["gold-sl", "sibling-en"]}
    )

    assert evidence.project_pages_to_concepts(["gold-sl"], page_concepts) == [
        "destinations/castle.md"
    ]
    assert evidence.project_pages_to_concepts(["sibling-en"], page_concepts) == [
        "destinations/castle.md"
    ]

    gold = evidence.project_pages_to_concepts(["gold-sl"], page_concepts)
    retrieved = evidence.project_pages_to_concepts(["sibling-en"], page_concepts)
    scores = score_rankings(
        [{"question_id": "q1", "chunk_id": gold[0]}],
        {"q1": retrieved},
        ks=(1,),
        recall_k=1,
    )

    assert scores["hit@1"] == 1.0
    assert scores["recall@1"] == 1.0


def test_large_collection_is_split_into_bounded_semantic_indexes(tmp_path):
    root = tmp_path / "bundle"
    for index in range(21):
        document = OKFDocument(
            frontmatter={
                **_document(f"Person {index:02d}").frontmatter,
                "type": "Person",
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
    assert "Keywords:" not in first_browse
