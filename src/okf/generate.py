from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.okf.bundle import (
    load_concepts,
    regenerate_indexes,
    validate_bundle,
    write_concept,
)
from src.okf.document import OKFDocument
from src.okf.paths import parse_concept_id
from src.okf.source import SourcePage, load_source_pages
from src.shared.env import ROOT, load_local_env, load_yaml
from src.shared.llm import AzureFoundryStructuredLlm

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
SOURCE_EVIDENCE_HEADING = "# Source evidence by page"
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class PagePart:
    index: int
    total: int
    heading_path: str
    markdown: str


class ConceptProposal(BaseModel):
    concept_id: str = Field(description="Proposed lowercase category/slug concept id")
    type: str = Field(description="Concrete semantic concept type")
    title: str
    description: str = Field(description="One factual sentence")
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(
        default_factory=list,
        description="Distinctive names, roles, places, dates, and topics",
    )

    @field_validator("concept_id")
    @classmethod
    def valid_concept_id(cls, value: str) -> str:
        parse_concept_id(value)
        return value


class PageProposal(ConceptProposal):
    page_id: str


class CanonicalConcept(ConceptProposal):
    pass


class ResolutionDecision(BaseModel):
    page_id: str
    canonical_concept_id: str

    @field_validator("canonical_concept_id")
    @classmethod
    def valid_concept_id(cls, value: str) -> str:
        parse_concept_id(value)
        return value


class ResolutionBatch(BaseModel):
    concepts: list[CanonicalConcept]
    decisions: list[ResolutionDecision]


class EnrichedConcept(BaseModel):
    title: str
    description: str = Field(description="One factual sentence")
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(
        default_factory=list,
        description="Additional supported names found on this source page",
    )
    search_terms: list[str] = Field(
        default_factory=list,
        description="Names, roles, places, periods, and events useful for retrieval",
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="Natural questions and paraphrases this source can answer",
    )
    source_summary: str = Field(
        default="",
        description="A source-specific factual summary with distinctive entities",
    )
    facts: list[str] = Field(
        default_factory=list,
        description="Atomic source-supported facts preserving names, dates, and numbers",
    )
    body: str = Field(
        default="", description="Factual Markdown body without YAML frontmatter"
    )


class GenerationCheckpoint(BaseModel):
    enriched_pages: dict[str, str] = Field(default_factory=dict)
    enriched_parts: dict[str, int] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)


class DiscoveryInventory(BaseModel):
    proposals: dict[str, PageProposal] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)


class CanonicalCatalog(BaseModel):
    concepts: dict[str, CanonicalConcept] = Field(default_factory=dict)
    assignments: dict[str, str] = Field(default_factory=dict)


def generate_bundle(
    *,
    source: str | None = None,
    limit: int | None = None,
    force: bool = False,
    refresh_retrieval: bool = False,
) -> dict:
    load_local_env()
    client = _azure_client()
    bundle_root = ROOT / CONFIG["bundle_path"]
    generation_checkpoint_path = ROOT / CONFIG["checkpoint_path"]
    checkpoint_path = (
        ROOT
        / CONFIG.get(
            "retrieval_checkpoint_path",
            str(Path(CONFIG["checkpoint_path"]).with_name("retrieval-refresh.json")),
        )
        if refresh_retrieval
        else generation_checkpoint_path
    )
    inventory_path = ROOT / CONFIG["inventory_path"]
    catalog_path = ROOT / CONFIG["catalog_path"]
    if force:
        checkpoint = GenerationCheckpoint()
    elif refresh_retrieval:
        checkpoint = _load_refresh_checkpoint(
            checkpoint_path,
            generation_checkpoint_path,
            bundle_root,
        )
    else:
        checkpoint = _load_checkpoint(checkpoint_path)
    inventory = (
        DiscoveryInventory()
        if force
        else _load_model(inventory_path, DiscoveryInventory)
    )
    catalog = (
        CanonicalCatalog() if force else _load_model(catalog_path, CanonicalCatalog)
    )
    _seed_catalog_from_bundle(bundle_root, catalog)
    pages = load_source_pages(ROOT / CONFIG["source_db"], source=source, limit=limit)

    discovered = _discover_pages(client, pages, inventory, inventory_path)
    resolved = _resolve_inventory(client, pages, inventory, catalog, catalog_path)
    processed, skipped = _enrich_catalog(
        client,
        pages,
        catalog,
        checkpoint,
        checkpoint_path,
        bundle_root,
        force=force,
        metadata_only=refresh_retrieval,
    )

    regenerate_indexes(bundle_root)
    report = validate_bundle(bundle_root)
    retrieval_stats = _retrieval_stats(bundle_root)
    selected_ids = {page.id for page in pages}
    selected_assignments = {
        page_id: concept_id
        for page_id, concept_id in catalog.assignments.items()
        if page_id in selected_ids
    }
    selected_failures = {
        **{
            page_id: error
            for page_id, error in inventory.failures.items()
            if page_id in selected_ids
        },
        **{
            page_id: error
            for page_id, error in checkpoint.failures.items()
            if page_id in selected_ids
        },
    }
    manifest = {
        "okf_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_db": CONFIG["source_db"],
        "source_filter": source,
        "generation_mode": (
            "force" if force else "refresh_retrieval" if refresh_retrieval else "resume"
        ),
        "selected_pages": len(pages),
        "discovered_pages": discovered,
        "resolved_pages": resolved,
        "processed_pages": processed,
        "resumed_pages": skipped,
        "covered_pages": len(checkpoint.enriched_pages),
        "failures": selected_failures,
        "concepts": len(set(selected_assignments.values())),
        "catalog_concepts": len(catalog.concepts),
        "retrieval_metadata": retrieval_stats,
        "model": os.environ.get("AZURE_AI_MODEL", ""),
        "validation_errors": report.errors,
        "validation_warnings": report.warnings,
    }
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if report.errors:
        raise RuntimeError(
            "Generated bundle failed validation: " + "; ".join(report.errors)
        )
    return manifest


def _discover_pages(
    client: AzureFoundryStructuredLlm,
    pages: list[SourcePage],
    inventory: DiscoveryInventory,
    inventory_path: Path,
) -> int:
    discovered = 0
    for index, page in enumerate(pages, start=1):
        if page.id in inventory.proposals:
            continue
        print(
            f"[discover {index}/{len(pages)}] {page.source}: {page.title or page.url}",
            flush=True,
        )
        try:
            parts = _page_parts(page.markdown)
            proposals = [
                client.structured_output(
                    _discovery_prompt(page, part),
                    ConceptProposal,
                    retries=CONFIG["retries"],
                )
                for part in parts
            ]
            proposal = (
                proposals[0]
                if len(proposals) == 1
                else client.structured_output(
                    _consolidation_prompt(page, proposals),
                    ConceptProposal,
                    retries=CONFIG["retries"],
                )
            )
            inventory.proposals[page.id] = PageProposal(
                page_id=page.id,
                **proposal.model_dump(),
            )
            inventory.failures.pop(page.id, None)
            discovered += 1
        except Exception as exc:
            inventory.failures[page.id] = _safe_error(exc)
            print(f"  failed: {inventory.failures[page.id]}", flush=True)
        _write_model(inventory_path, inventory)
    return discovered


def _seed_catalog_from_bundle(bundle_root: Path, catalog: CanonicalCatalog) -> None:
    for concept_id, document in load_concepts(bundle_root).items():
        if concept_id in catalog.concepts:
            continue
        frontmatter = document.frontmatter
        catalog.concepts[concept_id] = CanonicalConcept(
            concept_id=concept_id,
            type=str(frontmatter["type"]),
            title=str(frontmatter["title"]),
            description=str(frontmatter["description"]),
            tags=[str(value) for value in frontmatter.get("tags", [])],
            aliases=[str(value) for value in frontmatter.get("aliases", [])],
            search_terms=[str(value) for value in frontmatter.get("search_terms", [])],
        )


def _resolve_inventory(
    client: AzureFoundryStructuredLlm,
    pages: list[SourcePage],
    inventory: DiscoveryInventory,
    catalog: CanonicalCatalog,
    catalog_path: Path,
) -> int:
    selected_ids = {page.id for page in pages}
    pending = [
        proposal
        for page_id, proposal in inventory.proposals.items()
        if page_id in selected_ids and page_id not in catalog.assignments
    ]
    resolved = 0
    batch_size = CONFIG["resolution_batch_size"]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        _resolve_exact_matches(batch, catalog)
        unresolved = [item for item in batch if item.page_id not in catalog.assignments]
        if unresolved:
            _resolve_with_fallback(client, unresolved, catalog)
        resolved += len(batch)
        _write_model(catalog_path, catalog)
        print(
            f"[resolve] {min(start + batch_size, len(pending))}/{len(pending)}",
            flush=True,
        )
    return resolved


def _resolve_with_fallback(
    client: AzureFoundryStructuredLlm,
    proposals: list[PageProposal],
    catalog: CanonicalCatalog,
) -> None:
    try:
        result = client.structured_output(
            _resolution_prompt(proposals, catalog),
            ResolutionBatch,
            retries=CONFIG["retries"],
        )
        _apply_resolution(result, proposals, catalog)
    except RuntimeError:
        if len(proposals) == 1:
            raise
        midpoint = len(proposals) // 2
        print(
            f"  canonicalization response failed; retrying as {midpoint} and "
            f"{len(proposals) - midpoint} proposals",
            flush=True,
        )
        _resolve_with_fallback(client, proposals[:midpoint], catalog)
        _resolve_with_fallback(client, proposals[midpoint:], catalog)


def _resolve_exact_matches(
    proposals: list[PageProposal], catalog: CanonicalCatalog
) -> None:
    names: dict[tuple[str, str], str] = {}
    for concept_id, concept in catalog.concepts.items():
        for name in [concept.title, *concept.aliases]:
            names[(_normalize(name), _normalize(concept.type))] = concept_id
    for proposal in proposals:
        proposal_type = _normalize(proposal.type)
        matching = {
            names[(normalized, proposal_type)]
            for name in [proposal.title, *proposal.aliases]
            if (normalized := _normalize(name), proposal_type) in names
        }
        if len(matching) == 1:
            concept_id = matching.pop()
            catalog.assignments[proposal.page_id] = concept_id
            _merge_proposal_metadata(catalog, concept_id, proposal)


def _apply_resolution(
    result: ResolutionBatch,
    proposals: list[PageProposal],
    catalog: CanonicalCatalog,
) -> None:
    expected = {proposal.page_id for proposal in proposals}
    decisions = {
        decision.page_id: decision.canonical_concept_id for decision in result.decisions
    }
    if set(decisions) != expected:
        raise ValueError(
            "Canonicalization must assign every unresolved page exactly once"
        )
    returned = {concept.concept_id: concept for concept in result.concepts}
    known = set(catalog.concepts) | set(returned)
    unknown = set(decisions.values()) - known
    if unknown:
        raise ValueError(
            f"Canonicalization assigned unknown concepts: {sorted(unknown)}"
        )
    catalog.concepts.update(returned)
    catalog.assignments.update(decisions)
    proposals_by_page = {proposal.page_id: proposal for proposal in proposals}
    for page_id, concept_id in decisions.items():
        _merge_proposal_metadata(catalog, concept_id, proposals_by_page[page_id])


def _merge_proposal_metadata(
    catalog: CanonicalCatalog,
    concept_id: str,
    proposal: PageProposal,
) -> None:
    concept = catalog.concepts[concept_id]
    aliases = [*concept.aliases, *proposal.aliases]
    if _normalize(proposal.title) != _normalize(concept.title):
        aliases.append(proposal.title)
    catalog.concepts[concept_id] = concept.model_copy(
        update={
            "tags": _unique([*concept.tags, *proposal.tags]),
            "aliases": _unique(aliases),
            "search_terms": _unique([*concept.search_terms, *proposal.search_terms]),
        }
    )


def _enrich_catalog(
    client: AzureFoundryStructuredLlm,
    pages: list[SourcePage],
    catalog: CanonicalCatalog,
    checkpoint: GenerationCheckpoint,
    checkpoint_path: Path,
    bundle_root: Path,
    *,
    force: bool,
    metadata_only: bool = False,
) -> tuple[int, int]:
    pages_by_concept: dict[str, list[SourcePage]] = defaultdict(list)
    for page in pages:
        concept_id = catalog.assignments.get(page.id)
        if concept_id:
            pages_by_concept[concept_id].append(page)
    processed = 0
    skipped = 0
    for concept_index, (concept_id, concept_pages) in enumerate(
        sorted(pages_by_concept.items()), start=1
    ):
        plan = catalog.concepts[concept_id]
        path = bundle_root.joinpath(*parse_concept_id(concept_id)).with_suffix(".md")
        existing = (
            OKFDocument.parse(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )
        for page_index, page in enumerate(concept_pages, start=1):
            if page.id in checkpoint.enriched_pages and not force:
                skipped += 1
                continue
            print(
                f"[enrich {concept_index}/{len(pages_by_concept)} page {page_index}/{len(concept_pages)}] {concept_id}",
                flush=True,
            )
            try:
                parts = _page_parts(page.markdown)
                completed_part = checkpoint.enriched_parts.get(page.id, 0)
                for part in parts:
                    if part.index <= completed_part and not force:
                        continue
                    print(
                        f"  part {part.index}/{part.total}: "
                        f"{part.heading_path or 'untitled'} ({len(part.markdown)} chars)",
                        flush=True,
                    )
                    enriched = client.structured_output(
                        _enrichment_prompt(
                            page,
                            part,
                            plan,
                            existing,
                            catalog,
                            metadata_only=metadata_only and existing is not None,
                        ),
                        EnrichedConcept,
                        retries=CONFIG["retries"],
                    )
                    existing = _document(page, plan, enriched, existing)
                    write_concept(bundle_root, concept_id, existing)
                    checkpoint.enriched_parts[page.id] = part.index
                    checkpoint.failures.pop(page.id, None)
                    _write_model(checkpoint_path, checkpoint)
                checkpoint.enriched_pages[page.id] = concept_id
                checkpoint.enriched_parts.pop(page.id, None)
                checkpoint.failures.pop(page.id, None)
                processed += 1
            except Exception as exc:
                checkpoint.failures[page.id] = _safe_error(exc)
                print(f"  failed: {checkpoint.failures[page.id]}", flush=True)
            _write_model(checkpoint_path, checkpoint)
    return processed, skipped


def _azure_client() -> AzureFoundryStructuredLlm:
    missing = [
        name
        for name in ("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY", "AZURE_AI_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError("Missing Azure configuration: " + ", ".join(missing))
    return AzureFoundryStructuredLlm.from_env(
        timeout_seconds=CONFIG["request_timeout_seconds"],
        max_tokens=CONFIG["max_output_tokens"],
    )


def _discovery_prompt(page: SourcePage, part: PagePart) -> str:
    return f"""You organize complete tourism and cultural heritage pages into Open Knowledge Format concepts.
Propose the one primary, independently meaningful concept represented by the source page. You are reading part {part.index} of {part.total}; identify the page's main entity rather than treating this part or its heading as a separate concept. Do not mint concepts for headings, navigation labels, opening hours, contact details, incidental mentions, introductions, or generic "more information" sections; keep those inside their parent entity.
Return likely multilingual and historical names in aliases so a later global catalog pass can merge equivalent proposals.
Return search_terms with distinctive proper names, roles, locations, periods, dates, events, and terminology actually present on the page. Include useful variants in both the source language and English when the page supports them.
The concept id must be lowercase `category/slug`, ASCII, with underscores or hyphens. Prefer this compact top-level taxonomy: destinations, people, organizations, events, exhibitions, services, history, culture, infrastructure, geography, or references. Do not create a new top-level category merely for a narrow page topic.
Do not invent facts. Make the description one discriminative factual sentence that identifies the entity or topic by type, location or period, and its most distinctive role or feature. Avoid generic descriptions such as "a documented destination" or "information about a person".

Source: {page.source}
URL: {page.url}
Title: {page.title}
Language: {page.language}
Part heading context: {part.heading_path or "(none)"}

Page Markdown part {part.index}/{part.total}:
{part.markdown}
"""


def _consolidation_prompt(
    page: SourcePage,
    proposals: list[ConceptProposal],
) -> str:
    return f"""Consolidate part-level proposals for one source page into exactly one Open Knowledge Format concept.
The proposals describe different parts of the same page, not independent pages. Select the page's primary named entity or reusable topic. Merge supported aliases, tags, and search terms. Use a compact existing top-level taxonomy and a discriminative one-sentence description. Do not invent facts.

Source: {page.source}
URL: {page.url}
Title: {page.title}
Language: {page.language}

PART PROPOSALS:
{json.dumps([proposal.model_dump() for proposal in proposals], ensure_ascii=False, indent=2)}
"""


def _resolution_prompt(proposals: list[PageProposal], catalog: CanonicalCatalog) -> str:
    known = [
        {
            "concept_id": concept.concept_id,
            "type": concept.type,
            "title": concept.title,
            "aliases": concept.aliases,
            "description": concept.description,
            "search_terms": concept.search_terms,
        }
        for concept in catalog.concepts.values()
    ]
    proposed = [proposal.model_dump() for proposal in proposals]
    return f"""Resolve proposed tourism concepts into a canonical Open Knowledge Format catalog.
This is the global anti-duplication pass. Assign every proposed page exactly once.

Rules:
1. Reuse a known canonical concept when type and entity identity match, including translations, historical names, abbreviations, and minor spelling differences.
2. Merge equivalent proposals in this batch into one canonical concept.
3. Create a new concept only for an independently meaningful named entity or reusable knowledge topic. Do not mint concepts for headings, opening hours, contact details, incidental mentions, introductions, or generic navigation pages.
4. Preserve distinct entities even when their names are similar.
5. Canonical ids must be stable lowercase `category/slug` paths. Prefer locally canonical names; aliases retain translations and historical names.
6. Return `concepts` only for newly minted canonical concepts. Do not redefine known concepts.
7. Return exactly one decision for every page_id in PROPOSALS. A decision may target a known concept or a newly returned concept.
8. Do not invent page ids or concepts unsupported by the proposals.

KNOWN CANONICAL CONCEPTS:
{json.dumps(known, ensure_ascii=False, indent=2)}

PROPOSALS:
{json.dumps(proposed, ensure_ascii=False, indent=2)}
"""


def _enrichment_prompt(
    page: SourcePage,
    part: PagePart,
    plan: CanonicalConcept,
    existing: OKFDocument | None,
    catalog: CanonicalCatalog,
    *,
    metadata_only: bool = False,
) -> str:
    existing_text = existing.serialize() if existing else "(none)"
    known_targets = [
        {
            "concept_id": concept.concept_id,
            "title": concept.title,
            "type": concept.type,
        }
        for concept in catalog.concepts.values()
        if concept.concept_id != plan.concept_id
    ]
    body_instruction = (
        "This is a retrieval-metadata refresh. Return body as an empty string; "
        "the existing body is retained deterministically."
        if metadata_only
        else "Return Markdown body only in `body`; never include YAML frontmatter there."
    )
    return f"""Create or augment one Open Knowledge Format concept from a complete source page.
{body_instruction}
Preserve all supported facts and headings from the existing document. Add every concrete fact from the source, including details that may appear minor, retain multilingual names, and avoid duplication.
Return at most 30 concise search_terms containing important names, aliases, roles, places, periods, events, dates, and distinctive phrases supported by the source. Return additional aliases found on this page.
Return a source_summary for this part that distinguishes the source from related concepts. Extract atomic facts from this source part only: each fact must stand alone, retain exact proper names, dates, quantities, negation, and relationships, and contain no unsupported inference.
Return at most 40 atomic facts and 3–12 realistic retrieval_queries in the source language and, where useful, English. Cover explicit names as well as vague role-, place-, period-, and event-based questions. Queries must be answerable from this part; do not include answers in the query.
Use structural Markdown. Include a final `# Citations` section containing the exact source URL. Do not cite URLs absent from the input or existing document. Do not invent relationships or facts. You may link a clearly supported relationship only to a concept in KNOWN TARGETS, using a relative Markdown path from this concept document.

Concept plan:
{plan.model_dump_json(indent=2)}

Existing concept:
{existing_text}

Known targets:
{json.dumps(known_targets, ensure_ascii=False, indent=2)}

Source page id: {page.id}
Source name: {page.source}
Source URL: {page.url}
Language: {page.language}
Part: {part.index}/{part.total}
Part heading context: {part.heading_path or "(none)"}
Page Markdown part:
{part.markdown}
"""


def _document(
    page: SourcePage,
    plan: CanonicalConcept,
    enriched: EnrichedConcept,
    existing: OKFDocument | None,
) -> OKFDocument:
    old = existing.frontmatter if existing else {}
    page_ids = _unique([*old.get("source_page_ids", []), page.id])
    source_urls = _unique([*old.get("source_urls", []), page.url])
    source_names = _unique([*old.get("source_names", []), page.source])
    languages = _unique(
        [*old.get("languages", []), page.language]
        if page.language
        else old.get("languages", [])
    )
    tags = _unique([*old.get("tags", []), *plan.tags, *enriched.tags])
    search_terms = _bounded_unique(
        [
            *old.get("search_terms", []),
            *plan.aliases,
            *plan.tags,
            *plan.search_terms,
            *enriched.search_terms,
        ],
        40,
    )
    aliases = _unique([*old.get("aliases", []), *plan.aliases, *enriched.aliases])
    retrieval_queries = _bounded_unique(
        [*old.get("retrieval_queries", []), *enriched.retrieval_queries], 24
    )
    previous_evidence = old.get("source_evidence", {}).get(page.id, {})
    source_evidence = {
        **old.get("source_evidence", {}),
        page.id: {
            "title": page.title,
            "source": page.source,
            "url": page.url,
            "language": page.language,
            "summary": " ".join(
                _unique(
                    [
                        str(previous_evidence.get("summary", "")).strip(),
                        enriched.source_summary.strip(),
                    ]
                )
            ),
            "facts": _bounded_unique(
                [*previous_evidence.get("facts", []), *enriched.facts], 60
            ),
            "search_terms": _bounded_unique(
                [
                    *previous_evidence.get("search_terms", []),
                    *enriched.search_terms,
                ],
                40,
            ),
            "retrieval_queries": _bounded_unique(
                [
                    *previous_evidence.get("retrieval_queries", []),
                    *enriched.retrieval_queries,
                ],
                24,
            ),
        },
    }
    base_body = enriched.body.strip() or (existing.body if existing else "")
    if not base_body:
        raise ValueError("Enrichment did not produce a body for a new concept")
    body = _body_with_source_evidence(base_body, source_evidence)
    frontmatter = {
        **old,
        "type": old.get("type", plan.type),
        "title": enriched.title or old.get("title") or plan.title,
        "description": enriched.description
        or old.get("description")
        or plan.description,
        "tags": tags,
        "search_terms": search_terms,
        "retrieval_queries": retrieval_queries,
        "aliases": aliases,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "okf_version": "0.1",
        "source_page_ids": page_ids,
        "source_urls": source_urls,
        "source_names": source_names,
        "source_evidence": source_evidence,
        "languages": languages,
    }
    return OKFDocument(frontmatter=frontmatter, body=body)


def _body_with_source_evidence(body: str, source_evidence: dict) -> str:
    base = re.split(
        rf"\n(?:{re.escape(SOURCE_EVIDENCE_HEADING)}|# Citations)\s*\n",
        body.strip(),
        maxsplit=1,
    )[0].rstrip()
    evidence_lines = [SOURCE_EVIDENCE_HEADING, ""]
    citation_lines = ["# Citations", ""]
    for page_id, evidence in source_evidence.items():
        title = evidence.get("title") or evidence.get("source") or page_id
        evidence_lines.extend([f"## {title}", ""])
        summary = str(evidence.get("summary", "")).strip()
        if summary:
            evidence_lines.extend([summary, ""])
        for fact in evidence.get("facts", []):
            evidence_lines.append(f"- {fact}")
        evidence_lines.append("")
        url = str(evidence.get("url", "")).strip()
        if url:
            citation_lines.append(f"- [{title}]({url})")
    return "\n\n".join(
        [base, "\n".join(evidence_lines).rstrip(), "\n".join(citation_lines)]
    )


def _retrieval_stats(bundle_root: Path) -> dict[str, int]:
    concepts = load_concepts(bundle_root).values()
    return {
        "concepts_with_search_terms": sum(
            bool(document.frontmatter.get("search_terms")) for document in concepts
        ),
        "concepts_with_retrieval_queries": sum(
            bool(document.frontmatter.get("retrieval_queries")) for document in concepts
        ),
        "source_evidence_pages": sum(
            len(document.frontmatter.get("source_evidence", {}))
            for document in concepts
        ),
        "atomic_facts": sum(
            len(evidence.get("facts", []))
            for document in concepts
            for evidence in document.frontmatter.get("source_evidence", {}).values()
        ),
    }


def _page_parts(markdown: str) -> list[PagePart]:
    max_page_chars = int(CONFIG["max_page_chars"])
    if len(markdown) <= max_page_chars:
        return (PagePart(1, 1, _first_heading(markdown), markdown.strip()),)

    target_chars = min(int(CONFIG["page_part_chars"]), max_page_chars)
    blocks = _markdown_blocks(markdown)
    raw_parts: list[tuple[str, str]] = []
    current: list[str] = []
    current_chars = 0
    heading_stack: list[tuple[int, str]] = []
    part_heading = ""

    def flush() -> None:
        nonlocal current_chars, part_heading
        if current:
            raw_parts.append((part_heading, "\n\n".join(current).strip()))
            current.clear()
            current_chars = 0
            part_heading = " > ".join(title for _level, title in heading_stack)

    for block in blocks:
        heading = MARKDOWN_HEADING_RE.match(block.splitlines()[0].strip())
        if heading:
            level = len(heading.group(1))
            heading_stack[:] = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, heading.group(2).strip()))
        for piece in _split_oversized_block(block, target_chars):
            addition = len(piece) + (2 if current else 0)
            if current and current_chars + addition > target_chars:
                flush()
            if not current:
                part_heading = " > ".join(title for _level, title in heading_stack)
            current.append(piece)
            current_chars += len(piece) + (2 if len(current) > 1 else 0)
    flush()

    total = len(raw_parts)
    return [
        PagePart(index, total, heading_path, text)
        for index, (heading_path, text) in enumerate(raw_parts, start=1)
    ]


def _markdown_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]


def _split_oversized_block(block: str, max_chars: int) -> list[str]:
    pieces = []
    remaining = block.strip()
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = remaining.rfind(". ", 0, max_chars)
            if split_at >= max_chars // 2:
                split_at += 1
        if split_at < max_chars // 2:
            split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        heading = MARKDOWN_HEADING_RE.match(line.strip())
        if heading:
            return heading.group(2).strip()
    return ""


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _load_model(path: Path, model_type):
    if not path.exists():
        return model_type()
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> GenerationCheckpoint:
    if not path.exists():
        return GenerationCheckpoint()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "completed_pages" in raw and "enriched_pages" not in raw:
        raw["enriched_pages"] = raw.pop("completed_pages")
    return GenerationCheckpoint.model_validate(raw)


def _load_refresh_checkpoint(
    path: Path,
    generation_path: Path,
    bundle_root: Path,
) -> GenerationCheckpoint:
    if path.exists():
        return _load_checkpoint(path)

    generation = _load_checkpoint(generation_path)
    rich_page_ids = {
        str(page_id)
        for document in load_concepts(bundle_root).values()
        for page_id, evidence in document.frontmatter.get("source_evidence", {}).items()
        if evidence.get("facts") or evidence.get("retrieval_queries")
    }
    migrated = GenerationCheckpoint(
        enriched_pages={
            page_id: concept_id
            for page_id, concept_id in generation.enriched_pages.items()
            if page_id in rich_page_ids
        }
    )
    _write_model(path, migrated)
    return migrated


def _write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_error(exc: Exception) -> str:
    text = str(exc)
    api_key = os.environ.get("AZURE_AI_API_KEY")
    return text.replace(api_key, "<redacted>") if api_key else text


def _bounded_unique(values: list[str], limit: int) -> list[str]:
    return _unique(values)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--limit", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true")
    mode.add_argument(
        "--refresh-retrieval",
        action="store_true",
        help=(
            "Re-enrich assigned pages with current retrieval metadata while "
            "reusing discovery inventory and canonical assignments."
        ),
    )
    args = parser.parse_args()
    manifest = generate_bundle(
        source=args.source,
        limit=args.limit,
        force=args.force,
        refresh_retrieval=args.refresh_retrieval,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
