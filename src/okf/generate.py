from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
from src.shared.llm import LocalOllamaStructuredLlm, StructuredLlm

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
CITATION_LINK_RE = re.compile(r"^- \[([^]]+)\]\((https?://[^)]+)\)$", re.MULTILINE)


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

    @field_validator("concept_id", mode="before")
    @classmethod
    def valid_concept_id(cls, value: str) -> str:
        value = _safe_concept_id(value)
        parse_concept_id(value)
        return value


class PageProposal(ConceptProposal):
    page_id: str


class CanonicalConcept(ConceptProposal):
    pass


class ResolutionDecision(BaseModel):
    page_id: str
    canonical_concept_id: str

    @field_validator("canonical_concept_id", mode="before")
    @classmethod
    def valid_concept_id(cls, value: str) -> str:
        value = _safe_concept_id(value)
        parse_concept_id(value)
        return value


class ResolutionBatch(BaseModel):
    concepts: list[CanonicalConcept]
    decisions: list[ResolutionDecision]


class EnrichedConcept(BaseModel):
    title: str
    description: str = Field(description="One factual sentence")
    body: str = Field(
        default="", description="Factual Markdown body without YAML frontmatter"
    )


class GenerationCheckpoint(BaseModel):
    enriched_pages: dict[str, str] = Field(default_factory=dict)
    enriched_parts: dict[str, int] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)
    fallback_pages: dict[str, str] = Field(default_factory=dict)


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
    clean: bool = False,
) -> dict:
    if clean and (source is not None or limit is not None):
        raise ValueError("Clean generation must rebuild the full corpus")
    load_local_env()
    client = _llm_client()
    bundle_root = ROOT / CONFIG["bundle_path"]
    checkpoint_path = ROOT / CONFIG["checkpoint_path"]
    inventory_path = ROOT / CONFIG["inventory_path"]
    catalog_path = ROOT / CONFIG["catalog_path"]
    if clean:
        _reset_generation(bundle_root, checkpoint_path.parent)
    if clean:
        checkpoint = GenerationCheckpoint()
    else:
        checkpoint = _load_checkpoint(checkpoint_path)
    inventory = (
        DiscoveryInventory()
        if clean
        else _load_model(inventory_path, DiscoveryInventory)
    )
    catalog = (
        CanonicalCatalog() if clean else _load_model(catalog_path, CanonicalCatalog)
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
        overwrite=clean,
    )

    regenerate_indexes(bundle_root)
    report = validate_bundle(bundle_root)
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
        "generation_mode": "clean" if clean else "resume",
        "selected_pages": len(pages),
        "discovered_pages": discovered,
        "resolved_pages": resolved,
        "processed_pages": processed,
        "resumed_pages": skipped,
        "covered_pages": len(checkpoint.enriched_pages),
        "fallback_pages": {
            page_id: error
            for page_id, error in checkpoint.fallback_pages.items()
            if page_id in selected_ids
        },
        "failures": selected_failures,
        "concepts": len(set(selected_assignments.values())),
        "catalog_concepts": len(catalog.concepts),
        "model": CONFIG["model"],
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
    client: StructuredLlm,
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
        )


def _resolve_inventory(
    client: StructuredLlm,
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
    client: StructuredLlm,
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
    except (RuntimeError, ValueError):
        if len(proposals) == 1:
            raise
        midpoint = len(proposals) // 2
        print(
            f"  canonicalization response invalid; retrying as {midpoint} and "
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
        names[(_normalize(concept.title), _normalize(concept.type))] = concept_id
    for proposal in proposals:
        proposal_type = _normalize(proposal.type)
        matching = {
            names[(normalized, proposal_type)]
            for name in [proposal.title]
            if (normalized := _normalize(name), proposal_type) in names
        }
        if len(matching) == 1:
            concept_id = matching.pop()
            catalog.assignments[proposal.page_id] = concept_id


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


def _enrich_catalog(
    client: StructuredLlm,
    pages: list[SourcePage],
    catalog: CanonicalCatalog,
    checkpoint: GenerationCheckpoint,
    checkpoint_path: Path,
    bundle_root: Path,
    *,
    overwrite: bool,
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
            if page.id in checkpoint.enriched_pages and not overwrite:
                skipped += 1
                continue
            print(
                f"[enrich {concept_index}/{len(pages_by_concept)} page {page_index}/{len(concept_pages)}] {concept_id}",
                flush=True,
            )
            previous_failure = checkpoint.failures.get(page.id)
            try:
                parts = _page_parts(page.markdown)
                completed_part = checkpoint.enriched_parts.get(page.id, 0)
                for part in parts:
                    if part.index <= completed_part and not overwrite:
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
                checkpoint.fallback_pages.pop(page.id, None)
                processed += 1
            except Exception as exc:
                error = _safe_error(exc)
                if previous_failure:
                    existing = _source_fallback_document(page, plan, existing)
                    write_concept(bundle_root, concept_id, existing)
                    checkpoint.enriched_pages[page.id] = concept_id
                    checkpoint.enriched_parts.pop(page.id, None)
                    checkpoint.failures.pop(page.id, None)
                    checkpoint.fallback_pages[page.id] = error
                    processed += 1
                    print(f"  source fallback: {error}", flush=True)
                else:
                    checkpoint.failures[page.id] = error
                    print(f"  failed: {error}", flush=True)
            _write_model(checkpoint_path, checkpoint)
    return processed, skipped


def _llm_client() -> LocalOllamaStructuredLlm:
    return LocalOllamaStructuredLlm(
        model_id=CONFIG["model"],
        reasoning=CONFIG["model_reasoning"],
        num_ctx=CONFIG["model_num_ctx"],
        num_predict=CONFIG["model_num_predict"],
        method=CONFIG["model_structured_method"],
    )


def _discovery_prompt(page: SourcePage, part: PagePart) -> str:
    return f"""You organize complete tourism and cultural heritage pages into Open Knowledge Format concepts.
Propose the one primary, independently meaningful concept represented by the source page. You are reading part {part.index} of {part.total}; identify the page's main entity rather than treating this part or its heading as a separate concept. Do not mint concepts for headings, navigation labels, opening hours, contact details, incidental mentions, introductions, or generic "more information" sections; keep those inside their parent entity.
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
The proposals describe different parts of the same page, not independent pages. Select the page's primary named entity or reusable topic. Use a compact existing top-level taxonomy and a discriminative one-sentence description. Do not invent facts.

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
            "description": concept.description,
        }
        for concept in catalog.concepts.values()
    ]
    proposed = [proposal.model_dump() for proposal in proposals]
    return f"""Resolve proposed tourism concepts into a canonical Open Knowledge Format catalog.
This is the global anti-duplication pass. Assign every proposed page exactly once.

Rules:
1. Reuse a known canonical concept when type and entity identity match, including translations, historical names, abbreviations, and minor spelling differences evident in the title and description.
2. Merge equivalent proposals in this batch into one canonical concept.
3. Create a new concept only for an independently meaningful named entity or reusable knowledge topic. Do not mint concepts for headings, opening hours, contact details, incidental mentions, introductions, or generic navigation pages.
4. Preserve distinct entities even when their names are similar.
5. Canonical ids must be stable lowercase `category/slug` paths. Prefer locally canonical names.
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
) -> str:
    existing_text = existing.serialize() if existing else "(none)"
    return f"""Create or augment one Open Knowledge Format concept from a complete source page.
Return Markdown body only in `body`; never include YAML frontmatter there.
Preserve all supported facts and headings from the existing document. Add every concrete fact from the source, including details that may appear minor, retain multilingual names, and avoid duplication.
Use structural Markdown. Include a final `# Citations` section containing the exact source URL. Do not cite URLs absent from the input or existing document. Do not invent relationships or facts. Do not add internal concept links; directory indexes provide navigation.

Concept plan:
{plan.model_dump_json(indent=2)}

Existing concept:
{existing_text}

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
    base_body = enriched.body.strip() or (existing.body if existing else "")
    if not base_body:
        raise ValueError("Enrichment did not produce a body for a new concept")
    body = _body_with_citations(base_body, existing, page)
    frontmatter = {
        "type": old.get("type", plan.type),
        "title": enriched.title or old.get("title") or plan.title,
        "description": enriched.description
        or old.get("description")
        or plan.description,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_page_ids": page_ids,
    }
    return OKFDocument(frontmatter=frontmatter, body=body)


def _source_fallback_document(
    page: SourcePage,
    plan: CanonicalConcept,
    existing: OKFDocument | None,
) -> OKFDocument:
    old = existing.frontmatter if existing else {}
    old_page_ids = old.get("source_page_ids", [])
    page_ids = _unique([*old_page_ids, page.id])
    source_body = _body_with_citations(page.markdown, None, page)
    if existing and page.id not in old_page_ids:
        existing_base = re.split(
            r"\n# Citations\s*\n", existing.body.strip(), maxsplit=1
        )[0].rstrip()
        source_body = _body_with_citations(
            f"{existing_base}\n\n# Source: {page.title or page.source}\n\n{page.markdown}",
            existing,
            page,
        )
    return OKFDocument(
        frontmatter={
            "type": old.get("type", plan.type),
            "title": old.get("title", plan.title),
            "description": old.get("description", plan.description),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_page_ids": page_ids,
        },
        body=source_body,
    )


def _body_with_citations(
    body: str, existing: OKFDocument | None, page: SourcePage
) -> str:
    base = re.split(
        r"\n# Citations\s*\n",
        body.strip(),
        maxsplit=1,
    )[0].rstrip()
    citation_sources = [body, existing.body if existing else ""]
    citations = [
        (label, url)
        for source in citation_sources
        for label, url in CITATION_LINK_RE.findall(source)
    ]
    citations.append((page.title or page.source or page.id, page.url))
    unique_citations = list({url: label for label, url in reversed(citations)}.items())
    lines = ["# Citations", ""]
    lines.extend(f"- [{label}]({url})" for url, label in unique_citations if url)
    return f"{base}\n\n" + "\n".join(lines)


def _page_parts(markdown: str) -> list[PagePart]:
    max_page_chars = min(int(CONFIG["max_page_chars"]), int(CONFIG["page_part_chars"]))
    if len(markdown) <= max_page_chars:
        return (PagePart(1, 1, _first_heading(markdown), markdown.strip()),)

    target_chars = max_page_chars
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


def _safe_concept_id(value: str) -> str:
    if not isinstance(value, str) or value.count("/") != 1:
        return value
    category, slug = value.casefold().split("/", maxsplit=1)
    category = re.sub(r"[^a-z0-9_-]+", "-", category).strip("-_")
    slug = re.sub(r"[^a-z0-9_-]+", "-", slug).strip("-_")
    return f"{category}/{slug}"


def _load_model(path: Path, model_type):
    if not path.exists():
        return model_type()
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def _load_checkpoint(path: Path) -> GenerationCheckpoint:
    if not path.exists():
        return GenerationCheckpoint()
    return GenerationCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def _reset_generation(bundle_root: Path, state_root: Path) -> None:
    shutil.rmtree(bundle_root, ignore_errors=True)
    shutil.rmtree(state_root, ignore_errors=True)


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
    return str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the bundle and all OKF generation state before rebuilding.",
    )
    args = parser.parse_args()
    manifest = generate_bundle(
        source=args.source,
        limit=args.limit,
        clean=args.clean,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
