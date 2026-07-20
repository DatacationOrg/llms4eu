from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from src.okf.export import DEFAULT_BUNDLE
from src.retrieval.base import RankedChunk, Retriever, retrieve_batch_default

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class OkfSparseRetriever:
    """BM25 retriever over OKF concept Markdown files."""

    name: str = "okf_sparse"
    bundle: Path = DEFAULT_BUNDLE
    k1: float = 1.5
    b: float = 0.75

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        corpus = _corpus(self.bundle)
        query_terms = _tokens(query)
        scores = []
        for concept in corpus:
            score = 0.0
            for term in query_terms:
                term_frequency = concept.term_counts.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * concept.length / corpus.avgdl
                )
                score += corpus.idf.get(term, 0.0) * (
                    term_frequency * (self.k1 + 1) / denominator
                )
            if score:
                scores.append(RankedChunk(concept.id, score, concept.context_text))
        return sorted(scores, key=lambda item: item.score, reverse=True)[:limit]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return retrieve_batch_default(self, queries, limit)


@dataclass(frozen=True)
class OkfAgenticRetriever:
    """Agent-style OKF retriever that expands top concepts through links/backlinks."""

    name: str = "okf_agentic"
    base_retriever: Retriever | None = None
    bundle: Path = DEFAULT_BUNDLE
    seed_limit: int = 5
    neighbor_limit: int = 8

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        base = self.base_retriever or OkfSparseRetriever(bundle=self.bundle)
        seeds = base.retrieve(query, min(max(limit, self.seed_limit), self.seed_limit))
        if not seeds:
            return []

        corpus = _corpus(self.bundle)
        concepts_by_id = {concept.id: concept for concept in corpus.concepts}
        selected: dict[str, RankedChunk] = {chunk.id: chunk for chunk in seeds}

        for seed in seeds:
            concept = concepts_by_id.get(seed.id)
            if concept is None:
                continue
            candidate_ids = [*concept.links, *corpus.backlinks.get(concept.id, [])]
            for offset, concept_id in enumerate(candidate_ids[: self.neighbor_limit], start=1):
                if concept_id in selected:
                    continue
                neighbor = concepts_by_id.get(concept_id)
                if neighbor is None:
                    continue
                selected[concept_id] = RankedChunk(
                    id=neighbor.id,
                    score=seed.score * (0.85 / offset),
                    text=neighbor.context_text,
                )

        return sorted(selected.values(), key=lambda item: item.score, reverse=True)[:limit]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return retrieve_batch_default(self, queries, limit)


@dataclass(frozen=True)
class OkfConcept:
    id: str
    path: Path
    frontmatter: dict[str, Any]
    body: str
    context_text: str
    links: list[str]
    term_counts: Counter[str]
    length: int


@dataclass(frozen=True)
class OkfCorpus:
    concepts: list[OkfConcept]
    idf: dict[str, float]
    avgdl: float
    backlinks: dict[str, list[str]]

    def __iter__(self):
        return iter(self.concepts)


@cache
def _corpus(bundle: Path) -> OkfCorpus:
    bundle = Path(bundle).resolve()
    if not bundle.exists():
        return OkfCorpus([], {}, 1.0, {})

    concepts: list[OkfConcept] = []
    document_frequency: Counter[str] = Counter()
    raw_docs: list[tuple[str, Path, dict[str, Any], str]] = []
    for path in sorted(bundle.rglob("*.md")):
        concept_id = path.relative_to(bundle).with_suffix("").as_posix()
        frontmatter, body = _parse_doc(path.read_text(encoding="utf-8"))
        if frontmatter.get("type") == "Index" or path.name == "index.md":
            continue
        raw_docs.append((concept_id, path, frontmatter, body))

    known_ids = {concept_id for concept_id, *_ in raw_docs}
    backlinks: dict[str, list[str]] = {concept_id: [] for concept_id in known_ids}
    pending: list[OkfConcept] = []
    for concept_id, path, frontmatter, body in raw_docs:
        context_text = _context_text(concept_id, frontmatter, body)
        terms = _tokens(context_text)
        term_counts = Counter(terms)
        links = [link for link in _extract_links(body, concept_id) if link in known_ids]
        for link in links:
            backlinks.setdefault(link, []).append(concept_id)
        concept = OkfConcept(
            id=concept_id,
            path=path,
            frontmatter=frontmatter,
            body=body,
            context_text=context_text,
            links=links,
            term_counts=term_counts,
            length=len(terms) or 1,
        )
        pending.append(concept)
        document_frequency.update(term_counts)

    concepts = pending
    document_count = len(concepts)
    if not document_count:
        return OkfCorpus([], {}, 1.0, {})
    idf = {
        term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
        for term, count in document_frequency.items()
    }
    avgdl = sum(concept.length for concept in concepts) / document_count
    return OkfCorpus(concepts, idf, avgdl, backlinks)


def _parse_doc(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        frontmatter = {}
    body = text[match.end() :]
    return frontmatter if isinstance(frontmatter, dict) else {}, body


def _context_text(concept_id: str, frontmatter: dict[str, Any], body: str) -> str:
    title = frontmatter.get("title") or concept_id
    description = frontmatter.get("description") or ""
    typ = frontmatter.get("type") or "Concept"
    resource = frontmatter.get("resource") or ""
    tags = ", ".join(str(tag) for tag in frontmatter.get("tags") or [])
    return (
        f"id: {concept_id}\n"
        f"type: {typ}\n"
        f"title: {title}\n"
        f"description: {description}\n"
        f"resource: {resource}\n"
        f"tags: {tags}\n\n"
        f"{body.strip()}"
    ).strip()


def _extract_links(body: str, current_id: str) -> list[str]:
    links = []
    current_dir = Path(current_id).parent
    for raw_target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
        target = raw_target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if target.startswith("/"):
            normalized = target.lstrip("/")
        else:
            normalized = (current_dir / target).as_posix()
        if normalized.endswith(".md"):
            normalized = normalized[:-3]
        normalized = str(Path(normalized).as_posix()).strip("/")
        if normalized and normalized not in links:
            links.append(normalized)
    return links


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]
