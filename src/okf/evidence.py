from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.okf.answer import OKFAnswer, answer_question
from src.okf.document import OKFDocument
from src.retrieval.base import RankedChunk, retrieve_batch_default
from src.shared.env import ROOT

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class ConceptMetadata:
    path: str
    search_text: str
    page_ids: list[str]
    retrieval_ready_page_ids: list[str]
    page_search_text: dict[str, str]


class OKFPageEvidenceRetriever:
    """Adapt OKF navigation to page-ranked evidence for comparative evaluation."""

    name = "okf"

    def __init__(
        self,
        bundle_root: Path | None = None,
        *,
        metadata_search: bool = False,
        candidate_limit: int = 10,
        retrieval_ready_only: bool = False,
    ) -> None:
        self.bundle_root = (bundle_root or ROOT / "data/okf/tourism").resolve()
        self.catalog = load_bundle_catalog(self.bundle_root)
        self.page_map = {
            path: (
                metadata.retrieval_ready_page_ids
                if retrieval_ready_only
                else metadata.page_ids
            )
            for path, metadata in self.catalog.items()
        }
        self.allowed_page_ids = {
            page_id for page_ids in self.page_map.values() for page_id in page_ids
        }
        self.metadata_search = metadata_search
        self.candidate_limit = candidate_limit
        self._total_queries = 0
        self._question_count = 0
        self.failures = 0
        self.action_log: list[dict[str, object]] = []

    @property
    def covered_page_ids(self) -> set[str]:
        return {page_id for page_ids in self.page_map.values() for page_id in page_ids}

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        self._question_count += 1
        candidates = (
            _rank_concepts(query, self.catalog)[: self.candidate_limit]
            if self.metadata_search
            else []
        )
        try:
            result = answer_question(
                query,
                bundle_root=self.bundle_root,
                candidate_paths=candidates,
            )
        except Exception as exc:
            self.failures += 1
            self._total_queries += 1
            self.action_log.append(
                {
                    "question": query,
                    "candidates": candidates,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return []

        self._total_queries += result.query_count
        page_ids = [
            page_id
            for page_id in _rank_visited_pages(query, result, self.catalog)
            if page_id in self.allowed_page_ids
        ]
        self.action_log.append(
            {
                "question": query,
                "candidates": candidates,
                "visited": result.visited,
                "citations": result.citations,
                "sufficient": result.sufficient,
                "reason": result.reason,
                "query_count": result.query_count,
                "ranked_page_ids": page_ids[:limit],
                "trace": result.trace,
            }
        )
        return [
            RankedChunk(id=page_id, score=1.0 / rank, text="")
            for rank, page_id in enumerate(page_ids[:limit], start=1)
        ]

    def retrieve_batch(
        self, queries: list[str], limit: int
    ) -> dict[int, list[RankedChunk]]:
        return retrieve_batch_default(self, queries, limit)

    def total_queries(self) -> float:
        return float(self._total_queries)

    def average_queries_per_question(self) -> float:
        if not self._question_count:
            return 0.0
        return self._total_queries / self._question_count


class OKFSearchPageEvidenceRetriever(OKFPageEvidenceRetriever):
    """Metadata-assisted OKF retrieval that still reads complete concepts."""

    name = "okf_search"

    def __init__(
        self,
        bundle_root: Path | None = None,
        *,
        retrieval_ready_only: bool = False,
    ) -> None:
        super().__init__(
            bundle_root,
            metadata_search=True,
            retrieval_ready_only=retrieval_ready_only,
        )


def load_bundle_page_map(bundle_root: Path) -> dict[str, list[str]]:
    return {
        path: metadata.page_ids
        for path, metadata in load_bundle_catalog(bundle_root).items()
    }


def load_bundle_catalog(bundle_root: Path) -> dict[str, ConceptMetadata]:
    if not (bundle_root / "index.md").exists():
        raise RuntimeError(
            f"OKF bundle root index is missing: {bundle_root / 'index.md'}"
        )

    catalog = {}
    for path in sorted(bundle_root.rglob("*.md")):
        if path.name in {"index.md", "log.md"}:
            continue
        document = OKFDocument.parse(path.read_text(encoding="utf-8"))
        relative_path = path.relative_to(bundle_root).as_posix()
        page_ids = [
            str(page_id) for page_id in document.frontmatter.get("source_page_ids", [])
        ]
        search_values = [
            relative_path,
            document.frontmatter.get("title", ""),
            document.frontmatter.get("description", ""),
            *document.frontmatter.get("aliases", []),
            *document.frontmatter.get("tags", []),
            *document.frontmatter.get("search_terms", []),
            *document.frontmatter.get("retrieval_queries", []),
            *document.frontmatter.get("languages", []),
        ]
        source_urls = document.frontmatter.get("source_urls", [])
        source_names = document.frontmatter.get("source_names", [])
        source_evidence = document.frontmatter.get("source_evidence", {})
        retrieval_ready_page_ids = [
            page_id
            for page_id in page_ids
            if source_evidence.get(page_id, {}).get("facts")
            or source_evidence.get(page_id, {}).get("retrieval_queries")
        ]
        for details in source_evidence.values():
            search_values.extend(
                [
                    details.get("summary", ""),
                    *details.get("facts", []),
                    *details.get("search_terms", []),
                    *details.get("retrieval_queries", []),
                ]
            )
        page_search_text = {}
        for index, page_id in enumerate(page_ids):
            details = source_evidence.get(page_id, {})
            page_search_text[page_id] = " ".join(
                str(value)
                for value in [
                    details.get("title", ""),
                    details.get("source", ""),
                    details.get("url", ""),
                    details.get("language", ""),
                    details.get("summary", ""),
                    *details.get("facts", []),
                    *details.get("search_terms", []),
                    *details.get("retrieval_queries", []),
                    source_names[index] if index < len(source_names) else "",
                    source_urls[index] if index < len(source_urls) else "",
                ]
                if value
            )
        catalog[relative_path] = ConceptMetadata(
            path=relative_path,
            search_text=" ".join(str(value) for value in search_values if value),
            page_ids=page_ids,
            retrieval_ready_page_ids=retrieval_ready_page_ids,
            page_search_text=page_search_text,
        )
    return catalog


def _rank_concepts(query: str, catalog: dict[str, ConceptMetadata]) -> list[str]:
    return [
        path
        for path, _score in _bm25_rank(
            query,
            {path: metadata.search_text for path, metadata in catalog.items()},
        )
    ]


def _rank_visited_pages(
    query: str,
    result: OKFAnswer,
    catalog: dict[str, ConceptMetadata],
) -> list[str]:
    visited = [path for path in result.visited if path in catalog]
    concept_scores = dict(
        _bm25_rank(query, {path: catalog[path].search_text for path in visited})
    )
    ordered_concepts = sorted(
        visited,
        key=lambda path: (path in result.citations, concept_scores.get(path, 0.0)),
        reverse=True,
    )
    page_ids = []
    for path in ordered_concepts:
        metadata = catalog[path]
        page_scores = dict(_bm25_rank(query, metadata.page_search_text))
        page_ids.extend(
            sorted(
                metadata.page_ids,
                key=lambda page_id: page_scores.get(page_id, 0.0),
                reverse=True,
            )
        )
    return _unique(page_ids)


def _bm25_rank(query: str, documents: dict[str, str]) -> list[tuple[str, float]]:
    query_terms = _tokens(query)
    if not documents:
        return []
    term_counts = {key: Counter(_tokens(text)) for key, text in documents.items()}
    lengths = {key: sum(counts.values()) or 1 for key, counts in term_counts.items()}
    avg_length = sum(lengths.values()) / len(lengths)
    document_frequency = Counter(
        term for counts in term_counts.values() for term in counts
    )
    scores = {}
    for key, counts in term_counts.items():
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            count = document_frequency[term]
            idf = math.log(1 + (len(documents) - count + 0.5) / (count + 0.5))
            denominator = frequency + 1.5 * (0.25 + 0.75 * lengths[key] / avg_length)
            score += idf * frequency * 2.5 / denominator
        scores[key] = score
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))
