from __future__ import annotations

from pathlib import Path

from src.okf.answer import OKFAnswer, answer_question
from src.okf.document import OKFDocument
from src.retrieval.base import RankedChunk, retrieve_batch_default
from src.shared.env import ROOT


class OKFConceptRetriever:
    """Expose cited and visited OKF concepts as ranked evaluation results."""

    name = "okf"

    def __init__(
        self,
        bundle_root: Path | None = None,
    ) -> None:
        self.bundle_root = (bundle_root or ROOT / "data/okf/tourism").resolve()
        self.page_map = load_bundle_page_map(self.bundle_root)
        self._total_queries = 0
        self._question_count = 0
        self.failures = 0
        self.action_log: list[dict[str, object]] = []

    @property
    def covered_page_ids(self) -> set[str]:
        return {page_id for page_ids in self.page_map.values() for page_id in page_ids}

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        self._question_count += 1
        try:
            result = answer_question(
                query,
                bundle_root=self.bundle_root,
            )
        except Exception as exc:
            self.failures += 1
            self._total_queries += 1
            self.action_log.append(
                {
                    "question": query,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return []

        self._total_queries += result.query_count
        concepts = _rank_visited_concepts(result, self.page_map)
        self.action_log.append(
            {
                "question": query,
                "visited": result.visited,
                "citations": result.citations,
                "sufficient": result.sufficient,
                "reason": result.reason,
                "query_count": result.query_count,
                "ranked_concepts": concepts[:limit],
                "trace": result.trace,
            }
        )
        return [
            RankedChunk(id=concept, score=1.0 / rank, text="")
            for rank, concept in enumerate(concepts[:limit], start=1)
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


def load_bundle_page_map(bundle_root: Path) -> dict[str, list[str]]:
    if not (bundle_root / "index.md").exists():
        raise RuntimeError(
            f"OKF bundle root index is missing: {bundle_root / 'index.md'}"
        )

    page_map = {}
    for path in sorted(bundle_root.rglob("*.md")):
        if path.name in {"index.md", "log.md"}:
            continue
        document = OKFDocument.parse(path.read_text(encoding="utf-8"))
        relative_path = path.relative_to(bundle_root).as_posix()
        page_map[relative_path] = [
            str(page_id) for page_id in document.frontmatter.get("source_page_ids", [])
        ]
    return page_map


def invert_page_map(concept_pages: dict[str, list[str]]) -> dict[str, list[str]]:
    page_concepts: dict[str, list[str]] = {}
    for concept, page_ids in concept_pages.items():
        for page_id in page_ids:
            page_concepts.setdefault(page_id, []).append(concept)
    return page_concepts


def project_pages_to_concepts(
    page_ids: list[str], page_concepts: dict[str, list[str]]
) -> list[str]:
    return _unique(
        concept for page_id in page_ids for concept in page_concepts.get(page_id, [])
    )


def _rank_visited_concepts(
    result: OKFAnswer,
    page_map: dict[str, list[str]],
) -> list[str]:
    visited = [path for path in result.visited if path in page_map]
    cited = [path for path in result.citations if path in page_map]
    return _unique([*cited, *visited])


def _unique(values) -> list[str]:
    return list(dict.fromkeys(values))
