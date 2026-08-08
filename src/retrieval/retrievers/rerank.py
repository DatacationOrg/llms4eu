from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import httpx

from src.retrieval.base import RankedChunk, Retriever, retrieve_batch_default

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder import CrossEncoder


@dataclass(frozen=True)
class CrossEncoderRerankRetriever:
    """Wrap a first-stage retriever with a cross-encoder reranker."""

    name: str
    model_name: str
    base_retriever: Retriever
    candidate_limit: int
    device: str
    max_length: int
    local_files_only: bool
    batch_size: int
    prompt_name: str | None = None
    prompt: str | None = None

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        candidates = self.base_retriever.retrieve(query, self.candidate_limit)
        ranked = _rerank(
            query,
            candidates,
            self.model_name,
            self.device,
            self.max_length,
            self.local_files_only,
            self.batch_size,
            self.prompt_name,
            self.prompt,
        )
        return [
            RankedChunk(id=chunk.id, score=float(score), text=chunk.text)
            for chunk, score in ranked[:limit]
        ]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        candidate_rankings = self.base_retriever.retrieve_batch(
            queries,
            self.candidate_limit,
        )
        model = _cross_encoder(
            self.model_name,
            self.device,
            self.max_length,
            self.local_files_only,
            self.prompt_name,
            self.prompt,
        )
        pairs = []
        refs = []
        for query_index, candidates in candidate_rankings.items():
            for chunk in candidates:
                pairs.append((queries[query_index], chunk.text))
                refs.append((query_index, chunk))

        scores = model.predict(pairs, batch_size=self.batch_size)
        by_query: dict[int, list[tuple[RankedChunk, float]]] = {
            index: [] for index in range(len(queries))
        }
        for (query_index, chunk), score in zip(refs, scores, strict=True):
            by_query[query_index].append((chunk, float(score)))

        return {
            query_index: [
                RankedChunk(id=chunk.id, score=score, text=chunk.text)
                for chunk, score in sorted(
                    ranked,
                    key=lambda item: item[1],
                    reverse=True,
                )[:limit]
            ]
            for query_index, ranked in by_query.items()
        }


@dataclass(frozen=True)
class AzureCohereRerankRetriever:
    """Rerank first-stage candidates with an Azure-hosted Cohere endpoint."""

    name: str
    model_name: str
    endpoint: str
    api_key: str
    base_retriever: Retriever
    candidate_limit: int
    timeout_seconds: float = 90
    retries: int = 3

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        if limit <= 0:
            return []
        candidates = self.base_retriever.retrieve(query, self.candidate_limit)
        if not candidates:
            return []
        results = _cohere_rerank(
            endpoint=self.endpoint,
            api_key=self.api_key,
            model_name=self.model_name,
            query=query,
            documents=[chunk.text for chunk in candidates],
            top_n=min(limit, len(candidates)),
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
        )
        return [
            RankedChunk(
                id=candidates[index].id,
                score=score,
                text=candidates[index].text,
            )
            for index, score in results
        ]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return retrieve_batch_default(self, queries, limit)


def _cohere_rerank(
    *,
    endpoint: str,
    api_key: str,
    model_name: str,
    query: str,
    documents: list[str],
    top_n: int,
    timeout_seconds: float,
    retries: int,
) -> list[tuple[int, float]]:
    payload = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            response = _cohere_client(api_key, timeout_seconds).post(
                endpoint,
                json=payload,
            )
            _raise_for_status_with_body(response)
            return _parse_cohere_results(response.json(), len(documents), top_n)
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code != 429 and exc.response.status_code < 500:
                raise
        except (httpx.TransportError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
        if attempt + 1 == max(1, retries):
            break
    detail = f"{type(last_error).__name__}: {last_error}"
    raise RuntimeError(
        f"Azure Cohere rerank call failed after {max(1, retries)} attempts: {detail}"
    ) from last_error


def _parse_cohere_results(
    body: Any,
    document_count: int,
    top_n: int,
) -> list[tuple[int, float]]:
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise ValueError("Cohere rerank response must contain a results list")
    results = body["results"]
    if len(results) != top_n:
        raise ValueError(
            f"Cohere rerank returned {len(results)} results; expected {top_n}"
        )

    parsed = []
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Cohere rerank result must be an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Cohere rerank result index must be an integer")
        if index < 0 or index >= document_count:
            raise ValueError(f"Cohere rerank result index is out of range: {index}")
        if index in seen:
            raise ValueError(f"Cohere rerank returned duplicate index: {index}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Cohere rerank relevance score must be numeric")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Cohere rerank relevance score must be finite")
        seen.add(index)
        parsed.append((index, score))
    return parsed


def _raise_for_status_with_body(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()[:2000]
        raise httpx.HTTPStatusError(
            f"{exc}; response body: {detail or '(empty)'}",
            request=exc.request,
            response=exc.response,
        ) from exc


@cache
def _cohere_client(api_key: str, timeout_seconds: float) -> httpx.Client:
    return httpx.Client(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=timeout_seconds,
    )


def _rerank(
    query: str,
    candidates: list[RankedChunk],
    model_name: str,
    device: str,
    max_length: int,
    local_files_only: bool,
    batch_size: int,
    prompt_name: str | None,
    prompt: str | None,
) -> list[tuple[RankedChunk, float]]:
    scores = _cross_encoder(
        model_name,
        device,
        max_length,
        local_files_only,
        prompt_name,
        prompt,
    ).predict(
        [(query, chunk.text) for chunk in candidates],
        batch_size=batch_size,
    )
    return sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )


@cache
def _cross_encoder(
    model_name: str,
    device: str,
    max_length: int,
    local_files_only: bool,
    prompt_name: str | None,
    prompt: str | None,
) -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    prompt_kwargs = _prompt_kwargs(prompt_name, prompt)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            local_files_only=local_files_only,
            **prompt_kwargs,
        )


def _prompt_kwargs(prompt_name: str | None, prompt: str | None) -> dict:
    if not prompt_name or not prompt:
        return {}
    return {
        "prompts": {prompt_name: prompt},
        "default_prompt_name": prompt_name,
    }
