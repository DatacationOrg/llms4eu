from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from _cli import run_cli
from src.db.pages import connect_pages, initialize_page_artifacts_db
from src.eval.equivalence import (
    EvidenceDocument,
    EvidenceEquivalenceJudge,
)
from src.shared.env import ROOT, load_local_env
from src.shared.llm import AzureFoundryStructuredLlm, LocalOllamaStructuredLlm

DEFAULT_CHECKPOINT = Path("docs/retrieval-results-chunks-okf.md.checkpoint.json")
DEFAULT_OUTPUT = Path("docs/retrieval-equivalence-judge.md")
DEFAULT_LOCAL_MODEL = "gemma4:26b"


@dataclass
class IncrementalEquivalenceAudit:
    state: dict[str, Any]
    judge: EvidenceEquivalenceJudge
    model_id: str
    cutoff: int
    cache_path: Path
    questions: dict[str, dict[str, str]] = field(init=False)
    relevant_ids: dict[str, list[str]] = field(init=False)
    chunks: dict[str, str] = field(init=False)
    cache: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        if self.cutoff < 1:
            raise ValueError("cutoff must be at least 1")
        signature = self.state.get("signature", {})
        timed_ids = {str(value) for value in signature.get("timed_ids", [])}
        if not timed_ids:
            raise ValueError("Checkpoint has no timed questions")

        initialize_page_artifacts_db()
        (
            self.questions,
            self.relevant_ids,
            self.chunks,
        ) = _load_evidence(timed_ids)
        self.cache = _load_cache(
            self.cache_path,
            model_id=self.model_id,
            cutoff=self.cutoff,
        )

    def backfill(self) -> None:
        """Judge checkpointed predictions that predate incremental auditing."""
        timed_ids = [
            str(value) for value in self.state.get("signature", {}).get("timed_ids", [])
        ]
        for method_name, method_state in self.state.get("methods", {}).items():
            rankings = method_state.get("rankings", {})
            for question_id in timed_ids:
                ranked = rankings.get(question_id)
                if ranked is not None:
                    self.judge_prediction(method_name, question_id, ranked)

    def judge_prediction(
        self,
        method_name: str,
        question_id: str,
        ranked: list[str],
    ) -> None:
        question = self.questions.get(str(question_id))
        gold_ids = self.relevant_ids.get(str(question_id), [])
        if question is None or not gold_ids:
            return
        if _strict_hit(
            ranked=ranked,
            gold_ids=gold_ids,
            cutoff=self.cutoff,
        ):
            return

        retrieved = _retrieved_documents(
            ranked=ranked[: self.cutoff],
            chunks=self.chunks,
        )
        golden = [
            EvidenceDocument(id=chunk_id, text=self.chunks[chunk_id])
            for chunk_id in gold_ids
            if chunk_id in self.chunks
        ]
        if not golden or not retrieved:
            return

        key = _judgment_key(
            model_id=self.model_id,
            question=question,
            golden=golden,
            retrieved=retrieved,
        )
        judgments = self.cache.setdefault("judgments", {})
        record = judgments.get(key)
        if record is not None and "error" not in record:
            return
        try:
            result = self.judge.evaluate(
                question=question["question"],
                expected_answer=question["answer"],
                golden=golden,
                retrieved=retrieved,
            )
            judgments[key] = {
                "method": method_name,
                "question_id": str(question_id),
                "question": question["question"],
                "golden_ids": [document.id for document in golden],
                "retrieved_ids": [document.id for document in retrieved],
                **result.model_dump(),
            }
        except Exception as exc:
            judgments[key] = {
                "method": method_name,
                "question_id": str(question_id),
                "question": question["question"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.cache["updated"] = datetime.now().isoformat(timespec="seconds")
        _write_json(self.cache_path, self.cache)

    def summaries(
        self,
        method_names: list[str],
        question_ids: list[str],
    ) -> dict[str, dict[str, int]]:
        judgments = self.cache.get("judgments", {})
        summaries: dict[str, dict[str, int]] = {}
        for method_name in method_names:
            summary = {
                "questions": 0,
                "strict_hits": 0,
                "equivalent_misses": 0,
            }
            rankings = self.state["methods"][method_name].get("rankings", {})
            for question_id in question_ids:
                ranked = rankings.get(str(question_id))
                question = self.questions.get(str(question_id))
                gold_ids = self.relevant_ids.get(str(question_id), [])
                if ranked is None or question is None or not gold_ids:
                    continue
                summary["questions"] += 1
                if _strict_hit(
                    ranked=ranked,
                    gold_ids=gold_ids,
                    cutoff=self.cutoff,
                ):
                    summary["strict_hits"] += 1
                    continue
                retrieved = _retrieved_documents(
                    ranked=ranked[: self.cutoff],
                    chunks=self.chunks,
                )
                golden = [
                    EvidenceDocument(id=chunk_id, text=self.chunks[chunk_id])
                    for chunk_id in gold_ids
                    if chunk_id in self.chunks
                ]
                if not golden or not retrieved:
                    continue
                key = _judgment_key(
                    model_id=self.model_id,
                    question=question,
                    golden=golden,
                    retrieved=retrieved,
                )
                record = judgments.get(key, {})
                if record.get("verdict") == "equivalent":
                    summary["equivalent_misses"] += 1
            summaries[method_name] = summary
        return summaries


def main() -> None:
    load_local_env()
    args = _parse_args()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    output_path = (ROOT / args.output).resolve()
    cache_path = (
        (ROOT / args.cache).resolve()
        if args.cache
        else output_path.with_suffix(output_path.suffix + ".checkpoint.json")
    )
    state = _load_json(checkpoint_path)
    client, model_id = _build_client(args.provider, args.model)
    judge = EvidenceEquivalenceJudge(
        client=client,
        retries=args.retries,
        max_document_chars=args.max_document_chars,
    )
    report, cache, _ = judge_checkpoint(
        state=state,
        judge=judge,
        model_id=model_id,
        cutoff=args.k,
        cache_path=cache_path,
        limit=args.limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    _write_json(cache_path, cache)
    print(report)
    print(f"\nSaved report: {output_path}")
    print(f"Saved judgments: {cache_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge whether retrieved evidence can substitute for golden chunks in "
            "strict retrieval misses."
        )
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache")
    parser.add_argument("--provider", choices=("azure", "local"), default="azure")
    parser.add_argument(
        "--model",
        help="Local Ollama model. Azure always uses AZURE_AI_MODEL.",
    )
    parser.add_argument("-k", type=int, default=5, help="Retrieval cutoff to audit.")
    parser.add_argument(
        "--limit", type=int, help="Maximum strict misses to judge per method."
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-document-chars", type=int, default=12_000)
    return parser.parse_args()


def judge_checkpoint(
    *,
    state: dict[str, Any],
    judge: EvidenceEquivalenceJudge,
    model_id: str,
    cutoff: int,
    cache_path: Path,
    limit: int | None = None,
) -> tuple[str, dict[str, Any], dict[str, dict[str, int]]]:
    if cutoff < 1:
        raise ValueError("cutoff must be at least 1")
    signature = state.get("signature", {})
    timed_ids = [str(value) for value in signature.get("timed_ids", [])]
    methods = state.get("methods", {})
    if not timed_ids or not methods:
        raise ValueError("Checkpoint has no timed questions or retrieval methods")

    initialize_page_artifacts_db()
    questions, relevant_ids, chunks = _load_evidence(set(timed_ids))
    cache = _load_cache(cache_path, model_id=model_id, cutoff=cutoff)
    judgments = cache.setdefault("judgments", {})
    summaries: dict[str, dict[str, int]] = {}

    for method_name, method_state in methods.items():
        summary = {
            "questions": 0,
            "strict_hits": 0,
            "judged_misses": 0,
            "equivalent_misses": 0,
            "near_duplicate_misses": 0,
            "related_misses": 0,
            "different_misses": 0,
            "failures": 0,
        }
        judged_for_method = 0
        rankings = method_state.get("rankings", {})
        for question_id in timed_ids:
            ranked = rankings.get(question_id)
            question = questions.get(question_id)
            gold_ids = relevant_ids.get(question_id, [])
            if ranked is None or question is None or not gold_ids:
                continue
            summary["questions"] += 1
            if _strict_hit(
                ranked=ranked,
                gold_ids=gold_ids,
                cutoff=cutoff,
            ):
                summary["strict_hits"] += 1
                continue
            if limit is not None and judged_for_method >= limit:
                continue

            retrieved = _retrieved_documents(
                ranked=ranked[:cutoff],
                chunks=chunks,
            )
            golden = [
                EvidenceDocument(id=chunk_id, text=chunks[chunk_id])
                for chunk_id in gold_ids
                if chunk_id in chunks
            ]
            if not golden or not retrieved:
                summary["failures"] += 1
                continue

            key = _judgment_key(
                model_id=model_id,
                question=question,
                golden=golden,
                retrieved=retrieved,
            )
            record = judgments.get(key)
            if record is None or "error" in record:
                try:
                    result = judge.evaluate(
                        question=question["question"],
                        expected_answer=question["answer"],
                        golden=golden,
                        retrieved=retrieved,
                    )
                    record = {
                        "method": method_name,
                        "question_id": question_id,
                        "question": question["question"],
                        "golden_ids": [document.id for document in golden],
                        "retrieved_ids": [document.id for document in retrieved],
                        **result.model_dump(),
                    }
                    judgments[key] = record
                    _write_json(cache_path, cache)
                except Exception as exc:
                    record = {
                        "method": method_name,
                        "question_id": question_id,
                        "question": question["question"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    judgments[key] = record
                    _write_json(cache_path, cache)
            judged_for_method += 1
            if "error" in record:
                summary["failures"] += 1
                continue
            summary["judged_misses"] += 1
            verdict = str(record["verdict"])
            summary[f"{verdict}_misses"] += 1
            if record.get("near_duplicate_ids"):
                summary["near_duplicate_misses"] += 1
        summaries[method_name] = summary

    cache["updated"] = datetime.now().isoformat(timespec="seconds")
    return (
        _format_report(
            checkpoint=state,
            model_id=model_id,
            cutoff=cutoff,
            summaries=summaries,
            judgments=judgments,
        ),
        cache,
        summaries,
    )


def _load_evidence(
    timed_ids: set[str],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, list[str]],
    dict[str, str],
]:
    with connect_pages() as conn:
        questions = {
            str(row["id"]): {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in conn.execute("select id, question, answer from eval_questions")
            if str(row["id"]) in timed_ids
        }
        relevant_ids: dict[str, list[str]] = {}
        for row in conn.execute(
            "select question_id, chunk_id from eval_relevant_chunks order by question_id, chunk_id"
        ):
            question_id = str(row["question_id"])
            if question_id in timed_ids:
                relevant_ids.setdefault(question_id, []).append(str(row["chunk_id"]))
        rows = conn.execute("select id, page_id, text from page_chunks").fetchall()
        chunks = {str(row["id"]): str(row["text"]) for row in rows}
    return questions, relevant_ids, chunks


def _strict_hit(
    *,
    ranked: list[str],
    gold_ids: list[str],
    cutoff: int,
) -> bool:
    return bool(set(gold_ids).intersection(ranked[:cutoff]))


def _retrieved_documents(
    *,
    ranked: list[str],
    chunks: dict[str, str],
) -> list[EvidenceDocument]:
    return [
        EvidenceDocument(id=item, text=chunks[item])
        for item in ranked
        if item in chunks
    ]


def _judgment_key(
    *,
    model_id: str,
    question: dict[str, str],
    golden: list[EvidenceDocument],
    retrieved: list[EvidenceDocument],
) -> str:
    payload = {
        "prompt_version": 1,
        "model": model_id,
        "question": question,
        "golden": [(document.id, document.text) for document in golden],
        "retrieved": [(document.id, document.text) for document in retrieved],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(path: Path, *, model_id: str, cutoff: int) -> dict[str, Any]:
    if path.exists():
        cache = _load_json(path)
        if cache.get("model") == model_id and cache.get("cutoff") == cutoff:
            return cache
    return {
        "version": 1,
        "model": model_id,
        "cutoff": cutoff,
        "judgments": {},
    }


def _format_report(
    *,
    checkpoint: dict[str, Any],
    model_id: str,
    cutoff: int,
    summaries: dict[str, dict[str, int]],
    judgments: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Retrieval evidence-equivalence audit",
        "",
        f"Model: {model_id}",
        f"Cutoff: {cutoff}",
        f"Source scoring unit: {checkpoint.get('signature', {}).get('scoring_unit', 'chunk')}",
        "",
        (
            "Strict metrics remain authoritative. Judge-adjusted hit is diagnostic: "
            "it counts strict misses whose retrieved evidence can replace the golden evidence."
        ),
        "",
        "| method | assessed | strict hit | judged misses | equivalent misses | near-duplicate misses | judge-adjusted hit | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        assessed = summary["questions"]
        strict_rate = summary["strict_hits"] / assessed if assessed else 0.0
        adjusted = (
            (summary["strict_hits"] + summary["equivalent_misses"]) / assessed
            if assessed
            else 0.0
        )
        lines.append(
            f"| {method} | {assessed} | {strict_rate:.3f} | "
            f"{summary['judged_misses']} | {summary['equivalent_misses']} | "
            f"{summary['near_duplicate_misses']} | {adjusted:.3f} | "
            f"{summary['failures']} |"
        )

    examples = [
        record for record in judgments.values() if record.get("verdict") == "equivalent"
    ]
    if examples:
        lines.extend(["", "## Equivalent strict-miss examples", ""])
        for record in examples[:20]:
            lines.extend(
                [
                    f"- **{record.get('method')} / {record.get('question_id')}**: "
                    f"{record.get('question')}",
                    f"  - Near duplicates: {', '.join(record.get('near_duplicate_ids', [])) or 'collective evidence only'}",
                    f"  - Reason: {record.get('reason')}",
                ]
            )
    return "\n".join(lines)


def _build_client(provider: str, model: str | None):
    if provider == "azure":
        client = AzureFoundryStructuredLlm.from_env(
            timeout_seconds=180,
            max_tokens=1024,
        )
        return client, f"azure:{os.environ['AZURE_AI_MODEL']}"
    model_id = model or DEFAULT_LOCAL_MODEL
    return (
        LocalOllamaStructuredLlm(
            model_id=model_id,
            reasoning=False,
            num_ctx=32_768,
            num_predict=1024,
        ),
        f"ollama:{model_id}",
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_cli(main)
