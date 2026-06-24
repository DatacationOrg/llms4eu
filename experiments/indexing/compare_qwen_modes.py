from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Any

from _cli import run_cli
from src.db.pages import initialize_page_artifacts_db
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.evaluate import EvalRun, format_eval_report, load_eval_rows
from src.eval.metrics import score_rankings
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.shared.env import load_yaml

DEFAULT_OUTPUT = Path("docs/retrieval-results.md")
CONFIG = load_yaml(Path("experiments/indexing/config.yaml"))


def main() -> None:
    args = _parse_args()
    method_names = _resolve_methods(args.methods)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else _default_checkpoint_path(output_path)
    )
    warmup = _resolve_warmup(args.warmup, args.limit)
    run = _run_eval_with_checkpoint(
        method_names=method_names,
        limit=args.limit,
        category=args.category,
        warmup=warmup,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        resume=not args.no_resume,
        save_every=max(1, args.save_every),
    )
    report = _format_report(
        run,
        method_names=method_names,
        output_path=output_path,
        include_categories=args.category is None,
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval methods with live report and checkpoint resume.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated retriever names to compare. "
            "Use 'all' (default) to include every registered method."
        ),
    )
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=(
            "Optional number of warmup questions. If omitted, defaults to 0 when "
            "--limit is set and to experiments/indexing/config.yaml:default_warmup "
            "for full runs."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output file path for the full text report. Defaults to "
            "docs/retrieval-results.md."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Optional checkpoint file path used for crash-safe resume. "
            "Defaults to <output>.checkpoint.json."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start from scratch.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Persist checkpoint every N timed queries per method (default: 10).",
    )
    return parser.parse_args()


def _resolve_methods(raw_methods: str) -> list[str]:
    if raw_methods.strip().lower() == "all":
        return list_retrievers()

    methods = [name.strip() for name in raw_methods.split(",") if name.strip()]
    if not methods:
        raise ValueError(
            "No methods provided. Use --methods all or a comma-separated list."
        )

    available = set(list_retrievers())
    unknown = sorted(set(methods) - available)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _format_report(
    run,
    method_names: list[str],
    output_path: Path,
    include_categories: bool,
) -> str:
    lines = [
        f"Methods: {', '.join(method_names)}",
        f"Warmup: {run.warmup_count} queries",
        f"Output: {output_path}",
        "",
        format_eval_report(run, include_categories=include_categories),
    ]
    return "\n".join(lines)


def _resolve_warmup(requested_warmup: int | None, limit: int | None) -> int:
    if requested_warmup is not None:
        return requested_warmup
    if limit is not None:
        return 0
    return int(CONFIG["default_warmup"])


def _default_checkpoint_path(output_path: Path) -> Path:
    suffix = "".join(output_path.suffixes)
    if suffix:
        return output_path.with_name(f"{output_path.name}.checkpoint.json")
    return output_path.with_name(f"{output_path.name}.checkpoint.json")


def _run_eval_with_checkpoint(
    method_names: list[str],
    limit: int | None,
    category: str | None,
    warmup: int,
    checkpoint_path: Path,
    output_path: Path,
    resume: bool,
    save_every: int,
) -> EvalRun:
    initialize_page_artifacts_db()
    questions, relevance = load_eval_rows(limit=limit, category=category)
    if not questions:
        return EvalRun([], [], [], 0, {}, {}, [], {})

    warmup = min(warmup, max(len(questions) - 1, 0))
    warmup_questions = questions[:warmup]
    timed_questions = questions[warmup:]
    signature = {
        "version": 1,
        "method_names": method_names,
        "category": category,
        "limit": limit,
        "warmup": warmup,
        "warmup_ids": [row["id"] for row in warmup_questions],
        "timed_ids": [row["id"] for row in timed_questions],
        "result_limit": EVAL_CONFIG["result_limit"],
    }
    state = _load_or_initialize_state(
        checkpoint_path=checkpoint_path,
        signature=signature,
        method_names=method_names,
        resume=resume,
    )

    _write_live_report(
        state=state,
        method_names=method_names,
        warmup=warmup,
        timed_questions=timed_questions,
        relevance=relevance,
        category=category,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )

    ensure_retrievers_ready(method_names)
    retrievers = {name: build_retriever(name) for name in method_names}

    for name in method_names:
        _run_warmup_with_checkpoint(
            method_name=name,
            retriever=retrievers[name],
            warmup_questions=warmup_questions,
            state=state,
            checkpoint_path=checkpoint_path,
        )

    _run_timed_round_robin(
        method_names=method_names,
        retrievers=retrievers,
        timed_questions=timed_questions,
        state=state,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        relevance=relevance,
        warmup=warmup,
        category=category,
        save_every=save_every,
    )

    method_rankings = {
        name: {
            int(question_id): ranked
            for question_id, ranked in state["methods"][name]["rankings"].items()
        }
        for name in method_names
    }

    timings: dict[str, dict[str, float]] = {}
    for name in method_names:
        elapsed = float(state["methods"][name]["elapsed_seconds"])
        total_queries = float(state["methods"][name].get("timed_total_queries", 0.0))
        question_count = len(timed_questions)
        queries_per_query = total_queries / question_count if question_count else 0.0
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / len(timed_questions),
            "queries_per_query": queries_per_query,
            "total_queries": total_queries,
        }

    timed_question_ids = {row["id"] for row in timed_questions}
    timed_relevance = [
        row for row in relevance if row["question_id"] in timed_question_ids
    ]
    score_names = [f"hit@{k}" for k in EVAL_CONFIG["metric_ks"]]
    score_names.append(f"recall@{max(EVAL_CONFIG['metric_ks'])}")
    score_names.append(f"mrr@{EVAL_CONFIG['mrr_k']}")
    scores = {
        name: score_rankings(
            timed_relevance,
            method_rankings[name],
            ks=tuple(EVAL_CONFIG["metric_ks"]),
            mrr_k=EVAL_CONFIG["mrr_k"],
            recall_k=max(EVAL_CONFIG["metric_ks"]),
        )
        for name in method_names
    }
    return EvalRun(
        questions=timed_questions,
        relevance=timed_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=method_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    )


def _run_warmup_with_checkpoint(
    method_name: str,
    retriever: Retriever,
    warmup_questions: list[dict[str, Any]],
    state: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    method_state = state["methods"][method_name]
    for index in range(method_state["warmup_index"], len(warmup_questions)):
        retriever.retrieve(
            warmup_questions[index]["question"], EVAL_CONFIG["result_limit"]
        )
        method_state["warmup_index"] = index + 1
        _write_state(checkpoint_path, state)


def _run_timed_round_robin(
    method_names: list[str],
    retrievers: dict[str, Retriever],
    timed_questions: list[dict[str, Any]],
    state: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    relevance: list[dict[str, Any]],
    warmup: int,
    category: str | None,
    save_every: int,
) -> None:
    total_timed = len(timed_questions)
    for name in method_names:
        state["methods"][name].setdefault("timed_total_queries", 0.0)

    while any(
        state["methods"][name]["next_index"] < total_timed for name in method_names
    ):
        for method_name in method_names:
            method_state = state["methods"][method_name]
            next_index = int(method_state["next_index"])
            if next_index >= total_timed:
                continue

            retriever = retrievers[method_name]
            row = timed_questions[next_index]
            before_queries = _read_total_queries(retriever)
            started = time.perf_counter()
            chunks = retriever.retrieve(row["question"], EVAL_CONFIG["result_limit"])
            elapsed = time.perf_counter() - started
            after_queries = _read_total_queries(retriever)

            method_state["elapsed_seconds"] += elapsed
            method_state["timed_total_queries"] += _query_delta(
                before_queries, after_queries
            )
            method_state["rankings"][str(row["id"])] = [chunk.id for chunk in chunks]
            method_state["next_index"] = next_index + 1

            _write_live_report(
                state=state,
                method_names=method_names,
                warmup=warmup,
                timed_questions=timed_questions,
                relevance=relevance,
                category=category,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
            )

            completed = int(method_state["next_index"])
            if completed % save_every == 0 or completed == total_timed:
                print(
                    f"{method_name}: processed {completed}/{total_timed} timed questions"
                )
                _write_state(checkpoint_path, state)


def _load_or_initialize_state(
    checkpoint_path: Path,
    signature: dict[str, Any],
    method_names: list[str],
    resume: bool,
) -> dict[str, Any]:
    if resume and checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if state.get("signature") == signature:
                print(f"Resuming from checkpoint: {checkpoint_path}")
                return state
            print("Ignoring incompatible checkpoint and starting fresh.")
        except json.JSONDecodeError:
            print("Checkpoint file is corrupted; starting fresh.")

    state = {
        "signature": signature,
        "methods": {
            name: {
                "warmup_index": 0,
                "next_index": 0,
                "elapsed_seconds": 0.0,
                "timed_total_queries": 0.0,
                "rankings": {},
            }
            for name in method_names
        },
    }
    _write_state(checkpoint_path, state)
    return state


def _write_state(checkpoint_path: Path, state: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _read_total_queries(retriever: Retriever) -> float | None:
    if hasattr(retriever, "total_queries"):
        return float(retriever.total_queries())
    return None


def _query_delta(before: float | None, after: float | None) -> float:
    if before is None or after is None:
        return 1.0
    delta = after - before
    if delta <= 0:
        return 1.0
    return float(delta)


def _write_live_report(
    state: dict[str, Any],
    method_names: list[str],
    warmup: int,
    timed_questions: list[dict[str, Any]],
    relevance: list[dict[str, Any]],
    category: str | None,
    output_path: Path,
    checkpoint_path: Path,
) -> None:
    report = _build_live_report(
        state=state,
        method_names=method_names,
        warmup=warmup,
        timed_questions=timed_questions,
        relevance=relevance,
        include_categories=category is None,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")


def _build_live_report(
    state: dict[str, Any],
    method_names: list[str],
    warmup: int,
    timed_questions: list[dict[str, Any]],
    relevance: list[dict[str, Any]],
    include_categories: bool,
    output_path: Path,
    checkpoint_path: Path,
) -> str:
    total_timed = len(timed_questions)
    progress_by_method = {
        name: int(state["methods"][name]["next_index"]) for name in method_names
    }
    aligned = min(progress_by_method.values()) if progress_by_method else 0

    header = [
        f"Methods: {', '.join(method_names)}",
        f"Warmup: {warmup} queries",
        f"Output: {output_path}",
        f"Checkpoint: {checkpoint_path}",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "Status: in-progress",
        "",
        "Progress",
    ]
    for name in method_names:
        header.append(
            f"- {name}: {progress_by_method[name]}/{total_timed} timed queries"
        )
    header.append(f"- aligned_for_scoring: {aligned}/{total_timed}")

    if aligned == 0:
        header.extend(
            [
                "",
                "No shared scored queries yet. Detailed metrics will appear once all methods complete at least one timed query.",
            ]
        )
        return "\n".join(header)

    partial_questions = timed_questions[:aligned]
    partial_ids = {row["id"] for row in partial_questions}
    partial_ids_as_str = {str(question_id) for question_id in partial_ids}
    partial_relevance = [row for row in relevance if row["question_id"] in partial_ids]
    partial_rankings = {
        name: {
            question_id: rankings
            for question_id, rankings in state["methods"][name]["rankings"].items()
            if question_id in partial_ids_as_str
        }
        for name in method_names
    }
    score_names = [f"hit@{k}" for k in EVAL_CONFIG["metric_ks"]]
    score_names.append(f"recall@{max(EVAL_CONFIG['metric_ks'])}")
    score_names.append(f"mrr@{EVAL_CONFIG['mrr_k']}")
    scores = {
        name: score_rankings(
            partial_relevance,
            partial_rankings[name],
            ks=tuple(EVAL_CONFIG["metric_ks"]),
            mrr_k=EVAL_CONFIG["mrr_k"],
            recall_k=max(EVAL_CONFIG["metric_ks"]),
        )
        for name in method_names
    }
    timings: dict[str, dict[str, float]] = {}
    for name in method_names:
        elapsed = float(state["methods"][name]["elapsed_seconds"])
        total_queries = float(state["methods"][name].get("timed_total_queries", 0.0))
        timings[name] = {
            "seconds": elapsed,
            "ms_per_query": elapsed * 1000 / aligned,
            "queries_per_query": total_queries / aligned,
            "total_queries": total_queries,
        }

    partial_run = EvalRun(
        questions=partial_questions,
        relevance=partial_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=partial_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    )

    body = format_eval_report(partial_run, include_categories=include_categories)
    header.extend(["", f"Evaluating {aligned} questions", "", body])
    return "\n".join(header)


if __name__ == "__main__":
    run_cli(main)
