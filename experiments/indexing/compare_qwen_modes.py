from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from datetime import datetime
from typing import Any

from _cli import run_cli
from src.db.pages import connect_pages, initialize_page_artifacts_db
from src.eval.evaluate import CONFIG as EVAL_CONFIG
from src.eval.evaluate import EvalRun, format_eval_report, load_eval_rows
from src.eval.metrics import score_rankings
from src.okf.evidence import OKFPageEvidenceRetriever, OKFSearchPageEvidenceRetriever
from src.retrieval.base import Retriever
from src.retrieval.methods import (
    build_retriever,
    ensure_retrievers_ready,
    list_retrievers,
)
from src.retrieval.retrievers.agentic import AgenticRetriever
from src.shared.env import ROOT, load_local_env, load_yaml

DEFAULT_OUTPUT = Path("docs/retrieval-results-agentic.md")
OKF_METHOD = "okf"
OKF_SEARCH_METHOD = "okf_search"
OKF_METHODS = {OKF_METHOD, OKF_SEARCH_METHOD}
CONFIG = load_yaml(Path("experiments/indexing/config.yaml"))


def main() -> None:
    load_local_env()
    args = _parse_args()
    method_names = _resolve_methods(args.methods)
    limit = args.limit
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else _default_checkpoint_path(output_path)
    )
    warmup = _resolve_warmup(args.warmup, limit)
    run, state = _run_eval_with_checkpoint(
        method_names=method_names,
        limit=limit,
        category=args.category,
        warmup=warmup,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        resume=not args.no_resume,
        save_every=max(1, args.save_every),
        okf_bundle=Path(args.okf_bundle),
    )
    report = _format_report(
        run,
        method_names=method_names,
        output_path=output_path,
        include_categories=args.category is None,
        state=state,
    )
    print(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    _save_action_logs_from_state(state, method_names, output_path)
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
    parser.add_argument(
        "--okf-bundle",
        default="data/okf/tourism",
        help="OKF bundle used when an OKF method is selected.",
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


def _agentic_retrievers() -> list[str]:
    return sorted(name for name in list_retrievers() if "agentic" in name)


def _uses_okf(method_names: list[str]) -> bool:
    return bool(OKF_METHODS.intersection(method_names))


def _build_okf_retriever(name: str, bundle_root: Path) -> Retriever:
    if name == OKF_SEARCH_METHOD:
        return OKFSearchPageEvidenceRetriever(bundle_root, retrieval_ready_only=True)
    return OKFPageEvidenceRetriever(bundle_root, retrieval_ready_only=True)


def _resolve_methods(raw_methods: str) -> list[str]:
    if raw_methods.strip().lower() == "all":
        return _agentic_retrievers()

    methods = [name.strip() for name in raw_methods.split(",") if name.strip()]
    if not methods:
        raise ValueError(
            "No methods provided. Use --methods all or a comma-separated list."
        )

    available = set(list_retrievers()) | OKF_METHODS
    unknown = sorted(set(methods) - available)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _format_report(
    run,
    method_names: list[str],
    output_path: Path,
    include_categories: bool,
    state: dict[str, Any],
) -> str:
    lines = [
        f"Methods: {', '.join(method_names)}",
        f"Scoring unit: {'source page' if _uses_okf(method_names) else 'chunk'}",
        f"Warmup: {run.warmup_count} queries",
        f"Output: {output_path}",
    ]
    if _uses_okf(method_names):
        signature = state["signature"]
        lines.extend(
            [
                f"OKF bundle: {signature['okf_bundle']}",
                "OKF retrieval-ready source-page coverage: "
                f"{signature['okf_covered_pages']} pages",
                f"OKF-eligible questions: {signature['eligible_questions']}",
                "OKF failures: "
                + ", ".join(
                    f"{name}={state['methods'][name].get('failures', 0)}"
                    for name in method_names
                    if name in OKF_METHODS
                ),
            ]
        )
    lines.extend(["", format_eval_report(run, include_categories=include_categories)])
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


def _save_action_logs(
    retrievers: dict[str, Retriever],
    output_path: Path,
) -> None:
    log: dict[str, list] = {}
    for name, retriever in retrievers.items():
        if isinstance(retriever, AgenticRetriever) and retriever.batch_stats.action_log:
            log[name] = retriever.batch_stats.action_log
    if not log:
        return
    log_path = output_path.with_name(
        output_path.stem.replace(".", "_") + "-judge-actions.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge action log saved to: {log_path}")


def _save_action_logs_from_state(
    state: dict[str, Any],
    method_names: list[str],
    output_path: Path,
) -> None:
    log = {
        name: state["methods"][name].get("action_log", [])
        for name in method_names
        if state["methods"][name].get("action_log")
    }
    if not log:
        return
    log_path = output_path.with_name(
        output_path.stem.replace(".", "_") + "-judge-actions.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Judge action log saved to: {log_path}")


def _run_eval_with_checkpoint(
    method_names: list[str],
    limit: int | None,
    category: str | None,
    warmup: int,
    checkpoint_path: Path,
    output_path: Path,
    resume: bool,
    save_every: int,
    okf_bundle: Path = Path("data/okf/tourism"),
    retrievers_out: dict[str, Retriever] | None = None,
) -> tuple[EvalRun, dict[str, Any]]:
    initialize_page_artifacts_db()
    bundle_root = (ROOT / okf_bundle).resolve()
    questions, relevance, chunk_pages, covered_page_ids = _load_comparison_rows(
        method_names=method_names,
        limit=limit,
        category=category,
        bundle_root=bundle_root,
    )
    if not questions:
        return EvalRun([], [], [], 0, {}, {}, [], {}), {}

    warmup = min(warmup, max(len(questions) - 1, 0))
    warmup_questions = questions[:warmup]
    timed_questions = questions[warmup:]
    signature = {
        "version": 4 if _uses_okf(method_names) else 1,
        "method_names": method_names,
        "category": category,
        "limit": limit,
        "warmup": warmup,
        "warmup_ids": [row["id"] for row in warmup_questions],
        "timed_ids": [row["id"] for row in timed_questions],
        "result_limit": EVAL_CONFIG["result_limit"],
        "scoring_unit": "source_page" if _uses_okf(method_names) else "chunk",
        "okf_bundle": str(bundle_root) if _uses_okf(method_names) else None,
        "okf_covered_pages": len(covered_page_ids),
        "eligible_questions": len(questions),
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
        chunk_pages=chunk_pages,
        covered_page_ids=covered_page_ids,
    )

    rag_methods = [name for name in method_names if name not in OKF_METHODS]
    ensure_retrievers_ready(rag_methods)
    retrievers = {
        name: (
            _build_okf_retriever(name, bundle_root)
            if name in OKF_METHODS
            else build_retriever(name)
        )
        for name in method_names
    }
    if retrievers_out is not None:
        retrievers_out.update(retrievers)

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
        chunk_pages=chunk_pages,
        covered_page_ids=covered_page_ids,
    )

    method_rankings = {
        name: {
            question_id: ranked
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
    scoring_relevance, scoring_rankings = _scoring_inputs(
        relevance=timed_relevance,
        rankings=method_rankings,
        method_names=method_names,
        chunk_pages=chunk_pages,
        covered_page_ids=covered_page_ids,
    )
    scores = {
        name: score_rankings(
            scoring_relevance,
            scoring_rankings[name],
            ks=tuple(EVAL_CONFIG["metric_ks"]),
            mrr_k=EVAL_CONFIG["mrr_k"],
            recall_k=max(EVAL_CONFIG["metric_ks"]),
        )
        for name in method_names
    }
    return EvalRun(
        questions=timed_questions,
        relevance=scoring_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=scoring_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    ), state


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
    chunk_pages: dict[str, str],
    covered_page_ids: set[str],
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
            before_failures = int(getattr(retriever, "failures", 0))
            action_log = _retriever_action_log(retriever)
            prev_log_len = len(action_log) if action_log is not None else 0
            started = time.perf_counter()
            chunks = retriever.retrieve(row["question"], EVAL_CONFIG["result_limit"])
            elapsed = time.perf_counter() - started
            after_queries = _read_total_queries(retriever)
            after_failures = int(getattr(retriever, "failures", 0))

            if action_log is not None:
                new_entries = action_log[prev_log_len:]
                method_state.setdefault("action_log", []).extend(new_entries)

            method_state["elapsed_seconds"] += elapsed
            method_state["timed_total_queries"] += _query_delta(
                before_queries, after_queries
            )
            method_state["failures"] = int(method_state.get("failures", 0)) + max(
                after_failures - before_failures, 0
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
                chunk_pages=chunk_pages,
                covered_page_ids=covered_page_ids,
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
            if _extend_compatible_state(state, signature, method_names):
                print(f"Extending compatible checkpoint: {checkpoint_path}")
                _write_state(checkpoint_path, state)
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
                "action_log": [],
                "failures": 0,
            }
            for name in method_names
        },
    }
    _write_state(checkpoint_path, state)
    return state


def _extend_compatible_state(
    state: dict[str, Any],
    signature: dict[str, Any],
    method_names: list[str],
) -> bool:
    previous = state.get("signature", {})
    stable_keys = (
        "category",
        "limit",
        "warmup",
        "warmup_ids",
        "timed_ids",
        "result_limit",
    )
    if any(previous.get(key) != signature.get(key) for key in stable_keys):
        return False

    existing_methods = set(state.get("methods", {}))
    if not existing_methods or not existing_methods.issubset(method_names):
        return False

    for name in method_names:
        state["methods"].setdefault(name, _empty_method_state())
    state["signature"] = signature
    return True


def _empty_method_state() -> dict[str, Any]:
    return {
        "warmup_index": 0,
        "next_index": 0,
        "elapsed_seconds": 0.0,
        "timed_total_queries": 0.0,
        "rankings": {},
        "action_log": [],
        "failures": 0,
    }


def _load_comparison_rows(
    *,
    method_names: list[str],
    limit: int | None,
    category: str | None,
    bundle_root: Path,
) -> tuple[list[dict], list[dict], dict[str, str], set[str]]:
    if not _uses_okf(method_names):
        questions, relevance = load_eval_rows(limit=limit, category=category)
        return questions, relevance, {}, set()

    okf = OKFPageEvidenceRetriever(bundle_root, retrieval_ready_only=True)
    covered_page_ids = okf.covered_page_ids
    if not covered_page_ids:
        raise RuntimeError(
            f"OKF bundle contains no source page provenance: {bundle_root}"
        )

    questions, relevance = load_eval_rows(limit=None, category=category)
    with connect_pages() as conn:
        chunk_pages = {
            str(row["id"]): str(row["page_id"])
            for row in conn.execute("select id, page_id from page_chunks")
        }
    covered_question_ids = {
        row["question_id"]
        for row in relevance
        if chunk_pages.get(row["chunk_id"]) in covered_page_ids
    }
    questions = [row for row in questions if row["id"] in covered_question_ids]
    if limit is not None:
        questions = questions[:limit]
    selected_ids = {row["id"] for row in questions}
    relevance = [row for row in relevance if row["question_id"] in selected_ids]
    return questions, relevance, chunk_pages, covered_page_ids


def _scoring_inputs(
    *,
    relevance: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    method_names: list[str],
    chunk_pages: dict[str, str],
    covered_page_ids: set[str],
) -> tuple[list[dict], dict[str, dict[str, list[str]]]]:
    if not _uses_okf(method_names):
        return relevance, rankings

    page_relevance = []
    seen_relevance = set()
    for row in relevance:
        page_id = chunk_pages.get(row["chunk_id"])
        key = (row["question_id"], page_id)
        if page_id in covered_page_ids and key not in seen_relevance:
            page_relevance.append(
                {"question_id": row["question_id"], "chunk_id": page_id}
            )
            seen_relevance.add(key)

    page_rankings = {}
    for name in method_names:
        page_rankings[name] = {}
        for question_id, ranked in rankings[name].items():
            page_ids = (
                ranked
                if name in OKF_METHODS
                else [chunk_pages[item] for item in ranked if item in chunk_pages]
            )
            page_rankings[name][question_id] = list(
                dict.fromkeys(
                    page_id for page_id in page_ids if page_id in covered_page_ids
                )
            )
    return page_relevance, page_rankings


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


def _retriever_action_log(retriever: Retriever) -> list | None:
    if isinstance(retriever, AgenticRetriever):
        return retriever.batch_stats.action_log
    action_log = getattr(retriever, "action_log", None)
    return action_log if isinstance(action_log, list) else None


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
    chunk_pages: dict[str, str],
    covered_page_ids: set[str],
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
        chunk_pages=chunk_pages,
        covered_page_ids=covered_page_ids,
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
    chunk_pages: dict[str, str],
    covered_page_ids: set[str],
) -> str:
    total_timed = len(timed_questions)
    progress_by_method = {
        name: int(state["methods"][name]["next_index"]) for name in method_names
    }
    aligned = min(progress_by_method.values()) if progress_by_method else 0

    header = [
        f"Methods: {', '.join(method_names)}",
        f"Scoring unit: {'source page' if _uses_okf(method_names) else 'chunk'}",
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
    if _uses_okf(method_names):
        header.extend(
            [
                "",
                f"OKF source-page coverage: {len(covered_page_ids)} pages",
                "OKF failures: "
                + ", ".join(
                    f"{name}={state['methods'][name].get('failures', 0)}"
                    for name in method_names
                    if name in OKF_METHODS
                ),
            ]
        )

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
    scoring_relevance, scoring_rankings = _scoring_inputs(
        relevance=partial_relevance,
        rankings=partial_rankings,
        method_names=method_names,
        chunk_pages=chunk_pages,
        covered_page_ids=covered_page_ids,
    )
    scores = {
        name: score_rankings(
            scoring_relevance,
            scoring_rankings[name],
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
        relevance=scoring_relevance,
        methods=method_names,
        warmup_count=warmup,
        rankings=scoring_rankings,
        timings=timings,
        score_names=score_names,
        scores=scores,
    )

    body = format_eval_report(partial_run, include_categories=include_categories)
    header.extend(["", f"Evaluating {aligned} questions", "", body])
    return "\n".join(header)


if __name__ == "__main__":
    run_cli(main)
